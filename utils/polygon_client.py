"""
Polygon.io API client for options data
Handles API authentication, rate limiting, and retry logic
"""

import os
import time
import requests
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

# Import environment loader
from .env_loader import get_env_var

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolygonClient:
    """Client for Polygon.io API with rate limiting and retry logic"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon client
        
        Args:
            api_key: Polygon API key. If None, will try to load from environment
        """
        self.api_key = api_key or get_env_var('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key required. Set POLYGON_API_KEY environment variable or pass to constructor.")
        
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'BackTrader-LEAPS-Strategy/1.0'
        })
        
        # Rate limiting for free plan (~4 requests per minute to leave safety headroom)
        self.requests_per_minute = 4
        self.request_times = []
        
    def _rate_limit(self):
        """Implement rate limiting for free plan"""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until we can make another request
            sleep_time = 60 - (now - self.request_times[0]) + 1
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        # Add small random jitter to avoid multiple parallel loops firing at the same time
        import random
        jitter = random.uniform(0, 0.5)  # 0-0.5 second jitter
        self.request_times.append(now + jitter)
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     max_retries: int = 3, base_delay: float = 1.0) -> Dict[str, Any]:
        """
        Make a rate-limited request to Polygon API with retry logic
        
        Args:
            endpoint: API endpoint (e.g., '/v3/reference/options/contracts')
            params: Query parameters
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
        
        Returns:
            API response as dictionary
        
        Raises:
            requests.RequestException: If all retries fail
        """
        params = params or {}
        params['apiKey'] = self.api_key
        
        for attempt in range(max_retries + 1):
            try:
                self._rate_limit()
                
                url = f"{self.base_url}{endpoint}"
                logger.info(f"Making request to {url} with params {params}")
                logger.info(f"Full URL with params: {url}?{requests.compat.urlencode(params)}")
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Log response (truncated to 150 characters)
                response_str = str(data)
                if len(response_str) > 150:
                    response_str = response_str[:147] + "..."
                logger.debug(f"Response: {response_str}")
                
                # Check for API errors
                if 'error' in data:
                    raise requests.RequestException(f"Polygon API error: {data['error']}")
                
                return data
                
            except requests.RequestException as e:
                if attempt == max_retries:
                    logger.error(f"Request failed after {max_retries + 1} attempts: {e}")
                    raise
                
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}")
                time.sleep(delay)
    
    def list_expirations(self, underlying: str, start: str, end: str, 
                        include_expired: bool = True, max_contracts: int = None) -> List[str]:
        """
        Get list of available expiration dates for an underlying within a date window.
        Uses efficient pagination: processes contracts page by page, stopping when enough data is found.
        
        Args:
            underlying: Underlying symbol (e.g., 'QQQ')
            start: Start date for expiration window (YYYY-MM-DD)
            end: End date for expiration window (YYYY-MM-DD)
            include_expired: Whether to include expired expirations
            max_contracts: Maximum number of contracts to process before stopping (None = no limit)
            
        Returns:
            List of expiration dates in YYYY-MM-DD format
        """
        endpoint = "/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "expired": "true" if include_expired else "false",
            "expiration_date.gte": start,
            "expiration_date.lte": end,
            "sort": "expiration_date",
            "order": "desc",  # Newer contracts first
            "limit": 1000,
        }
        
        logger.info(f"list_expirations called with: underlying={underlying}, start={start}, end={end}, include_expired={include_expired}")
        logger.info(f"Requesting contracts with expiration dates between {start} and {end} (descending order)")
        logger.info(f"Initial params: {params}")
        
        expirations = set()
        next_url = None
        page_count = 0
        total_contracts_processed = 0
        
        while True:
            page_count += 1
            if next_url:
                # Extract endpoint from full URL
                if next_url.startswith(self.base_url):
                    endpoint = next_url[len(self.base_url):]
                else:
                    endpoint = next_url
                params = {}  # Clear params for subsequent requests
                logger.debug(f"Page {page_count}: Using next_url endpoint")
            else:
                logger.debug(f"Page {page_count}: Using initial endpoint: {endpoint}")
            
            logger.debug(f"Page {page_count}: Making request...")
            data = self._make_request(endpoint, params)
            
            # Log response details
            results_count = len(data.get("results", []))
            logger.info(f"Page {page_count}: Got {results_count} results")
            
            # Process contracts from this page
            page_contracts_processed = 0
            page_dates = set()
            
            for contract in data.get("results", []):
                if "expiration_date" in contract:
                    exp_date = contract["expiration_date"]
                    expirations.add(exp_date)
                    page_dates.add(exp_date)
                    page_contracts_processed += 1
                    total_contracts_processed += 1
            
            # Log what we found on this page
            if page_dates:
                unique_dates = sorted(page_dates)
                logger.info(f"Page {page_count}: Found {page_contracts_processed} contracts with dates: {unique_dates[0]} to {unique_dates[-1]} ({len(unique_dates)} unique dates)")
                logger.info(f"Page {page_count}: Total contracts processed so far: {total_contracts_processed}")
            
            # Check if we have enough data
            if max_contracts and total_contracts_processed >= max_contracts:
                logger.info(f"Reached target of {max_contracts} contracts, stopping pagination")
                break
            
            # Check if we're getting dates outside our requested range
            for date_str in page_dates:
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                    if date_obj < datetime.strptime(start, '%Y-%m-%d').date() or date_obj > datetime.strptime(end, '%Y-%m-%d').date():
                        logger.warning(f"Page {page_count}: Found date {date_str} outside requested range {start} to {end}")
                except ValueError:
                    pass
            
            next_url = data.get("next_url")
            if next_url:
                logger.debug(f"Page {page_count}: Has next_url, continuing to next page...")
            else:
                logger.info(f"Page {page_count}: No next_url, stopping pagination")
                break
            
            # Safety check: don't go beyond 10 pages to avoid infinite loops
            if page_count >= 10:
                logger.warning(f"Reached maximum page limit ({page_count}), stopping pagination")
                break
        
        final_expirations = sorted(list(expirations))
        logger.info(f"Total unique expiration dates found: {len(final_expirations)}")
        logger.info(f"Total contracts processed: {total_contracts_processed}")
        if final_expirations:
            logger.info(f"Final date range: {final_expirations[0]} to {final_expirations[-1]}")
        
        return final_expirations
    
    def get_options_chain(self, underlying: str, expiration_date: str, 
                           contract_type: Optional[str] = None,
                           as_of: Optional[str] = None, expired: Optional[bool] = None) -> Dict[str, Any]:
        """
        Get options chain for a specific underlying and expiration
        
        Args:
            underlying: Underlying symbol (e.g., 'QQQ')
            expiration_date: Expiration date in YYYY-MM-DD format
            contract_type: Filter by contract type ('call', 'put', or None for both)
            as_of: Discover contracts "as of" a past date (YYYY-MM-DD)
            expired: Filter by expired status (True/False/None for all)
        
        Returns:
            Options chain data with pagination handled
        """
        endpoint = "/v3/reference/options/contracts"
        params = {
            'underlying_ticker': underlying,  # Fixed: was 'underlying_asset'
            'expiration_date': expiration_date
        }
        
        if contract_type:
            params['contract_type'] = contract_type
        if as_of:
            params['as_of'] = as_of
        if expired is not None:
            params['expired'] = str(expired).lower()
        
        # Handle pagination
        all_results = []
        next_url = None
        
        while True:
            if next_url:
                # Extract endpoint from full URL
                if next_url.startswith(self.base_url):
                    endpoint = next_url[len(self.base_url):]
                else:
                    endpoint = next_url
                params = {}  # Clear params for subsequent requests
            
            data = self._make_request(endpoint, params)
            
            if 'results' in data:
                all_results.extend(data['results'])
            
            # Check for next page
            next_url = data.get('next_url')
            if not next_url:
                break
        
        # Return in same format as original, but with all pages combined
        return {
            'results': all_results,
            'status': data.get('status', 'OK'),
            'request_id': data.get('request_id')
        }
    
    def get_options_snapshot(self, option_id: str) -> Dict[str, Any]:
        """
        Get current snapshot for a specific option contract
        
        Args:
            option_id: OCC option identifier
        
        Returns:
            Option snapshot data
        """
        endpoint = f"/v3/snapshot/options/{option_id}"
        return self._make_request(endpoint)
    
    def get_options_aggregates(self, option_id: str, from_date: str, to_date: str,
                              timespan: str = 'day', multiplier: int = 1) -> Dict[str, Any]:
        """
        Get historical aggregates for an option contract
        
        Args:
            option_id: OCC option identifier
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            timespan: Timespan for aggregation ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
            multiplier: Size of the timespan multiplier
        
        Returns:
            Historical aggregates data
        """
        endpoint = f"/v2/aggs/ticker/{option_id}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        return self._make_request(endpoint)
    
    def get_options_previous_close(self, option_id: str) -> Dict[str, Any]:
        """
        Get previous day's close for an option contract
        
        Args:
            option_id: OCC option identifier
        
        Returns:
            Previous close data
        """
        endpoint = f"/v2/aggs/ticker/{option_id}/prev"
        return self._make_request(endpoint)
    
    def get_options_trades(self, option_id: str, from_date: str, to_date: str,
                          limit: int = 50000) -> Dict[str, Any]:
        """
        Get trades for an option contract
        
        Args:
            option_id: OCC option identifier
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            limit: Maximum number of trades to return
        
        Returns:
            Trades data
        """
        endpoint = f"/v3/trades/{option_id}"
        params = {
            'timestamp.gte': from_date,
            'timestamp.lte': to_date,
            'limit': limit
        }
        return self._make_request(endpoint, params)
    
    def get_options_quotes(self, option_id: str, from_date: str, to_date: str,
                          limit: int = 50000) -> Dict[str, Any]:
        """
        Get quotes for an option contract
        
        Args:
            option_id: OCC option identifier
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            limit: Maximum number of quotes to return
        
        Returns:
            Quotes data
        """
        endpoint = f"/v3/quotes/{option_id}"
        params = {
            'timestamp.gte': from_date,
            'timestamp.lte': to_date,
            'limit': limit
        }
        return self._make_request(endpoint, params)
    
    def close(self):
        """Close the session"""
        if self.session:
            self.session.close()


def get_polygon_client(api_key: Optional[str] = None) -> PolygonClient:
    """
    Factory function to get a Polygon client
    
    Args:
        api_key: Optional API key override
    
    Returns:
        Configured PolygonClient instance
    """
    return PolygonClient(api_key)


# Convenience function for simple JSON requests
def get_json(url: str, params: Optional[Dict[str, Any]] = None, 
             api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Simple function to make a JSON request to Polygon API
    
    Args:
        url: Full URL to request
        params: Query parameters
        api_key: Polygon API key
    
    Returns:
        API response as dictionary
    """
    client = get_polygon_client(api_key)
    try:
        # Extract endpoint from full URL
        if url.startswith(client.base_url):
            endpoint = url[len(client.base_url):]
        else:
            endpoint = url
        
        return client._make_request(endpoint, params)
    finally:
        client.close()

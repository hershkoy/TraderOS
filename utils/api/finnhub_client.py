"""
Finnhub API client for EPS estimates and growth data
Handles API authentication, rate limiting, and retry logic
"""

import os
import time
import requests
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import environment loader
try:
    from ..config.env_loader import get_env_var
except ImportError:
    # Fallback for direct execution
    try:
        from utils.config.env_loader import get_env_var
    except ImportError:
        from dotenv import load_dotenv
        load_dotenv()
        def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
            return os.getenv(key, default)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinnhubClient:
    """Client for Finnhub API with rate limiting and retry logic"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub client
        
        Args:
            api_key: Finnhub API key. If None, will try to load from environment
        """
        self.api_key = api_key or get_env_var('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError("Finnhub API key required. Set FINNHUB_API_KEY environment variable or pass to constructor.")
        
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BackTrader-PEG-Screener/1.0'
        })
        
        # Rate limiting - free plan is ~60 calls per minute
        # Default to 50 req/min to stay well below limit
        self.requests_per_minute = int(os.getenv("FINNHUB_REQUESTS_PER_MINUTE", "50"))
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
                     max_retries: int = 5, base_delay: float = 1.0) -> Dict[str, Any]:
        """
        Make a rate-limited request to Finnhub API with retry logic
        
        Args:
            endpoint: API endpoint (e.g., '/stock/eps-estimate')
            params: Query parameters
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
        
        Returns:
            API response as dictionary
        
        Raises:
            requests.RequestException: If all retries fail
        """
        params = params or {}
        params['token'] = self.api_key
        
        for attempt in range(max_retries + 1):
            try:
                self._rate_limit()
                
                url = f"{self.base_url}{endpoint}"
                
                # Create a safe version of params for logging (hide API key)
                safe_params = params.copy()
                if 'token' in safe_params:
                    safe_params['token'] = '***HIDDEN***'
                
                logger.debug(f"Making request to {url} with params {safe_params}")
                
                response = self.session.get(url, params=params, timeout=30)
                
                # Handle 429 rate limit errors specifically
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded (429). Waiting {retry_after} seconds before retry...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                
                data = response.json()
                
                # Log response (truncated)
                response_str = str(data)
                if len(response_str) > 200:
                    response_str = response_str[:200] + "..."
                logger.debug(f"Response: {response_str}")
                
                return data
                
            except requests.RequestException as e:
                if attempt == max_retries:
                    logger.error(f"Request failed after {max_retries + 1} attempts: {e}")
                    raise
                
                # Handle 429 errors with longer delays
                if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded (429). Waiting {retry_after} seconds before retry...")
                    time.sleep(retry_after)
                    continue
                
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}")
                time.sleep(delay)
    
    def get_eps_estimates(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get EPS estimates for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
        
        Returns:
            Dictionary containing EPS estimates, or None if not available
            Structure: {
                'data': [
                    {
                        'period': '2024-12-31',
                        'epsEstimate': 6.50,
                        'epsActual': None,
                        'numberOfAnalysts': 25,
                        ...
                    },
                    ...
                ],
                'freq': 'quarterly'
            }
        """
        try:
            data = self._make_request('/stock/eps-estimate', {'symbol': symbol.upper()})
            
            # Check if we got valid data
            if not data or 'data' not in data:
                logger.warning(f"No EPS estimates data for {symbol}")
                return None
            
            if not data.get('data'):
                logger.warning(f"Empty EPS estimates data for {symbol}")
                return None
            
            logger.info(f"Retrieved EPS estimates for {symbol}: {len(data['data'])} periods")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching EPS estimates for {symbol}: {e}")
            return None
    
    def get_eps_growth_rate(self, symbol: str, periods: int = 4) -> Optional[float]:
        """
        Calculate EPS growth rate from estimates
        
        Args:
            symbol: Stock symbol
            periods: Number of future periods to use for growth calculation (default: 4 quarters)
        
        Returns:
            Annualized EPS growth rate as a percentage (e.g., 15.5 for 15.5% growth),
            or None if insufficient data
        """
        estimates_data = self.get_eps_estimates(symbol)
        
        if not estimates_data or not estimates_data.get('data'):
            return None
        
        estimates = estimates_data['data']
        
        # Sort by period (earliest first)
        sorted_estimates = sorted(estimates, key=lambda x: x.get('period', ''))
        
        # Get the first N periods with valid estimates
        valid_estimates = [e for e in sorted_estimates[:periods] if e.get('epsEstimate') is not None]
        
        if len(valid_estimates) < 2:
            logger.warning(f"Insufficient EPS estimates for {symbol} (need at least 2, got {len(valid_estimates)})")
            return None
        
        # Calculate growth rate
        first_eps = valid_estimates[0].get('epsEstimate')
        last_eps = valid_estimates[-1].get('epsEstimate')
        
        if first_eps is None or last_eps is None or first_eps <= 0:
            logger.warning(f"Invalid EPS values for {symbol}: first={first_eps}, last={last_eps}")
            return None
        
        # Calculate compound annual growth rate (CAGR)
        # If we have quarterly data, we need to annualize
        num_periods = len(valid_estimates) - 1
        if num_periods == 0:
            return None
        
        # For quarterly data, each period is 0.25 years
        # For annual data, each period is 1 year
        freq = estimates_data.get('freq', 'quarterly')
        if freq == 'quarterly':
            years = num_periods * 0.25
        else:
            years = num_periods
        
        if years <= 0:
            return None
        
        # CAGR formula: ((last/first)^(1/years) - 1) * 100
        cagr = ((last_eps / first_eps) ** (1 / years) - 1) * 100
        
        logger.info(f"Calculated EPS growth rate for {symbol}: {cagr:.2f}% (from {first_eps:.2f} to {last_eps:.2f} over {years:.2f} years)")
        
        return cagr
    
    def close(self):
        """Close the session (cleanup)"""
        self.session.close()


def get_finnhub_client() -> FinnhubClient:
    """
    Get a configured Finnhub client instance
    
    Returns:
        FinnhubClient instance
    """
    return FinnhubClient()


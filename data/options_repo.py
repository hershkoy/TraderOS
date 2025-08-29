"""
Options Repository API Module

This module provides functions to fetch option chain data for backtesting and analysis.
It serves as the main API for accessing options data from the database.
"""

import pandas as pd
import logging
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor

from utils.database import get_database_connection

logger = logging.getLogger(__name__)

class OptionsRepository:
    """
    Repository class for accessing options data from the database.
    """
    
    def __init__(self, conn=None):
        """
        Initialize the options repository.
        
        Args:
            conn: Database connection (optional, will create one if not provided)
        """
        self.conn = conn
    
    def _get_connection(self):
        """Get database connection, creating one if needed."""
        if self.conn is None:
            return get_database_connection()
        return self.conn
    
    def get_chain_at(self, ts: datetime, underlying: str = 'QQQ') -> pd.DataFrame:
        """
        Get the complete option chain at a specific timestamp.
        
        Args:
            ts: Timestamp for the chain
            underlying: Underlying symbol (default: 'QQQ')
        
        Returns:
            DataFrame with option chain data
        """
        query = """
            SELECT 
                ts,
                underlying,
                expiration,
                strike_cents,
                option_right,
                bid,
                ask,
                last,
                volume,
                open_interest,
                iv,
                delta,
                gamma,
                theta,
                vega,
                option_id,
                multiplier,
                strike_price,
                underlying_close,
                moneyness,
                days_to_expiration
            FROM option_chain_with_underlying
            WHERE underlying = %s
              AND DATE(ts) = DATE(%s)
              AND snapshot_type = 'eod'
            ORDER BY expiration, strike_cents, option_right
        """
        
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=[underlying, ts])
            return df
        except Exception as e:
            logger.error(f"Error fetching option chain at {ts}: {e}")
            return pd.DataFrame()
    
    def select_leaps(self, ts: datetime, underlying: str = 'QQQ', 
                    delta_band: Tuple[float, float] = (0.6, 0.85),
                    moneyness_band: Tuple[float, float] = (0.9, 1.1)) -> pd.DataFrame:
        """
        Select LEAPS (long-term call options) for a given timestamp.
        
        Args:
            ts: Timestamp for selection
            underlying: Underlying symbol (default: 'QQQ')
            delta_band: Delta range for selection (default: (0.6, 0.85))
            moneyness_band: Moneyness range for fallback (default: (0.9, 1.1))
        
        Returns:
            DataFrame with LEAPS candidates
        """
        query = """
            SELECT 
                ts,
                underlying,
                expiration,
                strike_cents,
                option_right,
                bid,
                ask,
                last,
                volume,
                open_interest,
                iv,
                delta,
                gamma,
                theta,
                vega,
                option_id,
                multiplier,
                strike_price,
                underlying_close,
                moneyness,
                days_to_expiration,
                suitability_score
            FROM leaps_candidates
            WHERE underlying = %s
              AND DATE(ts) = DATE(%s)
              AND (
                  (delta IS NOT NULL AND delta BETWEEN %s AND %s)
                  OR 
                  (delta IS NULL AND moneyness BETWEEN %s AND %s)
              )
            ORDER BY 
                CASE 
                    WHEN delta IS NOT NULL THEN ABS(delta - 0.7)  -- Prefer delta around 0.7
                    ELSE ABS(moneyness - 1.0)  -- Prefer moneyness around 1.0
                END,
                days_to_expiration DESC
        """
        
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=[
                underlying, ts, delta_band[0], delta_band[1], 
                moneyness_band[0], moneyness_band[1]
            ])
            return df
        except Exception as e:
            logger.error(f"Error selecting LEAPS at {ts}: {e}")
            return pd.DataFrame()
    
    def select_short_calls(self, ts: datetime, underlying: str = 'QQQ',
                          dte_band: Tuple[int, int] = (25, 45),
                          delta_band: Tuple[float, float] = (0.15, 0.35),
                          moneyness_band: Tuple[float, float] = (1.02, 1.08)) -> pd.DataFrame:
        """
        Select short-term call options for covered calls.
        
        Args:
            ts: Timestamp for selection
            underlying: Underlying symbol (default: 'QQQ')
            dte_band: Days to expiration range (default: (25, 45))
            delta_band: Delta range for selection (default: (0.15, 0.35))
            moneyness_band: Moneyness range for fallback (default: (1.02, 1.08))
        
        Returns:
            DataFrame with short call candidates
        """
        query = """
            SELECT 
                ts,
                underlying,
                expiration,
                strike_cents,
                option_right,
                bid,
                ask,
                last,
                volume,
                open_interest,
                iv,
                delta,
                gamma,
                theta,
                vega,
                option_id,
                multiplier,
                strike_price,
                underlying_close,
                moneyness,
                days_to_expiration,
                suitability_score
            FROM short_call_candidates
            WHERE underlying = %s
              AND DATE(ts) = DATE(%s)
              AND days_to_expiration BETWEEN %s AND %s
              AND (
                  (delta IS NOT NULL AND delta BETWEEN %s AND %s)
                  OR 
                  (delta IS NULL AND moneyness BETWEEN %s AND %s)
              )
            ORDER BY 
                CASE 
                    WHEN delta IS NOT NULL THEN ABS(delta - 0.25)  -- Prefer delta around 0.25
                    ELSE ABS(moneyness - 1.05)  -- Prefer moneyness around 1.05
                END,
                days_to_expiration ASC
        """
        
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=[
                underlying, ts, dte_band[0], dte_band[1],
                delta_band[0], delta_band[1], 
                moneyness_band[0], moneyness_band[1]
            ])
            return df
        except Exception as e:
            logger.error(f"Error selecting short calls at {ts}: {e}")
            return pd.DataFrame()
    
    def get_pmcc_candidates(self, ts: datetime, underlying: str = 'QQQ') -> Dict[str, pd.DataFrame]:
        """
        Get both LEAPS and short call candidates for PMCC strategy.
        
        Args:
            ts: Timestamp for selection
            underlying: Underlying symbol (default: 'QQQ')
        
        Returns:
            Dictionary with 'leaps' and 'short_calls' DataFrames
        """
        leaps = self.select_leaps(ts, underlying)
        short_calls = self.select_short_calls(ts, underlying)
        
        return {
            'leaps': leaps,
            'short_calls': short_calls
        }
    
    def get_option_by_id(self, option_id: str, ts: datetime) -> Optional[pd.Series]:
        """
        Get a specific option by ID and timestamp.
        
        Args:
            option_id: Option identifier
            ts: Timestamp
        
        Returns:
            Series with option data or None if not found
        """
        query = """
            SELECT 
                ts,
                underlying,
                expiration,
                strike_cents,
                option_right,
                bid,
                ask,
                last,
                volume,
                open_interest,
                iv,
                delta,
                gamma,
                theta,
                vega,
                option_id,
                multiplier,
                strike_price,
                underlying_close,
                moneyness,
                days_to_expiration
            FROM option_chain_with_underlying
            WHERE option_id = %s
              AND DATE(ts) = DATE(%s)
        """
        
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=[option_id, ts])
            if not df.empty:
                return df.iloc[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching option {option_id} at {ts}: {e}")
            return None
    
    def get_historical_quotes(self, option_id: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get historical quotes for a specific option.
        
        Args:
            option_id: Option identifier
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with historical quotes
        """
        query = """
            SELECT 
                q.ts,
                q.bid,
                q.ask,
                q.last,
                q.volume,
                q.open_interest,
                q.iv,
                q.delta,
                q.gamma,
                q.theta,
                q.vega,
                c.underlying,
                c.expiration,
                c.strike_cents,
                c.option_right,
                c.multiplier,
                (c.strike_cents / 100.0) as strike_price,
                m.close as underlying_close
            FROM option_quotes q
            JOIN option_contracts c USING (option_id)
            LEFT JOIN market_data m ON (
                m.symbol = c.underlying 
                AND DATE(m.ts) = DATE(q.ts)
                AND m.timeframe = '1d'
            )
            WHERE q.option_id = %s
              AND DATE(q.ts) BETWEEN %s AND %s
              AND q.snapshot_type = 'eod'
            ORDER BY q.ts
        """
        
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=[option_id, start_date, end_date])
            return df
        except Exception as e:
            logger.error(f"Error fetching historical quotes for {option_id}: {e}")
            return pd.DataFrame()
    
    def get_available_dates(self, underlying: str = 'QQQ', start_date: date = None, end_date: date = None) -> List[date]:
        """
        Get list of available trading dates with options data.
        
        Args:
            underlying: Underlying symbol (default: 'QQQ')
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
        
        Returns:
            List of available dates
        """
        query = """
            SELECT DISTINCT DATE(ts) as trade_date
            FROM option_quotes q
            JOIN option_contracts c USING (option_id)
            WHERE c.underlying = %s
              AND q.snapshot_type = 'eod'
        """
        
        params = [underlying]
        
        if start_date:
            query += " AND DATE(ts) >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(ts) <= %s"
            params.append(end_date)
        
        query += " ORDER BY trade_date"
        
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            # Convert to date objects if needed
            if 'trade_date' in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                    return df['trade_date'].dt.date.tolist()
                else:
                    return df['trade_date'].tolist()
            return []
        except Exception as e:
            logger.error(f"Error fetching available dates for {underlying}: {e}")
            return []

# Convenience functions for backward compatibility
def get_chain_at(ts: datetime, underlying: str = 'QQQ') -> pd.DataFrame:
    """Get option chain at specific timestamp."""
    repo = OptionsRepository()
    return repo.get_chain_at(ts, underlying)

def select_leaps(ts: datetime, underlying: str = 'QQQ', 
                delta_band: Tuple[float, float] = (0.6, 0.85)) -> pd.DataFrame:
    """Select LEAPS candidates."""
    repo = OptionsRepository()
    return repo.select_leaps(ts, underlying, delta_band)

def select_short_calls(ts: datetime, underlying: str = 'QQQ',
                      dte_band: Tuple[int, int] = (25, 45),
                      delta_band: Tuple[float, float] = (0.15, 0.35)) -> pd.DataFrame:
    """Select short call candidates."""
    repo = OptionsRepository()
    return repo.select_short_calls(ts, underlying, dte_band, delta_band)

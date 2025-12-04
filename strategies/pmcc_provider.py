#!/usr/bin/env python3
"""
PMCC (Poor Man's Covered Call) Strategy Provider

This module provides a data provider for backtesting PMCC strategies using
the options data repository. It selects LEAPS and short call candidates
for each trading day and provides pricing information.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from data.options_repo import OptionsRepository
from utils.backtesting.assignment import AssignmentRiskHelper

logger = logging.getLogger(__name__)


class PMCCProvider:
    """
    Data provider for PMCC strategy backtesting.
    
    This provider integrates with the options data repository to select
    LEAPS and short call candidates for each trading day, providing
    pricing and risk assessment for backtesting.
    """
    
    def __init__(self, db_config: Dict[str, str], underlying: str = 'QQQ'):
        """
        Initialize the PMCC provider.
        
        Args:
            db_config: Database connection configuration
            underlying: Underlying symbol (default: 'QQQ')
        """
        self.db_config = db_config
        self.underlying = underlying
        self.options_repo = OptionsRepository(db_config)
        self.assignment_helper = AssignmentRiskHelper()
        
        # Strategy parameters
        self.leaps_delta_band = (0.6, 0.85)
        self.short_call_dte_band = (25, 45)
        self.short_call_delta_band = (0.15, 0.35)
        self.spread_haircut_pct = 0.5  # 0.5% haircut on bid-ask spread
        
    def get_available_dates(self) -> List[date]:
        """
        Get list of available trading dates with options data.
        
        Returns:
            List of available dates
        """
        return self.options_repo.get_available_dates(self.underlying)
    
    def get_pmcc_candidates(self, ts: datetime) -> Dict[str, Any]:
        """
        Get PMCC candidates for a specific trading day.
        
        Args:
            ts: Trading day timestamp
            
        Returns:
            Dictionary with LEAPS and short call candidates
        """
        try:
            # Get LEAPS candidates
            leaps_df = self.options_repo.select_leaps(
                ts, 
                delta_band=self.leaps_delta_band
            )
            
            # Get short call candidates
            short_calls_df = self.options_repo.select_short_calls(
                ts,
                dte_band=self.short_call_dte_band,
                delta_band=self.short_call_delta_band
            )
            
            # Select best candidates
            leaps_candidate = self._select_best_leaps(leaps_df)
            short_call_candidate = self._select_best_short_call(short_calls_df)
            
            # Calculate pricing and risk metrics
            result = {
                'date': ts,
                'underlying': self.underlying,
                'leaps': leaps_candidate,
                'short_call': short_call_candidate,
                'strategy_metrics': self._calculate_strategy_metrics(
                    leaps_candidate, short_call_candidate
                )
            }
            
            logger.info(f"PMCC candidates for {ts.date()}: "
                       f"LEAPS={leaps_candidate.get('option_id', 'None')}, "
                       f"Short Call={short_call_candidate.get('option_id', 'None')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting PMCC candidates for {ts}: {e}")
            return {
                'date': ts,
                'underlying': self.underlying,
                'leaps': None,
                'short_call': None,
                'strategy_metrics': self._calculate_strategy_metrics(None, None),
                'error': str(e)
            }
    
    def _select_best_leaps(self, leaps_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Select the best LEAPS candidate from available options.
        
        Args:
            leaps_df: DataFrame with LEAPS candidates
            
        Returns:
            Best LEAPS candidate or None
        """
        if leaps_df.empty:
            return None
        
        # Sort by delta (closer to 0.75 is better) and liquidity
        leaps_df = leaps_df.copy()
        leaps_df['delta_deviation'] = abs(leaps_df['delta'] - 0.75)
        leaps_df['liquidity_score'] = (
            leaps_df['volume'].fillna(0) + 
            leaps_df['open_interest'].fillna(0) * 0.1
        )
        
        # Sort by delta deviation (lower is better) and liquidity (higher is better)
        leaps_df = leaps_df.sort_values(['delta_deviation', 'liquidity_score'], 
                                       ascending=[True, False])
        
        best_leaps = leaps_df.iloc[0].to_dict()
        
        # Calculate pricing
        best_leaps['mid_price'] = (best_leaps['bid'] + best_leaps['ask']) / 2
        best_leaps['spread_pct'] = (
            (best_leaps['ask'] - best_leaps['bid']) / best_leaps['mid_price'] * 100
        )
        best_leaps['fill_price'] = best_leaps['mid_price'] * (1 + self.spread_haircut_pct / 100)
        
        return best_leaps
    
    def _select_best_short_call(self, short_calls_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Select the best short call candidate from available options.
        
        Args:
            short_calls_df: DataFrame with short call candidates
            
        Returns:
            Best short call candidate or None
        """
        if short_calls_df.empty:
            return None
        
        # Sort by delta (closer to 0.25 is better) and liquidity
        short_calls_df = short_calls_df.copy()
        short_calls_df['delta_deviation'] = abs(short_calls_df['delta'] - 0.25)
        short_calls_df['liquidity_score'] = (
            short_calls_df['volume'].fillna(0) + 
            short_calls_df['open_interest'].fillna(0) * 0.1
        )
        
        # Sort by delta deviation (lower is better) and liquidity (higher is better)
        short_calls_df = short_calls_df.sort_values(['delta_deviation', 'liquidity_score'], 
                                                   ascending=[True, False])
        
        best_short_call = short_calls_df.iloc[0].to_dict()
        
        # Calculate pricing
        best_short_call['mid_price'] = (best_short_call['bid'] + best_short_call['ask']) / 2
        best_short_call['spread_pct'] = (
            (best_short_call['ask'] - best_short_call['bid']) / best_short_call['mid_price'] * 100
        )
        best_short_call['fill_price'] = best_short_call['mid_price'] * (1 - self.spread_haircut_pct / 100)
        
        # Calculate assignment risk
        days_to_exp = (best_short_call['expiration'] - best_short_call['ts'].date()).days
        assignment_risk = self.assignment_helper.assess_assignment_risk(
            delta=best_short_call['delta'],
            moneyness=best_short_call['moneyness'],
            days_to_exp=days_to_exp,
            ex_dividend_dates=[]
        )
        best_short_call['assignment_risk'] = assignment_risk
        
        return best_short_call
    
    def _calculate_strategy_metrics(self, leaps: Optional[Dict], short_call: Optional[Dict]) -> Dict[str, Any]:
        """
        Calculate strategy-level metrics for the PMCC position.
        
        Args:
            leaps: LEAPS candidate
            short_call: Short call candidate
            
        Returns:
            Strategy metrics dictionary
        """
        if not leaps or not short_call:
            return {
                'net_debit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'breakeven': 0,
                'probability_of_profit': 0
            }
        
        # Calculate position metrics
        leaps_cost = leaps['fill_price'] * 100  # 100 shares equivalent
        short_call_credit = short_call['fill_price'] * 100
        
        net_debit = leaps_cost - short_call_credit
        
        # Calculate profit/loss scenarios
        leaps_strike = leaps['strike_cents'] / 100
        short_strike = short_call['strike_cents'] / 100
        
        # Max profit occurs when underlying is at short call strike
        max_profit = (short_strike - leaps_strike) * 100 - net_debit
        
        # Max loss is the net debit (if underlying goes to zero)
        max_loss = net_debit
        
        # Breakeven is leaps strike + net debit per share
        breakeven = leaps_strike + (net_debit / 100)
        
        # Rough probability of profit (simplified)
        # This is a very rough estimate based on delta
        probability_of_profit = 1 - abs(leaps['delta'])
        
        return {
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'probability_of_profit': probability_of_profit,
            'leaps_cost': leaps_cost,
            'short_call_credit': short_call_credit,
            'leaps_strike': leaps_strike,
            'short_strike': short_strike
        }
    
    def get_historical_pmcc_data(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        Get historical PMCC candidates for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of PMCC candidate data for each trading day
        """
        available_dates = self.get_available_dates()
        
        # Filter dates in range
        trading_dates = [
            d for d in available_dates 
            if start_date <= d <= end_date
        ]
        
        results = []
        for trading_date in trading_dates:
            ts = datetime.combine(trading_date, datetime.min.time())
            pmcc_data = self.get_pmcc_candidates(ts)
            results.append(pmcc_data)
        
        return results
    
    def set_strategy_parameters(self, 
                               leaps_delta_band: Tuple[float, float] = None,
                               short_call_dte_band: Tuple[int, int] = None,
                               short_call_delta_band: Tuple[float, float] = None,
                               spread_haircut_pct: float = None):
        """
        Update strategy parameters.
        
        Args:
            leaps_delta_band: Delta range for LEAPS selection
            short_call_dte_band: Days to expiration range for short calls
            short_call_delta_band: Delta range for short call selection
            spread_haircut_pct: Spread haircut percentage
        """
        if leaps_delta_band:
            self.leaps_delta_band = leaps_delta_band
        if short_call_dte_band:
            self.short_call_dte_band = short_call_dte_band
        if short_call_delta_band:
            self.short_call_delta_band = short_call_delta_band
        if spread_haircut_pct is not None:
            self.spread_haircut_pct = spread_haircut_pct

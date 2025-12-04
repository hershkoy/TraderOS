"""
Assignment Risk Helper for Options Trading

This module provides functions to assess assignment risk for short options positions,
particularly for covered calls and PMCC strategies.
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DividendInfo:
    """Information about a dividend payment."""
    ex_date: date
    amount: float
    payment_date: date
    symbol: str

@dataclass
class AssignmentRisk:
    """Assessment of assignment risk for a short option."""
    is_high_risk: bool
    risk_factors: List[str]
    days_to_expiration: int
    is_itm: bool
    moneyness: float
    days_to_ex_div: Optional[int]
    recommendation: str

class AssignmentRiskHelper:
    """
    Helper class for assessing assignment risk on short options positions.
    """
    
    def __init__(self, dividend_calendar: Optional[List[DividendInfo]] = None):
        """
        Initialize the assignment risk helper.
        
        Args:
            dividend_calendar: List of dividend information (optional)
        """
        self.dividend_calendar = dividend_calendar or []
    
    def add_dividend(self, symbol: str, ex_date: date, amount: float, payment_date: date):
        """
        Add dividend information to the calendar.
        
        Args:
            symbol: Stock/ETF symbol
            ex_date: Ex-dividend date
            amount: Dividend amount per share
            payment_date: Payment date
        """
        dividend = DividendInfo(
            ex_date=ex_date,
            amount=amount,
            payment_date=payment_date,
            symbol=symbol
        )
        self.dividend_calendar.append(dividend)
    
    def get_next_dividend(self, symbol: str, current_date: date) -> Optional[DividendInfo]:
        """
        Get the next dividend for a symbol after the current date.
        
        Args:
            symbol: Stock/ETF symbol
            current_date: Current date
        
        Returns:
            Next dividend info or None if no upcoming dividends
        """
        upcoming_dividends = [
            div for div in self.dividend_calendar
            if div.symbol == symbol and div.ex_date > current_date
        ]
        
        if upcoming_dividends:
            return min(upcoming_dividends, key=lambda x: x.ex_date)
        return None
    
    def should_flag_assignment(self, short_delta: float, moneyness: float, 
                              days_to_exp: int, ex_div_calendar: Optional[List[DividendInfo]] = None,
                              symbol: str = 'QQQ', current_date: Optional[date] = None) -> bool:
        """
        Determine if a short option position should be flagged for assignment risk.
        
        Args:
            short_delta: Delta of the short option (negative for short calls)
            moneyness: Underlying price / strike price
            days_to_exp: Days to expiration
            ex_div_calendar: Dividend calendar (optional, uses instance calendar if None)
            symbol: Underlying symbol
            current_date: Current date (defaults to today)
        
        Returns:
            True if assignment risk is high
        """
        if current_date is None:
            current_date = date.today()
        
        # Use provided calendar or instance calendar
        calendar = ex_div_calendar or self.dividend_calendar
        
        # Check for immediate assignment risk (ITM and very close to expiration)
        if days_to_exp <= 3 and moneyness > 1.0:
            return True
        
        # Check for dividend-related assignment risk
        next_dividend = self.get_next_dividend(symbol, current_date)
        if next_dividend:
            days_to_ex_div = (next_dividend.ex_date - current_date).days
            
            # High risk if ITM and ex-dividend is within 5 days
            if days_to_ex_div <= 5 and moneyness > 1.0:
                return True
            
            # Moderate risk if ITM and ex-dividend is within 10 days
            if days_to_ex_div <= 10 and moneyness > 1.02:
                return True
        
        # Check for high delta (deep ITM)
        if abs(short_delta) > 0.8 and moneyness > 1.05:
            return True
        
        return False
    
    def assess_assignment_risk(self, short_delta: float, moneyness: float, 
                              days_to_exp: int, symbol: str = 'QQQ', 
                              current_date: Optional[date] = None) -> AssignmentRisk:
        """
        Comprehensive assessment of assignment risk.
        
        Args:
            short_delta: Delta of the short option (negative for short calls)
            moneyness: Underlying price / strike price
            days_to_exp: Days to expiration
            symbol: Underlying symbol
            current_date: Current date (defaults to today)
        
        Returns:
            AssignmentRisk object with detailed assessment
        """
        if current_date is None:
            current_date = date.today()
        
        risk_factors = []
        is_itm = moneyness > 1.0
        
        # Check expiration risk
        if days_to_exp <= 3:
            risk_factors.append(f"Very close to expiration ({days_to_exp} days)")
        
        # Check moneyness risk
        if moneyness > 1.05:
            risk_factors.append(f"Deep ITM (moneyness: {moneyness:.3f})")
        elif moneyness > 1.0:
            risk_factors.append(f"ITM (moneyness: {moneyness:.3f})")
        
        # Check delta risk
        if abs(short_delta) > 0.8:
            risk_factors.append(f"High delta ({short_delta:.3f})")
        elif abs(short_delta) > 0.6:
            risk_factors.append(f"Moderate delta ({short_delta:.3f})")
        
        # Check dividend risk
        next_dividend = self.get_next_dividend(symbol, current_date)
        days_to_ex_div = None
        if next_dividend:
            days_to_ex_div = (next_dividend.ex_date - current_date).days
            if days_to_ex_div <= 5 and is_itm:
                risk_factors.append(f"Ex-dividend in {days_to_ex_div} days")
            elif days_to_ex_div <= 10 and moneyness > 1.02:
                risk_factors.append(f"Ex-dividend approaching ({days_to_ex_div} days)")
        
        # Determine overall risk
        is_high_risk = self.should_flag_assignment(
            short_delta, moneyness, days_to_exp, 
            symbol=symbol, current_date=current_date
        )
        
        # Generate recommendation
        if is_high_risk:
            if days_to_exp <= 3:
                recommendation = "Close position immediately - high assignment risk"
            elif days_to_ex_div and days_to_ex_div <= 5:
                recommendation = "Consider closing before ex-dividend date"
            elif abs(short_delta) > 0.8:
                recommendation = "Consider rolling to higher strike or closing position"
            else:
                recommendation = "Monitor closely - assignment risk elevated"
        else:
            if len(risk_factors) > 0:
                recommendation = "Monitor position - some risk factors present"
            else:
                recommendation = "Low assignment risk"
        
        return AssignmentRisk(
            is_high_risk=is_high_risk,
            risk_factors=risk_factors,
            days_to_expiration=days_to_exp,
            is_itm=is_itm,
            moneyness=moneyness,
            days_to_ex_div=days_to_ex_div,
            recommendation=recommendation
        )
    
    def get_roll_recommendations(self, current_strike: float, current_expiration: date,
                                underlying_price: float, current_date: Optional[date] = None) -> Dict:
        """
        Get recommendations for rolling short options to reduce assignment risk.
        
        Args:
            current_strike: Current strike price
            current_expiration: Current expiration date
            underlying_price: Current underlying price
            current_date: Current date (defaults to today)
        
        Returns:
            Dictionary with roll recommendations
        """
        if current_date is None:
            current_date = date.today()
        
        days_to_exp = (current_expiration - current_date).days
        moneyness = underlying_price / current_strike
        
        recommendations = {
            'should_roll': False,
            'reason': '',
            'suggested_strike': None,
            'suggested_expiration': None,
            'risk_reduction': ''
        }
        
        # Determine if rolling is recommended
        if days_to_exp <= 5 and moneyness > 1.0:
            recommendations['should_roll'] = True
            recommendations['reason'] = 'Close to expiration and ITM'
            
            # Suggest higher strike for same expiration
            if moneyness > 1.05:
                recommendations['suggested_strike'] = current_strike * 1.05
                recommendations['risk_reduction'] = 'Higher strike reduces assignment probability'
        
        elif days_to_exp <= 10 and moneyness > 1.02:
            recommendations['should_roll'] = True
            recommendations['reason'] = 'ITM with approaching expiration'
            
            # Suggest rolling out in time
            new_expiration = current_expiration + timedelta(days=30)
            recommendations['suggested_expiration'] = new_expiration
            recommendations['risk_reduction'] = 'More time reduces assignment pressure'
        
        return recommendations

# Convenience functions
def should_flag_assignment(short_delta: float, moneyness: float, days_to_exp: int, 
                          ex_div_calendar: Optional[List[DividendInfo]] = None) -> bool:
    """
    Convenience function to check assignment risk.
    
    Args:
        short_delta: Delta of the short option
        moneyness: Underlying price / strike price
        days_to_exp: Days to expiration
        ex_div_calendar: Dividend calendar
    
    Returns:
        True if assignment risk is high
    """
    helper = AssignmentRiskHelper(ex_div_calendar)
    return helper.should_flag_assignment(short_delta, moneyness, days_to_exp)

def assess_assignment_risk(short_delta: float, moneyness: float, days_to_exp: int, 
                          symbol: str = 'QQQ') -> AssignmentRisk:
    """
    Convenience function to assess assignment risk.
    
    Args:
        short_delta: Delta of the short option
        moneyness: Underlying price / strike price
        days_to_exp: Days to expiration
        symbol: Underlying symbol
    
    Returns:
        AssignmentRisk object
    """
    helper = AssignmentRiskHelper()
    return helper.assess_assignment_risk(short_delta, moneyness, days_to_exp, symbol)

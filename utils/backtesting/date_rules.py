"""
Date utility functions for options trading
Handles calculation of standard option expiration dates
"""

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


def third_friday(year: int, month: int) -> date:
    """
    Find the 3rd Friday of (year, month)
    
    Args:
        year: Year (e.g., 2025)
        month: Month (1-12)
    
    Returns:
        Date of the 3rd Friday
    """
    d = date(year, month, 1)
    # weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    first_friday = d + timedelta(days=((4 - d.weekday()) % 7))
    return first_friday + timedelta(days=14)  # +2 weeks


def main_monthly_expiry_one_year_ahead(backtest_date: date) -> date:
    """
    Get the main monthly expiry (3rd Friday) exactly one year ahead of backtest date
    
    Args:
        backtest_date: The date we're backtesting from
    
    Returns:
        Date of the 3rd Friday one year ahead
    """
    # 1y ahead by calendar, then take the main monthly (3rd Friday) of that month
    one_year = backtest_date + relativedelta(years=1)
    return third_friday(one_year.year, one_year.month)


def is_friday(date_obj: date) -> bool:
    """
    Check if a date is a Friday
    
    Args:
        date_obj: Date to check
    
    Returns:
        True if Friday, False otherwise
    """
    return date_obj.weekday() == 4  # Friday = 4


def get_next_friday_after(date_obj: date) -> date:
    """
    Get the next Friday after the given date
    
    Args:
        date_obj: Starting date
    
    Returns:
        Next Friday date
    """
    days_ahead = (4 - date_obj.weekday()) % 7
    if days_ahead == 0:  # Already Friday
        days_ahead = 7
    return date_obj + timedelta(days=days_ahead)

#!/usr/bin/env python3
"""
Daily Options Data Integrity Checks

This script performs daily integrity checks on the options data pipeline
to ensure data quality and consistency.
"""

import logging
import sys
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
os.makedirs('logs/data/options', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data/options/options_data_checks.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OptionsDataChecker:
    """Performs integrity checks on options data."""
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize the data checker.
        
        Args:
            db_config: Database connection parameters
        """
        self.db_config = db_config
    
    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_config)
    
    def check_contract_count_growth(self, check_date: date) -> Dict[str, Any]:
        """
        Check if contract count growth is reasonable vs yesterday.
        
        Args:
            check_date: Date to check
            
        Returns:
            Dict with check results
        """
        yesterday = check_date - timedelta(days=1)
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get contract counts for today and yesterday
                cursor.execute("""
                    SELECT 
                        DATE(last_seen) as date,
                        COUNT(DISTINCT option_id) as contract_count
                    FROM option_contracts 
                    WHERE DATE(last_seen) IN (%s, %s)
                    GROUP BY DATE(last_seen)
                    ORDER BY date
                """, (yesterday, check_date))
                
                results = cursor.fetchall()
                
                if len(results) < 2:
                    return {
                        'status': 'WARNING',
                        'message': f'Insufficient data for comparison: {len(results)} days found',
                        'today_count': 0,
                        'yesterday_count': 0,
                        'growth_pct': 0
                    }
                
                # Find today and yesterday counts
                today_count = 0
                yesterday_count = 0
                
                for row in results:
                    if row['date'] == check_date:
                        today_count = row['contract_count']
                    elif row['date'] == yesterday:
                        yesterday_count = row['contract_count']
                
                if yesterday_count == 0:
                    growth_pct = 100 if today_count > 0 else 0
                else:
                    growth_pct = ((today_count - yesterday_count) / yesterday_count) * 100
                
                # Check if growth is reasonable (±20%)
                status = 'PASS'
                if abs(growth_pct) > 20:
                    status = 'WARNING'
                    if abs(growth_pct) > 50:
                        status = 'FAIL'
                
                return {
                    'status': status,
                    'message': f'Contract count growth: {growth_pct:.1f}%',
                    'today_count': today_count,
                    'yesterday_count': yesterday_count,
                    'growth_pct': growth_pct
                }
    
    def check_monotonic_last_seen(self, sample_size: int = 10) -> Dict[str, Any]:
        """
        Spot-check random option_ids for monotonic last_seen.
        
        Args:
            sample_size: Number of contracts to check
            
        Returns:
            Dict with check results
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get random sample of option_ids
                cursor.execute("""
                    SELECT option_id, underlying, expiration
                    FROM option_contracts 
                    ORDER BY RANDOM() 
                    LIMIT %s
                """, (sample_size,))
                
                sample_contracts = cursor.fetchall()
                
                issues = []
                checked_count = 0
                
                for contract in sample_contracts:
                    option_id = contract['option_id']
                    
                    # Check if last_seen is monotonic (should be >= first_seen)
                    cursor.execute("""
                        SELECT first_seen, last_seen
                        FROM option_contracts 
                        WHERE option_id = %s
                    """, (option_id,))
                    
                    result = cursor.fetchone()
                    if result and result['first_seen'] and result['last_seen']:
                        if result['last_seen'] < result['first_seen']:
                            issues.append({
                                'option_id': option_id,
                                'first_seen': result['first_seen'],
                                'last_seen': result['last_seen']
                            })
                        checked_count += 1
                
                status = 'PASS'
                if issues:
                    status = 'FAIL' if len(issues) > sample_size // 2 else 'WARNING'
                
                return {
                    'status': status,
                    'message': f'Monotonic last_seen check: {len(issues)} issues found in {checked_count} samples',
                    'issues': issues,
                    'checked_count': checked_count
                }
    
    def check_quotes_contract_consistency(self, check_date: date, tolerance_pct: float = 10.0) -> Dict[str, Any]:
        """
        Ensure option_quotes has reasonable number of rows vs active contracts.
        
        Args:
            check_date: Date to check
            tolerance_pct: Tolerance percentage for consistency check
            
        Returns:
            Dict with check results
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Count active contracts for the date
                cursor.execute("""
                    SELECT COUNT(DISTINCT option_id) as active_contracts
                    FROM option_contracts 
                    WHERE DATE(last_seen) >= %s
                """, (check_date,))
                
                active_contracts = cursor.fetchone()['active_contracts']
                
                # Count quotes for the date
                cursor.execute("""
                    SELECT COUNT(DISTINCT option_id) as quoted_contracts
                    FROM option_quotes 
                    WHERE DATE(ts) = %s
                """, (check_date,))
                
                quoted_contracts = cursor.fetchone()['quoted_contracts']
                
                if active_contracts == 0:
                    consistency_pct = 0
                else:
                    consistency_pct = (quoted_contracts / active_contracts) * 100
                
                # Check if consistency is within tolerance
                status = 'PASS'
                if consistency_pct < (100 - tolerance_pct):
                    status = 'WARNING'
                    if consistency_pct < 50:
                        status = 'FAIL'
                
                return {
                    'status': status,
                    'message': f'Quotes consistency: {consistency_pct:.1f}% ({quoted_contracts}/{active_contracts})',
                    'active_contracts': active_contracts,
                    'quoted_contracts': quoted_contracts,
                    'consistency_pct': consistency_pct
                }
    
    def check_data_quality(self, check_date: date) -> Dict[str, Any]:
        """
        Check for common data quality issues.
        
        Args:
            check_date: Date to check
            
        Returns:
            Dict with check results
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                issues = []
                
                # Check for contracts with NULL expiration
                cursor.execute("""
                    SELECT COUNT(*) as null_expiration_count
                    FROM option_contracts 
                    WHERE expiration IS NULL
                """)
                
                null_expiration = cursor.fetchone()['null_expiration_count']
                if null_expiration > 0:
                    issues.append(f'{null_expiration} contracts with NULL expiration')
                
                # Check for quotes with invalid bid/ask (bid > ask)
                cursor.execute("""
                    SELECT COUNT(*) as invalid_spread_count
                    FROM option_quotes 
                    WHERE DATE(ts) = %s 
                      AND bid IS NOT NULL 
                      AND ask IS NOT NULL 
                      AND bid > ask
                """, (check_date,))
                
                invalid_spread = cursor.fetchone()['invalid_spread_count']
                if invalid_spread > 0:
                    issues.append(f'{invalid_spread} quotes with bid > ask')
                
                # Check for contracts with missing underlying
                cursor.execute("""
                    SELECT COUNT(*) as null_underlying_count
                    FROM option_contracts 
                    WHERE underlying IS NULL OR underlying = ''
                """)
                
                null_underlying = cursor.fetchone()['null_underlying_count']
                if null_underlying > 0:
                    issues.append(f'{null_underlying} contracts with NULL/empty underlying')
                
                status = 'PASS'
                if issues:
                    status = 'WARNING' if len(issues) <= 2 else 'FAIL'
                
                return {
                    'status': status,
                    'message': f'Data quality check: {len(issues)} issues found',
                    'issues': issues
                }
    
    def run_all_checks(self, check_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Run all integrity checks.
        
        Args:
            check_date: Date to check (defaults to today)
            
        Returns:
            Dict with all check results
        """
        if check_date is None:
            check_date = date.today()
        
        logger.info(f"Running options data integrity checks for {check_date}")
        
        results = {
            'check_date': check_date,
            'timestamp': datetime.now(),
            'checks': {}
        }
        
        # Run all checks
        try:
            results['checks']['contract_growth'] = self.check_contract_count_growth(check_date)
            results['checks']['monotonic_last_seen'] = self.check_monotonic_last_seen()
            results['checks']['quotes_consistency'] = self.check_quotes_contract_consistency(check_date)
            results['checks']['data_quality'] = self.check_data_quality(check_date)
            
        except Exception as e:
            logger.error(f"Error running checks: {e}")
            results['error'] = str(e)
            return results
        
        # Determine overall status
        statuses = [check['status'] for check in results['checks'].values()]
        if 'FAIL' in statuses:
            overall_status = 'FAIL'
        elif 'WARNING' in statuses:
            overall_status = 'WARNING'
        else:
            overall_status = 'PASS'
        
        results['overall_status'] = overall_status
        
        # Log results
        logger.info(f"Overall status: {overall_status}")
        for check_name, check_result in results['checks'].items():
            logger.info(f"{check_name}: {check_result['status']} - {check_result['message']}")
        
        return results


def main():
    """Main function to run the data integrity checks."""
    # Database configuration (should be loaded from environment or config file)
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'backtrader_db',
        'user': 'backtrader_user',
        'password': 'backtrader_password'
    }
    
    checker = OptionsDataChecker(db_config)
    results = checker.run_all_checks()
    
    # Exit with appropriate code
    if results.get('overall_status') == 'FAIL':
        sys.exit(1)
    elif results.get('overall_status') == 'WARNING':
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()

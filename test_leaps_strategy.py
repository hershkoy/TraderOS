#!/usr/bin/env python3
"""
Test script for LEAPS strategy with mock data
"""

import sys
import os
from datetime import date

# Add project root to path
sys.path.append('.')

from strategies.pmcc_provider import PMCCProvider
from data.options_repo import OptionsRepository

def test_leaps_strategy():
    """Test the LEAPS strategy with mock data"""
    print("Testing LEAPS Strategy with Mock Data")
    print("="*50)
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'backtrader',
        'user': 'backtrader_user',
        'password': 'backtrader_password'
    }
    
    try:
        # Initialize components
        options_repo = OptionsRepository(db_config)
        pmcc_provider = PMCCProvider(options_repo)
        
        print("✅ Components initialized successfully")
        
        # Test data availability
        print("\n📊 Checking data availability...")
        
        # Check contracts
        contracts_count = options_repo.get_contracts_count()
        print(f"   Total contracts: {contracts_count}")
        
        # Check quotes
        quotes_count = options_repo.get_quotes_count()
        print(f"   Total quotes: {quotes_count}")
        
        # Check LEAPS availability
        leaps_count = options_repo.get_leaps_count()
        print(f"   LEAPS contracts: {leaps_count}")
        
        # Check short-term calls
        short_calls_count = options_repo.get_short_calls_count()
        print(f"   Short-term calls: {short_calls_count}")
        
        if contracts_count == 0:
            print("❌ No contracts found. Please run the mock data generator first.")
            return
        
        # Test PMCC candidate selection
        print("\n🎯 Testing PMCC candidate selection...")
        
        # Get today's date for testing
        test_date = date.today()
        
        # Get PMCC candidates
        candidates = pmcc_provider.get_pmcc_candidates(test_date)
        
        if candidates:
            print(f"✅ Found {len(candidates)} PMCC candidates")
            
            # Display first few candidates
            print("\n📋 Sample PMCC Candidates:")
            for i, candidate in enumerate(candidates[:5]):
                print(f"   {i+1}. LEAPS: {candidate.leaps_contract.option_id}")
                print(f"      Strike: ${candidate.leaps_contract.strike_cents/100:.2f}")
                print(f"      Delta: {candidate.leaps_contract.delta:.3f}")
                print(f"      Expiration: {candidate.leaps_contract.expiration}")
                print(f"      Short Call: {candidate.short_call.option_id}")
                print(f"      Strike: ${candidate.short_call.strike_cents/100:.2f}")
                print(f"      Delta: {candidate.short_call.delta:.3f}")
                print(f"      Expiration: {candidate.short_call.expiration}")
                print(f"      Net Debit: ${candidate.net_debit:.2f}")
                print(f"      Max Profit: ${candidate.max_profit:.2f}")
                print(f"      Max Loss: ${candidate.max_loss:.2f}")
                print(f"      Breakeven: ${candidate.breakeven:.2f}")
                print()
        else:
            print("❌ No PMCC candidates found")
            
            # Let's check what's available
            print("\n🔍 Checking available data...")
            
            # Get some LEAPS
            leaps = options_repo.select_leaps(test_date, limit=5)
            print(f"   Available LEAPS: {len(leaps)}")
            for leap in leaps[:3]:
                print(f"     - {leap.option_id}: Delta={leap.delta:.3f}, Strike=${leap.strike_cents/100:.2f}")
            
            # Get some short calls
            short_calls = options_repo.select_short_calls(test_date, limit=5)
            print(f"   Available Short Calls: {len(short_calls)}")
            for call in short_calls[:3]:
                print(f"     - {call.option_id}: Delta={call.delta:.3f}, Strike=${call.strike_cents/100:.2f}")
        
        # Test strategy metrics calculation
        print("\n📈 Testing strategy metrics calculation...")
        
        # Get a sample of contracts to test with
        sample_leaps = options_repo.select_leaps(test_date, limit=1)
        sample_calls = options_repo.select_short_calls(test_date, limit=1)
        
        if sample_leaps and sample_calls:
            leap = sample_leaps[0]
            call = sample_calls[0]
            
            # Calculate metrics manually
            net_debit = leap.fill_price - call.fill_price
            max_profit = call.strike_cents/100 - leap.strike_cents/100 - net_debit
            max_loss = net_debit
            breakeven = leap.strike_cents/100 + net_debit
            
            print(f"   Sample Strategy Metrics:")
            print(f"     Net Debit: ${net_debit:.2f}")
            print(f"     Max Profit: ${max_profit:.2f}")
            print(f"     Max Loss: ${max_loss:.2f}")
            print(f"     Breakeven: ${breakeven:.2f}")
        
        print("\n✅ LEAPS strategy test completed!")
        
    except Exception as e:
        print(f"❌ Error testing LEAPS strategy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_leaps_strategy()

#!/usr/bin/env python3
"""
Test script to verify Polygon API functionality
"""

import sys
import json

# Add project root to path
sys.path.append('.')

from utils.api.polygon_client import PolygonClient

def test_polygon_api():
    """Test Polygon API functionality"""
    print("Testing Polygon API...")
    
    try:
        # Create Polygon client
        client = PolygonClient()
        print("✅ Polygon client created successfully")
        
        # Test a simple API call - get current QQQ price
        print("\nTesting QQQ current price...")
        response = client._make_request('/v2/aggs/ticker/QQQ/prev')
        print(f"Response status: {response.get('status', 'N/A')}")
        print(f"Response keys: {list(response.keys())}")
        
        if 'results' in response and response['results']:
            result = response['results'][0]
            print(f"QQQ previous close: ${result.get('c', 'N/A')}")
        
        # Test options contracts endpoint with different parameters
        print("\nTesting options contracts endpoint...")
        
        # Try different expiration dates
        test_dates = ['2024-12-20', '2025-01-17', '2025-06-20']
        
        for test_date in test_dates:
            print(f"\nTesting date: {test_date}")
            params = {
                'underlying_ticker': 'QQQ',
                'expiration_date': test_date,
                'contract_type': 'call',
                'limit': 5
            }
            
            response = client._make_request('/v3/reference/options/contracts', params)
            print(f"Response status: {response.get('status', 'N/A')}")
            
            if 'results' in response:
                print(f"Found {len(response['results'])} contracts")
                if response['results']:
                    contract = response['results'][0]
                    print(f"Sample contract: {contract.get('ticker', 'N/A')}")
                    break
            else:
                print("No results found")
        
        # Try without contract_type filter
        print("\nTesting without contract_type filter...")
        params = {
            'underlying_ticker': 'QQQ',
            'expiration_date': '2025-01-17',
            'limit': 5
        }
        
        response = client._make_request('/v3/reference/options/contracts', params)
        print(f"Response status: {response.get('status', 'N/A')}")
        
        if 'results' in response:
            print(f"Found {len(response['results'])} contracts")
            if response['results']:
                contract = response['results'][0]
                print(f"Sample contract: {contract.get('ticker', 'N/A')}")
        else:
            print("No results found in options response")
            print(f"Full response: {json.dumps(response, indent=2)}")
        
        # Try with SPY instead of QQQ
        print("\nTesting with SPY...")
        params = {
            'underlying_ticker': 'SPY',
            'expiration_date': '2025-01-17',
            'limit': 5
        }
        
        response = client._make_request('/v3/reference/options/contracts', params)
        print(f"SPY response status: {response.get('status', 'N/A')}")
        
        if 'results' in response:
            print(f"Found {len(response['results'])} SPY contracts")
            if response['results']:
                contract = response['results'][0]
                print(f"Sample SPY contract: {contract.get('ticker', 'N/A')}")
        else:
            print("No SPY results found")
        
        # Try a different endpoint - options chain
        print("\nTesting options chain endpoint...")
        params = {
            'underlying_ticker': 'QQQ',
            'expiration_date': '2025-01-17'
        }
        
        response = client._make_request('/v3/snapshot/options/QQQ', params)
        print(f"Options chain response status: {response.get('status', 'N/A')}")
        print(f"Options chain response keys: {list(response.keys())}")
        
        if 'results' in response:
            print(f"Found {len(response['results'])} options chain results")
        else:
            print("No options chain results found")
            print(f"Full response: {json.dumps(response, indent=2)}")
        
    except Exception as e:
        print(f"❌ Error testing Polygon API: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_polygon_api()

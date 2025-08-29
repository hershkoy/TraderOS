#!/usr/bin/env python3
"""
Test script to verify environment variable loading
"""

import sys
import os

# Add project root to path
sys.path.append('.')

from utils.env_loader import get_env_var, load_env_file

def test_env_loading():
    """Test environment variable loading"""
    print("Testing environment variable loading...")
    
    # Try to load .env file
    env_loaded = load_env_file()
    print(f"Environment file loaded: {env_loaded}")
    
    # Check for Polygon API key
    polygon_key = get_env_var('POLYGON_API_KEY')
    if polygon_key:
        print(f"✅ POLYGON_API_KEY found: {polygon_key[:10]}...")
    else:
        print("❌ POLYGON_API_KEY not found")
    
    # Check for other common environment variables
    db_host = get_env_var('DB_HOST', 'localhost')
    db_port = get_env_var('DB_PORT', '5432')
    db_name = get_env_var('DB_NAME', 'backtrader')
    db_user = get_env_var('DB_USER', 'backtrader_user')
    
    print(f"Database config:")
    print(f"  Host: {db_host}")
    print(f"  Port: {db_port}")
    print(f"  Database: {db_name}")
    print(f"  User: {db_user}")
    
    # List all environment variables that start with POLYGON or DB
    print("\nAll environment variables:")
    for key, value in os.environ.items():
        if key.startswith(('POLYGON', 'DB')):
            print(f"  {key}: {value[:20]}{'...' if len(value) > 20 else ''}")

if __name__ == '__main__':
    test_env_loading()

#!/usr/bin/env python3
"""
TimescaleDB Setup Script for BackTrader Framework
Helps users set up and test TimescaleDB integration
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Docker found: {result.stdout.strip()}")
            return True
        else:
            print("✗ Docker not found or not working")
            return False
    except FileNotFoundError:
        print("✗ Docker not installed")
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Docker Compose found: {result.stdout.strip()}")
            return True
        else:
            # Try docker compose (newer version)
            result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Docker Compose found: {result.stdout.strip()}")
                return True
            else:
                print("✗ Docker Compose not found")
                return False
    except FileNotFoundError:
        print("✗ Docker Compose not installed")
        return False

def start_timescaledb():
    """Start TimescaleDB using Docker Compose"""
    print("\n🚀 Starting TimescaleDB...")
    
    try:
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Start the services
        result = subprocess.run(['docker-compose', 'up', '-d'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ TimescaleDB started successfully")
            return True
        else:
            print(f"✗ Failed to start TimescaleDB: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error starting TimescaleDB: {e}")
        return False

def wait_for_timescaledb():
    """Wait for TimescaleDB to be ready"""
    print("⏳ Waiting for TimescaleDB to be ready...")
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            result = subprocess.run([
                'docker-compose', 'exec', '-T', 'timescaledb', 
                'pg_isready', '-U', 'backtrader_user', '-d', 'backtrader'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ TimescaleDB is ready!")
                return True
            else:
                print(f"⏳ Attempt {attempt + 1}/{max_attempts}: TimescaleDB not ready yet...")
                time.sleep(2)
        except Exception as e:
            print(f"⏳ Attempt {attempt + 1}/{max_attempts}: Error checking readiness: {e}")
            time.sleep(2)
    
    print("✗ TimescaleDB failed to start within timeout")
    return False

def test_connection():
    """Test the TimescaleDB connection"""
    print("\n🔍 Testing TimescaleDB connection...")
    
    try:
        from utils.db.timescaledb_client import test_connection
        if test_connection():
            print("✓ TimescaleDB connection test successful")
            return True
        else:
            print("✗ TimescaleDB connection test failed")
            return False
    except ImportError as e:
        print(f"✗ Cannot import TimescaleDB client: {e}")
        print("Make sure you have installed the requirements: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"✗ Connection test error: {e}")
        return False

def show_status():
    """Show the status of TimescaleDB services"""
    print("\n📊 TimescaleDB Status:")
    
    try:
        result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("✗ Failed to get service status")
    except Exception as e:
        print(f"✗ Error getting status: {e}")

def show_usage():
    """Show usage examples"""
    print("\n📖 Usage Examples:")
    print("\n1. Fetch data from Alpaca:")
    print("   python utils/fetch_data.py --symbol NFLX --provider alpaca --timeframe 1h --bars 1000")
    
    print("\n2. Run backtest with TimescaleDB:")
    print("   python backtrader_runner_yaml.py --symbol NFLX --provider ALPACA --timeframe 1h --strategy mean_reversion")
    
    print("\n3. List available data:")
    print("   python -c \"from utils.db.timescaledb_loader import get_available_data; print(get_available_data())\"")
    
    print("\n4. Access pgAdmin (optional):")
    print("   Open http://localhost:8080 in your browser")
    print("   Email: admin@backtrader.com")
    print("   Password: admin")

def main():
    """Main setup function"""
    print("🔧 TimescaleDB Setup for BackTrader Framework")
    print("=" * 50)
    
    # Check prerequisites
    print("\n1. Checking prerequisites...")
    if not check_docker():
        print("\n❌ Docker is required but not found.")
        print("Please install Docker from: https://docs.docker.com/get-docker/")
        return False
    
    if not check_docker_compose():
        print("\n❌ Docker Compose is required but not found.")
        print("Please install Docker Compose or ensure it's available with your Docker installation.")
        return False
    
    # Start TimescaleDB
    print("\n2. Starting TimescaleDB...")
    if not start_timescaledb():
        print("\n❌ Failed to start TimescaleDB.")
        return False
    
    # Wait for it to be ready
    if not wait_for_timescaledb():
        print("\n❌ TimescaleDB failed to start properly.")
        return False
    
    # Test connection
    print("\n3. Testing connection...")
    if not test_connection():
        print("\n❌ Connection test failed.")
        print("You may need to install requirements: pip install -r requirements.txt")
        return False
    
    # Show status
    show_status()
    
    # Show usage
    show_usage()
    
    print("\n✅ TimescaleDB setup completed successfully!")
    print("\nNext steps:")
    print("1. Fetch some data using the fetch_data.py script")
    print("2. Run backtests using the new --symbol parameter")
    print("3. Access pgAdmin at http://localhost:8080 for database management")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

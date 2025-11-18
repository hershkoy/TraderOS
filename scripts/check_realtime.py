import argparse
import os
from ib_insync import *

def main():
    parser = argparse.ArgumentParser(description='Check real-time market data from IB')
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='IB Gateway/TWS port number (default: from IB_PORT env var or 7496). Common ports: 4001 (Gateway paper), 4002 (Gateway live), 7497 (TWS paper), 7496 (TWS live)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Symbol to check (default: AAPL)'
    )
    
    args = parser.parse_args()
    
    # Get port from argument, environment variable, or default
    if args.port is None:
        port = int(os.getenv("IB_PORT", "7496"))
    else:
        port = args.port
    
    HOST = '127.0.0.1'
    CLIENT_ID = 1
    
    ib = IB()
    print(f"Connecting to {HOST}:{port} with client ID {CLIENT_ID}...")
    ib.connect(HOST, port, clientId=CLIENT_ID, timeout=10)
    
    ib.reqMarketDataType(1)  # request real-time data
    
    contract = Stock(args.symbol, 'SMART', 'USD')
    ticker = ib.reqMktData(contract, '', False, False)
    
    ib.sleep(2)
    print(ticker)
    
    ib.disconnect()
    print("\nDisconnected.")

if __name__ == '__main__':
    main()

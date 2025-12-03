import argparse
import os
from ib_insync import IB, util, Stock 

def main():
    parser = argparse.ArgumentParser(description='Test IB connection and market data')
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='IB Gateway/TWS port number (default: from IB_PORT env var or 4002). Common ports: 4001 (Gateway paper), 4002 (Gateway live), 7497 (TWS paper), 7496 (TWS live)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Symbol to test (default: AAPL)'
    )
    
    args = parser.parse_args()
    
    # Get port from argument, environment variable, or default
    if args.port is None:
        port = int(os.getenv("IB_PORT", "4002"))
    else:
        port = args.port
    
    HOST = '127.0.0.1'
    CLIENT_ID = 1
    
    util.logToConsole()  # see low-level messages
    
    ib = IB()
    print(f"Connecting to {HOST}:{port} with client ID {CLIENT_ID}...")
    ib.connect(HOST, port, clientId=CLIENT_ID, timeout=10)
    print('Connected:', ib.isConnected())
    
    
    # --- Request Server Time ---
    print("\n=== Server Time ===")
    print(ib.reqCurrentTime())
    
    
    # --- TEST: Request contract details ---
    contract = Stock(args.symbol, 'SMART', 'USD')
    
    print(f"\n=== Contract Details ===")
    details = ib.reqContractDetails(contract)
    for d in details:
        print(d)
    
    if not details:
        print("❌ Contract lookup failed — cannot request market data.")
        ib.disconnect()
        raise SystemExit()
    
    # --- TEST: Request market data snapshot ---
    print(f"\n=== Market Data Snapshot for {args.symbol} ===")
    
    # Market data snapshot (reqMktData) returns quickly; needs US market hours or snapshot permissions
    ticker = ib.reqMktData(contract, snapshot=True)
    
    # wait for snapshot
    ib.sleep(2)
    
    print("Last Price:", ticker.last)
    print("Bid:", ticker.bid)
    print("Ask:", ticker.ask)
    print("Close:", ticker.close)
    
    ib.disconnect()
    print("\nDisconnected.")

if __name__ == '__main__':
    main()
# pip install ib-insync
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)   # or 4002 for IB Gateway

contract = Stock('AAPL', 'SMART', 'USD')    # swap instrument as needed
ib.qualifyContracts(contract)

# Subscribe to true tick-by-tick trades and bid/ask
ib.reqTickByTickData(contract, 'Last', 0, True)     # trades
ib.reqTickByTickData(contract, 'BidAsk', 0, True)   # quotes

def on_tick(t):
    print(t)

ib.pendingTickersEvent += lambda: None       # keep event loop happy
ib.run()

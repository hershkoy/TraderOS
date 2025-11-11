from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1)

ib.reqMarketDataType(1)  # request real-time data

contract = Stock('AAPL', 'SMART', 'USD')
ticker = ib.reqMktData(contract, '', False, False)

ib.sleep(2)
print(ticker)

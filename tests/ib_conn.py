from ib_insync import IB, util, Stock 

HOST = '127.0.0.1'
PORT = 4001      # make sure this really changed
CLIENT_ID = 1

util.logToConsole()  # see low-level messages

ib = IB()
ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
print('Connected:', ib.isConnected())


# --- Request Server Time ---
print("\n=== Server Time ===")
print(ib.reqCurrentTime())


# --- TEST: Request contract details ---
contract = Stock('AAPL', 'SMART', 'USD')

print("\n=== Contract Details ===")
details = ib.reqContractDetails(contract)
for d in details:
    print(d)

if not details:
    print("❌ Contract lookup failed — cannot request market data.")
    ib.disconnect()
    raise SystemExit()

# --- TEST: Request market data snapshot ---
print("\n=== Market Data Snapshot for AAPL ===")

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
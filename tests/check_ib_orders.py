#!/usr/bin/env python3
"""
check_ib_orders.py

Check current IB orders and their structure via API.
Useful for debugging bracket orders, take profit, stop loss.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fetch_data import get_ib_connection, cleanup_ib_connection
from utils.ib_port_detector import detect_ib_port


def main():
    port = detect_ib_port()
    if not port:
        print("Could not detect IB port")
        return
    
    print(f"Connecting to IB on port {port}...")
    ib = get_ib_connection(port=port)
    
    # Get all orders using different methods
    print("\n" + "=" * 80)
    print("OPEN ORDERS")
    print("=" * 80)
    
    orders = ib.openOrders()
    trades = ib.openTrades()
    all_orders = ib.orders()  # All orders including inactive
    
    print(f"\nFound {len(orders)} open orders, {len(trades)} open trades, {len(all_orders)} total orders\n")
    
    # Request all orders from TWS (includes orders created in TWS UI)
    ib.reqAllOpenOrders()
    ib.sleep(1)
    
    orders_after = ib.openOrders()
    trades_after = ib.openTrades()
    print(f"After reqAllOpenOrders: {len(orders_after)} open orders, {len(trades_after)} open trades\n")
    
    trades = trades_after  # Use refreshed list
    
    for i, trade in enumerate(trades):
        order = trade.order
        contract = trade.contract
        status = trade.orderStatus
        
        print(f"[{i+1}] Order ID: {order.orderId}")
        print(f"    Parent ID: {order.parentId}")
        print(f"    Action: {order.action}")
        print(f"    Type: {order.orderType}")
        print(f"    Quantity: {order.totalQuantity}")
        print(f"    Limit Price: {order.lmtPrice}")
        print(f"    Aux Price: {order.auxPrice}")
        print(f"    TIF: {order.tif}")
        print(f"    Status: {status.status}")
        print(f"    Filled: {status.filled}/{order.totalQuantity}")
        print(f"    Contract: {contract.symbol} {contract.secType}")
        
        if contract.secType == "BAG" and contract.comboLegs:
            print(f"    Combo Legs:")
            for leg in contract.comboLegs:
                print(f"      - ConId: {leg.conId}, Action: {leg.action}, Ratio: {leg.ratio}")
        
        print()
    
    # Group by parent ID to show bracket structure
    print("\n" + "=" * 80)
    print("BRACKET ORDER STRUCTURE")
    print("=" * 80)
    
    parent_orders = {}
    child_orders = {}
    
    for trade in trades:
        order = trade.order
        if order.parentId == 0:
            parent_orders[order.orderId] = trade
        else:
            if order.parentId not in child_orders:
                child_orders[order.parentId] = []
            child_orders[order.parentId].append(trade)
    
    for parent_id, parent_trade in parent_orders.items():
        order = parent_trade.order
        contract = parent_trade.contract
        print(f"\nPARENT [{parent_id}]: {order.action} {order.totalQuantity}x {contract.symbol} @ {order.lmtPrice} ({order.orderType})")
        print(f"  Status: {parent_trade.orderStatus.status}")
        
        if parent_id in child_orders:
            for child_trade in child_orders[parent_id]:
                child_order = child_trade.order
                child_type = "TAKE PROFIT" if child_order.lmtPrice and abs(child_order.lmtPrice) < abs(order.lmtPrice) else "STOP LOSS"
                if child_order.orderType == "STP":
                    child_type = "STOP LOSS"
                print(f"  CHILD [{child_order.orderId}] {child_type}: {child_order.action} @ {child_order.lmtPrice} ({child_order.orderType})")
                print(f"    Aux Price: {child_order.auxPrice}")
                print(f"    Status: {child_trade.orderStatus.status}")
    
    # Show orphan child orders (parent already filled)
    orphan_children = [t for t in trades if t.order.parentId != 0 and t.order.parentId not in parent_orders]
    if orphan_children:
        print("\n" + "=" * 80)
        print("CHILD ORDERS (Parent filled or not found)")
        print("=" * 80)
        for trade in orphan_children:
            order = trade.order
            contract = trade.contract
            print(f"\n[{order.orderId}] Parent: {order.parentId}")
            print(f"  {order.action} {order.totalQuantity}x {contract.symbol} @ {order.lmtPrice} ({order.orderType})")
            print(f"  Status: {trade.orderStatus.status}")
    
    # Raw order dump
    print("\n" + "=" * 80)
    print("RAW ORDER DATA")
    print("=" * 80)
    for trade in trades:
        print(f"\n{trade}")
    
    cleanup_ib_connection()


if __name__ == "__main__":
    main()


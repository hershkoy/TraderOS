# test_bt.py
import backtrader as bt
import pandas as pd

class TestStrategy(bt.Strategy):
    def next(self):
        print(self.data.datetime.date(0), self.data.close[0])

# Load sample data (Pandas DataFrame)
data = pd.DataFrame({
    'datetime': pd.date_range(start="2024-06-01", periods=5, freq="D"),
    'open': [1,2,3,4,5],
    'high': [2,3,4,5,6],
    'low':  [0.5,1.5,2.5,3.5,4.5],
    'close':[1.5,2.5,3.5,4.5,5.5],
    'volume':[100,200,300,400,500],
})

datafeed = bt.feeds.PandasData(dataname=data, datetime='datetime')

cerebro = bt.Cerebro()
cerebro.adddata(datafeed)
cerebro.addstrategy(TestStrategy)
cerebro.run()

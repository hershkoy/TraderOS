
# Setup



# Fetch

## Fetch 5 years of NFLX 1-hour bars from Alpaca (max 10,000 capped internally)
python fetch_data.py --symbol NFLX --provider alpaca --timeframe 1h --bars 10000

## Fetch 1d bars from IB (capped to 3000 inside)
python fetch_data.py --symbol NFLX --provider ib --timeframe 1d --bars 9999

# Run 

python backtrader_runner_yaml.py ^
  --config default.yaml ^
  --parquet "data\ALPACA\NFLX\1h\nflx_1h.parquet" ^
  --strategy mean_reversion ^
  --log-level DEBUG
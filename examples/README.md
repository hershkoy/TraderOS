# Examples

This directory contains example scripts and alternative implementations for the BackTrader Testing Framework.

## Example Scripts

- **mean_reversion_simple.py** - Simple mean reversion strategy example with basic implementation
- **pnf_backtrader.py** - Point & Figure strategy implementation (alternative to the main runner)
- **backtrader_runner.py** - Alternative backtrader runner without YAML configuration
- **simple_test.py** - Basic test script for verifying the framework setup
- **fetch_data_examples.py** - Examples demonstrating data fetching with `--bars max` functionality

## Usage

These examples can be run from the project root directory:

```bash
# Run simple mean reversion example
python examples/mean_reversion_simple.py --parquet data/ALPACA/NFLX/1h/nflx_1h.parquet --lookback 30 --std 1.5 --size 2

# Run alternative backtrader runner
python examples/backtrader_runner.py --parquet data/ALPACA/NFLX/1h/nflx_1h.parquet --strategy mean_reversion

# Run PnF strategy
python examples/pnf_backtrader.py --parquet data/ALPACA/NFLX/1h/nflx_1h.parquet --strategy pnf

# View fetch data examples (demonstration only)
python examples/fetch_data_examples.py

## Note

These are example implementations and may not have all the features of the main `backtrader_runner_yaml.py` script. For production use, use the main script in the root directory.

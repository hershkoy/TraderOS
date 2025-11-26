# Options Strategy Trader

Automated options spread trading system with modular strategy support.

## File Structure

```
scripts/
  options_strategy_trader.py    # Main trading script

strategies/
  option_strategies.py          # Strategy classes and registry

crons/
  daily_spreads.yaml            # Configuration for scheduled trades

tests/
  test_option_strategies.py     # Unit tests for strategy classes
```

## Available Strategies

### 1. `otm_credit_spreads`

Sells an OTM put spread for credit.

- **Structure**: Sell higher strike put, buy lower strike put
- **Target**: ~10 delta short leg
- **Risk**: Limited to spread width minus credit received
- **Profit**: Credit received if stock stays above short strike

### 2. `vertical_spread_with_hedging`

Put ratio spread (1x2 structure).

- **Structure**: Buy 1 ATM put (~50 delta), sell 2 OTM puts (~25 and ~15 delta)
- **Risk**: Limited between short strikes, unlimited below lower breakeven
- **Profit**: Net credit if stock stays above upper breakeven

Example from TSLA:
- Buy 1x 400P
- Sell 1x 382.5P
- Sell 1x 375P
- Net Credit: $109, Max Loss: $1,641, Breakevens: $358.59 - $398.91

## Command Line Options

```bash
python scripts/options_strategy_trader.py [OPTIONS]
```

### Required

| Option | Description |
|--------|-------------|
| `--symbol` | Underlying symbol (SPY, QQQ, etc.) |

### Strategy Selection

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `otm_credit_spreads` | Strategy to use |
| `--dte` | - | Days to expiration for auto-fetch |
| `--expiry` | - | Specific expiration (YYYYMMDD) |

### Spread Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--spread-width` | 4.0 | Width of vertical spread |
| `--target-delta` | 0.10 | Target delta for short leg |
| `--min-credit` | 0.10 | Minimum acceptable credit |
| `--num-candidates` | 3 | Number of candidates to show |

### Order Execution

| Option | Default | Description |
|--------|---------|-------------|
| `--create-orders-en` | false | Enable order creation |
| `--quantity` | 1 | Number of spreads to trade |
| `--risk-profile` | interactive | Selection mode: interactive, conservative, balanced, risky |
| `--account` | auto | IB account ID |
| `--monitor-order` | false | Monitor order until filled |

### Risk Management

| Option | Description |
|--------|-------------|
| `--take-profit` | Take profit price (e.g., -0.02 for 2 cents) |
| `--stop-loss-multiplier` | Stop loss as multiplier of limit price |

## Configuration File (YAML)

```yaml
orders:
  - symbol: QQQ
    strategy: otm_credit_spreads
    dte: 7
    create_orders_en: true
    quantity: 2
    risk_profile: balanced
    take_profit: -0.02
    stop_loss_multiplier: 1.5
    
  - symbol: IWM
    strategy: otm_credit_spreads
    dte: 7
    create_orders_en: true
    quantity: 2
    risk_profile: balanced
```

Run with config file:

```bash
python scripts/options_strategy_trader.py --conf-file crons/daily_spreads.yaml
```

## Tests

Run strategy tests:

```bash
python -m pytest tests/test_option_strategies.py -v
```

Tests cover:
- OptionRow mid price calculation
- OTMCreditSpreadsStrategy candidate finding
- VerticalSpreadWithHedgingStrategy candidate finding
- Strategy registry functions
- Data class creation

## Adding New Strategies

1. Create a new class in `strategies/option_strategies.py`:

```python
class MyNewStrategy(OptionStrategy):
    @property
    def name(self) -> str:
        return "my_new_strategy"
    
    def find_candidates(self, rows: List[OptionRow], **kwargs) -> List:
        # Implementation
        pass
    
    def describe_candidate(self, label: str, candidate) -> str:
        # Implementation
        pass
```

2. Register in `STRATEGY_REGISTRY`:

```python
STRATEGY_REGISTRY: Dict[str, type] = {
    "otm_credit_spreads": OTMCreditSpreadsStrategy,
    "vertical_spread_with_hedging": VerticalSpreadWithHedgingStrategy,
    "my_new_strategy": MyNewStrategy,
}
```

3. Add tests in `tests/test_option_strategies.py`


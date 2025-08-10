# Strategies package
from .pnf_strategy import PnF_MTF_Strategy
from .mean_reversion_strategy import MeanReversionStrategy

# Strategy registry
STRATEGIES = {
    'pnf': PnF_MTF_Strategy,
    'mean_reversion': MeanReversionStrategy,
}

def get_strategy(name):
    """Get strategy class by name"""
    if name not in STRATEGIES:
        available = ', '.join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {available}")
    return STRATEGIES[name]

def list_strategies():
    """List all available strategies"""
    return list(STRATEGIES.keys())

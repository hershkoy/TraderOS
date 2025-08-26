# Strategies package
import os
import importlib
import inspect
from pathlib import Path

# Strategy registry - will be populated dynamically
STRATEGIES = {}

def _discover_strategies():
    """Dynamically discover all strategy classes from Python files in this directory"""
    strategies_dir = Path(__file__).parent
    
    for file_path in strategies_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue
            
        module_name = file_path.stem
        
        try:
            # Import the module
            module = importlib.import_module(f".{module_name}", package="strategies")
            
            # Find all classes in the module that inherit from bt.Strategy
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    hasattr(obj, '__bases__') and 
                    any('Strategy' in base.__name__ for base in obj.__bases__)):
                    
                    # Create multiple strategy keys for better discoverability
                    # 1. Original class name (lowercase)
                    # 2. Module name
                    # 3. Simplified name (remove 'Strategy' suffix and common prefixes)
                    
                    class_key = name.lower()
                    module_key = module_name
                    
                    # Simplified key - remove 'Strategy' suffix and common prefixes
                    simple_key = name.lower()
                    if simple_key.endswith('strategy'):
                        simple_key = simple_key[:-8]  # Remove 'strategy'
                    if simple_key.startswith('mean'):
                        simple_key = simple_key[4:]  # Remove 'mean' prefix
                    
                    # Register with multiple keys for flexibility
                    STRATEGIES[class_key] = obj
                    STRATEGIES[module_key] = obj
                    if simple_key and simple_key != class_key:
                        STRATEGIES[simple_key] = obj
                    
                    print(f"DEBUG: Discovered strategy '{name}' from {module_name} (keys: {class_key}, {module_key}, {simple_key})")
                    
        except ImportError as e:
            print(f"Warning: Could not import module {module_name}: {e}")
        except Exception as e:
            print(f"Warning: Could not load strategy from {module_name}: {e}")

# Discover strategies when module is imported
_discover_strategies()

def get_strategy(name):
    """Get strategy class by name"""
    if name not in STRATEGIES:
        available = ', '.join(sorted(set(STRATEGIES.keys())))
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {available}")
    return STRATEGIES[name]

def list_strategies():
    """List all available strategies"""
    return sorted(set(STRATEGIES.keys()))

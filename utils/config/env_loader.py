"""
Environment variable loader utility
Loads environment variables from .env file if it exists
"""

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_file_path: Optional[str] = None) -> bool:
    """
    Load environment variables from a .env file
    
    Args:
        env_file_path: Path to .env file. If None, looks for .env in current directory
        
    Returns:
        True if .env file was loaded, False otherwise
    """
    if env_file_path is None:
        # Look for .env file in current directory and parent directories
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            env_file = parent / '.env'
            if env_file.exists():
                env_file_path = str(env_file)
                break
    
    if env_file_path is None or not os.path.exists(env_file_path):
        return False
    
    try:
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # Only set if not already in environment
                    if key and key not in os.environ:
                        os.environ[key] = value
        
        return True
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")
        return False


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable, loading from .env file if needed
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    # Try to load .env file if variable not found
    if key not in os.environ:
        load_env_file()
    
    return os.environ.get(key, default)


# Auto-load .env file when module is imported
load_env_file()

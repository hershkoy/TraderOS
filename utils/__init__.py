# Utils package for BackTrader Testing Framework
# 
# The utils folder has been reorganized into subdirectories:
# - utils/db/ - Database utilities
# - utils/data/ - Data fetching and management
# - utils/api/ - API clients
# - utils/scanners/ - Scanner logic
# - utils/screeners/ - Screener logic
# - utils/reporting/ - Reporting utilities
# - utils/options/ - Options utilities
# - utils/backtesting/ - Backtesting utilities
# - utils/config/ - Configuration utilities
#
# For backward compatibility, old imports like "from utils.fetch_data import ..."
# will still work because Python resolves them through the package structure.
# However, you can also use the new paths: "from utils.data.fetch_data import ..."
#
# We don't eagerly import everything here to avoid circular dependencies.

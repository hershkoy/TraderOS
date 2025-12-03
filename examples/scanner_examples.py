#!/usr/bin/env python3
"""
Scanner Examples
Shows different ways to use the unified scanner runner
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show the description"""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    # For demonstration, just show the command
    # In real usage, you would run: subprocess.run(cmd, shell=True)
    print("(This would execute the command in a real scenario)")

def main():
    """Show different scanner usage examples"""
    print("🔍 Unified Scanner Runner - Usage Examples")
    print("=" * 60)
    
    # Example 1: Basic HL After LL scanner
    run_command(
        "python scanner_runner.py --scanner hl_after_ll",
        "Basic HL After LL scanner with default configuration"
    )
    
    # Example 2: HL After LL with specific symbols
    run_command(
        "python scanner_runner.py --scanner hl_after_ll --symbols AAPL MSFT GOOGL",
        "HL After LL scanner on specific symbols"
    )
    
    # Example 3: HL After LL with custom config and debug logging
    run_command(
        "python scanner_runner.py --config scanner_config.yaml --scanner hl_after_ll --log-level DEBUG",
        "HL After LL scanner with custom config and debug logging"
    )
    
    # Example 4: VCP scanner (future)
    run_command(
        "python scanner_runner.py --scanner vcp --log-level INFO",
        "VCP scanner (Volume Contraction Pattern)"
    )
    
    # Example 5: Liquidity Sweep scanner (future)
    run_command(
        "python scanner_runner.py --scanner liquidity_sweep --symbols AAPL TSLA",
        "Liquidity Sweep scanner on specific symbols"
    )
    
    # Example 6: Using batch file (Windows)
    run_command(
        "run_scanner.bat",
        "Using Windows batch file for easy execution"
    )
    
    # Example 7: Advanced configuration
    run_command(
        "python scanner_runner.py --config scanner_config.yaml --scanner hl_after_ll --provider ALPACA --log-level DEBUG",
        "Advanced configuration with provider and debug logging"
    )
    
    print(f"\n{'='*60}")
    print("Configuration File Examples")
    print(f"{'='*60}")
    
    print("\n1. Basic scanner_config.yaml:")
    print("""
scanner:
  type: "hl_after_ll"
  hl_after_ll:
    left_bars: 1
    right_bars: 2
    monday_check:
      enabled: true

universe:
  source: "database"  # or "manual" with symbols list

output:
  console:
    show_summary: true
    show_details: true
  files:
    save_csv: true
""")
    
    print("\n2. Advanced scanner_config.yaml:")
    print("""
scanner:
  type: "hl_after_ll"
  hl_after_ll:
    left_bars: 1
    right_bars: 2
    patterns: ["LL→HH→HL", "LL→HH→HH→HL"]
    monday_check:
      enabled: true
    data:
      timeframe: "1d"
      min_weeks: 52

universe:
  source: "manual"
  symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
  filters:
    min_price: 10.0
    min_volume: 1000000

output:
  console:
    show_summary: true
    show_details: true
    show_monday_check: true
  files:
    save_csv: true
    csv_filename: "hl_after_ll_scan_{timestamp}.csv"

logging:
  level: "INFO"
  file_rotation: true
""")
    
    print(f"\n{'='*60}")
    print("Scanner Types Available")
    print(f"{'='*60}")
    
    scanners = [
        ("hl_after_ll", "Detects LL → HH → HL reversal patterns", "✅ Implemented"),
        ("vcp", "Volume Contraction Pattern scanner", "🚧 Future"),
        ("liquidity_sweep", "Liquidity sweep detection", "🚧 Future")
    ]
    
    for scanner_type, description, status in scanners:
        print(f"\n{scanner_type.upper()}:")
        print(f"  Description: {description}")
        print(f"  Status: {status}")
    
    print(f"\n{'='*60}")
    print("Quick Start Guide")
    print(f"{'='*60}")
    
    print("""
1. Test the scanner:
   python scanner_runner.py --scanner hl_after_ll --symbols AAPL

2. Run on your universe:
   python scanner_runner.py --scanner hl_after_ll

3. Use custom configuration:
   python scanner_runner.py --config scanner_config.yaml --scanner hl_after_ll

4. Enable debug logging:
   python scanner_runner.py --scanner hl_after_ll --log-level DEBUG

5. Use Windows batch file:
   run_scanner.bat
""")
    
    print(f"\n{'='*60}")
    print("Troubleshooting")
    print(f"{'='*60}")
    
    print("""
Common Issues:
1. No data loaded: Check TimescaleDB connection
2. No patterns found: Try different time periods
3. Database errors: Verify credentials in config
4. Import errors: Ensure virtual environment is activated

Debug Mode:
python scanner_runner.py --scanner hl_after_ll --log-level DEBUG

Check logs:
- Console output for real-time feedback
- logs/scanner_runner.log for detailed logs
- reports/ directory for CSV exports
""")

if __name__ == "__main__":
    main()

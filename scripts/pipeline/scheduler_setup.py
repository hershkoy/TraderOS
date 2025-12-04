#!/usr/bin/env python3
"""
Daily Scheduler Setup for Options Data Pipeline

This script sets up a daily scheduler to run the options data pipeline tasks:
1. Contract discovery (after market close)
2. EOD quotes ingestion (after market settle)
3. Greeks backfill (optional)

The scheduler can be configured to run on Windows Task Scheduler or cron.
"""

import os
import sys
import logging
from datetime import datetime, time
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
os.makedirs('logs/pipeline', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline/scheduler_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_windows_task_scheduler_script():
    """
    Create a Windows batch script for Task Scheduler.
    """
    script_content = """@echo off
REM Options Data Pipeline - Daily Tasks
REM This script should be scheduled to run daily after market close

set PROJECT_DIR=%~dp0
set VENV_DIR=%PROJECT_DIR%venv\\Scripts
set LOG_DIR=%PROJECT_DIR%logs

REM Set environment variables
set PYTHONPATH=%PROJECT_DIR%
set POLYGON_API_KEY=your_polygon_api_key_here

REM Create log directory if it doesn't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Get current date for logging
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YYYY=%dt:~2,2%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%"
set "MIN=%dt:~10,2%"
set "SEC=%dt:~12,2%"
set "datestamp=%YYYY%%MM%%DD%_%HH%%MIN%%SEC%"

echo [%datestamp%] Starting Options Data Pipeline >> "%LOG_DIR%\\daily_pipeline_%YYYY%%MM%%DD%.log"

REM Step 1: Discover new contracts (run after market close ~4:00 PM ET)
echo [%datestamp%] Step 1: Discovering contracts... >> "%LOG_DIR%\\daily_pipeline_%YYYY%%MM%%DD%.log"
call "%VENV_DIR%\\activate.bat" && python "%PROJECT_DIR%scripts\\api\\polygon\\polygon_discover_contracts.py" --underlying QQQ >> "%LOG_DIR%\\daily_pipeline_%YYYY%%MM%%DD%.log" 2>&1

REM Step 2: Ingest EOD quotes (run after market settle ~5:00 PM ET)
echo [%datestamp%] Step 2: Ingesting EOD quotes... >> "%LOG_DIR%\\daily_pipeline_%YYYY%%MM%%DD%.log"
call "%VENV_DIR%\\activate.bat" && python "%PROJECT_DIR%scripts\\api\\polygon\\polygon_ingest_eod_quotes.py" --underlying QQQ --date %YYYY%-%MM%-%DD% >> "%LOG_DIR%\\daily_pipeline_%YYYY%%MM%%DD%.log" 2>&1

REM Step 3: Optional: Backfill missing Greeks
echo [%datestamp%] Step 3: Backfilling missing Greeks... >> "%LOG_DIR%\\daily_pipeline_%YYYY%%MM%%DD%.log"
call "%VENV_DIR%\\activate.bat" && python "%PROJECT_DIR%scripts\\db\\greeks_fill.py" --start-date %YYYY%-%MM%-%DD% --end-date %YYYY%-%MM%-%DD% --underlying QQQ >> "%LOG_DIR%\\daily_pipeline_%YYYY%%MM%%DD%.log" 2>&1

echo [%datestamp%] Options Data Pipeline completed >> "%LOG_DIR%\\daily_pipeline_%YYYY%%MM%%DD%.log"
"""
    
    script_path = os.path.join(os.path.dirname(__file__), 'run_daily_pipeline.bat')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Created Windows batch script: {script_path}")
    return script_path

def create_cron_script():
    """
    Create a shell script for cron scheduling (Linux/macOS).
    """
    script_content = """#!/bin/bash
# Options Data Pipeline - Daily Tasks
# This script should be added to crontab to run daily after market close

# Set project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv/bin"
LOG_DIR="$PROJECT_DIR/logs"

# Set environment variables
export PYTHONPATH="$PROJECT_DIR"
export POLYGON_API_KEY="your_polygon_api_key_here"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Get current date for logging
DATESTAMP=$(date '+%Y%m%d_%H%M%S')
DATE=$(date '+%Y-%m-%d')

echo "[$DATESTAMP] Starting Options Data Pipeline" >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log"

# Step 1: Discover new contracts (run after market close ~4:00 PM ET)
echo "[$DATESTAMP] Step 1: Discovering contracts..." >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log"
source "$VENV_DIR/activate" && python "$PROJECT_DIR/scripts/api/polygon/polygon_discover_contracts.py" --underlying QQQ >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log" 2>&1

# Step 2: Ingest EOD quotes (run after market settle ~5:00 PM ET)
echo "[$DATESTAMP] Step 2: Ingesting EOD quotes..." >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log"
source "$VENV_DIR/activate" && python "$PROJECT_DIR/scripts/api/polygon/polygon_ingest_eod_quotes.py" --underlying QQQ --date $DATE >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log" 2>&1

# Step 3: Optional: Backfill missing Greeks
echo "[$DATESTAMP] Step 3: Backfilling missing Greeks..." >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log"
source "$VENV_DIR/activate" && python "$PROJECT_DIR/scripts/db/greeks_fill.py" --start-date $DATE --end-date $DATE --underlying QQQ >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log" 2>&1

echo "[$DATESTAMP] Options Data Pipeline completed" >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log"
"""
    
    script_path = os.path.join(os.path.dirname(__file__), 'run_daily_pipeline.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"Created cron script: {script_path}")
    return script_path

def create_taskfile():
    """
    Create a Taskfile.yml for task automation.
    """
    taskfile_content = """# Taskfile for Options Data Pipeline
# Install Task: https://taskfile.dev/installation/

version: '3'

vars:
  PROJECT_DIR: '{{.PROJECT_DIR | default "."}}'
  VENV_DIR: '{{.VENV_DIR | default "venv"}}'
  LOG_DIR: '{{.LOG_DIR | default "logs"}}'
  UNDERLYING: '{{.UNDERLYING | default "QQQ"}}'

tasks:
  # Discover new option contracts
  discover-contracts:
    desc: Discover new option contracts from Polygon
    cmds:
      - '{{.VENV_DIR}}/Scripts/python.exe scripts/api/polygon/polygon_discover_contracts.py --underlying {{.UNDERLYING}}'
    env:
      PYTHONPATH: '{{.PROJECT_DIR}}'
      POLYGON_API_KEY: '{{.POLYGON_API_KEY}}'

  # Ingest EOD quotes for a specific date
  ingest-quotes:
    desc: Ingest EOD quotes for a specific date
    cmds:
      - '{{.VENV_DIR}}/Scripts/python.exe scripts/api/polygon/polygon_ingest_eod_quotes.py --underlying {{.UNDERLYING}} --date {{.DATE | default .TODAY}}'
    env:
      PYTHONPATH: '{{.PROJECT_DIR}}'
      POLYGON_API_KEY: '{{.POLYGON_API_KEY}}'
    vars:
      TODAY: '{{now | date "2006-01-02"}}'

  # Backfill missing Greeks
  backfill-greeks:
    desc: Backfill missing Greeks for option quotes
    cmds:
      - '{{.VENV_DIR}}/Scripts/python.exe scripts/db/greeks_fill.py --start-date {{.START_DATE | default .TODAY}} --end-date {{.END_DATE | default .TODAY}} --underlying {{.UNDERLYING}}'
    env:
      PYTHONPATH: '{{.PROJECT_DIR}}'
    vars:
      TODAY: '{{now | date "2006-01-02"}}'

  # Run complete daily pipeline
  daily-pipeline:
    desc: Run complete daily options data pipeline
    deps: [discover-contracts, ingest-quotes, backfill-greeks]
    cmds:
      - echo "Daily pipeline completed successfully"

  # Run pipeline for a specific date range
  backfill-range:
    desc: Backfill data for a date range
    cmds:
      - '{{.VENV_DIR}}/Scripts/python.exe scripts/api/polygon/polygon_discover_contracts.py --underlying {{.UNDERLYING}}'
      - '{{.VENV_DIR}}/Scripts/python.exe scripts/polygon_ingest_eod_quotes.py --underlying {{.UNDERLYING}} --start-date {{.START_DATE}} --end-date {{.END_DATE}}'
      - '{{.VENV_DIR}}/Scripts/python.exe scripts/greeks_fill.py --start-date {{.START_DATE}} --end-date {{.END_DATE}} --underlying {{.UNDERLYING}}'
    env:
      PYTHONPATH: '{{.PROJECT_DIR}}'
      POLYGON_API_KEY: '{{.POLYGON_API_KEY}}'
"""
    
    taskfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Taskfile.yml')
    with open(taskfile_path, 'w') as f:
        f.write(taskfile_content)
    
    logger.info(f"Created Taskfile: {taskfile_path}")
    return taskfile_path

def print_scheduling_instructions(platform: str):
    """
    Print instructions for setting up the scheduler.
    """
    if platform.lower() == 'windows':
        print("\n=== Windows Task Scheduler Setup ===")
        print("1. Open Task Scheduler (taskschd.msc)")
        print("2. Create Basic Task:")
        print("   - Name: 'Options Data Pipeline'")
        print("   - Trigger: Daily at 5:00 PM")
        print("   - Action: Start a program")
        print("   - Program: path/to/run_daily_pipeline.bat")
        print("3. Set the working directory to your project folder")
        print("4. Configure to run whether user is logged on or not")
        print("5. Set POLYGON_API_KEY environment variable in the task")
        
    elif platform.lower() == 'linux':
        print("\n=== Linux/macOS Cron Setup ===")
        print("1. Open crontab: crontab -e")
        print("2. Add the following line:")
        print("   0 17 * * 1-5 /path/to/run_daily_pipeline.sh")
        print("   (Runs at 5:00 PM on weekdays)")
        print("3. Make sure the script is executable: chmod +x run_daily_pipeline.sh")
        print("4. Set POLYGON_API_KEY environment variable in your shell profile")
        
    elif platform.lower() == 'taskfile':
        print("\n=== Taskfile Usage ===")
        print("Install Task: https://taskfile.dev/installation/")
        print("Available commands:")
        print("  task discover-contracts     # Discover new contracts")
        print("  task ingest-quotes          # Ingest EOD quotes for today")
        print("  task backfill-greeks        # Backfill missing Greeks")
        print("  task daily-pipeline         # Run complete daily pipeline")
        print("  task backfill-range START_DATE=2024-01-01 END_DATE=2024-01-31")

def main():
    """Main function to set up the scheduler."""
    parser = argparse.ArgumentParser(description='Set up daily scheduler for options data pipeline')
    parser.add_argument('--platform', choices=['windows', 'linux', 'taskfile', 'all'], 
                       default='all', help='Platform to create scheduler for')
    
    args = parser.parse_args()
    
    logger.info("Setting up daily scheduler for options data pipeline")
    
    try:
        if args.platform in ['windows', 'all']:
            create_windows_task_scheduler_script()
            print_scheduling_instructions('windows')
            
        if args.platform in ['linux', 'all']:
            create_cron_script()
            print_scheduling_instructions('linux')
            
        if args.platform in ['taskfile', 'all']:
            create_taskfile()
            print_scheduling_instructions('taskfile')
            
        logger.info("Scheduler setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error setting up scheduler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

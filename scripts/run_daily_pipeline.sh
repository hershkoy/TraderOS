#!/bin/bash
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
source "$VENV_DIR/activate" && python "$PROJECT_DIR/scripts/polygon_discover_contracts.py" --underlying QQQ >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log" 2>&1

# Step 2: Ingest EOD quotes (run after market settle ~5:00 PM ET)
echo "[$DATESTAMP] Step 2: Ingesting EOD quotes..." >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log"
source "$VENV_DIR/activate" && python "$PROJECT_DIR/scripts/polygon_ingest_eod_quotes.py" --underlying QQQ --date $DATE >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log" 2>&1

# Step 3: Optional: Backfill missing Greeks
echo "[$DATESTAMP] Step 3: Backfilling missing Greeks..." >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log"
source "$VENV_DIR/activate" && python "$PROJECT_DIR/scripts/greeks_fill.py" --start-date $DATE --end-date $DATE --underlying QQQ >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log" 2>&1

echo "[$DATESTAMP] Options Data Pipeline completed" >> "$LOG_DIR/daily_pipeline_$(date '+%Y%m%d').log"

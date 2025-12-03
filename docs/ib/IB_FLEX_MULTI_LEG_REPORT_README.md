# IB Flex Query Multi-Leg Strategy Report Generator

A tool for fetching trade history from Interactive Brokers Flex Query Web Service and generating comprehensive reports of multi-leg option strategies grouped by OrderID.

## Purpose

This script automates the process of retrieving and analyzing multi-leg option strategies from your Interactive Brokers account. It:

- Fetches trade data via IB Flex Query Web Service
- Groups trades by OrderID to identify multi-leg strategies (spreads, collars, iron condors, etc.)
- Generates formatted reports in HTML or CSV format
- Filters trades by date range
- Provides summary statistics and detailed leg-by-leg breakdowns

## Features

- **Automated Flex Query Processing**: Handles the two-step process of requesting and downloading Flex Query results
- **Multi-Leg Strategy Grouping**: Automatically groups option trades by OrderID to identify complex strategies
- **Date Filtering**: Filter trades by date range (e.g., all trades since a specific date)
- **Multiple Output Formats**: Generate reports in HTML (with styling) or CSV (for spreadsheet analysis)
- **Comprehensive Trade Details**: Includes symbol, strike, expiry, put/call, buy/sell, quantity, price, net cash, and commissions
- **Summary Statistics**: Total strategies, legs, net cash, and commissions
- **Flexible Column Handling**: Automatically handles various column name formats from IB Flex Queries
- **Error Handling**: Robust error handling with retry logic for statement downloads

## Files

- `scripts/api/ib/ib_flex_multi_leg_report.py` - Main script
- `docs/ib/IB_FLEX_MULTI_LEG_REPORT_README.md` - This documentation

## Setup

### Prerequisites

1. **Interactive Brokers Account**: Active IB account with Flex Query Web Service access
2. **Python Dependencies**: 
   - `requests` - For HTTP requests to IB Flex Query Web Service
   - `pandas` - For data processing
   - `python-dotenv` - For environment variable management

Install dependencies:
```bash
pip install requests pandas python-dotenv
```

### Configuration

The script requires two environment variables to authenticate with IB Flex Query Web Service:

#### 1. Get Flex Query Web Service Token

1. Log in to Interactive Brokers Client Portal
2. Navigate to: **Reports** → **Flex Web Service** → **Token**
3. Copy your Flex Web Service token
4. Set it as an environment variable:

**Windows (Command Prompt):**
```cmd
set IB_FLEX_QUERY_TOKEN=your_token_here
```

**Windows (PowerShell):**
```powershell
$env:IB_FLEX_QUERY_TOKEN="your_token_here"
```

**Using .env file (Recommended):**
Create or edit `.env` file in the project root:
```
IB_FLEX_QUERY_TOKEN=your_token_here
IB_FLEX_QUERY_ID=your_query_id_here
```

#### 2. Create and Get Flex Query ID

1. Log in to Interactive Brokers Client Portal
2. Navigate to: **Reports** → **Flex Queries**
3. Create a new Flex Query or use an existing one that includes:
   - **TradeConfirms** section
   - Fields: OrderID, Symbol, Strike, Expiry, Put/Call, Buy/Sell, Quantity, Price, NetCash, Commission, Date/Time
4. Copy the Flex Query ID (usually a number like `123456`)
5. Set it as an environment variable:

**Windows (Command Prompt):**
```cmd
set IB_FLEX_QUERY_ID=123456
```

**Windows (PowerShell):**
```powershell
$env:IB_FLEX_QUERY_ID="123456"
```

**Using .env file:**
Add to your `.env` file (see above)

### Flex Query Configuration

Your Flex Query should include the following sections and fields:

**Required Sections:**
- `TradeConfirms` - Contains trade confirmation data

**Recommended Fields:**
- `OrderID` - Groups trades into multi-leg strategies
- `Symbol` - Underlying symbol
- `Strike` - Option strike price
- `Expiry` - Option expiration date
- `Put/Call` - Option type (P or C)
- `Buy/Sell` - Trade direction
- `Quantity` - Number of contracts
- `Price` - Trade price
- `NetCash` - Net cash amount
- `Commission` - Commission charged
- `Date/Time` or `Date` - Trade date/time

**Example Flex Query Setup:**
1. In Client Portal, go to Reports → Flex Queries
2. Create a new query or edit existing
3. Add "TradeConfirms" section
4. Select the fields listed above
5. Save the query and note the Query ID

## Usage

### Basic Usage

**Generate HTML report for all trades:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --type html
```

**Generate CSV report for all trades:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --type csv
```

### Date Filtering

**Generate report for trades since a specific date:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html
```

**Date format:** YYYY-MM-DD (e.g., `2025-01-01`)

### Custom Output Path

**Specify custom output file:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --output reports/my_strategies.html
```

**Default output location:** If `--output` is not specified, reports are saved to:
- `reports/ib_multi_leg_strategies_YYYYMMDD_HHMMSS.html` (or `.csv`)

### Command-Line Options

```
--since DATE          Filter trades on or after this date (YYYY-MM-DD format)
--type {html,csv}     Output format: html or csv (default: html)
--output PATH         Output file path (default: auto-generated in reports/)
```

### Examples

**Generate HTML report for all trades in 2025:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html
```

**Generate CSV report for recent trades (last 30 days):**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-15 --type csv --output reports/recent_strategies.csv
```

**Generate HTML report with custom filename:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --type html --output reports/q1_2025_strategies.html
```

## Output Formats

### HTML Report

The HTML report provides a visually formatted view with:

- **Summary Section**: Total strategies, total legs, total net cash, total commissions
- **Strategy Cards**: Each multi-leg strategy displayed in a card format showing:
  - Order ID
  - Number of legs
  - Underlying symbol
  - Trade date/time
  - Net cash and commission
  - Detailed leg-by-leg breakdown

**Example HTML Output:**
- Summary statistics at the top
- Each strategy in a styled card
- Legs displayed in a formatted list
- Professional styling with colors and spacing

### CSV Report

The CSV report provides a tabular format suitable for spreadsheet analysis:

**Columns:**
- `OrderID` - Order identifier
- `NumLegs` - Number of legs in the strategy
- `Underlying` - Underlying symbol
- `When` - Trade date/time
- `Legs` - Pipe-separated list of leg descriptions
- `NetCash` - Total net cash for the strategy
- `Commission` - Total commission for the strategy
- `TotalCost` - Net cash + commission

**Example CSV Output:**
```csv
OrderID,NumLegs,Underlying,When,Legs,NetCash,Commission,TotalCost
12345,2,SPY,2025-01-15 10:30:00,"BUY 1 x SPY 2025-02-21 450C @ 5.20 | SELL 1 x SPY 2025-02-21 455C @ 2.10",-310.00,1.00,-311.00
```

## How It Works

### Process Flow

1. **Request Flex Query**: Sends request to IB Flex Query Web Service with token and query ID
2. **Get Reference Code**: Receives reference code for the query execution
3. **Download Statement**: Polls for statement readiness and downloads XML/CSV data
4. **Parse Data**: Parses XML or CSV format into pandas DataFrame
5. **Filter by Date**: Optionally filters trades by date range
6. **Group Strategies**: Groups option trades by OrderID to identify multi-leg strategies
7. **Generate Report**: Creates HTML or CSV report with formatted output

### Strategy Grouping Logic

- Filters to option trades (trades with Put/Call column)
- Groups trades by OrderID
- Only includes strategies with 2+ legs (multi-leg strategies)
- Sorts legs by Symbol, Strike, Put/Call for consistent display
- Calculates totals (NetCash, Commission) across all legs

### Leg Description Format

Each leg is formatted as:
```
{Buy/Sell} {Quantity} x {Symbol} {Expiry} {Strike}{Put/Call} @ {Price}
```

**Example:**
```
BUY 1 x SPY 2025-02-21 450C @ 5.20
SELL 1 x SPY 2025-02-21 455C @ 2.10
```

## Troubleshooting

### Common Issues

**Error: "IB_FLEX_QUERY_TOKEN environment variable not set"**
- **Solution**: Set the `IB_FLEX_QUERY_TOKEN` environment variable or add it to your `.env` file
- **Check**: Verify token is correct in Client Portal → Reports → Flex Web Service → Token

**Error: "IB_FLEX_QUERY_ID environment variable not set"**
- **Solution**: Set the `IB_FLEX_QUERY_ID` environment variable or add it to your `.env` file
- **Check**: Verify query ID exists in Client Portal → Reports → Flex Queries

**Error: "Flex Query request failed"**
- **Solution**: 
  - Verify token and query ID are correct
  - Check that Flex Query is active and accessible
  - Ensure query includes TradeConfirms section

**Error: "Statement not ready after 30 seconds"**
- **Solution**: 
  - This is usually temporary - try running again
  - IB may need more time to process large queries
  - Check IB system status

**Warning: "No trade data found in Flex Query"**
- **Solution**: 
  - Verify your Flex Query includes TradeConfirms section
  - Check that query date range includes trades
  - Review query configuration in Client Portal

**Warning: "No multi-leg strategies found"**
- **Solution**: 
  - This is normal if you don't have multi-leg option trades
  - Verify OrderID field is included in Flex Query
  - Check that trades have matching OrderIDs

**Error: "No OrderID column found"**
- **Solution**: 
  - Ensure your Flex Query includes OrderID field
  - Check column names in Flex Query output
  - Script handles case-insensitive matching, but field must exist

### Debugging

Enable debug logging by modifying the script or setting logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

This will show:
- XML structure details
- Column names found
- Parsing steps
- Retry attempts

### Verification Steps

1. **Test Environment Variables:**
   ```bash
   echo %IB_FLEX_QUERY_TOKEN%
   echo %IB_FLEX_QUERY_ID%
   ```

2. **Test Flex Query in Client Portal:**
   - Manually run the Flex Query in Client Portal
   - Verify it returns trade data
   - Check that OrderID field is present

3. **Test Script with Minimal Date Range:**
   ```bash
   python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type csv
   ```

## Integration

### Scheduled Execution

You can schedule this script to run automatically:

**Windows Task Scheduler:**
```cmd
# Create a batch file: crons/ib_flex_report.bat
@echo off
cd /d D:\WORK\Projs\IB\backTraderTest
venv\Scripts\activate
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html
```

### Programmatic Usage

The script can be imported and used programmatically:

```python
from scripts.api.ib.ib_flex_multi_leg_report import (
    request_flex_query,
    download_flex_statement,
    parse_flex_xml,
    group_multi_leg_strategies
)

# Use individual functions as needed
```

## Limitations

- Requires active Flex Query Web Service access
- Flex Query must be pre-configured in Client Portal
- Only processes multi-leg strategies (2+ legs)
- Date filtering depends on date column format in Flex Query
- Statement download may take time for large queries (up to 30 seconds wait)

## Future Enhancements

Potential improvements:
- Support for single-leg option trades
- Additional output formats (JSON, Excel)
- Strategy type detection (spread, collar, iron condor, etc.)
- P&L calculation and tracking
- Integration with database storage
- Email report delivery
- Charting and visualization

## Related Documentation

- [IB API Connection Rules](.cursor/rules/ib_api.mdc) - How to connect to IB API
- [Project Structure](../PROJECT_STRUCTURE.md) - Overall project organization

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Flex Query configuration in Client Portal
3. Verify environment variables are set correctly
4. Check script logs for detailed error messages


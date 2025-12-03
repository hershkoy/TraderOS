# Polygon API Setup for Options Pipeline

This document explains how to set up the Polygon.io API key for the LEAPS strategy options pipeline.

## Getting a Polygon API Key

1. Visit [Polygon.io](https://polygon.io/) and create an account
2. Choose a plan:
   - **Free Plan**: ~5 requests per minute, limited data access
   - **Paid Plans**: Higher rate limits, more data access

## Environment Configuration

Create a `.env` file in the project root with your API key:

```bash
# Polygon API Configuration
POLYGON_API_KEY=your_actual_api_key_here

# Optional: Override database settings if not using docker-compose defaults
# TIMESCALEDB_HOST=localhost
# TIMESCALEDB_PORT=5432
# TIMESCALEDB_DATABASE=backtrader
# TIMESCALEDB_USER=backtrader_user
# TIMESCALEDB_PASSWORD=backtrader_password
```

## Rate Limiting

The Polygon client automatically handles rate limiting:

- **Free Plan**: 5 requests per minute with automatic delays
- **Paid Plans**: Higher limits based on your subscription

## API Endpoints Used

The options pipeline uses these Polygon endpoints:

- `/v3/reference/options/contracts` - Discover option contracts
- `/v3/snapshot/options/{option_id}` - Get current option data
- `/v2/aggs/ticker/{option_id}/prev` - Get previous day's close
- `/v3/quotes/{option_id}` - Get historical quotes

## Testing the Connection

You can test your API key setup by running:

```python
from utils.polygon_client import get_polygon_client

# This will raise an error if POLYGON_API_KEY is not set
client = get_polygon_client()

# Test a simple request
try:
    # Get QQQ options for a specific date
    data = client.get_options_chain('QQQ', '2025-01-17')
    print(f"Found {len(data.get('results', []))} contracts")
except Exception as e:
    print(f"Error: {e}")
finally:
    client.close()
```

## Troubleshooting

### Common Issues

1. **"Polygon API key required" error**
   - Make sure you have a `.env` file with `POLYGON_API_KEY=your_key`
   - Or set the environment variable directly: `export POLYGON_API_KEY=your_key`

2. **Rate limit errors**
   - The client automatically handles rate limiting
   - For free plans, expect delays between requests

3. **Authentication errors**
   - Verify your API key is correct
   - Check if your account is active and has the required permissions

### Support

- [Polygon.io Documentation](https://polygon.io/docs/)
- [Polygon Support](https://polygon.io/support)
- Check the logs for detailed error messages

# Stock Tools MCP Server

An MCP (Model Context Protocol) server providing financial analysis and stock market utility tools. Designed for use with AI assistants to perform technical analysis, portfolio calculations, and market data operations.

## Features

- **Returns Analysis** - Calculate total returns, Sharpe ratio, volatility, and max drawdown
- **Volatility Analysis** - Rolling volatility, VaR, CVaR, and volatility regime classification
- **Stock Comparison** - Compare multiple stocks with correlation analysis and rankings
- **Stock Screening** - Filter stocks by financial criteria (P/E, market cap, sector, etc.)
- **Portfolio Metrics** - Portfolio performance, weights, and beta calculation
- **Technical Levels** - Pivot points, Fibonacci retracements, and moving average support/resistance
- **Moving Averages** - SMA and EMA calculations with distance from current price
- **Pivot Points** - Standard, Woodie's, and Camarilla pivot calculations

## Installation

Requires Python 3.10-3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repository
cd stock_tools_mcp

# Install dependencies
uv sync

# Install with dev dependencies (for testing)
uv sync --extra dev
```

## Usage

### Running the Server

```bash
# Using uv (recommended)
uv run stock-tools-mcp-server

# Using the launcher script (Unix/macOS)
./launch-stock-tools.sh

# Using the Python launcher (cross-platform)
python launch.py

# With SSE transport
uv run stock-tools-mcp-server --sse

# With streamable HTTP transport
uv run stock-tools-mcp-server --streamable
```

### MCP Client Configuration

Add to your MCP client configuration. Use `launch.py` as it can locate `uv` even when PATH is not available (common when launched from AI assistants):

```json
{
  "mcpServers": {
    "stock-tools": {
      "command": "python3",
      "args": ["/path/to/stock_tools_mcp/launch.py"]
    }
  }
}
```

Alternative using uv directly (requires uv in PATH):

```json
{
  "mcpServers": {
    "stock-tools": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/stock_tools_mcp", "stock-tools-mcp-server"]
    }
  }
}
```

## Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STOCK_TOOLS_DEBUG` | `false` | Enable debug mode |
| `STOCK_TOOLS_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `STOCK_TOOLS_RISK_FREE_RATE` | `0.02` | Risk-free rate for Sharpe ratio (2%) |
| `STOCK_TOOLS_TRADING_DAYS` | `252` | Trading days per year for annualization |
| `STOCK_TOOLS_PORT` | `8003` | Server port for SSE/streamable transports |
| `STOCK_TOOLS_MAX_DATA_POINTS` | `50` | Max data points returned in time series |
| `MCP_TRANSPORT` | `stdio` | Transport protocol (stdio, sse, streamable) |

Example:
```bash
STOCK_TOOLS_DEBUG=true STOCK_TOOLS_LOG_LEVEL=DEBUG uv run stock-tools-mcp-server
```

## Available Tools

### calculate_returns
Calculate stock returns and performance metrics from a price series.

```python
prices = [100.0, 102.0, 101.0, 105.0, 108.0]
periods = [3, 5]  # Optional: specific period returns
```

Returns: total return, average daily return, volatility, Sharpe ratio, max drawdown, period returns.

### analyze_volatility
Calculate volatility and risk metrics.

```python
prices = [100.0, 102.0, ...]  # At least window+1 prices
window = 20  # Rolling window size
```

Returns: current/historical volatility, VaR (95%), CVaR, volatility regime (high/normal/low).

### compare_stocks
Compare multiple stocks side by side.

```python
stock_data = {
    "AAPL": [150.0, 152.0, 155.0, ...],
    "MSFT": [300.0, 305.0, 310.0, ...]
}
```

Returns: metrics per stock, correlation matrix, rankings by return and Sharpe ratio.

### screen_stocks
Filter stocks based on financial criteria.

```python
criteria = {
    "stocks": [
        {"symbol": "AAPL", "sector": "Technology", "market_cap": 3000000000000, ...}
    ],
    "min_market_cap": 1000000000000,
    "max_pe_ratio": 30.0,
    "sectors": ["Technology", "Healthcare"]
}
```

### calculate_portfolio_metrics
Analyze portfolio performance.

```python
holdings = {"AAPL": 10, "MSFT": 5}
prices = {"AAPL": [...], "MSFT": [...]}
benchmark_prices = [...]  # Optional: for beta calculation
```

Returns: total return, volatility, Sharpe ratio, max drawdown, beta, portfolio weights.

### calculate_technical_levels
Calculate support and resistance levels.

```python
prices = [100.0, 102.0, ...]
method = "pivot"  # or "fibonacci", "moving_average"
```

### calculate_moving_averages
Calculate SMA and EMA for given periods.

```python
prices = [100.0, 102.0, ...]
periods = [5, 10, 20, 50, 200]  # Optional, these are defaults
```

### calculate_pivot_points
Calculate pivot points from OHLC data.

```python
high = 110.0
low = 100.0
close = 105.0
```

Returns: standard, Woodie's, and Camarilla pivot levels.

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_server.py

# Run specific test
uv run pytest -k "test_basic_returns"
```

### Project Structure

```
stock_tools_mcp/
├── server.py           # Main MCP server implementation
├── launch.py           # Cross-platform launcher
├── launch-stock-tools.sh  # Unix launcher script
├── pyproject.toml      # Project configuration
├── uv.lock             # Dependency lock file
├── shared/
│   ├── __init__.py
│   └── types.py        # MCPResponse type definition
└── tests/
    ├── __init__.py
    ├── conftest.py     # Test fixtures
    └── test_server.py  # Test suite (41 tests)
```

## License

Proprietary - Onyx R&D

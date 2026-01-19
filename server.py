"""Stock Tools MCP Server implementation."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from shared.types import MCPResponse

# Configuration from environment variables
DEBUG_MODE = os.getenv("STOCK_TOOLS_DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("STOCK_TOOLS_LOG_LEVEL", "INFO").upper()

# Financial constants (configurable via environment)
RISK_FREE_RATE = float(os.getenv("STOCK_TOOLS_RISK_FREE_RATE", "0.02"))
TRADING_DAYS_PER_YEAR = int(os.getenv("STOCK_TOOLS_TRADING_DAYS", "252"))
MAX_DATA_POINTS_FOR_LLM = int(os.getenv("STOCK_TOOLS_MAX_DATA_POINTS", "50"))
SERVER_PORT = int(os.getenv("STOCK_TOOLS_PORT", "8003"))

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("stock-tools-mcp")


def calculate_sharpe_ratio(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Series of periodic returns

    Returns:
        Annualized Sharpe ratio
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    return float(excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_volatility(returns: pd.Series) -> float:
    """Calculate annualized volatility as percentage.

    Args:
        returns: Series of periodic returns

    Returns:
        Annualized volatility percentage
    """
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown as percentage.

    Args:
        prices: Series of prices

    Returns:
        Maximum drawdown percentage (negative value)
    """
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak * 100
    return float(drawdown.min())


def calculate_total_return(prices: list) -> float:
    """Calculate total return as percentage.

    Args:
        prices: List of prices

    Returns:
        Total return percentage
    """
    return (prices[-1] / prices[0] - 1) * 100


def validate_prices(prices: List[float]) -> Optional[str]:
    """Validate a list of prices for NaN, Inf, and negative values.

    Args:
        prices: List of prices to validate

    Returns:
        Error message if validation fails, None if valid
    """
    if not prices:
        return "Prices list cannot be empty"

    price_array = np.array(prices)
    if np.any(np.isnan(price_array)):
        return "Prices contain NaN values"
    if np.any(np.isinf(price_array)):
        return "Prices contain infinite values"
    if np.any(price_array < 0):
        return "Prices cannot be negative"

    return None


class StockToolsMCPServer:
    """MCP Server for general stock analysis and utility tools."""

    def __init__(self):
        """Initialize the Stock Tools MCP server."""
        self.app = FastMCP(
            "stock-tools",
            debug=DEBUG_MODE,
            json_response=True,
            port=SERVER_PORT,
            log_level=LOG_LEVEL,
        )
        logger.info(f"Initializing Stock Tools MCP Server (debug={DEBUG_MODE})")
        self._register_tools()

    def _register_tools(self):
        """Register all MCP tools."""

        @self.app.tool()
        async def calculate_returns(
            prices: List[float], periods: Optional[List[int]] = None
        ) -> MCPResponse:
            """Calculate stock returns and performance metrics.

            Args:
                prices: List of stock prices in chronological order
                periods: Optional list of periods to calculate returns for
            """
            try:
                validation_error = validate_prices(prices)
                if validation_error:
                    return MCPResponse(success=False, error=validation_error)

                if len(prices) < 2:
                    return MCPResponse(
                        success=False,
                        error="Need at least 2 prices to calculate returns",
                    )

                price_series = pd.Series(prices)
                simple_returns = price_series.pct_change().dropna()
                cumulative_returns = (price_series / price_series.iloc[0] - 1) * 100

                # Calculate specific period returns if provided
                period_returns = {}
                if periods:
                    for period in periods:
                        if 0 < period < len(prices):
                            period_return = (prices[-1] / prices[-period] - 1) * 100
                            period_returns[f"{period}_period"] = period_return

                # Limit time series data to prevent LLM processing issues
                recent_daily_returns = simple_returns.tail(MAX_DATA_POINTS_FOR_LLM).tolist()
                recent_cumulative_returns = cumulative_returns.tail(MAX_DATA_POINTS_FOR_LLM).tolist()

                results = {
                    "total_return_pct": float(cumulative_returns.iloc[-1]),
                    "average_daily_return_pct": float(simple_returns.mean() * 100),
                    "volatility_pct": calculate_volatility(simple_returns),
                    "sharpe_ratio": calculate_sharpe_ratio(simple_returns),
                    "max_drawdown_pct": calculate_max_drawdown(price_series),
                    "period_returns": period_returns,
                    "total_data_points": len(simple_returns),
                    "recent_daily_returns": recent_daily_returns,
                    "recent_cumulative_returns": recent_cumulative_returns,
                    "data_summary": {
                        "min_daily_return": float(simple_returns.min()) if len(simple_returns) > 0 else None,
                        "max_daily_return": float(simple_returns.max()) if len(simple_returns) > 0 else None,
                        "median_daily_return": float(simple_returns.median()) if len(simple_returns) > 0 else None,
                        "recent_trend": (
                            "increasing" if len(recent_daily_returns) > 1 and recent_daily_returns[-1] > recent_daily_returns[0]
                            else "decreasing" if len(recent_daily_returns) > 1
                            else "stable"
                        ),
                    },
                }

                return MCPResponse(success=True, data=results)
            except Exception as e:
                logger.exception("Error in calculate_returns")
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def analyze_volatility(
            prices: List[float], window: int = 20
        ) -> MCPResponse:
            """Calculate volatility and risk metrics.

            Args:
                prices: List of stock prices in chronological order
                window: Rolling window for volatility calculation
            """
            try:
                validation_error = validate_prices(prices)
                if validation_error:
                    return MCPResponse(success=False, error=validation_error)

                if len(prices) < window + 1:
                    return MCPResponse(
                        success=False,
                        error=f"Need at least {window + 1} prices for window size {window}",
                    )

                price_series = pd.Series(prices)
                returns = price_series.pct_change().dropna()

                # Rolling volatility (annualized)
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

                # Historical volatility
                historical_vol = calculate_volatility(returns)

                # Value at Risk (VaR) at 95% confidence level
                var_95 = float(np.percentile(returns, 5) * 100)

                # Conditional Value at Risk (CVaR)
                cvar_95 = float(returns[returns <= np.percentile(returns, 5)].mean() * 100)

                # Calculate volatility regime
                current_vol = float(rolling_vol.iloc[-1]) if not rolling_vol.empty else historical_vol
                vol_percentile = float((rolling_vol < current_vol).mean() * 100) if not rolling_vol.empty else 50.0

                regime = "high" if vol_percentile > 75 else "low" if vol_percentile < 25 else "normal"

                # Limit rolling volatility data
                recent_rolling_vol = rolling_vol.dropna().tail(MAX_DATA_POINTS_FOR_LLM).tolist()

                results = {
                    "current_volatility_pct": current_vol,
                    "historical_volatility_pct": historical_vol,
                    "volatility_percentile": vol_percentile,
                    "volatility_regime": regime,
                    "var_95_pct": var_95,
                    "cvar_95_pct": cvar_95,
                    "total_data_points": len(rolling_vol.dropna()),
                    "recent_rolling_volatility": recent_rolling_vol,
                    "volatility_statistics": {
                        "min_volatility": float(rolling_vol.min()) if not rolling_vol.empty else None,
                        "max_volatility": float(rolling_vol.max()) if not rolling_vol.empty else None,
                        "median_volatility": float(rolling_vol.median()) if not rolling_vol.empty else None,
                        "volatility_trend": (
                            "increasing" if len(recent_rolling_vol) > 1 and recent_rolling_vol[-1] > recent_rolling_vol[0]
                            else "decreasing" if len(recent_rolling_vol) > 1
                            else "stable"
                        ),
                    },
                }

                return MCPResponse(success=True, data=results)
            except Exception as e:
                logger.exception("Error in analyze_volatility")
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def compare_stocks(stock_data: Dict[str, List[float]]) -> MCPResponse:
            """Compare multiple stocks side by side.

            Args:
                stock_data: Dictionary mapping stock symbols to price lists
            """
            try:
                if len(stock_data) < 2:
                    return MCPResponse(
                        success=False, error="Need at least 2 stocks to compare"
                    )

                # Validate all price lists
                for symbol, prices in stock_data.items():
                    validation_error = validate_prices(prices)
                    if validation_error:
                        return MCPResponse(
                            success=False,
                            error=f"Invalid prices for {symbol}: {validation_error}",
                        )

                comparison_results = {}

                for symbol, prices in stock_data.items():
                    if len(prices) < 2:
                        continue

                    price_series = pd.Series(prices)
                    returns = price_series.pct_change().dropna()

                    comparison_results[symbol] = {
                        "total_return_pct": calculate_total_return(prices),
                        "volatility_pct": calculate_volatility(returns),
                        "sharpe_ratio": calculate_sharpe_ratio(returns),
                        "max_drawdown_pct": calculate_max_drawdown(price_series),
                        "current_price": float(prices[-1]),
                        "price_change_pct": float((prices[-1] / prices[-2] - 1) * 100) if len(prices) > 1 else 0,
                    }

                # Calculate correlations
                price_df = pd.DataFrame({
                    symbol: prices
                    for symbol, prices in stock_data.items()
                    if len(prices) >= 2
                })
                returns_df = price_df.pct_change().dropna()
                correlation_matrix = returns_df.corr().to_dict()

                # Rank stocks by performance
                ranked_by_return = sorted(
                    comparison_results.items(),
                    key=lambda x: x[1]["total_return_pct"],
                    reverse=True,
                )
                ranked_by_sharpe = sorted(
                    comparison_results.items(),
                    key=lambda x: x[1]["sharpe_ratio"],
                    reverse=True,
                )

                results = {
                    "stock_metrics": comparison_results,
                    "correlation_matrix": correlation_matrix,
                    "ranked_by_return": [
                        {"symbol": s, "return_pct": m["total_return_pct"]}
                        for s, m in ranked_by_return
                    ],
                    "ranked_by_sharpe": [
                        {"symbol": s, "sharpe_ratio": m["sharpe_ratio"]}
                        for s, m in ranked_by_sharpe
                    ],
                }

                return MCPResponse(success=True, data=results)
            except Exception as e:
                logger.exception("Error in compare_stocks")
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def screen_stocks(criteria: Dict[str, Any]) -> MCPResponse:
            """Screen stocks based on financial criteria.

            Args:
                criteria: Dictionary containing 'stocks' array and filter criteria.
                    Required: stocks (array of stock objects)
                    Optional filters: min/max_market_cap, min/max_pe_ratio,
                    min/max_dividend_yield, sectors, exclude_sectors, etc.
            """
            try:
                stocks = criteria.get("stocks", [])
                if not stocks:
                    return MCPResponse(
                        success=False,
                        error="No stocks array provided in criteria. Please include 'stocks' array with stock data to screen.",
                    )

                # Required fields for each stock
                required_fields = [
                    "symbol", "sector", "market_cap", "price", "pe_ratio",
                    "dividend_yield", "debt_to_equity", "roe", "current_ratio",
                    "revenue_growth", "volume"
                ]

                # Validate stock data
                for i, stock in enumerate(stocks):
                    missing = [f for f in required_fields if f not in stock]
                    if missing:
                        return MCPResponse(
                            success=False,
                            error=f"Stock at index {i} missing required fields: {missing}",
                        )

                # Limit stocks to prevent processing issues
                max_stocks = 500
                warning_message = None
                if len(stocks) > max_stocks:
                    stocks = stocks[:max_stocks]
                    warning_message = f"Stock list truncated to {max_stocks} stocks"

                # Extract filter criteria with defaults
                filters = {
                    "min_market_cap": criteria.get("min_market_cap", 0),
                    "max_market_cap": criteria.get("max_market_cap", float("inf")),
                    "min_pe_ratio": criteria.get("min_pe_ratio", 0),
                    "max_pe_ratio": criteria.get("max_pe_ratio", float("inf")),
                    "min_dividend_yield": criteria.get("min_dividend_yield", 0),
                    "max_dividend_yield": criteria.get("max_dividend_yield", float("inf")),
                    "min_debt_to_equity": criteria.get("min_debt_to_equity", 0),
                    "max_debt_to_equity": criteria.get("max_debt_to_equity", float("inf")),
                    "min_roe": criteria.get("min_roe", 0),
                    "max_roe": criteria.get("max_roe", float("inf")),
                    "min_current_ratio": criteria.get("min_current_ratio", 0),
                    "max_current_ratio": criteria.get("max_current_ratio", float("inf")),
                    "min_revenue_growth": criteria.get("min_revenue_growth", float("-inf")),
                    "max_revenue_growth": criteria.get("max_revenue_growth", float("inf")),
                    "min_volume": criteria.get("min_volume", 0),
                    "min_price": criteria.get("min_price", 0),
                    "max_price": criteria.get("max_price", float("inf")),
                }
                sectors = criteria.get("sectors", [])
                exclude_sectors = criteria.get("exclude_sectors", [])

                # Apply screening criteria
                screened_stocks = []
                for stock in stocks:
                    # Numeric filters
                    if not (filters["min_market_cap"] <= stock["market_cap"] <= filters["max_market_cap"]):
                        continue
                    if not (filters["min_pe_ratio"] <= stock["pe_ratio"] <= filters["max_pe_ratio"]):
                        continue
                    if not (filters["min_dividend_yield"] <= stock["dividend_yield"] <= filters["max_dividend_yield"]):
                        continue
                    if not (filters["min_debt_to_equity"] <= stock["debt_to_equity"] <= filters["max_debt_to_equity"]):
                        continue
                    if not (filters["min_roe"] <= stock["roe"] <= filters["max_roe"]):
                        continue
                    if not (filters["min_current_ratio"] <= stock["current_ratio"] <= filters["max_current_ratio"]):
                        continue
                    if not (filters["min_revenue_growth"] <= stock["revenue_growth"] <= filters["max_revenue_growth"]):
                        continue
                    if not (filters["min_volume"] <= stock["volume"]):
                        continue
                    if not (filters["min_price"] <= stock["price"] <= filters["max_price"]):
                        continue

                    # Sector filters
                    if sectors and stock["sector"] not in sectors:
                        continue
                    if exclude_sectors and stock["sector"] in exclude_sectors:
                        continue

                    screened_stocks.append(stock)

                # Sort by market cap descending
                screened_stocks.sort(key=lambda x: x["market_cap"], reverse=True)

                results = {
                    "criteria_applied": criteria,
                    "stocks_found": len(screened_stocks),
                    "total_stocks_screened": len(stocks),
                    "screened_stocks": screened_stocks,
                    "available_sectors": list(set(stock.get("sector", "Unknown") for stock in stocks)),
                    "warning": warning_message,
                }

                return MCPResponse(success=True, data=results)
            except Exception as e:
                logger.exception("Error in screen_stocks")
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def calculate_portfolio_metrics(
            holdings: Dict[str, float],
            prices: Dict[str, List[float]],
            benchmark_prices: Optional[List[float]] = None,
        ) -> MCPResponse:
            """Calculate portfolio analysis and optimization metrics.

            Args:
                holdings: Dictionary mapping stock symbols to quantities held
                prices: Dictionary mapping stock symbols to price histories
                benchmark_prices: Optional benchmark price history for comparison
            """
            try:
                # Validate all price lists
                for symbol, price_list in prices.items():
                    validation_error = validate_prices(price_list)
                    if validation_error:
                        return MCPResponse(
                            success=False,
                            error=f"Invalid prices for {symbol}: {validation_error}",
                        )

                if benchmark_prices:
                    validation_error = validate_prices(benchmark_prices)
                    if validation_error:
                        return MCPResponse(
                            success=False,
                            error=f"Invalid benchmark prices: {validation_error}",
                        )

                # Calculate portfolio value over time
                portfolio_values = []
                min_length = min(len(price_list) for price_list in prices.values())

                for i in range(min_length):
                    total_value = sum(
                        holdings[symbol] * prices[symbol][i]
                        for symbol in holdings.keys()
                        if symbol in prices
                    )
                    portfolio_values.append(total_value)

                if len(portfolio_values) < 2:
                    return MCPResponse(
                        success=False,
                        error="Need at least 2 data points for portfolio analysis",
                    )

                portfolio_series = pd.Series(portfolio_values)
                portfolio_returns = portfolio_series.pct_change().dropna()

                # Current portfolio composition
                current_values = {
                    symbol: holdings[symbol] * prices[symbol][-1]
                    for symbol in holdings.keys()
                    if symbol in prices
                }
                total_current_value = sum(current_values.values())

                if total_current_value == 0:
                    return MCPResponse(
                        success=False,
                        error="Portfolio has zero total value - cannot calculate weights",
                    )

                portfolio_weights = {
                    symbol: value / total_current_value
                    for symbol, value in current_values.items()
                }

                # Beta calculation (if benchmark provided)
                beta = None
                if benchmark_prices and len(benchmark_prices) >= min_length:
                    benchmark_series = pd.Series(benchmark_prices[:min_length])
                    benchmark_returns = benchmark_series.pct_change().dropna()

                    if len(portfolio_returns) == len(benchmark_returns):
                        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
                        benchmark_variance = np.var(benchmark_returns)
                        if benchmark_variance != 0:
                            beta = float(covariance / benchmark_variance)

                results = {
                    "total_return_pct": calculate_total_return(portfolio_values),
                    "volatility_pct": calculate_volatility(portfolio_returns),
                    "sharpe_ratio": calculate_sharpe_ratio(portfolio_returns),
                    "max_drawdown_pct": calculate_max_drawdown(portfolio_series),
                    "beta": beta,
                    "portfolio_value": float(total_current_value),
                    "portfolio_weights": {k: float(v) for k, v in portfolio_weights.items()},
                    "holdings": holdings,
                    "portfolio_values": portfolio_values,
                }

                return MCPResponse(success=True, data=results)
            except Exception as e:
                logger.exception("Error in calculate_portfolio_metrics")
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def calculate_technical_levels(
            prices: List[float], method: str = "pivot"
        ) -> MCPResponse:
            """Calculate support and resistance levels.

            Args:
                prices: List of stock prices
                method: Method to use ('pivot', 'fibonacci', 'moving_average')
            """
            try:
                validation_error = validate_prices(prices)
                if validation_error:
                    return MCPResponse(success=False, error=validation_error)

                if len(prices) < 3:
                    return MCPResponse(
                        success=False,
                        error="Need at least 3 prices for technical analysis",
                    )

                price_series = pd.Series(prices)
                current_price = prices[-1]

                if method == "pivot":
                    if len(prices) < 10:
                        return MCPResponse(
                            success=False,
                            error="Need at least 10 prices for reliable pivot point calculation",
                        )

                    lookback_period = min(10, len(prices) - 1)
                    previous_period_prices = prices[-(lookback_period + 1) : -1]

                    high = max(previous_period_prices)
                    low = min(previous_period_prices)
                    close = previous_period_prices[-1]

                    pivot = (high + low + close) / 3
                    r1 = 2 * pivot - low
                    r2 = pivot + (high - low)
                    r3 = high + 2 * (pivot - low)
                    s1 = 2 * pivot - high
                    s2 = pivot - (high - low)
                    s3 = low - 2 * (high - pivot)

                    levels = {
                        "pivot_point": round(pivot, 2),
                        "resistance_1": round(r1, 2),
                        "resistance_2": round(r2, 2),
                        "resistance_3": round(r3, 2),
                        "support_1": round(s1, 2),
                        "support_2": round(s2, 2),
                        "support_3": round(s3, 2),
                        "high_prev": round(high, 2),
                        "low_prev": round(low, 2),
                        "close_prev": round(close, 2),
                    }

                elif method == "fibonacci":
                    recent_high = max(prices[-20:])
                    recent_low = min(prices[-20:])
                    diff = recent_high - recent_low

                    levels = {
                        "100%": round(recent_high, 2),
                        "78.6%": round(recent_high - 0.786 * diff, 2),
                        "61.8%": round(recent_high - 0.618 * diff, 2),
                        "50.0%": round(recent_high - 0.5 * diff, 2),
                        "38.2%": round(recent_high - 0.382 * diff, 2),
                        "23.6%": round(recent_high - 0.236 * diff, 2),
                        "0%": round(recent_low, 2),
                    }

                elif method == "moving_average":
                    if len(prices) < 20:
                        return MCPResponse(
                            success=False,
                            error="Need at least 20 prices for moving average analysis",
                        )

                    ma_20 = price_series.rolling(window=20).mean().iloc[-1]
                    std_20 = price_series.rolling(window=20).std().iloc[-1]

                    levels = {
                        "ma_20": round(ma_20, 2),
                        "upper_band": round(ma_20 + 2 * std_20, 2),
                        "lower_band": round(ma_20 - 2 * std_20, 2),
                    }

                    if len(prices) >= 50:
                        ma_50 = price_series.rolling(window=50).mean().iloc[-1]
                        levels["ma_50"] = round(ma_50, 2)

                else:
                    return MCPResponse(success=False, error=f"Unknown method: {method}")

                # Determine nearest support and resistance
                levels_above = [v for v in levels.values() if isinstance(v, (int, float)) and v > current_price]
                levels_below = [v for v in levels.values() if isinstance(v, (int, float)) and v < current_price]

                results = {
                    "method": method,
                    "current_price": current_price,
                    "levels": levels,
                    "nearest_support": max(levels_below) if levels_below else None,
                    "nearest_resistance": min(levels_above) if levels_above else None,
                }

                return MCPResponse(success=True, data=results)
            except Exception as e:
                logger.exception("Error in calculate_technical_levels")
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def calculate_moving_averages(
            prices: List[float], periods: Optional[List[int]] = None
        ) -> MCPResponse:
            """Calculate moving averages for given periods.

            Args:
                prices: List of stock prices in chronological order
                periods: List of periods to calculate (default: [5, 10, 20, 50, 200])
            """
            try:
                validation_error = validate_prices(prices)
                if validation_error:
                    return MCPResponse(success=False, error=validation_error)

                if len(prices) < 5:
                    return MCPResponse(
                        success=False,
                        error="Need at least 5 prices for moving average calculation",
                    )

                if periods is None:
                    periods = [5, 10, 20, 50, 200]

                price_series = pd.Series(prices)
                current_price = prices[-1]
                moving_averages = {}

                for period in periods:
                    if len(prices) >= period:
                        ma = price_series.rolling(window=period).mean().iloc[-1]
                        moving_averages[f"ma_{period}"] = round(ma, 2)
                        moving_averages[f"ma_{period}_distance"] = round(((current_price - ma) / ma) * 100, 2)

                # Calculate EMAs for periods <= 50
                ema_periods = [12, 26] if not periods else [p for p in periods if p <= 50]
                for period in ema_periods:
                    if len(prices) >= period:
                        ema = price_series.ewm(span=period).mean().iloc[-1]
                        moving_averages[f"ema_{period}"] = round(ema, 2)
                        moving_averages[f"ema_{period}_distance"] = round(((current_price - ema) / ema) * 100, 2)

                results = {
                    "current_price": current_price,
                    "moving_averages": moving_averages,
                    "data_points": len(prices),
                }

                return MCPResponse(success=True, data=results)
            except Exception as e:
                logger.exception("Error in calculate_moving_averages")
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def calculate_pivot_points(
            high: float, low: float, close: float
        ) -> MCPResponse:
            """Calculate standard daily pivot points using exact OHLC data.

            Args:
                high: Previous day's high price
                low: Previous day's low price
                close: Previous day's close price
            """
            try:
                pivot = (high + low + close) / 3

                # Standard pivots
                r1 = 2 * pivot - low
                r2 = pivot + (high - low)
                r3 = high + 2 * (pivot - low)
                s1 = 2 * pivot - high
                s2 = pivot - (high - low)
                s3 = low - 2 * (high - pivot)

                # Woodie's pivots
                woodie_pivot = (high + low + 2 * close) / 4
                woodie_r1 = 2 * woodie_pivot - low
                woodie_s1 = 2 * woodie_pivot - high

                # Camarilla pivots
                cam_range = high - low
                cam_r1 = close + cam_range * 1.1 / 12
                cam_r2 = close + cam_range * 1.1 / 6
                cam_r3 = close + cam_range * 1.1 / 4
                cam_s1 = close - cam_range * 1.1 / 12
                cam_s2 = close - cam_range * 1.1 / 6
                cam_s3 = close - cam_range * 1.1 / 4

                results = {
                    "input_data": {
                        "high": round(high, 2),
                        "low": round(low, 2),
                        "close": round(close, 2),
                        "range": round(high - low, 2),
                    },
                    "standard_pivots": {
                        "pivot_point": round(pivot, 2),
                        "resistance_1": round(r1, 2),
                        "resistance_2": round(r2, 2),
                        "resistance_3": round(r3, 2),
                        "support_1": round(s1, 2),
                        "support_2": round(s2, 2),
                        "support_3": round(s3, 2),
                    },
                    "woodie_pivots": {
                        "pivot_point": round(woodie_pivot, 2),
                        "resistance_1": round(woodie_r1, 2),
                        "support_1": round(woodie_s1, 2),
                    },
                    "camarilla_pivots": {
                        "resistance_1": round(cam_r1, 2),
                        "resistance_2": round(cam_r2, 2),
                        "resistance_3": round(cam_r3, 2),
                        "support_1": round(cam_s1, 2),
                        "support_2": round(cam_s2, 2),
                        "support_3": round(cam_s3, 2),
                    },
                }

                return MCPResponse(success=True, data=results)
            except Exception as e:
                logger.exception("Error in calculate_pivot_points")
                return MCPResponse(success=False, error=str(e))


def main():
    """Main entry point for the Stock Tools MCP server."""
    import sys

    transport = "stdio"

    if "--sse" in sys.argv or os.getenv("MCP_TRANSPORT") == "sse":
        transport = "sse"
    elif "--streamable" in sys.argv or os.getenv("MCP_TRANSPORT") == "streamable":
        transport = "streamable-http"

    logger.info(f"Starting Stock Tools MCP server with transport={transport}")
    server = StockToolsMCPServer()
    server.app.run(transport=transport)


if __name__ == "__main__":
    main()

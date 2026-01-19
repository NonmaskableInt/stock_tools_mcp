"""Tests for Stock Tools MCP Server tools."""

import pytest
from datetime import datetime


class TestCalculateReturns:
    """Tests for calculate_returns tool."""

    @pytest.mark.asyncio
    async def test_basic_returns(self, server, sample_prices):
        """Test basic return calculations."""
        tool = server.app._tool_manager._tools["calculate_returns"]
        result = await tool.fn(prices=sample_prices)

        assert result.success is True
        assert result.data is not None
        assert "total_return_pct" in result.data
        assert "average_daily_return_pct" in result.data
        assert "volatility_pct" in result.data
        assert "sharpe_ratio" in result.data
        assert "max_drawdown_pct" in result.data

    @pytest.mark.asyncio
    async def test_returns_with_periods(self, server, sample_prices):
        """Test returns with specific periods."""
        tool = server.app._tool_manager._tools["calculate_returns"]
        result = await tool.fn(prices=sample_prices, periods=[3, 5])

        assert result.success is True
        assert "period_returns" in result.data
        assert "3_period" in result.data["period_returns"]
        assert "5_period" in result.data["period_returns"]

    @pytest.mark.asyncio
    async def test_returns_insufficient_data(self, server):
        """Test returns with insufficient data."""
        tool = server.app._tool_manager._tools["calculate_returns"]
        result = await tool.fn(prices=[100.0])

        assert result.success is False
        assert "at least 2 prices" in result.error.lower()

    @pytest.mark.asyncio
    async def test_returns_positive_trend(self, server):
        """Test returns with clear positive trend."""
        prices = [100.0, 105.0, 110.0, 115.0, 120.0]
        tool = server.app._tool_manager._tools["calculate_returns"]
        result = await tool.fn(prices=prices)

        assert result.success is True
        assert result.data["total_return_pct"] == pytest.approx(20.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_returns_data_summary(self, server, sample_prices):
        """Test that data summary is included."""
        tool = server.app._tool_manager._tools["calculate_returns"]
        result = await tool.fn(prices=sample_prices)

        assert result.success is True
        assert "data_summary" in result.data
        assert "min_daily_return" in result.data["data_summary"]
        assert "max_daily_return" in result.data["data_summary"]
        assert "median_daily_return" in result.data["data_summary"]


class TestAnalyzeVolatility:
    """Tests for analyze_volatility tool."""

    @pytest.mark.asyncio
    async def test_basic_volatility(self, server, extended_prices):
        """Test basic volatility analysis."""
        tool = server.app._tool_manager._tools["analyze_volatility"]
        result = await tool.fn(prices=extended_prices, window=20)

        assert result.success is True
        assert "current_volatility_pct" in result.data
        assert "historical_volatility_pct" in result.data
        assert "var_95_pct" in result.data
        assert "cvar_95_pct" in result.data
        assert "volatility_regime" in result.data

    @pytest.mark.asyncio
    async def test_volatility_regime_classification(self, server, extended_prices):
        """Test that volatility regime is properly classified."""
        tool = server.app._tool_manager._tools["analyze_volatility"]
        result = await tool.fn(prices=extended_prices, window=20)

        assert result.success is True
        assert result.data["volatility_regime"] in ["high", "normal", "low"]

    @pytest.mark.asyncio
    async def test_volatility_insufficient_data(self, server):
        """Test volatility with insufficient data for window."""
        tool = server.app._tool_manager._tools["analyze_volatility"]
        result = await tool.fn(prices=[100.0, 101.0, 102.0], window=20)

        assert result.success is False
        assert "at least" in result.error.lower()

    @pytest.mark.asyncio
    async def test_volatility_custom_window(self, server, extended_prices):
        """Test volatility with custom window size."""
        tool = server.app._tool_manager._tools["analyze_volatility"]
        result = await tool.fn(prices=extended_prices, window=10)

        assert result.success is True
        assert result.data is not None


class TestCompareStocks:
    """Tests for compare_stocks tool."""

    @pytest.mark.asyncio
    async def test_basic_comparison(self, server):
        """Test basic stock comparison."""
        stock_data = {
            "AAPL": [150.0, 152.0, 151.0, 155.0, 158.0],
            "MSFT": [300.0, 305.0, 302.0, 310.0, 315.0],
        }
        tool = server.app._tool_manager._tools["compare_stocks"]
        result = await tool.fn(stock_data=stock_data)

        assert result.success is True
        assert "stock_metrics" in result.data
        assert "AAPL" in result.data["stock_metrics"]
        assert "MSFT" in result.data["stock_metrics"]
        assert "correlation_matrix" in result.data
        assert "ranked_by_return" in result.data
        assert "ranked_by_sharpe" in result.data

    @pytest.mark.asyncio
    async def test_comparison_metrics(self, server):
        """Test that all metrics are calculated for each stock."""
        stock_data = {
            "AAPL": [150.0, 152.0, 151.0, 155.0, 158.0],
            "MSFT": [300.0, 305.0, 302.0, 310.0, 315.0],
        }
        tool = server.app._tool_manager._tools["compare_stocks"]
        result = await tool.fn(stock_data=stock_data)

        assert result.success is True
        for symbol in ["AAPL", "MSFT"]:
            metrics = result.data["stock_metrics"][symbol]
            assert "total_return_pct" in metrics
            assert "volatility_pct" in metrics
            assert "sharpe_ratio" in metrics
            assert "max_drawdown_pct" in metrics
            assert "current_price" in metrics

    @pytest.mark.asyncio
    async def test_comparison_insufficient_stocks(self, server):
        """Test comparison with only one stock."""
        stock_data = {"AAPL": [150.0, 152.0, 151.0, 155.0, 158.0]}
        tool = server.app._tool_manager._tools["compare_stocks"]
        result = await tool.fn(stock_data=stock_data)

        assert result.success is False
        assert "at least 2 stocks" in result.error.lower()


class TestScreenStocks:
    """Tests for screen_stocks tool."""

    @pytest.mark.asyncio
    async def test_basic_screening(self, server, sample_stock_data):
        """Test basic stock screening."""
        criteria = {"stocks": sample_stock_data}
        tool = server.app._tool_manager._tools["screen_stocks"]
        result = await tool.fn(criteria=criteria)

        assert result.success is True
        assert "screened_stocks" in result.data
        assert "stocks_found" in result.data
        assert len(result.data["screened_stocks"]) == 4

    @pytest.mark.asyncio
    async def test_screening_by_sector(self, server, sample_stock_data):
        """Test screening by sector."""
        criteria = {"stocks": sample_stock_data, "sectors": ["Technology"]}
        tool = server.app._tool_manager._tools["screen_stocks"]
        result = await tool.fn(criteria=criteria)

        assert result.success is True
        assert result.data["stocks_found"] == 2
        for stock in result.data["screened_stocks"]:
            assert stock["sector"] == "Technology"

    @pytest.mark.asyncio
    async def test_screening_exclude_sector(self, server, sample_stock_data):
        """Test screening with sector exclusion."""
        criteria = {"stocks": sample_stock_data, "exclude_sectors": ["Energy"]}
        tool = server.app._tool_manager._tools["screen_stocks"]
        result = await tool.fn(criteria=criteria)

        assert result.success is True
        assert result.data["stocks_found"] == 3
        for stock in result.data["screened_stocks"]:
            assert stock["sector"] != "Energy"

    @pytest.mark.asyncio
    async def test_screening_pe_ratio(self, server, sample_stock_data):
        """Test screening by P/E ratio range."""
        criteria = {
            "stocks": sample_stock_data,
            "min_pe_ratio": 10.0,
            "max_pe_ratio": 20.0,
        }
        tool = server.app._tool_manager._tools["screen_stocks"]
        result = await tool.fn(criteria=criteria)

        assert result.success is True
        for stock in result.data["screened_stocks"]:
            assert 10.0 <= stock["pe_ratio"] <= 20.0

    @pytest.mark.asyncio
    async def test_screening_market_cap(self, server, sample_stock_data):
        """Test screening by market cap."""
        criteria = {
            "stocks": sample_stock_data,
            "min_market_cap": 1000000000000,  # 1 trillion
        }
        tool = server.app._tool_manager._tools["screen_stocks"]
        result = await tool.fn(criteria=criteria)

        assert result.success is True
        assert result.data["stocks_found"] == 2
        for stock in result.data["screened_stocks"]:
            assert stock["market_cap"] >= 1000000000000

    @pytest.mark.asyncio
    async def test_screening_no_stocks(self, server):
        """Test screening with no stocks provided."""
        criteria = {}
        tool = server.app._tool_manager._tools["screen_stocks"]
        result = await tool.fn(criteria=criteria)

        assert result.success is False
        assert "no stocks" in result.error.lower()

    @pytest.mark.asyncio
    async def test_screening_missing_required_fields(self, server):
        """Test screening with stocks missing required fields."""
        criteria = {
            "stocks": [
                {"symbol": "AAPL", "price": 175.0}  # Missing most required fields
            ]
        }
        tool = server.app._tool_manager._tools["screen_stocks"]
        result = await tool.fn(criteria=criteria)

        assert result.success is False
        assert "missing required fields" in result.error.lower()


class TestCalculatePortfolioMetrics:
    """Tests for calculate_portfolio_metrics tool."""

    @pytest.mark.asyncio
    async def test_basic_portfolio(self, server):
        """Test basic portfolio metrics calculation."""
        holdings = {"AAPL": 10, "MSFT": 5}
        prices = {
            "AAPL": [150.0, 152.0, 151.0, 155.0, 158.0],
            "MSFT": [300.0, 305.0, 302.0, 310.0, 315.0],
        }
        tool = server.app._tool_manager._tools["calculate_portfolio_metrics"]
        result = await tool.fn(holdings=holdings, prices=prices)

        assert result.success is True
        assert "total_return_pct" in result.data
        assert "volatility_pct" in result.data
        assert "sharpe_ratio" in result.data
        assert "max_drawdown_pct" in result.data
        assert "portfolio_value" in result.data
        assert "portfolio_weights" in result.data

    @pytest.mark.asyncio
    async def test_portfolio_with_benchmark(self, server):
        """Test portfolio metrics with benchmark comparison."""
        holdings = {"AAPL": 10, "MSFT": 5}
        prices = {
            "AAPL": [150.0, 152.0, 151.0, 155.0, 158.0],
            "MSFT": [300.0, 305.0, 302.0, 310.0, 315.0],
        }
        benchmark = [400.0, 405.0, 402.0, 410.0, 415.0]
        tool = server.app._tool_manager._tools["calculate_portfolio_metrics"]
        result = await tool.fn(
            holdings=holdings, prices=prices, benchmark_prices=benchmark
        )

        assert result.success is True
        assert "beta" in result.data

    @pytest.mark.asyncio
    async def test_portfolio_weights_sum_to_one(self, server):
        """Test that portfolio weights sum to 1."""
        holdings = {"AAPL": 10, "MSFT": 5, "GOOGL": 3}
        prices = {
            "AAPL": [150.0, 152.0, 151.0, 155.0, 158.0],
            "MSFT": [300.0, 305.0, 302.0, 310.0, 315.0],
            "GOOGL": [140.0, 142.0, 141.0, 145.0, 148.0],
        }
        tool = server.app._tool_manager._tools["calculate_portfolio_metrics"]
        result = await tool.fn(holdings=holdings, prices=prices)

        assert result.success is True
        weights_sum = sum(result.data["portfolio_weights"].values())
        assert weights_sum == pytest.approx(1.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_portfolio_insufficient_data(self, server):
        """Test portfolio with insufficient data points."""
        holdings = {"AAPL": 10}
        prices = {"AAPL": [150.0]}
        tool = server.app._tool_manager._tools["calculate_portfolio_metrics"]
        result = await tool.fn(holdings=holdings, prices=prices)

        assert result.success is False
        assert "at least 2 data points" in result.error.lower()


class TestCalculateTechnicalLevels:
    """Tests for calculate_technical_levels tool."""

    @pytest.mark.asyncio
    async def test_pivot_method(self, server, extended_prices):
        """Test pivot point calculation."""
        tool = server.app._tool_manager._tools["calculate_technical_levels"]
        result = await tool.fn(prices=extended_prices, method="pivot")

        assert result.success is True
        assert "levels" in result.data
        assert "pivot_point" in result.data["levels"]
        assert "resistance_1" in result.data["levels"]
        assert "support_1" in result.data["levels"]
        assert "nearest_support" in result.data
        assert "nearest_resistance" in result.data

    @pytest.mark.asyncio
    async def test_fibonacci_method(self, server, extended_prices):
        """Test Fibonacci retracement levels."""
        tool = server.app._tool_manager._tools["calculate_technical_levels"]
        result = await tool.fn(prices=extended_prices, method="fibonacci")

        assert result.success is True
        assert "levels" in result.data
        assert "61.8%" in result.data["levels"]
        assert "38.2%" in result.data["levels"]
        assert "50.0%" in result.data["levels"]

    @pytest.mark.asyncio
    async def test_moving_average_method(self, server, extended_prices):
        """Test moving average support/resistance."""
        tool = server.app._tool_manager._tools["calculate_technical_levels"]
        result = await tool.fn(prices=extended_prices, method="moving_average")

        assert result.success is True
        assert "levels" in result.data
        assert "ma_20" in result.data["levels"]
        assert "upper_band" in result.data["levels"]
        assert "lower_band" in result.data["levels"]

    @pytest.mark.asyncio
    async def test_unknown_method(self, server, extended_prices):
        """Test with unknown method."""
        tool = server.app._tool_manager._tools["calculate_technical_levels"]
        result = await tool.fn(prices=extended_prices, method="unknown")

        assert result.success is False
        assert "unknown method" in result.error.lower()

    @pytest.mark.asyncio
    async def test_insufficient_data(self, server):
        """Test with insufficient data."""
        tool = server.app._tool_manager._tools["calculate_technical_levels"]
        result = await tool.fn(prices=[100.0, 101.0], method="pivot")

        assert result.success is False


class TestCalculateMovingAverages:
    """Tests for calculate_moving_averages tool."""

    @pytest.mark.asyncio
    async def test_default_periods(self, server, extended_prices):
        """Test moving averages with default periods."""
        tool = server.app._tool_manager._tools["calculate_moving_averages"]
        result = await tool.fn(prices=extended_prices)

        assert result.success is True
        assert "moving_averages" in result.data
        assert "current_price" in result.data
        assert "ma_5" in result.data["moving_averages"]
        assert "ma_10" in result.data["moving_averages"]
        assert "ma_20" in result.data["moving_averages"]

    @pytest.mark.asyncio
    async def test_custom_periods(self, server, extended_prices):
        """Test moving averages with custom periods."""
        tool = server.app._tool_manager._tools["calculate_moving_averages"]
        result = await tool.fn(prices=extended_prices, periods=[7, 14, 21])

        assert result.success is True
        assert "ma_7" in result.data["moving_averages"]
        assert "ma_14" in result.data["moving_averages"]
        assert "ma_21" in result.data["moving_averages"]

    @pytest.mark.asyncio
    async def test_ema_included(self, server, extended_prices):
        """Test that EMA is included based on periods <= 50."""
        tool = server.app._tool_manager._tools["calculate_moving_averages"]
        result = await tool.fn(prices=extended_prices)

        assert result.success is True
        # EMA uses periods <= 50 from the input periods list
        # Default periods are [5, 10, 20, 50, 200], so EMA uses [5, 10, 20, 50]
        assert "ema_5" in result.data["moving_averages"]
        assert "ema_10" in result.data["moving_averages"]
        assert "ema_20" in result.data["moving_averages"]
        assert "ema_50" in result.data["moving_averages"]

    @pytest.mark.asyncio
    async def test_distance_from_ma(self, server, extended_prices):
        """Test that distance from MA is calculated."""
        tool = server.app._tool_manager._tools["calculate_moving_averages"]
        result = await tool.fn(prices=extended_prices)

        assert result.success is True
        assert "ma_20_distance" in result.data["moving_averages"]

    @pytest.mark.asyncio
    async def test_insufficient_data(self, server):
        """Test with insufficient data."""
        tool = server.app._tool_manager._tools["calculate_moving_averages"]
        result = await tool.fn(prices=[100.0, 101.0, 102.0])

        assert result.success is False
        assert "at least 5 prices" in result.error.lower()


class TestCalculatePivotPoints:
    """Tests for calculate_pivot_points tool."""

    @pytest.mark.asyncio
    async def test_standard_pivots(self, server):
        """Test standard pivot point calculation."""
        tool = server.app._tool_manager._tools["calculate_pivot_points"]
        result = await tool.fn(high=110.0, low=100.0, close=105.0)

        assert result.success is True
        assert "standard_pivots" in result.data
        pivots = result.data["standard_pivots"]
        assert "pivot_point" in pivots
        assert "resistance_1" in pivots
        assert "resistance_2" in pivots
        assert "resistance_3" in pivots
        assert "support_1" in pivots
        assert "support_2" in pivots
        assert "support_3" in pivots

    @pytest.mark.asyncio
    async def test_woodie_pivots(self, server):
        """Test Woodie's pivot points."""
        tool = server.app._tool_manager._tools["calculate_pivot_points"]
        result = await tool.fn(high=110.0, low=100.0, close=105.0)

        assert result.success is True
        assert "woodie_pivots" in result.data

    @pytest.mark.asyncio
    async def test_camarilla_pivots(self, server):
        """Test Camarilla pivot points."""
        tool = server.app._tool_manager._tools["calculate_pivot_points"]
        result = await tool.fn(high=110.0, low=100.0, close=105.0)

        assert result.success is True
        assert "camarilla_pivots" in result.data

    @pytest.mark.asyncio
    async def test_pivot_formula(self, server):
        """Test that pivot point formula is correct: (H + L + C) / 3."""
        tool = server.app._tool_manager._tools["calculate_pivot_points"]
        result = await tool.fn(high=120.0, low=100.0, close=110.0)

        assert result.success is True
        expected_pivot = (120.0 + 100.0 + 110.0) / 3
        assert result.data["standard_pivots"]["pivot_point"] == pytest.approx(
            expected_pivot, rel=0.01
        )

    @pytest.mark.asyncio
    async def test_input_data_echoed(self, server):
        """Test that input data is echoed back."""
        tool = server.app._tool_manager._tools["calculate_pivot_points"]
        result = await tool.fn(high=110.0, low=100.0, close=105.0)

        assert result.success is True
        assert "input_data" in result.data
        assert result.data["input_data"]["high"] == 110.0
        assert result.data["input_data"]["low"] == 100.0
        assert result.data["input_data"]["close"] == 105.0


class TestMCPResponseFormat:
    """Tests for MCPResponse format consistency."""

    @pytest.mark.asyncio
    async def test_success_response_has_data(self, server, sample_prices):
        """Test that successful responses have data."""
        tool = server.app._tool_manager._tools["calculate_returns"]
        result = await tool.fn(prices=sample_prices)

        assert result.success is True
        assert result.data is not None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_error_response_has_message(self, server):
        """Test that error responses have error message."""
        tool = server.app._tool_manager._tools["calculate_returns"]
        result = await tool.fn(prices=[100.0])

        assert result.success is False
        assert result.error is not None
        assert result.data is None

    @pytest.mark.asyncio
    async def test_response_has_timestamp(self, server, sample_prices):
        """Test that responses have timestamp."""
        tool = server.app._tool_manager._tools["calculate_returns"]
        result = await tool.fn(prices=sample_prices)

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

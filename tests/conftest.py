"""Pytest configuration and fixtures for Stock Tools MCP Server tests."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import StockToolsMCPServer


@pytest.fixture
def server():
    """Create a StockToolsMCPServer instance for testing."""
    return StockToolsMCPServer()


@pytest.fixture
def sample_prices():
    """Sample price data for testing."""
    return [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0]


@pytest.fixture
def extended_prices():
    """Extended price data for tests requiring more data points."""
    import numpy as np
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices


@pytest.fixture
def sample_stock_data():
    """Sample stock data for screening tests."""
    return [
        {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "market_cap": 3000000000000,
            "price": 175.50,
            "pe_ratio": 25.5,
            "dividend_yield": 0.005,
            "debt_to_equity": 1.8,
            "roe": 0.25,
            "current_ratio": 1.2,
            "revenue_growth": 0.08,
            "volume": 50000000,
            "52_week_high": 198.23,
            "52_week_low": 124.17,
        },
        {
            "symbol": "MSFT",
            "company_name": "Microsoft Corporation",
            "sector": "Technology",
            "market_cap": 2800000000000,
            "price": 380.00,
            "pe_ratio": 35.0,
            "dividend_yield": 0.008,
            "debt_to_equity": 0.5,
            "roe": 0.35,
            "current_ratio": 1.8,
            "revenue_growth": 0.12,
            "volume": 25000000,
            "52_week_high": 420.00,
            "52_week_low": 280.00,
        },
        {
            "symbol": "JNJ",
            "company_name": "Johnson & Johnson",
            "sector": "Healthcare",
            "market_cap": 400000000000,
            "price": 160.00,
            "pe_ratio": 15.0,
            "dividend_yield": 0.03,
            "debt_to_equity": 0.4,
            "roe": 0.20,
            "current_ratio": 1.5,
            "revenue_growth": 0.03,
            "volume": 8000000,
            "52_week_high": 180.00,
            "52_week_low": 140.00,
        },
        {
            "symbol": "XOM",
            "company_name": "Exxon Mobil",
            "sector": "Energy",
            "market_cap": 450000000000,
            "price": 105.00,
            "pe_ratio": 12.0,
            "dividend_yield": 0.035,
            "debt_to_equity": 0.3,
            "roe": 0.18,
            "current_ratio": 1.3,
            "revenue_growth": -0.02,
            "volume": 15000000,
            "52_week_high": 120.00,
            "52_week_low": 85.00,
        },
    ]

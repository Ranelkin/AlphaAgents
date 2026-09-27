from src.util.log_config import setup_logging

from .metrics import (
    annualized_return,
    annualized_volatility,
    cumulative_return_from_prices,
    daily_returns_from_prices,
    max_drawdown,
    sharpe_ratio,
)

logger = setup_logging("experiments.index_benchmark")

DEFAULT_INDEX_SYMBOL = "SPY"


def index_metrics_from_prices(prices, risk_free_rate_annual):
    daily = daily_returns_from_prices(prices)
    cumulative = cumulative_return_from_prices(prices)
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized_return(cumulative, len(daily)),
        "annualized_volatility": annualized_volatility(daily),
        "sharpe_ratio": sharpe_ratio(
            daily, risk_free_rate_annual=risk_free_rate_annual
        ),
        "max_drawdown": max_drawdown(prices),
        "n_prices": len(prices),
    }


def _fetch_index_prices(symbol, start_date, end_date):
    import pandas as pd
    import yfinance as yf

    end_exclusive = (pd.Timestamp(end_date) + pd.DateOffset(days=1)).strftime(
        "%Y-%m-%d"
    )
    history = yf.Ticker(symbol).history(
        start=start_date, end=end_exclusive, interval="1d", auto_adjust=True
    )
    return [float(value) for value in history["Close"].tolist()]


def compute_index_benchmark(*, symbol, start_date, end_date, risk_free_rate_annual):
    try:
        prices = _fetch_index_prices(symbol, start_date, end_date)
    except Exception as exc:
        logger.warning(
            "Index benchmark %s fetch failed (%s); omitting index columns", symbol, exc
        )
        return None
    if len(prices) < 2:
        logger.warning(
            "Index benchmark %s: <2 prices in %s..%s; omitting",
            symbol,
            start_date,
            end_date,
        )
        return None
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "prices": prices,
        **index_metrics_from_prices(prices, risk_free_rate_annual),
    }

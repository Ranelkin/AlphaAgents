from collections import defaultdict

from src.util.log_config import setup_logging

from .metrics import (
    TRADING_DAYS,
    annualized_return,
    annualized_volatility,
    cumulative_return_from_prices,
    daily_returns_from_prices,
    max_drawdown,
    rolling_sharpe,
    sharpe_ratio,
)
from .schemas import Portfolio, PortfolioMetrics

logger = setup_logging("experiments.portfolio")


def build_equal_weight_portfolio(name, recommendations, include_holds=False):
    # include the stocks with allowed signals and give them the same weight
    members = [
        ticker
        for ticker, signal in recommendations.items()
        if signal == "BUY" or (include_holds and signal == "HOLD")
    ]
    if not members:
        return Portfolio(name=name, members=[], weights={}, signals=recommendations)
    equal_weight = 1.0 / len(members)
    weights = {ticker: equal_weight for ticker in members}
    return Portfolio(
        name=name, members=members, weights=weights, signals=recommendations
    )


def build_benchmark_portfolio(manifest, name="benchmark"):
    members = [
        ticker for ticker in manifest.benchmark_universe if ticker in manifest.stocks
    ]
    if not members:
        return Portfolio(name=name, members=[], weights={}, signals={})
    equal_weight = 1.0 / len(members)
    weights = {ticker: equal_weight for ticker in members}

    signals = {ticker: "BUY" for ticker in members}
    return Portfolio(
        name=name,
        members=members,
        weights=weights,
        signals=signals,
        metadata={"kind": "benchmark", "construction": "equal_weight_universe"},
    )


def build_portfolios_from_results(results, manifest=None):
    # make one portfolio for each agent and one for the final decision
    analyst_signals = defaultdict(dict)
    consensus_signals = {}
    for ticker, result in results.items():
        if result.consensus is not None:
            consensus_signals[ticker] = result.consensus.signal
        for analyst_name, decision in result.recommendations.items():
            analyst_signals[analyst_name][ticker] = decision.signal

    portfolios = {
        "consensus": build_equal_weight_portfolio("consensus", consensus_signals)
    }
    for analyst_name, signals in analyst_signals.items():
        key = analyst_name.lower().replace(" ", "_")
        portfolios[key] = build_equal_weight_portfolio(key, signals)
    if manifest is not None:
        portfolios["benchmark"] = build_benchmark_portfolio(manifest)
    return portfolios


def _usable_members(manifest, tickers):
    usable_tickers, dropped_tickers = [], []
    for ticker in tickers:
        snapshot = manifest.stocks.get(ticker)
        has_prices = (
            snapshot is not None
            and len(snapshot.evaluation_prices) >= 2
            and snapshot.evaluation_prices[0].close != 0
        )
        if has_prices:
            usable_tickers.append(ticker)
        else:
            dropped_tickers.append(ticker)
    if dropped_tickers:
        logger.warning(
            "Dropping members without usable evaluation prices: %s", dropped_tickers
        )
    return usable_tickers


def _portfolio_prices(manifest, portfolio):
    members = _usable_members(manifest, list(portfolio.weights))
    if not members:
        return []
    length = min(len(manifest.stocks[ticker].evaluation_prices) for ticker in members)
    if length < 2:
        return []
    total_weight = sum(portfolio.weights[ticker] for ticker in members)
    base = {
        ticker: manifest.stocks[ticker].evaluation_prices[0].close for ticker in members
    }
    values = []
    for index in range(length):
        total = 0.0
        for ticker in members:
            equal_weight = portfolio.weights[ticker] / total_weight
            total += (
                manifest.stocks[ticker].evaluation_prices[index].close
                / base[ticker]
                * equal_weight
            )
        values.append(total)
    return values


def _benchmark_prices(manifest):
    return _portfolio_prices(manifest, build_benchmark_portfolio(manifest))


def evaluate_portfolios(manifest, portfolios, rolling_window, risk_free_rate_annual):
    # calculate the same set of numbers for every portfolio
    benchmark_prices = _benchmark_prices(manifest)
    benchmark_return = cumulative_return_from_prices(benchmark_prices)
    # Start with an empty dictionary.
    results = {}
    for name, portfolio in portfolios.items():
        if not portfolio.members:
            # an empty portfolio earns the daily cash return in this experiment
            periods = max(len(benchmark_prices) - 1, 0)
            rf_daily = risk_free_rate_annual / TRADING_DAYS
            daily_returns = [rf_daily] * periods

            cumulative = (1.0 + rf_daily) ** periods - 1.0 if periods else 0.0

            results[name] = PortfolioMetrics(
                cumulative_return=cumulative,
                annualized_return=(
                    annualized_return(cumulative, periods) if periods else 0.0
                ),
                annualized_volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                benchmark_relative_return=cumulative - benchmark_return,
                rolling_sharpe=[],
                daily_returns=daily_returns,
            )
            continue
        prices = _portfolio_prices(manifest, portfolio)
        if len(prices) < 2:
            # two prices are needed to calculate a return
            results[name] = PortfolioMetrics(
                cumulative_return=0.0,
                annualized_return=0.0,
                annualized_volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                benchmark_relative_return=-benchmark_return,
                rolling_sharpe=[],
                daily_returns=[],
            )
            continue
        daily_returns = daily_returns_from_prices(prices)

        cumulative_return = cumulative_return_from_prices(prices)
        results[name] = PortfolioMetrics(
            cumulative_return=cumulative_return,
            annualized_return=annualized_return(cumulative_return, len(daily_returns)),
            annualized_volatility=annualized_volatility(daily_returns),
            sharpe_ratio=sharpe_ratio(
                daily_returns, risk_free_rate_annual=risk_free_rate_annual
            ),
            max_drawdown=max_drawdown(prices),
            benchmark_relative_return=cumulative_return - benchmark_return,
            rolling_sharpe=rolling_sharpe(
                daily_returns,
                window=rolling_window,
                risk_free_rate_annual=risk_free_rate_annual,
            ),
            daily_returns=daily_returns,
        )
    return results

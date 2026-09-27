from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any, Literal

Signal = Literal["BUY", "SELL", "HOLD"]


def to_primitive(value):
    if is_dataclass(value):
        return {
            key: to_primitive(nested_value)
            for key, nested_value in asdict(value).items()
        }

    if isinstance(value, dict):
        return {key: to_primitive(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [to_primitive(item) for item in value]
    return value


@dataclass
class NewsArticle:
    title: str
    published_at: str | None = None
    source: str | None = None
    url: str | None = None
    summary: str | None = None
    sentiment: float | None = None
    provider: str = "yahoo"


@dataclass
class FilingChunk:
    form_type: str
    chunk_id: str
    text: str
    source_label: str
    score: float | None = None


@dataclass
class PriceBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class StockSnapshot:
    ticker: str
    news_source: str
    pre_window_prices: list[PriceBar]
    evaluation_prices: list[PriceBar]
    news: list[NewsArticle] = field(default_factory=list)
    filings: list[FilingChunk] = field(default_factory=list)
    analyst_price_targets: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentManifest:
    manifest_version: str
    generated_at: str
    information_cutoff_date: str
    portfolio_start_date: str
    portfolio_end_date: str
    universe: list[str]
    benchmark_universe: list[str]
    risk_free_rate_series: list[float]
    stocks: dict[str, StockSnapshot]
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str | None = None


@dataclass
class AgentDecision:
    signal: Signal
    conviction: int
    thesis: str
    evidence: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    expected_excess_return_pct: float | None = None


@dataclass
class StockRunResult:
    ticker: str
    workflow_mode: str
    risk_profile: str
    tooling_mode: str
    recommendations: dict[str, AgentDecision]
    consensus: AgentDecision | None
    prompt_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Portfolio:
    name: str
    members: list[str]
    weights: dict[str, float]
    signals: dict[str, Signal]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioMetrics:
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    benchmark_relative_return: float
    rolling_sharpe: list[float]
    daily_returns: list[float]


@dataclass
class ExperimentRunResult:
    config: dict[str, Any]
    manifest_hash: str
    generated_at: str
    stock_results: dict[str, StockRunResult]
    portfolios: dict[str, Portfolio]
    metrics: dict[str, PortfolioMetrics]
    leaderboard_row: dict[str, Any]
    run_metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def utc_now():
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

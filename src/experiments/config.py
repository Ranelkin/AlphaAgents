from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

WorkflowMode = Literal[
    "single_agent", "collaboration", "debate", "baseline_random", "baseline_buy"
]
RiskProfile = Literal["risk_neutral", "risk_averse", "risk_seeking"]

ToolingMode = Literal["minimal", "paper_like", "no_rag", "no_news"]

NewsSource = Literal["gdelt", "edgar_8k", "yahoo", "bloomberg_like_adapter"]

DecisionMode = Literal["buy_sell", "buy_sell_hold"]

DecisionRule = Literal["agent", "median_split", "llm_rank_topk"]

# early 2024 one month treasury bill rate used by the experiments referenced
DEFAULT_RISK_FREE_RATE_ANNUAL = 0.053


@dataclass
class ExperimentConfig:
    model: str
    provider: str
    workflow_mode: WorkflowMode = "debate"
    risk_profile: RiskProfile = "risk_neutral"
    tooling_mode: ToolingMode = "paper_like"
    news_source: NewsSource = "gdelt"
    decision_mode: DecisionMode = "buy_sell"
    decision_rule: DecisionRule = "agent"
    rank_buy_count: int | None = None
    news_lookback_days: int = 30
    valuation_window_months: int = 3
    information_cutoff_date: str = "2024-01-31"
    portfolio_start_date: str = "2024-02-01"
    portfolio_end_date: str = "2024-05-31"
    universe: list[str] = field(default_factory=list)
    benchmark_universe: list[str] = field(default_factory=list)
    max_news_articles: int = 8
    filing_chunk_chars: int = 1600
    filing_top_k: int = 6
    debate_max_rounds: int = 5
    debate_min_rounds: int = 2
    max_tool_iterations: int = 8
    rolling_sharpe_window: int = 21
    index_benchmark_symbol: str | None = "SPY"
    risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL
    output_dir: str = "experiments/output"
    enable_judge_eval: bool = False
    judge_max_traces: int | None = 18

    def as_dict(self):
        return asdict(self)

    @property
    def output_path(self):

        return Path(self.output_dir)

import json
from collections.abc import Mapping
from typing import Any

from src.util.log_config import setup_logging

from .freeze import freeze_experiment_dataset, load_manifest, save_manifest
from .index_benchmark import compute_index_benchmark
from .inference import PROMPT_VERSION, model_settings, run_inference
from .metrics import spearman_correlation
from .phoenix_eval import evaluate_stock_results
from .portfolio import build_portfolios_from_results, evaluate_portfolios
from .runtime import collect_runtime_metadata, get_git_commit_sha
from .schemas import ExperimentManifest, ExperimentRunResult, to_primitive, utc_now

logger = setup_logging("experiments.runner")


def _realized_returns(manifest: ExperimentManifest) -> dict[str, float]:
    # use the first and last saved price for each stock
    returns = {}

    for ticker, snapshot in manifest.stocks.items():
        prices = snapshot.evaluation_prices
        if len(prices) >= 2 and prices[0].close:
            returns[ticker] = prices[-1].close / prices[0].close - 1.0
    return returns


def compute_decision_quality(
    manifest: ExperimentManifest, stock_results: Mapping[str, Any]
) -> dict[str, Any]:
    realized_returns = _realized_returns(manifest)
    if not realized_returns:
        return {"n_decisions": 0}
    benchmark_return = sum(realized_returns.values()) / len(realized_returns)
    positive_return_share = sum(
        1 for value in realized_returns.values() if value > 0
    ) / len(realized_returns)

    signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}

    buy_hits = buy_total = sell_hits = sell_total = 0

    signed_scores = []

    forward_returns = []

    failed_decisions = 0
    parse_failures = 0

    for ticker, result in stock_results.items():
        if isinstance(result, dict):
            metadata = result["metadata"]
            consensus = result["consensus"]
        else:
            metadata = result.metadata
            consensus = result.consensus
        parse_failures += int(metadata.get("parse_failures", 0) or 0)
        if consensus is None:
            failed_decisions += 1
            continue
        if ticker not in realized_returns:
            continue
        if isinstance(consensus, dict):
            signal = consensus["signal"]
            conviction = consensus["conviction"]

            expected_return = consensus.get("expected_excess_return_pct")
        else:
            signal = consensus.signal
            conviction = consensus.conviction
            expected_return = consensus.expected_excess_return_pct
        excess = realized_returns[ticker] - benchmark_return
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
        if signal == "BUY":
            buy_total += 1

            buy_hits += int(excess > 0)
            conviction = float(conviction)
        elif signal == "SELL":
            sell_total += 1
            sell_hits += int(excess < 0)
            conviction = -float(conviction)
        else:
            conviction = 0.0
        signed_scores.append(
            float(expected_return) if expected_return is not None else conviction
        )
        forward_returns.append(realized_returns[ticker])

    n_decisions = buy_total + sell_total + signal_counts.get("HOLD", 0)
    successful_decisions = buy_hits + sell_hits
    return {
        "n_decisions": n_decisions,
        "n_failed": failed_decisions,
        "parse_failures": parse_failures,
        "signal_counts": signal_counts,
        "universe_up_share": positive_return_share,
        "benchmark_window_return": benchmark_return,
        "buy_hit_rate": buy_hits / buy_total if buy_total else None,
        "sell_hit_rate": sell_hits / sell_total if sell_total else None,
        "hit_rate": (
            successful_decisions / (buy_total + sell_total)
            if (buy_total + sell_total)
            else None
        ),
        "rank_ic": spearman_correlation(signed_scores, forward_returns),
    }


class ExperimentRunner:
    def __init__(self, config):
        self.config = config

    def freeze(self):
        return freeze_experiment_dataset(self.config)

    def _build_run_metadata(
        self, manifest, *, cli_args=None, manifest_path=None, run_label=None
    ):
        metadata = collect_runtime_metadata()
        metadata.update(
            {
                "git_commit_sha": get_git_commit_sha(),
                "manifest_path": manifest_path,
                "manifest_hash": manifest.manifest_hash,
                "cli_args": list(cli_args or []),
                "run_label": run_label,
                "prompt_version": PROMPT_VERSION,
                **model_settings(),
            }
        )
        return metadata

    def _build_leaderboard_row(
        self, manifest, metrics, *, status="completed", error=None, run_label=None
    ):
        def cell(portfolio_name, field):
            entry = metrics.get(portfolio_name) if metrics else None

            return float(getattr(entry, field, 0.0)) if entry is not None else 0.0

        leaderboard_row = {
            "status": status,
            "error": error,
            "run_label": run_label,
            "model": self.config.model,
            "provider": self.config.provider,
            "workflow_mode": self.config.workflow_mode,
            "risk_profile": self.config.risk_profile,
            "tooling_mode": self.config.tooling_mode,
            "decision_rule": self.config.decision_rule,
            "manifest_hash": manifest.manifest_hash,
        }
        portfolio_names = (
            "consensus",
            "benchmark",
            "fundamental",
            "sentiment",
            "valuation",
            "single_agent",
        )
        for portfolio_name in portfolio_names:
            if metrics is None or portfolio_name not in metrics:
                continue
            leaderboard_row[f"{portfolio_name}_cumulative_return"] = cell(
                portfolio_name, "cumulative_return"
            )
            leaderboard_row[f"{portfolio_name}_sharpe"] = cell(
                portfolio_name, "sharpe_ratio"
            )
            leaderboard_row[f"{portfolio_name}_max_drawdown"] = cell(
                portfolio_name, "max_drawdown"
            )
            leaderboard_row[f"{portfolio_name}_ann_return"] = cell(
                portfolio_name, "annualized_return"
            )

            leaderboard_row[f"{portfolio_name}_ann_vol"] = cell(
                portfolio_name, "annualized_volatility"
            )
        return leaderboard_row

    def run(self, manifest, *, cli_args=None, manifest_path=None, run_label=None):
        stock_results = run_inference(manifest, self.config)
        portfolios = build_portfolios_from_results(stock_results, manifest)
        metrics = evaluate_portfolios(
            manifest,
            portfolios,
            rolling_window=self.config.rolling_sharpe_window,
            risk_free_rate_annual=self.config.risk_free_rate_annual,
        )
        decision_quality = compute_decision_quality(manifest, stock_results)
        leaderboard_row = self._build_leaderboard_row(
            manifest, metrics, run_label=run_label
        )
        counts = decision_quality.get("signal_counts", {})

        leaderboard_row.update(
            {
                "consensus_n_buy": counts.get("BUY", 0),
                "consensus_n_sell": counts.get("SELL", 0),
                "consensus_n_hold": counts.get("HOLD", 0),
                "decision_failures": decision_quality.get("n_failed", 0),
                "parse_failures": decision_quality.get("parse_failures", 0),
                "universe_up_share": decision_quality.get("universe_up_share"),
                "consensus_buy_hit_rate": decision_quality.get("buy_hit_rate"),
                "consensus_sell_hit_rate": decision_quality.get("sell_hit_rate"),
                "consensus_hit_rate": decision_quality.get("hit_rate"),
                "consensus_rank_ic": decision_quality.get("rank_ic"),
            }
        )
        index_benchmark = None
        if self.config.index_benchmark_symbol:
            index_benchmark = compute_index_benchmark(
                symbol=self.config.index_benchmark_symbol,
                start_date=manifest.portfolio_start_date,
                end_date=manifest.portfolio_end_date,
                risk_free_rate_annual=self.config.risk_free_rate_annual,
            )
        if index_benchmark is not None:
            leaderboard_row.update(
                {
                    "index_symbol": index_benchmark["symbol"],
                    "index_cumulative_return": index_benchmark["cumulative_return"],
                    "index_sharpe": index_benchmark["sharpe_ratio"],
                    "index_max_drawdown": index_benchmark["max_drawdown"],
                }
            )
        judge_eval = {"enabled": False, "reason": "disabled"}
        if self.config.enable_judge_eval:
            judge_eval = evaluate_stock_results(
                stock_results, max_per_run=self.config.judge_max_traces
            )
            if judge_eval.get("per_agent"):
                for agent_name, scores in judge_eval["per_agent"].items():
                    if scores.get("faithfulness_mean") is not None:
                        key = f"judge_{agent_name}_faithfulness"
                        leaderboard_row[key] = scores["faithfulness_mean"]
                    if scores.get("relevance_mean") is not None:
                        leaderboard_row[f"judge_{agent_name}_relevance"] = scores[
                            "relevance_mean"
                        ]
        return ExperimentRunResult(
            config=self.config.as_dict(),
            manifest_hash=manifest.manifest_hash or "",
            generated_at=utc_now(),
            stock_results=stock_results,
            portfolios=portfolios,
            metrics=metrics,
            leaderboard_row=leaderboard_row,
            run_metadata=self._build_run_metadata(
                manifest,
                cli_args=cli_args,
                manifest_path=manifest_path,
                run_label=run_label,
            )
            | {
                "judge_eval": judge_eval,
                "decision_quality": decision_quality,
                "index_benchmark": index_benchmark,
            },
        )

    def persist(self, manifest, result):
        base = self.config.output_path / (manifest.manifest_hash or "unknown-manifest")
        rule_suffix = (
            ""
            if self.config.decision_rule == "agent"
            else f"__{self.config.decision_rule}"
        )
        run_dir = base / (
            f"{self.config.model.replace('/', '_')}"
            f"__{self.config.workflow_mode}"
            f"__{self.config.risk_profile}"
            f"__{self.config.tooling_mode}"
            f"{rule_suffix}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = save_manifest(manifest, base / "manifest.json")
        result_path = run_dir / "result.json"
        leaderboard_path = base / "leaderboard.jsonl"
        result.artifacts = {
            "manifest": str(manifest_path),
            "result": str(result_path),
            "leaderboard": str(leaderboard_path),
        }
        result_path.write_text(
            json.dumps(to_primitive(result), indent=2), encoding="utf-8"
        )
        with leaderboard_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.leaderboard_row) + "\n")
        logger.info("Saved experiment artifacts under %s", run_dir)
        return {
            "manifest": manifest_path,
            "result": result_path,
            "leaderboard": leaderboard_path,
        }

    def freeze_and_persist(self):
        manifest = self.freeze()
        base = self.config.output_path / (manifest.manifest_hash or "unknown-manifest")
        manifest_path = save_manifest(manifest, base / "manifest.json")
        logger.info("Saved experiment manifest under %s", manifest_path)
        return manifest, manifest_path

    def run_and_persist(self, manifest_path=None, *, cli_args=None, run_label=None):
        manifest = load_manifest(manifest_path) if manifest_path else self.freeze()
        result = self.run(
            manifest,
            cli_args=cli_args,
            manifest_path=manifest_path,
            run_label=run_label,
        )
        artifacts = self.persist(manifest, result)
        return manifest, result, artifacts

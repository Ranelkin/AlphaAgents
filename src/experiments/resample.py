from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from src.util.log_config import setup_logging

from .config import DEFAULT_RISK_FREE_RATE_ANNUAL, ExperimentConfig
from .freeze import MissingPriceDataError, freeze_experiment_dataset, save_manifest
from .metrics import holm_bonferroni
from .suite import run_experiment_suite

logger = setup_logging("experiments.resample")


@dataclass
class ResampleSpec:
    ticker_pool: list[str]
    universe_size: int
    window_days: int = 122
    earliest_start: str = "2022-02-01"
    latest_start: str = "2024-09-01"
    valuation_window_months: int = 3
    seed: int = 17
    n_draws: int = 15
    risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL


@dataclass
class DrawResult:
    draw_id: str
    universe: list[str]
    information_cutoff_date: str
    portfolio_start_date: str
    portfolio_end_date: str
    suite_rows: list[dict[str, Any]] = field(default_factory=list)
    manifest_hash: str = ""


def _date_range(spec):
    return date.fromisoformat(spec.earliest_start), date.fromisoformat(
        spec.latest_start
    )


def _generate_draws(spec):
    # pick a stock group and a start date for every draw
    random_number_generator = random.Random(spec.seed)

    earliest, latest = _date_range(spec)
    span_days = (latest - earliest).days
    if span_days < 0:
        raise ValueError("earliest_start must be on/before latest_start")

    if len(spec.ticker_pool) < spec.universe_size:
        raise ValueError("ticker_pool smaller than universe_size")
    draws = []
    for _ in range(spec.n_draws):
        universe = random_number_generator.sample(spec.ticker_pool, spec.universe_size)
        start = earliest + timedelta(
            days=random_number_generator.randint(0, max(span_days, 0))
        )
        draws.append((universe, start))

    return draws


def _bootstrap_p_value(deltas, n_bootstrap=5000, seed=23):
    # compare the observed average with averages from random samples
    if not deltas:
        return None
    random_number_generator = random.Random(seed)

    observed = sum(deltas) / len(deltas)
    if observed == 0.0:
        return 1.0
    centered = [d - observed for d in deltas]
    extreme = 0
    for _ in range(n_bootstrap):
        sample = [
            centered[random_number_generator.randrange(len(centered))]
            for _ in range(len(centered))
        ]
        mean = sum(sample) / len(sample)
        if abs(mean) >= abs(observed):
            extreme += 1
    return extreme / n_bootstrap


def _mean_std(values):
    items = [sample_value for sample_value in values if sample_value is not None]
    if not items:
        return 0.0, 0.0
    mean = sum(items) / len(items)
    if len(items) < 2:
        return mean, 0.0
    sample_variance = sum((sample_value - mean) ** 2 for sample_value in items) / (
        len(items) - 1
    )
    return mean, math.sqrt(sample_variance)


def run_resample(*, spec, suite_config_path, output_dir, run_label):
    output_root = Path(output_dir) / "resample" / run_label
    output_root.mkdir(parents=True, exist_ok=True)

    n_suite_runs = len(
        json.loads(Path(suite_config_path).read_text(encoding="utf-8"))["runs"]
    )

    draws = _generate_draws(spec)
    results = []
    failed_draws = []
    for index, (universe, start) in enumerate(draws):
        draw_id = f"{index:02d}_{start.isoformat()}_{universe[0]}"
        end = start + timedelta(days=spec.window_days)
        cutoff = start - timedelta(days=1)
        draw_dir = output_root / draw_id
        draw_dir.mkdir(parents=True, exist_ok=True)

        existing_lb = next(
            iter(draw_dir.glob(f"*/suites/{draw_id}/leaderboard.jsonl")), None
        )
        if existing_lb is not None and existing_lb.exists():
            rows = [
                json.loads(line)
                for line in existing_lb.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            completed = [row for row in rows if row.get("status") == "completed"]

            if len(completed) >= n_suite_runs:
                logger.info(
                    "Draw %s: reusing %d completed rows, skipping",
                    draw_id,
                    len(completed),
                )
                results.append(
                    DrawResult(
                        draw_id=draw_id,
                        universe=universe,
                        information_cutoff_date=cutoff.isoformat(),
                        portfolio_start_date=start.isoformat(),
                        portfolio_end_date=end.isoformat(),
                        suite_rows=completed,
                        manifest_hash=completed[0].get("manifest_hash", ""),
                    )
                )
                continue

        try:
            replacement_rng = random.Random(f"{spec.seed}:{draw_id}")
            manifest = None
            for attempt in range(4):
                config = ExperimentConfig(
                    model="resample-freezer",
                    provider="together",
                    universe=universe,
                    benchmark_universe=universe,
                    information_cutoff_date=cutoff.isoformat(),
                    portfolio_start_date=start.isoformat(),
                    portfolio_end_date=end.isoformat(),
                    valuation_window_months=spec.valuation_window_months,
                    risk_free_rate_annual=spec.risk_free_rate_annual,
                    output_dir=str(output_root),
                )
                try:
                    manifest = freeze_experiment_dataset(config)
                    break
                except MissingPriceDataError as exc:
                    if attempt == 3:
                        raise
                    candidates = [
                        ticker
                        for ticker in spec.ticker_pool
                        if ticker not in universe and ticker not in exc.tickers
                    ]
                    if len(candidates) < len(exc.tickers):
                        raise
                    replacements = replacement_rng.sample(candidates, len(exc.tickers))
                    logger.warning(
                        "Draw %s: no price data for %s in window; replacing with %s",
                        draw_id,
                        exc.tickers,
                        replacements,
                    )
                    universe = [
                        replacements.pop() if ticker in exc.tickers else ticker
                        for ticker in universe
                    ]
                except Exception as exc:
                    if attempt == 3:
                        raise
                    wait = 60 * (attempt + 1)
                    logger.warning(
                        "Draw %s freeze attempt %d failed (%s); retrying in %ds",
                        draw_id,
                        attempt + 1,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
            manifest_path = draw_dir / "manifest.json"
            save_manifest(manifest, manifest_path)

            logger.info(
                "Draw %s: universe=%s window=%s..%s", draw_id, universe, start, end
            )

            summary = run_experiment_suite(
                manifest_path=manifest_path,
                suite_config_path=suite_config_path,
                output_dir=str(draw_dir),
                run_label=draw_id,
            )
            results.append(
                DrawResult(
                    draw_id=draw_id,
                    universe=universe,
                    information_cutoff_date=cutoff.isoformat(),
                    portfolio_start_date=start.isoformat(),
                    portfolio_end_date=end.isoformat(),
                    suite_rows=list(summary.rows),
                    manifest_hash=summary.manifest_hash,
                )
            )
        except Exception:
            logger.exception("Draw %s failed; continuing with remaining draws", draw_id)
            failed_draws.append(draw_id)

    by_label = {}
    for result in results:
        for row in result.suite_rows:
            by_label.setdefault(row.get("label") or "unlabeled", []).append(
                {"draw_id": result.draw_id, **row}
            )

    per_label_summary = {}
    for label, rows in by_label.items():
        sharpes = [row.get("consensus_sharpe", 0.0) for row in rows]
        returns = [row.get("consensus_cumulative_return", 0.0) for row in rows]
        sharpe_mean, sharpe_std = _mean_std(sharpes)
        ret_mean, ret_std = _mean_std(returns)

        per_label_summary[label] = {
            "n": len(rows),
            "consensus_sharpe_mean": sharpe_mean,
            "consensus_sharpe_std": sharpe_std,
            "consensus_cumulative_return_mean": ret_mean,
            "consensus_cumulative_return_std": ret_std,
        }

    single_label = next((lab for lab in by_label if "single" in lab.lower()), None)
    pairwise = {}

    if single_label:
        single_by_draw = {row["draw_id"]: row for row in by_label[single_label]}
        for label, rows in by_label.items():
            if label == single_label:
                continue
            deltas_sharpe = []

            deltas_return = []

            for row in rows:
                other = single_by_draw.get(row["draw_id"])
                if not other:
                    continue
                deltas_sharpe.append(
                    row.get("consensus_sharpe", 0.0)
                    - other.get("consensus_sharpe", 0.0)
                )
                deltas_return.append(
                    row.get("consensus_cumulative_return", 0.0)
                    - other.get("consensus_cumulative_return", 0.0)
                )
            mean_s, std_s = _mean_std(deltas_sharpe)
            mean_r, std_r = _mean_std(deltas_return)
            pairwise[label] = {
                "vs": single_label,
                "n_paired": len(deltas_sharpe),
                "sharpe_delta_mean": mean_s,
                "sharpe_delta_std": std_s,
                "sharpe_delta_bootstrap_p": _bootstrap_p_value(deltas_sharpe),
                "return_delta_mean": mean_r,
                "return_delta_std": std_r,
                "return_delta_bootstrap_p": _bootstrap_p_value(deltas_return),
            }

    labels = list(pairwise)
    for metric in ("sharpe_delta_bootstrap_p", "return_delta_bootstrap_p"):
        adjusted = holm_bonferroni([pairwise[label][metric] for label in labels])
        for label, adj in zip(labels, adjusted):
            pairwise[label][metric + "_holm"] = adj

    summary_path = output_root / "resample_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "spec": spec.__dict__,
                "draws": [
                    {
                        "draw_id": resample_result.draw_id,
                        "universe": resample_result.universe,
                        "information_cutoff_date": resample_result.information_cutoff_date,
                        "portfolio_start_date": resample_result.portfolio_start_date,
                        "portfolio_end_date": resample_result.portfolio_end_date,
                        "manifest_hash": resample_result.manifest_hash,
                        "rows": resample_result.suite_rows,
                    }
                    for resample_result in results
                ],
                "per_label": per_label_summary,
                "pairwise_vs_single": pairwise,
                "failed_draws": failed_draws,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote resample summary to %s", summary_path)
    return summary_path

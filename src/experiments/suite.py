from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.util.log_config import setup_logging

from .config import DEFAULT_RISK_FREE_RATE_ANNUAL, ExperimentConfig
from .freeze import load_manifest
from .runner import ExperimentRunner

logger = setup_logging("experiments.suite")


@dataclass
class SuiteRunSummary:
    manifest_hash: str
    suite_dir: Path
    leaderboard_jsonl: Path
    leaderboard_csv: Path
    rows: list[dict[str, Any]]


def _load_suite_matrix(path):
    suite_path = Path(path)
    suite_text = suite_path.read_text(encoding="utf-8")
    if suite_path.suffix.lower() == ".json":
        return json.loads(suite_text)
    if suite_path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyYAML is required to load YAML suite files.") from exc
        return yaml.safe_load(suite_text)
    raise ValueError(f"Unsupported suite file format: {suite_path.suffix}")


def _suite_dir(output_dir, manifest_hash, run_label):
    safe_label = run_label.replace("/", "_").replace(" ", "_")
    return Path(output_dir) / manifest_hash / "suites" / safe_label


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for leaderboard_row in rows:
            handle.write(json.dumps(leaderboard_row) + "\n")


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for leaderboard_row in rows:
        for key in leaderboard_row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)


def run_experiment_suite(
    *, manifest_path, suite_config_path, output_dir, run_label, cli_args=None
):
    manifest = load_manifest(manifest_path)
    suite_config = _load_suite_matrix(suite_config_path)
    matrix = suite_config.get("runs", [])
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("Suite config must define a non-empty 'runs' list.")

    suite_dir = _suite_dir(
        output_dir, manifest.manifest_hash or "unknown-manifest", run_label
    )
    suite_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_jsonl = suite_dir / "leaderboard.jsonl"
    leaderboard_csv = suite_dir / "leaderboard.csv"

    rows = []
    for index, entry in enumerate(matrix, start=1):
        logger.info("Running suite entry %s/%s", index, len(matrix))
        options = {**suite_config, **entry}
        config = ExperimentConfig(
            model=entry["model"],
            provider=entry["provider"],
            workflow_mode=entry.get("workflow_mode", "debate"),
            risk_profile=entry.get("risk_profile", "risk_neutral"),
            tooling_mode=entry.get("tooling_mode", "paper_like"),
            news_source=options.get("news_source", "gdelt"),
            decision_mode=options.get("decision_mode", "buy_sell"),
            decision_rule=options.get("decision_rule", "agent"),
            rank_buy_count=options.get("rank_buy_count"),
            universe=list(manifest.universe),
            benchmark_universe=list(manifest.benchmark_universe),
            information_cutoff_date=manifest.information_cutoff_date,
            portfolio_start_date=manifest.portfolio_start_date,
            portfolio_end_date=manifest.portfolio_end_date,
            output_dir=str(output_dir),
            enable_judge_eval=bool(options.get("enable_judge_eval", False)),
            judge_max_traces=int(options.get("judge_max_traces", 18)),
            risk_free_rate_annual=float(
                options.get("risk_free_rate_annual", DEFAULT_RISK_FREE_RATE_ANNUAL)
            ),
        )
        row_label = (
            entry.get("label") or f"{index:02d}_{config.model}_{config.workflow_mode}"
        )
        try:
            runner = ExperimentRunner(config)
            _, result, artifacts = runner.run_and_persist(
                manifest_path=str(manifest_path), cli_args=cli_args, run_label=row_label
            )
            leaderboard_row = dict(result.leaderboard_row)
            leaderboard_row["result_path"] = str(artifacts["result"])
            leaderboard_row["label"] = row_label
        except Exception as exc:
            logger.exception("Suite entry failed: %s", row_label)
            leaderboard_row = {
                "status": "failed",
                "error": str(exc),
                "label": row_label,
                "model": config.model,
                "provider": config.provider,
                "workflow_mode": config.workflow_mode,
                "risk_profile": config.risk_profile,
                "tooling_mode": config.tooling_mode,
                "manifest_hash": manifest.manifest_hash,
            }
        rows.append(leaderboard_row)

    _write_jsonl(leaderboard_jsonl, rows)
    _write_csv(leaderboard_csv, rows)
    return SuiteRunSummary(
        manifest_hash=manifest.manifest_hash or "",
        suite_dir=suite_dir,
        leaderboard_jsonl=leaderboard_jsonl,
        leaderboard_csv=leaderboard_csv,
        rows=rows,
    )

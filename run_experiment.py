import argparse
import json
import sys

from dotenv import load_dotenv

from src.experiments.config import DEFAULT_RISK_FREE_RATE_ANNUAL, ExperimentConfig
from src.experiments.runner import ExperimentRunner

load_dotenv()


def _add_shared_config_args(parser, *, require_universe):
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo"
    )
    parser.add_argument("--provider", type=str, default="together")
    parser.add_argument(
        "--workflow-mode",
        choices=[
            "single_agent",
            "collaboration",
            "debate",
            "baseline_random",
            "baseline_buy",
        ],
        default="debate",
    )
    parser.add_argument(
        "--risk-profile",
        choices=["risk_neutral", "risk_averse", "risk_seeking"],
        default="risk_neutral",
    )
    parser.add_argument(
        "--tooling-mode",
        choices=["minimal", "paper_like", "no_rag", "no_news"],
        default="paper_like",
    )
    parser.add_argument(
        "--enable-judge-eval",
        action="store_true",
        help="Run the faithfulness and relevance judge (adds LLM calls).",
    )
    parser.add_argument(
        "--judge-max-traces",
        type=int,
        default=18,
        help="Maximum traces sent to the judge per run.",
    )

    parser.add_argument(
        "--risk-free-rate-annual",
        type=float,
        default=DEFAULT_RISK_FREE_RATE_ANNUAL,
        help="Annual risk-free rate used in Sharpe calculations.",
    )
    parser.add_argument(
        "--news-source",
        choices=["gdelt", "edgar_8k", "yahoo", "bloomberg_like_adapter"],
        default="gdelt",
    )
    parser.add_argument(
        "--news-lookback-days",
        type=int,
        default=30,
        help="News lookback from the information cutoff, in days.",
    )

    parser.add_argument(
        "--decision-mode", choices=["buy_sell", "buy_sell_hold"], default="buy_sell"
    )
    parser.add_argument(
        "--decision-rule",
        choices=["agent", "median_split", "llm_rank_topk"],
        default="agent",
    )
    parser.add_argument(
        "--rank-buy-count",
        type=int,
        default=None,
        help="Number of BUY signals assigned by llm_rank_topk.",
    )
    parser.add_argument("--universe", nargs="+", required=require_universe)
    parser.add_argument("--benchmark-universe", nargs="*", default=[])

    parser.add_argument("--information-cutoff-date", default="2024-01-31")
    parser.add_argument("--portfolio-start-date", default="2024-02-01")
    parser.add_argument("--portfolio-end-date", default="2024-05-31")
    parser.add_argument("--output-dir", default="experiments/output")
    parser.add_argument("--run-label", default=None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a reproducible AlphaAgents experiment."
    )
    subparsers = parser.add_subparsers(dest="command")

    freeze_parser = subparsers.add_parser(
        "freeze", help="Freeze and persist a manifest only."
    )
    _add_shared_config_args(freeze_parser, require_universe=True)

    run_parser = subparsers.add_parser(
        "run", help="Run inference and evaluation, optionally from a frozen manifest."
    )
    _add_shared_config_args(run_parser, require_universe=False)
    run_parser.add_argument(
        "--manifest", type=str, default=None, help="Path to a frozen manifest JSON."
    )

    args = parser.parse_args(argv)

    if args.command is None:
        legacy_args = parser.parse_args(["run", *(argv or sys.argv[1:])])
        if legacy_args.manifest is None and not legacy_args.universe:
            parser.error("Either --manifest or --universe must be provided.")
        return legacy_args
    if args.command == "run" and args.manifest is None and not args.universe:
        parser.error("Either --manifest or --universe must be provided.")
    return args


def _build_config(args):
    universe = args.universe or []
    benchmark_universe = args.benchmark_universe or universe
    return ExperimentConfig(
        model=args.model,
        provider=args.provider,
        workflow_mode=args.workflow_mode,
        risk_profile=args.risk_profile,
        tooling_mode=args.tooling_mode,
        news_source=args.news_source,
        news_lookback_days=args.news_lookback_days,
        decision_mode=args.decision_mode,
        decision_rule=args.decision_rule,
        rank_buy_count=args.rank_buy_count,
        universe=universe,
        benchmark_universe=benchmark_universe,
        information_cutoff_date=args.information_cutoff_date,
        portfolio_start_date=args.portfolio_start_date,
        portfolio_end_date=args.portfolio_end_date,
        output_dir=args.output_dir,
        enable_judge_eval=getattr(args, "enable_judge_eval", False),
        judge_max_traces=getattr(args, "judge_max_traces", 18),
        risk_free_rate_annual=getattr(
            args, "risk_free_rate_annual", DEFAULT_RISK_FREE_RATE_ANNUAL
        ),
    )


def main(argv=None):
    args = parse_args(argv)
    config = _build_config(args)
    runner = ExperimentRunner(config)
    cli_args = list(argv or sys.argv[1:])

    if args.command == "freeze":
        manifest, manifest_path = runner.freeze_and_persist()
        print(
            json.dumps(
                {
                    "command": "freeze",
                    "manifest_hash": manifest.manifest_hash,
                    "artifacts": {"manifest": str(manifest_path)},
                },
                indent=2,
            )
        )
        return

    manifest, result, artifacts = runner.run_and_persist(
        manifest_path=args.manifest, cli_args=cli_args, run_label=args.run_label
    )
    print(
        json.dumps(
            {
                "command": "run",
                "manifest_hash": manifest.manifest_hash,
                "leaderboard_row": result.leaderboard_row,
                "artifacts": {name: str(path) for name, path in artifacts.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.experiments.config import DEFAULT_RISK_FREE_RATE_ANNUAL
from src.experiments.resample import ResampleSpec, run_resample

load_dotenv()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Resample AlphaAgents universes and evaluation windows."
    )

    parser.add_argument(
        "--ticker-pool",
        required=True,
        help="Path to a newline-separated ticker file (e.g. stocks.txt).",
    )
    parser.add_argument(
        "--suite-config",
        required=True,
        help="Suite config to run on each (universe, window) draw.",
    )

    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-dir", default="experiments/output")
    parser.add_argument("--n-draws", type=int, default=15)
    parser.add_argument("--universe-size", type=int, default=15)
    parser.add_argument("--window-days", type=int, default=122)
    parser.add_argument("--earliest-start", default="2022-02-01")
    parser.add_argument("--latest-start", default="2024-09-01")
    parser.add_argument("--valuation-window-months", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--risk-free-rate-annual", type=float, default=DEFAULT_RISK_FREE_RATE_ANNUAL
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pool = [
        line.strip()
        for line in Path(args.ticker_pool).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not pool:
        raise SystemExit(f"No tickers found in {args.ticker_pool}")
    spec = ResampleSpec(
        ticker_pool=pool,
        universe_size=args.universe_size,
        window_days=args.window_days,
        earliest_start=args.earliest_start,
        latest_start=args.latest_start,
        valuation_window_months=args.valuation_window_months,
        seed=args.seed,
        n_draws=args.n_draws,
        risk_free_rate_annual=args.risk_free_rate_annual,
    )
    summary_path = run_resample(
        spec=spec,
        suite_config_path=args.suite_config,
        output_dir=args.output_dir,
        run_label=args.run_label,
    )
    print(json.dumps({"resample_summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])

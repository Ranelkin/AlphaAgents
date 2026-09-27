import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from src.experiments.news_store import months_for_window
from src.experiments.resample import ResampleSpec, _generate_draws

NEWS_LOOKBACK_DAYS = 30

TECH_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "META",
    "AMZN",
    "TSLA",
    "AVGO",
    "ORCL",
    "CRM",
    "AMD",
    "QCOM",
    "TXN",
    "MU",
    "INTC",
]

REGIME_CUTOFFS = ["2024-01-31", "2022-01-31", "2023-03-31", "2020-01-31"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plan the news cells needed by all experiments."
    )
    parser.add_argument("--ticker-pool", default="stocks.txt")
    parser.add_argument("--out", default="experiments/news_cells.txt")
    parser.add_argument("--n-draws", type=int, default=15)
    parser.add_argument("--universe-size", type=int, default=15)
    parser.add_argument("--window-days", type=int, default=122)
    parser.add_argument("--earliest-start", default="2022-02-01")
    parser.add_argument("--latest-start", default="2024-09-01")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def cells_for_cutoff(tickers, cutoff):
    start = (
        date.fromisoformat(cutoff) - timedelta(days=NEWS_LOOKBACK_DAYS)
    ).isoformat()
    months = months_for_window(start, cutoff)
    return {(ticker, month) for ticker in tickers for month in months}


def main(argv=None):
    args = parse_args(argv)
    pool = [
        line.strip()
        for line in Path(args.ticker_pool).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    spec = ResampleSpec(
        ticker_pool=pool,
        universe_size=args.universe_size,
        window_days=args.window_days,
        earliest_start=args.earliest_start,
        latest_start=args.latest_start,
        seed=args.seed,
        n_draws=args.n_draws,
    )

    planned_cells = set()
    for universe, start in _generate_draws(spec):
        cutoff = (start - timedelta(days=1)).isoformat()
        planned_cells |= cells_for_cutoff(universe, cutoff)
    for cutoff in REGIME_CUTOFFS:
        planned_cells |= cells_for_cutoff(TECH_UNIVERSE, cutoff)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines = [f"{ticker} {month}" for ticker, month in sorted(planned_cells)]
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    tickers = {ticker for ticker, _ in planned_cells}
    print(f"{len(planned_cells)} cells across {len(tickers)} tickers -> {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])

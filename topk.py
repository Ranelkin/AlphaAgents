import json
import random
from pathlib import Path

from src.experiments.freeze import load_manifest
from src.experiments.portfolio import evaluate_portfolios
from src.experiments.resample import _bootstrap_p_value
from src.experiments.runner import compute_decision_quality
from src.experiments.schemas import Portfolio

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "output"
E1_MANIFEST_DIR = OUT / "3e06f140a601f3cb"
E4_BATCHES = ["e4_full_n15", "e4_full_n15_v2"]
K_GRID = [3, 5, 7]
BOOTSTRAP_B = 10_000
BASELINE = ("single_agent", "risk_neutral")


def _bootstrap_ci(sharpe_differences):
    rng = random.Random(23)
    n = len(sharpe_differences)
    bootstrap_means = sorted(
        sum(sharpe_differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_B)
    )
    return (
        bootstrap_means[int(0.025 * BOOTSTRAP_B)],
        bootstrap_means[int(0.975 * BOOTSTRAP_B)],
    )


def _topk_portfolios(stock_results):
    stocks_with_scores = []
    for ticker, stock_result in stock_results.items():
        consensus = stock_result["consensus"]
        if consensus is None:
            continue
        score = consensus.get("expected_excess_return_pct")

        if score is None:
            signal_direction = {"BUY": 1, "SELL": -1}.get(consensus["signal"], 0)
            score = signal_direction * float(consensus["conviction"])
        stocks_with_scores.append((float(score), ticker))
    stocks_with_scores.sort(key=lambda item: (-item[0], item[1]))
    portfolios = {}
    for number_of_stocks in K_GRID:
        members = [
            ticker
            for score, ticker in stocks_with_scores[:number_of_stocks]
            if score > 0
        ]
        boundary_tie = 0
        if (
            0 < number_of_stocks < len(stocks_with_scores)
            and stocks_with_scores[number_of_stocks - 1][0]
            == stocks_with_scores[number_of_stocks][0]
        ):
            boundary_tie = sum(
                1
                for score, _ in stocks_with_scores
                if score == stocks_with_scores[number_of_stocks - 1][0]
            )
        weight = 1.0 / len(members) if members else 0.0
        portfolios[f"top{number_of_stocks}"] = Portfolio(
            name=f"top{number_of_stocks}",
            members=members,
            weights={ticker: weight for ticker in members},
            signals={ticker: "BUY" for ticker in members},
            metadata={
                "kind": "topk",
                "k": number_of_stocks,
                "boundary_tie_size": boundary_tie,
            },
        )
    return portfolios


def evaluate_run(result_path, manifest):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = result["config"]
    stock_results = result["stock_results"]
    portfolios = _topk_portfolios(stock_results)
    metrics = evaluate_portfolios(
        manifest,
        portfolios,
        rolling_window=config["rolling_sharpe_window"],
        risk_free_rate_annual=config["risk_free_rate_annual"],
    )
    committed = result["metrics"]
    row = {
        "run_dir": result_path.parent.name,
        "model": config["model"],
        "workflow_mode": config["workflow_mode"],
        "risk_profile": config["risk_profile"],
        "tooling_mode": config["tooling_mode"],
        "consensus_sharpe": committed["consensus"]["sharpe_ratio"],
        "benchmark_sharpe": committed["benchmark"]["sharpe_ratio"],
        "decision_quality": compute_decision_quality(manifest, stock_results),
    }
    for name, portfolio_metrics in metrics.items():
        row[f"{name}_sharpe"] = portfolio_metrics.sharpe_ratio
        row[f"{name}_cumulative_return"] = portfolio_metrics.cumulative_return

        row[f"{name}_max_drawdown"] = portfolio_metrics.max_drawdown
        row[f"{name}_n_members"] = len(portfolios[name].members)
        row[f"{name}_boundary_tie_size"] = portfolios[name].metadata[
            "boundary_tie_size"
        ]
    return row


def collect_e1_rows():
    manifest = load_manifest(E1_MANIFEST_DIR / "manifest.json")
    rows = []
    for run_dir in sorted(E1_MANIFEST_DIR.iterdir()):
        result_path = run_dir / "result.json"
        if result_path.is_file():
            rows.append(evaluate_run(result_path, manifest))

    return rows


def collect_e4_rows():
    rows = []
    for batch in E4_BATCHES:
        batch_dir = OUT / "resample" / batch
        summary = json.loads(
            (batch_dir / "resample_summary.json").read_text(encoding="utf-8")
        )

        for draw in summary["draws"]:
            draw_dir = batch_dir / draw["draw_id"]

            manifest_path = draw_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = load_manifest(manifest_path)
            for result_path in sorted(draw_dir.glob("*/*/result.json")):
                row = evaluate_run(result_path, manifest)
                row["batch"] = batch
                row["draw_id"] = draw["draw_id"]
                rows.append(row)
    return rows


def aggregate_e4(rows):
    by_draw = {}
    for row in rows:
        key = (row["workflow_mode"], row["risk_profile"])
        by_draw.setdefault(f"{row['batch']}/{row['draw_id']}", {})[key] = row

    aggregate = {}

    configs = sorted(
        {
            (result_row["workflow_mode"], result_row["risk_profile"])
            for result_row in rows
        }
    )
    for config in configs:
        if config == BASELINE:
            continue
        label = f"{config[0]}_{config[1]}"
        entry = {}
        for number_of_stocks in K_GRID:
            field = f"top{number_of_stocks}_sharpe"
            sharpe_differences = [
                runs[config][field] - runs[BASELINE][field]
                for runs in by_draw.values()
                if config in runs and BASELINE in runs
            ]
            if not sharpe_differences:
                continue
            mean = sum(sharpe_differences) / len(sharpe_differences)
            confidence_interval = _bootstrap_ci(sharpe_differences)
            entry[f"top{number_of_stocks}"] = {
                "n_paired": len(sharpe_differences),
                "sharpe_delta_mean": mean,
                "sharpe_delta_p": _bootstrap_p_value(
                    sharpe_differences, n_bootstrap=BOOTSTRAP_B
                ),
                "sharpe_delta_ci95": confidence_interval,
            }
        aggregate[label] = entry

    decision_quality = {}
    for config in configs:
        label = f"{config[0]}_{config[1]}"
        hit_rates = [
            result_row["decision_quality"]["hit_rate"]
            for result_row in rows
            if (result_row["workflow_mode"], result_row["risk_profile"]) == config
            and result_row["decision_quality"].get("hit_rate") is not None
        ]
        rank_correlations = [
            result_row["decision_quality"]["rank_ic"]
            for result_row in rows
            if (result_row["workflow_mode"], result_row["risk_profile"]) == config
            and result_row["decision_quality"].get("rank_ic") is not None
        ]
        decision_quality[label] = {
            "n_hit": len(hit_rates),
            "hit_rate_mean": sum(hit_rates) / len(hit_rates) if hit_rates else None,
            "n_ic": len(rank_correlations),
            "rank_ic_mean": (
                sum(rank_correlations) / len(rank_correlations)
                if rank_correlations
                else None
            ),
            "rank_ic_p": (
                _bootstrap_p_value(rank_correlations, n_bootstrap=BOOTSTRAP_B)
                if rank_correlations
                else None
            ),
            "rank_ic_ci95": (
                _bootstrap_ci(rank_correlations) if rank_correlations else None
            ),
        }
    return {"pairwise_topk_vs_single": aggregate, "decision_quality": decision_quality}


def format_number(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def main():
    e1_rows = collect_e1_rows()
    e4_rows = collect_e4_rows()
    e4_aggregate = aggregate_e4(e4_rows)

    posthoc_dir = OUT / "posthoc"
    posthoc_dir.mkdir(parents=True, exist_ok=True)
    out_path = posthoc_dir / "topk_summary.json"

    out_path.write_text(
        json.dumps(
            {
                "k_grid": K_GRID,
                "bootstrap_B": BOOTSTRAP_B,
                "baseline": f"{BASELINE[0]}_{BASELINE[1]}",
                "e1_manifest_rows": e1_rows,
                "e4_rows": e4_rows,
                "e4_aggregate": e4_aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path}\n")

    print("===== E1/E3 manifest 3e06f140a601f3cb: top-k Sharpe by run =====")
    print(
        "run  &  BUY-rule S  &  top3 S (n)  &  top5 S (n)  &  top7 S (n)  &  rank-IC \\\\"
    )
    for row in e1_rows:
        decision_quality_values = row["decision_quality"]
        cells = " & ".join(
            f"{format_number(row[f'top{number_of_stocks}_sharpe'])} ({row[f'top{number_of_stocks}_n_members']})"
            for number_of_stocks in K_GRID
        )
        print(
            f"{row['run_dir']} & {format_number(row['consensus_sharpe'])} & {cells}"
            f" & {format_number(decision_quality_values.get('rank_ic'), 3)} \\\\"
        )

    print("\n===== E4: paired top-k Sharpe delta vs single_agent_risk_neutral =====")
    for label, entry in e4_aggregate["pairwise_topk_vs_single"].items():
        for number_of_stocks in K_GRID:
            stats = entry.get(f"top{number_of_stocks}")
            if not stats:
                continue
            lower_limit, upper_limit = stats["sharpe_delta_ci95"]
            print(
                f"{label} top{number_of_stocks}: n={stats['n_paired']} "
                f"mean dS={stats['sharpe_delta_mean']:+.2f} "
                f"CI95=[{lower_limit:+.2f},{upper_limit:+.2f}] p={stats['sharpe_delta_p']:.3f}"
            )

    print("\n===== E4: decision quality per configuration (across draws) =====")

    for label, stats in e4_aggregate["decision_quality"].items():
        confidence_interval = stats["rank_ic_ci95"]
        ci_s = (
            f"[{confidence_interval[0]:+.2f},{confidence_interval[1]:+.2f}]"
            if confidence_interval
            else "--"
        )
        print(
            f"{label}: hit_rate={format_number(stats['hit_rate_mean'], 3)} (n={stats['n_hit']}) "
            f"rank_IC={format_number(stats['rank_ic_mean'], 3)} CI95={ci_s} "
            f"p={format_number(stats['rank_ic_p'], 3)} (n={stats['n_ic']})"
        )


if __name__ == "__main__":
    main()

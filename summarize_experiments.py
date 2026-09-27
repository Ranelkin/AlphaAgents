import json
from pathlib import Path

from src.experiments.metrics import holm_bonferroni, mean_ci95

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "output"
E1_HASH = "609920784c6d59df"

REGIMES = [
    ("bull_2024H1", "Tech bull rally (2024 H1)"),
    ("bear_2022H1", "Tech bear / rate shock (2022 H1)"),
    ("sideways_2023Q2", "Range-bound (2023 Q2)"),
    ("highvol_2020H1", "COVID shock / V-recovery (2020 H1)"),
]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_one(pattern):
    hits = sorted(OUT.rglob(pattern))
    return hits[0] if hits else None


def find_most_completed(pattern):
    def n_completed(path):
        return sum(1 for row in read_jsonl(path) if row.get("status") == "completed")

    hits = sorted(
        OUT.rglob(pattern),
        key=lambda candidate_path: (n_completed(candidate_path), str(candidate_path)),
    )
    return hits[-1] if hits else None


def e2_ablation():
    print("\n===== E2: workflow x tooling consensus Sharpe (tab:ablation) =====")
    leaderboard_path = (
        find_one("*ablation*/leaderboard_merged.jsonl")
        or find_most_completed("*ablation*/leaderboard.jsonl")
        or OUT / E1_HASH / "suites" / "ablation_v1" / "leaderboard.jsonl"
    )

    if not leaderboard_path or not leaderboard_path.exists():
        print("  [missing] run ablation.json first")
        return

    rows = [
        result_row
        for result_row in read_jsonl(leaderboard_path)
        if result_row.get("status") == "completed"
    ]
    workflows = ["single_agent", "collaboration", "debate"]
    tooling = ["paper_like", "no_news", "no_rag", "minimal"]

    grid = {
        (result_row["workflow_mode"], result_row["tooling_mode"]): result_row[
            "consensus_sharpe"
        ]
        for result_row in rows
    }
    print(f"  (n={len(rows)} runs)")
    for workflow_name in workflows:
        cells = " & ".join(
            (
                f"${grid[(workflow_name, tooling_name)]:.2f}$"
                if (workflow_name, tooling_name) in grid
                else "n/a"
            )
            for tooling_name in tooling
        )
        print(f"    {workflow_name:13s} & {cells}\\\\")


def e3_cross_model():
    print("\n===== E3: cross-model consensus Sharpe (tab:e3) =====")
    leaderboard_path = (
        find_most_completed("*cross_model*/leaderboard.jsonl")
        or OUT / E1_HASH / "suites" / "cross_model_v1" / "leaderboard.jsonl"
    )
    if not leaderboard_path or not leaderboard_path.exists():
        print("  [missing] run cross_model.json first")
        return
    rows = [
        result_row
        for result_row in read_jsonl(leaderboard_path)
        if result_row.get("status") == "completed"
    ]
    for result_row in rows:
        print(
            f"    {result_row['model']:45s} & ${result_row['consensus_sharpe']:.2f}$\\\\"
        )


def e4_resample():
    print("\n===== E4: resampling deltas + bootstrap p (tab:e4) =====")
    summary_path = None
    for c in OUT.rglob("resample_summary.json"):
        if "e4_full" in str(c):
            summary_path = c
            break
        summary_path = summary_path or c
    if not summary_path:
        print("  [missing] run resampling first")
        return
    data = json.loads(summary_path.read_text())
    print(
        f"  source: {summary_path.relative_to(ROOT)}  (n_draws={data['spec']['n_draws']})"
    )
    pairwise_values = data.get("pairwise_vs_single", {})
    entries = [
        (label, comparison_data)
        for label, comparison_data in pairwise_values.items()
        if label != comparison_data.get("vs")
    ]
    holm = holm_bonferroni(
        [comparison_data["sharpe_delta_bootstrap_p"] for _, comparison_data in entries]
    )
    for (label, comparison_data), p_adj in zip(entries, holm):
        p_adj_s = f"${p_adj:.2f}$" if p_adj is not None else "n/a"

        print(
            f"    {label:32s} & ${comparison_data['sharpe_delta_mean']:+.2f}$ & "
            f"${comparison_data['sharpe_delta_std']:.2f}$ & ${comparison_data['sharpe_delta_bootstrap_p']:.2f}$ & "
            f"{p_adj_s}\\\\"
        )


def e5_regime():
    print("\n===== E5: regime sensitivity consensus Sharpe (tab:e5) =====")
    print("  cols: regime & benchmark & single & debate(neutral) & debate(averse)")
    any_found = False
    for window_label, name in REGIMES:
        leaderboard_path = find_most_completed(
            f"*regime_{window_label}*/leaderboard.jsonl"
        )

        if not leaderboard_path or not leaderboard_path.exists():
            missing = f"[missing] freeze + run window {window_label}"
            print(f"    {name:34s} & \\multicolumn{{4}}{{c}}{{{missing}}}\\\\")
            continue
        any_found = True
        rows = {
            result_row.get("run_label"): result_row
            for result_row in read_jsonl(leaderboard_path)
            if result_row.get("status") == "completed"
        }
        benchmark_sharpe = next(
            (result_row.get("benchmark_sharpe") for result_row in rows.values()), None
        )

        def cell(label):
            result_row = rows.get(label)
            return f"${result_row['consensus_sharpe']:.2f}$" if result_row else "n/a"

        bench_s = f"${benchmark_sharpe:.2f}$" if benchmark_sharpe is not None else "n/a"
        print(
            f"    {name:34s} & {bench_s} & {cell('single_neutral')} & "
            f"{cell('debate_neutral')} & {cell('debate_averse')}\\\\"
        )
    if not any_found:
        print(
            "  [missing] no e5_regime_* suite leaderboards found -> run regime.json per window"
        )


def repeats_ci():
    print("\n===== Repeats: consensus Sharpe mean +- 95% CI (tab:repeats) =====")

    # Start with an empty dictionary.
    by_config = {}
    for leaderboard_path in sorted(OUT.glob("repeats/*/*/suites/*/leaderboard.jsonl")):
        for row in read_jsonl(leaderboard_path):
            if row.get("status") != "completed":
                continue
            key = (
                row.get("model"),
                row.get("workflow_mode"),
                row.get("risk_profile"),
                row.get("tooling_mode"),
            )
            by_config.setdefault(key, []).append(row["consensus_sharpe"])
    if not by_config:
        print(
            "  [missing] no repeats found -> run repeat suites into "
            "experiments/output/repeats/rep<i>"
        )

        return

    for (model, workflow, risk, tooling), sharpes in sorted(by_config.items()):
        mean, lower_limit, upper_limit = mean_ci95(sharpes)

        label = f"{workflow}/{risk}/{tooling}"
        print(
            f"    {label:45s} & {len(sharpes):2d} & ${mean:.2f}$ & $[{lower_limit:.2f}, {upper_limit:.2f}]$\\\\"
            f"  % {model}"
        )


def signal_distribution():
    print("\n===== Signal distribution & decision quality per run =====")
    print(
        f"  {'run':66s} {'B':>3s} {'S':>3s} {'H':>3s} {'fail':>4s} "
        f"{'up%':>4s} {'hitB':>5s} {'hitS':>5s} {'IC':>6s}"
    )
    for result_path in sorted(OUT.rglob("result.json")):
        try:
            result = json.loads(result_path.read_text())
            manifest_path = result_path.parent.parent / "manifest.json"
            realized = {}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                for ticker, snap in manifest.get("stocks", {}).items():
                    prices = snap.get("evaluation_prices", [])
                    if len(prices) >= 2 and prices[0]["close"]:
                        realized[ticker] = (
                            prices[-1]["close"] / prices[0]["close"] - 1.0
                        )
        except (json.JSONDecodeError, OSError, KeyError):
            continue
        counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        failed = 0
        benchmark_sharpe = sum(realized.values()) / len(realized) if realized else 0.0
        buy_hits = buy_n = sell_hits = sell_n = 0
        for ticker, stock_result in result.get("stock_results", {}).items():
            consensus = stock_result.get("consensus")
            if not consensus:
                failed += 1
                continue
            signal = consensus.get("signal")
            counts[signal] = counts.get(signal, 0) + 1
            if ticker in realized:
                excess = realized[ticker] - benchmark_sharpe
                if signal == "BUY":
                    buy_n += 1
                    buy_hits += int(excess > 0)
                elif signal == "SELL":
                    sell_n += 1

                    sell_hits += int(excess < 0)
        up = (
            100 * sum(1 for v in realized.values() if v > 0) / len(realized)
            if realized
            else 0
        )
        decision_quality_values = result.get("run_metadata", {}).get(
            "decision_quality", {}
        )
        rank_correlation = decision_quality_values.get("rank_ic")
        label = f"{result_path.parent.parent.name[:14]}/{result_path.parent.name[:50]}"
        hit_b = f"{buy_hits/buy_n:5.2f}" if buy_n else "  n/a"
        hit_s = f"{sell_hits/sell_n:5.2f}" if sell_n else "  n/a"
        ic_s = (
            f"{rank_correlation:+6.2f}"
            if isinstance(rank_correlation, (int, float))
            else "   n/a"
        )
        print(
            f"  {label:66s} {counts['BUY']:3d} {counts['SELL']:3d} {counts['HOLD']:3d} "
            f"{failed:4d} {up:4.0f} {hit_b} {hit_s} {ic_s}"
        )
    print(
        "  NOTE: an arm with (near-)constant signals measures prompt compliance,"
        " not stock discrimination - flag it in the thesis."
    )


if __name__ == "__main__":
    e2_ablation()
    e3_cross_model()
    e4_resample()
    e5_regime()
    repeats_ci()
    signal_distribution()

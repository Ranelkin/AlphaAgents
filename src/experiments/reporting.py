import json
from pathlib import Path


def load_saved_result(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _portfolio_metric(saved_result, portfolio_name, metric_name):
    metrics = saved_result.get("metrics", {}).get(portfolio_name, {})
    value = metrics.get(metric_name)
    return float(value) if value is not None else None


def summarize_saved_results(result_paths):
    result_rows = []
    # Start with an empty dictionary.
    by_risk = {}
    by_workflow = {}
    for result_path in result_paths:
        saved_result = load_saved_result(result_path)
        saved_configuration = saved_result.get("config", {})
        cumulative_return_value = _portfolio_metric(
            saved_result, "consensus", "cumulative_return"
        )
        sharpe_ratio_value = _portfolio_metric(
            saved_result, "consensus", "sharpe_ratio"
        )
        max_drawdown = _portfolio_metric(saved_result, "consensus", "max_drawdown")
        rolling_sharpe_values = (
            saved_result.get("metrics", {})
            .get("consensus", {})
            .get("rolling_sharpe", [])
        )

        result_row = {
            "result_path": str(result_path),
            "model": saved_configuration.get("model"),
            "provider": saved_configuration.get("provider"),
            "workflow_mode": saved_configuration.get("workflow_mode"),
            "risk_profile": saved_configuration.get("risk_profile"),
            "tooling_mode": saved_configuration.get("tooling_mode"),
            "manifest_hash": saved_result.get("manifest_hash"),
            "consensus_cumulative_return": cumulative_return_value,
            "consensus_sharpe": sharpe_ratio_value,
            "consensus_max_drawdown": max_drawdown,
            "rolling_sharpe_points": len(rolling_sharpe_values),
        }
        result_rows.append(result_row)
        risk_key = saved_configuration.get("risk_profile")
        if risk_key and cumulative_return_value is not None:
            by_risk.setdefault(risk_key, []).append(cumulative_return_value)
        workflow_key = saved_configuration.get("workflow_mode")

        if workflow_key and sharpe_ratio_value is not None:
            by_workflow.setdefault(workflow_key, []).append(sharpe_ratio_value)

    def average(values):
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "runs": result_rows,
        "comparisons": {
            "risk_profile_average_cumulative_return": {
                key: average(values) for key, values in by_risk.items()
            },
            "workflow_average_sharpe": {
                key: average(values) for key, values in by_workflow.items()
            },
        },
    }


def write_report(result_paths, output_path):
    report_summary = summarize_saved_results(result_paths)
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text(json.dumps(report_summary, indent=2), encoding="utf-8")
    return output_file_path

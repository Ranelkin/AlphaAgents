import argparse
import json
import sys

from dotenv import load_dotenv

from src.experiments.suite import run_experiment_suite

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Run a configuration matrix against one manifest."
    )

    parser.add_argument(
        "--manifest", required=True, help="Path to a frozen manifest JSON."
    )
    parser.add_argument(
        "--suite-config", required=True, help="Path to a JSON or YAML suite definition."
    )
    parser.add_argument("--output-dir", default="experiments/output")
    parser.add_argument(
        "--run-label", required=True, help="Label used for the suite output directory."
    )
    args = parser.parse_args()
    summary = run_experiment_suite(
        manifest_path=args.manifest,
        suite_config_path=args.suite_config,
        output_dir=args.output_dir,
        run_label=args.run_label,
        cli_args=sys.argv[1:],
    )
    print(
        json.dumps(
            {
                "manifest_hash": summary.manifest_hash,
                "suite_dir": str(summary.suite_dir),
                "leaderboard_jsonl": str(summary.leaderboard_jsonl),
                "leaderboard_csv": str(summary.leaderboard_csv),
                "rows": summary.rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

import argparse
import json

from src.experiments.reporting import write_report


def main():
    parser = argparse.ArgumentParser(
        description="Summarize saved results without rerunning models."
    )
    parser.add_argument(
        "--result",
        dest="results",
        action="append",
        required=True,
        help="Path to a saved result.json.",
    )
    parser.add_argument(
        "--output", required=True, help="Path to the generated report JSON."
    )
    args = parser.parse_args()

    output_path = write_report(args.results, args.output)
    print(json.dumps({"report": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()

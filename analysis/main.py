import argparse
from pathlib import Path

from src.analyzer import analyze
from src.plotting import plot
from src.results import write_output


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "output"
PREDICTIONS_FILENAME = "cost_predictions.json"
SCATTER_FILENAME = "cost_prediction_scatter.png"
PIPELINE_PLOT_FILENAME = "cost_pipeline_counts.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the cost-analysis pass over TTGIR metadata and plot throughput "
            "predictions."
        )
    )
    parser.add_argument("results", type=Path, help="benchmark result.json")
    parser.add_argument("ttgir_dir", type=Path, help="directory containing TTGIR files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / PREDICTIONS_FILENAME

    rows, skips = analyze(args.results, args.ttgir_dir)
    write_output(predictions_path, rows, skips)
    plot(
        rows,
        output_dir / SCATTER_FILENAME,
        output_dir / PIPELINE_PLOT_FILENAME,
    )
    print(f"wrote {len(rows)} predictions to {predictions_path}")
    if skips:
        print(f"skipped {len(skips)} records")


if __name__ == "__main__":
    main()

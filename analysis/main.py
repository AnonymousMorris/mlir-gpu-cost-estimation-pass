from __future__ import annotations

import argparse
from pathlib import Path

from cost_analysis.dataset import DatasetStore, fit_global, metrics
from cost_analysis.plots import write_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Triton cost analysis plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for generated plot images.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild the dataset before plotting.",
    )
    args = parser.parse_args()

    dataset = DatasetStore().load(refresh=args.refresh)
    values = fit_global(dataset)
    written = write_plots(dataset, values, args.output_dir)
    fit_metrics = metrics(dataset, values)

    print(f"Wrote {len(written)} plot files to {args.output_dir}")
    print(f"RMSE: {fit_metrics['rmse']:.6g} ms")
    print(f"MAE: {fit_metrics['mae']:.6g} ms")

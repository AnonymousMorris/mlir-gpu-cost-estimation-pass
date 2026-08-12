import argparse

from benchmarking import storage
from benchmarking.runner import run_sweep
from kernels import KERNEL_MODULES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate Triton TTGIR files and benchmark timings."
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=100,
        help="Approximate warmup duration per benchmark case.",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=1000,
        help="Approximate measured repetition duration per benchmark case.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=f"Discard {storage.RESUME_PATH} and start from the first kernel.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_sweep(
        KERNEL_MODULES,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()

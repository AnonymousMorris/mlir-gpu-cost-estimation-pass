import argparse

from benchmarking.storage import OutputPaths


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_int(value: str) -> int:
    number = nonnegative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate Triton TTGIR files and benchmark timings."
    )
    parser.add_argument(
        "--warmup-ms",
        type=nonnegative_int,
        default=100,
        help="Approximate warmup duration per benchmark case.",
    )
    parser.add_argument(
        "--rep-ms",
        type=positive_int,
        default=1000,
        help="Approximate measured repetition duration per benchmark case.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=f"Discard {OutputPaths().resume_path} and start from the first kernel.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from benchmarking.runner import run_sweep
    from kernels import KERNEL_MODULES

    run_sweep(
        KERNEL_MODULES,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()

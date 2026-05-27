# Triton NVGPU

Generate Triton TTGIR programs and benchmark kernel launch configurations. The generated data can be compared against estimates from an MLIR pass.

## Setup

With uv:

```bash
uv sync
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
uv run main.py
```

For lower-noise timings, increase the benchmark duration per kernel configuration:

```bash
uv run main.py --warmup-ms 500 --rep-ms 5000
```

Or, if you installed with pip:

```bash
python main.py
```

The script writes:

- `results/result.json`: timing records grouped by kernel name
- `results/ttgir/*.ttgir`: one TTGIR file per kernel argument configuration

Each timing record stores the median runtime as `time_ms` plus p20/p80 timings
and a simple spread ratio. Large spread ratios are a sign that the kernel is too
short or noisy for reliable fitting.

The current kernel grids emit 167 TTGIR files on standard CUDA targets, plus
8 more block-scaled matmul files on targets that support block scaling.

## Adding Kernels

Create a kernel module that defines:

```python
KERNEL = your_triton_kernel

def iter_args(device):
    yield args, kwargs, grid
```

Then import the module in `main.py` and add it to `KERNEL_MODULES`.

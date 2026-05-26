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

Or, if you installed with pip:

```bash
python main.py
```

The script writes:

- `results/result.json`: timing records grouped by kernel name
- `results/ttgir/*.ttgir`: one TTGIR file per kernel argument configuration

## Adding Kernels

Create a kernel module that defines:

```python
KERNEL = your_triton_kernel

def iter_args(device):
    yield args, kwargs, grid
```

Then import the module in `main.py` and add it to `KERNEL_MODULES`.

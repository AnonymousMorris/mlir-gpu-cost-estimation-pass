## MLIR Cost UI

Run the web UI with:

```bash
MLIR_PYTHON_PACKAGE_DIR=/home/morris/llvm-project/build/tools/mlir/python_packages/mlir_core \
  uv run python main.py
```

Open `http://127.0.0.1:8000`, paste generated MLIR containing
`func.func @__cost_expr`, and edit the variable inputs to update the displayed
cost in real time. The server parses the MLIR with the MLIR Python binding,
converts the cost function into a SymPy equation, simplifies it, and sends the
browser a JavaScript-ready equation string.

The app validates input with the LLVM MLIR Python bindings:

```python
from mlir.ir import Context, Module
```

If your MLIR build has Python bindings enabled, point the app at the generated
package directory. On this machine, that is:

```bash
MLIR_PYTHON_PACKAGE_DIR=/home/morris/llvm-project/build/tools/mlir/python_packages/mlir_core \
  uv run python main.py
```

For an installed MLIR tree, the equivalent path is commonly:

```bash
MLIR_PYTHON_PACKAGE_DIR=/path/to/install/python_packages/mlir_core \
  uv run python main.py
```

The bindings under `/home/morris/llvm-project/build` were built for CPython
3.14, so this project is pinned to Python 3.14 in `.python-version`.

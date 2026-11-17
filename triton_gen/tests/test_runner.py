import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

import main
from benchmarking import runner, storage


class FakeProgress:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)

    def set_description(self, description):
        pass

    def write(self, message):
        pass


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paths = storage.OutputPaths(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_sweep_writes_timing_record_and_ttgir(self):
        def add_kernel():
            pass

        add_kernel.arg_names = ("input_ptr", "n_elements", "BLOCK_SIZE")
        tensor = torch.empty(0)
        module = SimpleNamespace(
            KERNEL=add_kernel,
            iter_args=lambda device: iter(
                [
                    (
                        (tensor, 4096),
                        {"BLOCK_SIZE": 1024},
                        lambda meta: (4096 // meta["BLOCK_SIZE"],),
                    )
                ]
            ),
        )
        compiled_kernel = SimpleNamespace(
            name="add_kernel",
            asm={"ttgir": "module { test }"},
        )
        fake_triton = SimpleNamespace(
            runtime=SimpleNamespace(
                driver=SimpleNamespace(
                    active=SimpleNamespace(
                        get_active_torch_device=lambda: "cuda:0",
                    )
                )
            )
        )

        with (
            patch.object(runner, "triton", fake_triton),
            patch.object(
                runner,
                "tqdm",
                side_effect=lambda items, **kwargs: FakeProgress(items),
            ),
            patch.object(
                runner,
                "benchmark_kernel",
                return_value=runner.BenchmarkResult(
                    compiled_kernel=compiled_kernel,
                    time_ms=0.01,
                    time_p20_ms=0.009,
                    time_p80_ms=0.011,
                    time_cv=0.2,
                ),
            ),
            patch.object(runner, "clear_cuda_cache"),
        ):
            runner.run_sweep(
                [module], warmup_ms=100, rep_ms=1000, paths=self.paths
            )

        result = json.loads(self.paths.result_path.read_text())
        self.assertEqual(
            result["add_kernel"][0],
            {
                "args": ["4096"],
                "kwargs": {"BLOCK_SIZE": "1024"},
                "scalar_args": {"n_elements": 4096, "BLOCK_SIZE": 1024},
                "grid_size": [4],
                "block_size": {"BLOCK_SIZE": "1024"},
                "compiled_name": "add_kernel",
                "ttgir_filename": "add_kernel_4096_BLOCK_SIZE=1024.ttgir",
                "time_ms": 0.01,
                "time_p20_ms": 0.009,
                "time_p80_ms": 0.011,
                "time_cv": 0.2,
                "status": "ok",
                "error": None,
            },
        )
        ttgir_path = self.paths.ttgir_dir / result["add_kernel"][0]["ttgir_filename"]
        self.assertEqual(ttgir_path.read_text(), "module { test }")
        self.assertFalse(self.paths.resume_path.exists())

    def test_main_forwards_cli_options(self):
        from kernels import KERNEL_MODULES

        with patch.object(runner, "run_sweep") as run_sweep:
            main.main(["--warmup-ms", "25", "--rep-ms", "250", "--restart"])

        run_sweep.assert_called_once_with(
            KERNEL_MODULES,
            warmup_ms=25,
            rep_ms=250,
            restart=True,
        )


if __name__ == "__main__":
    unittest.main()

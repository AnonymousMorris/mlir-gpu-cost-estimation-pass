import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarking import storage
from kernels import grouped_gemm


class ResumeStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paths = storage.OutputPaths(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resume_state_round_trip(self):
        result = {"add_kernel": [{"args": ["4096"], "kwargs": {"BLOCK_SIZE": "1024"}}]}

        storage.save_resume_state(
            result, ["add_kernel"], 100, 1000, 0, 1, paths=self.paths
        )
        state = storage.load_resume_state(["add_kernel"], 100, 1000, paths=self.paths)

        self.assertEqual(state["result"], result)
        self.assertEqual(state["next_kernel_index"], 0)
        self.assertEqual(state["next_case_index"], 1)
        self.assertFalse(self.paths.resume_path.with_suffix(".tmp.new").exists())

    def test_resume_state_rejects_changed_run_settings(self):
        storage.save_resume_state(
            {}, ["add_kernel"], 100, 1000, 0, 0, paths=self.paths
        )

        with self.assertRaisesRegex(RuntimeError, "rep_ms"):
            storage.load_resume_state(["add_kernel"], 100, 2000, paths=self.paths)

    def test_output_paths_keep_independent_runs_isolated(self):
        other_paths = storage.OutputPaths(self.paths.root / "other")
        storage.write_result({"first": []}, paths=self.paths)
        storage.write_result({"second": []}, paths=other_paths)
        storage.write_ttgir("kernel", "first module", paths=self.paths)
        storage.write_ttgir("kernel", "second module", paths=other_paths)
        storage.save_resume_state(
            {}, ["add_kernel"], 100, 1000, 0, 0, paths=self.paths
        )

        self.assertIn('"first"', self.paths.result_path.read_text())
        self.assertIn('"second"', other_paths.result_path.read_text())
        self.assertEqual(
            (self.paths.ttgir_dir / "kernel.ttgir").read_text(), "first module"
        )
        self.assertEqual(
            (other_paths.ttgir_dir / "kernel.ttgir").read_text(), "second module"
        )
        self.assertFalse(other_paths.resume_path.exists())

    def test_final_result_is_written_atomically(self):
        result = {"add_kernel": []}

        storage.write_result(result, paths=self.paths)

        self.assertEqual(self.paths.result_path.read_text(), '{\n  "add_kernel": []\n}')
        self.assertFalse(self.paths.result_path.with_suffix(".json.new").exists())


class GroupedGemmLifetimeTests(unittest.TestCase):
    def tearDown(self):
        grouped_gemm.release_args()

    def test_iter_args_releases_each_cases_tensor_owners(self):
        owners = []

        def make_args(*args, **kwargs):
            owner = object()
            owners.append(owner)
            grouped_gemm._KEEPALIVE = owner
            return (), {}, (1,)

        with patch.object(grouped_gemm, "make_args", side_effect=make_args):
            cases = grouped_gemm.iter_args(None)
            next(cases)
            self.assertIs(grouped_gemm._KEEPALIVE, owners[0])

            next(cases)
            self.assertIs(grouped_gemm._KEEPALIVE, owners[1])

            cases.close()
            self.assertIsNone(grouped_gemm._KEEPALIVE)


if __name__ == "__main__":
    unittest.main()

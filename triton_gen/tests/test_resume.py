import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarking import storage
from kernels import grouped_gemm


class ResumeStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_result_path = storage.RESULT_PATH
        self.original_resume_path = storage.RESUME_PATH
        root = Path(self.temp_dir.name)
        storage.RESULT_PATH = root / "result.json"
        storage.RESUME_PATH = root / "run_state.json.tmp"

    def tearDown(self):
        storage.RESULT_PATH = self.original_result_path
        storage.RESUME_PATH = self.original_resume_path
        self.temp_dir.cleanup()

    def test_resume_state_round_trip(self):
        result = {"add_kernel": [{"args": ["4096"], "kwargs": {"BLOCK_SIZE": "1024"}}]}

        storage.save_resume_state(result, ["add_kernel"], 100, 1000, 0, 1)
        state = storage.load_resume_state(["add_kernel"], 100, 1000)

        self.assertEqual(state["result"], result)
        self.assertEqual(state["next_kernel_index"], 0)
        self.assertEqual(state["next_case_index"], 1)
        self.assertFalse(storage.RESUME_PATH.with_name("run_state.json.tmp.new").exists())

    def test_resume_state_rejects_changed_run_settings(self):
        storage.save_resume_state({}, ["add_kernel"], 100, 1000, 0, 0)

        with self.assertRaisesRegex(RuntimeError, "rep_ms"):
            storage.load_resume_state(["add_kernel"], 100, 2000)

    def test_final_result_is_written_atomically(self):
        result = {"add_kernel": []}

        storage.write_result(result)

        self.assertEqual(storage.RESULT_PATH.read_text(), '{\n  "add_kernel": []\n}')
        self.assertFalse(storage.RESULT_PATH.with_name("result.json.new").exists())


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

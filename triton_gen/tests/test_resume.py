import tempfile
import unittest
from pathlib import Path

import main


class ResumeStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_result_path = main.RESULT_PATH
        self.original_resume_path = main.RESUME_PATH
        root = Path(self.temp_dir.name)
        main.RESULT_PATH = root / "result.json"
        main.RESUME_PATH = root / "run_state.json.tmp"

    def tearDown(self):
        main.RESULT_PATH = self.original_result_path
        main.RESUME_PATH = self.original_resume_path
        self.temp_dir.cleanup()

    def test_resume_state_round_trip(self):
        result = {"add_kernel": [{"args": ["4096"], "kwargs": {"BLOCK_SIZE": "1024"}}]}

        main.save_resume_state(result, ["add_kernel"], 100, 1000, 0, 1)
        state = main.load_resume_state(["add_kernel"], 100, 1000)

        self.assertEqual(state["result"], result)
        self.assertEqual(state["next_kernel_index"], 0)
        self.assertEqual(state["next_case_index"], 1)
        self.assertFalse(main.RESUME_PATH.with_name("run_state.json.tmp.new").exists())

    def test_resume_state_rejects_changed_run_settings(self):
        main.save_resume_state({}, ["add_kernel"], 100, 1000, 0, 0)

        with self.assertRaisesRegex(RuntimeError, "rep_ms"):
            main.load_resume_state(["add_kernel"], 100, 2000)

    def test_final_result_is_written_atomically(self):
        result = {"add_kernel": []}

        main.write_result(result)

        self.assertEqual(main.RESULT_PATH.read_text(), '{\n  "add_kernel": []\n}')
        self.assertFalse(main.RESULT_PATH.with_name("result.json.new").exists())


if __name__ == "__main__":
    unittest.main()

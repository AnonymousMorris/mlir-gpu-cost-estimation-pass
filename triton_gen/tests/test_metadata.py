import unittest
from dataclasses import asdict

import main


class MetadataTests(unittest.TestCase):
    def test_records_callable_grid_dimensions(self):
        grid = lambda meta: (meta["N"] // meta["BLOCK_SIZE"], 2)

        self.assertEqual(
            main.record_grid_size(grid, {"N": 4096, "BLOCK_SIZE": 1024}),
            [4, 2],
        )

    def test_normalizes_scalar_grid(self):
        self.assertEqual(main.record_grid_size(8, {}), [8])

    def test_records_logical_block_parameters(self):
        self.assertEqual(
            main.record_block_size(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "num_warps": 4,
                }
            ),
            {
                "BLOCK_SIZE_M": "64",
                "BLOCK_SIZE_N": "128",
            },
        )

    def test_successful_record_schema_includes_launch_metadata(self):
        record = main.KernelRunRecord(
            args=["4096"],
            kwargs={"BLOCK_SIZE": "1024"},
            grid_size=[4],
            block_size={"BLOCK_SIZE": "1024"},
            compiled_name="add_kernel",
            ttgir_filename="add.ttgir",
            time_ms=0.01,
            time_p20_ms=0.009,
            time_p80_ms=0.011,
            time_cv=0.2,
        )

        payload = asdict(record)
        self.assertEqual(payload["grid_size"], [4])
        self.assertEqual(payload["block_size"], {"BLOCK_SIZE": "1024"})
        self.assertEqual(payload["status"], "ok")
        self.assertIsNone(payload["error"])


if __name__ == "__main__":
    unittest.main()

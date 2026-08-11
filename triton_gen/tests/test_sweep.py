from contextlib import ExitStack
import inspect
import unittest
from unittest.mock import patch

from kernels import KERNEL_MODULES
from kernels import attention
from kernels import block_scaled_matmul
from kernels import dropout
from kernels import fp32_fma
from kernels import fp64_fma
from kernels import grouped_gemm
from kernels import integer_hash
from kernels import layer_norm
from kernels import libdevice_asin
from kernels import matmul
from kernels import persistent_matmul
from kernels import sfu_exp2
from kernels import softmax
from kernels import vec_add


def collect_case_arguments(module, *, patches=()):
    signature = inspect.signature(module.make_args)
    sentinel = ((), {}, (1,))
    with patch.object(module, "make_args", return_value=sentinel) as make_args:
        with ExitStack() as stack:
            for target, value in patches:
                stack.enter_context(patch.object(module, target, return_value=value))
            list(module.iter_args(None))

    cases = []
    for invocation in make_args.call_args_list:
        bound = signature.bind(*invocation.args, **invocation.kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        del arguments["device"]
        cases.append(arguments)
    return cases


class SweepTests(unittest.TestCase):
    def test_matches_laptop_sweep(self):
        matmul_configs = (
            (128, 256, 64, 8, 8, 3),
            (64, 256, 32, 8, 4, 4),
            (128, 128, 32, 8, 4, 4),
            (128, 64, 32, 8, 4, 4),
            (64, 128, 32, 8, 4, 4),
            (128, 32, 32, 8, 4, 4),
            (64, 32, 32, 8, 2, 5),
            (32, 64, 32, 8, 2, 5),
            (128, 256, 128, 8, 8, 3),
            (256, 128, 128, 8, 8, 3),
            (256, 64, 128, 8, 4, 4),
            (64, 256, 128, 8, 4, 4),
            (128, 128, 128, 8, 4, 4),
            (128, 64, 64, 8, 4, 4),
            (64, 128, 64, 8, 4, 4),
            (128, 32, 64, 8, 4, 4),
        )
        grouped_shapes = (
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (128, 8192, 8192),
            (256, 8192, 8192),
            (512, 8192, 8192),
            (1024, 8192, 8192),
        )
        expected = {
            vec_add: [
                {"size": 2**exponent, "block_size": 1024}
                for exponent in range(12, 28)
            ],
            softmax: [
                {"n_rows": 4096, "n_cols": n_cols, "num_warps": 8, "pipeline_stages": 2}
                for n_cols in range(256, 12673, 128)
            ],
            matmul: [
                {
                    "M": size,
                    "N": size,
                    "K": size,
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_k": block_k,
                    "group_m": group_m,
                    "activation": "",
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                }
                for size in range(256, 4097, 128)
                for block_m, block_n, block_k, group_m, num_warps, num_stages in matmul_configs
            ],
            fp32_fma: [
                {"size": size, "iterations": 256, "block_size": 256}
                for size in (2**18, 2**20, 2**22)
            ],
            fp64_fma: [
                {"size": size, "iterations": 32, "block_size": 256}
                for size in (2**16, 2**18, 2**20)
            ],
            sfu_exp2: [
                {"size": size, "iterations": 32, "block_size": 256}
                for size in (2**18, 2**20, 2**22)
            ],
            integer_hash: [
                {"size": size, "iterations": 64, "block_size": 256}
                for size in (2**18, 2**20, 2**22)
            ],
            dropout: [
                {"size": 10, "p": 0.5, "seed": seed, "block_size": 1024}
                for seed in (123, 512)
            ],
            layer_norm: [
                {
                    "M": 4096,
                    "N": N,
                    "eps": 1e-5,
                    "block_size": 1 << (N - 1).bit_length(),
                    "num_warps": 4 if N == 1024 else 8,
                }
                for N in range(1024, 15873, 512)
            ],
            attention: [
                {"Z": 4, "H": 32, "N_CTX": N_CTX, "HEAD_DIM": HEAD_DIM, "causal": causal}
                for HEAD_DIM in (64, 128)
                for N_CTX in (1024, 2048, 4096, 8192, 16384)
                for causal in (True, False)
            ],
            libdevice_asin: [
                {"size": 98432, "block_size": 1024},
            ],
            grouped_gemm: [
                {"group_size": 4, "shape": shape, "num_sm": 30}
                for shape in grouped_shapes
            ],
            persistent_matmul: [
                {"M": 8192, "N": 8192, "K": K, "num_sms": 30}
                for K in range(128, 1025, 128)
            ],
        }
        expected_counts = {
            vec_add: 16,
            softmax: 98,
            matmul: 496,
            fp32_fma: 3,
            fp64_fma: 3,
            sfu_exp2: 3,
            integer_hash: 3,
            dropout: 2,
            layer_norm: 30,
            attention: 20,
            libdevice_asin: 1,
            grouped_gemm: 8,
            persistent_matmul: 8,
        }

        for module, expected_cases in expected.items():
            with self.subTest(kernel=module.KERNEL.__name__):
                actual_cases = collect_case_arguments(module)
                self.assertEqual(actual_cases, expected_cases)
                self.assertEqual(len(actual_cases), expected_counts[module])

        self.assertEqual(sum(expected_counts.values()), 691)

    def test_registered_kernel_order(self):
        self.assertEqual(
            KERNEL_MODULES,
            [
                vec_add,
                softmax,
                matmul,
                fp32_fma,
                fp64_fma,
                sfu_exp2,
                integer_hash,
                dropout,
                layer_norm,
                attention,
                libdevice_asin,
                grouped_gemm,
                persistent_matmul,
                block_scaled_matmul,
            ],
        )

    def test_block_scaled_targets_add_eight_cases(self):
        cases = collect_case_arguments(
            block_scaled_matmul,
            patches=(("supports_block_scaling", True),),
        )

        self.assertEqual(len(cases), 8)


if __name__ == "__main__":
    unittest.main()

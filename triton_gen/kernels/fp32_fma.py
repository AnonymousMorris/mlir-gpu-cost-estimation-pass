import torch
import triton
import triton.language as tl


@triton.jit
def fp32_fma_kernel(
    x_ptr,
    factor_ptr,
    output_ptr,
    n_elements,
    ITERATIONS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    factor = tl.load(factor_ptr + offsets, mask=mask, other=1.0)

    # Independent recurrence chains expose enough instruction-level parallelism
    # to sustain the FP32 pipelines while loading and storing each value once.
    acc0 = x
    acc1 = x + 0.25
    acc2 = x - 0.25
    acc3 = x * 0.5
    factor0 = factor
    factor1 = 2.0 - factor
    factor2 = 0.5 + 0.5 * factor
    factor3 = 1.5 - 0.5 * factor
    for _ in range(ITERATIONS):
        acc0 = acc0 * factor0 + 0.0001
        acc1 = acc1 * factor1 + 0.0002
        acc2 = acc2 * factor2 - 0.0001
        acc3 = acc3 * factor3 - 0.0002

    tl.store(output_ptr + offsets, acc0 + acc1 + acc2 + acc3, mask=mask)


KERNEL = fp32_fma_kernel


COMPUTE_ITERATIONS = 256
BENCHMARK_SIZES = (2**18, 2**20, 2**22)


def init_args(device):
    return make_args(device, size=2**18, iterations=COMPUTE_ITERATIONS, block_size=256)


def iter_args(device):
    for size in BENCHMARK_SIZES:
        yield make_args(device, size=size, iterations=COMPUTE_ITERATIONS, block_size=256)


def make_args(device, size, iterations, block_size):
    x = torch.rand(size, device=device, dtype=torch.float32)
    factor = torch.empty(size, device=device, dtype=torch.float32).uniform_(0.999, 1.001)
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    args = (x, factor, output, size)
    kwargs = {
        "ITERATIONS": iterations,
        "BLOCK_SIZE": block_size,
        "num_warps": 8,
    }
    return args, kwargs, grid

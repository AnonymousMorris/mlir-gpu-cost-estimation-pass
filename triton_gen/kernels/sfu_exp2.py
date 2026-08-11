import torch
import triton
import triton.language as tl


@triton.jit
def sfu_exp2_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    ITERATIONS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Four dependent chains provide SFU instruction-level parallelism. The
    # recurrences remain bounded, and every exp2 contributes to the output.
    acc0 = x
    acc1 = x + 0.1
    acc2 = x + 0.2
    acc3 = x + 0.3
    for _ in range(ITERATIONS):
        acc0 = tl.exp2(0.0 - acc0)
        acc1 = tl.exp2(0.1 - acc1)
        acc2 = tl.exp2(0.2 - acc2)
        acc3 = tl.exp2(0.3 - acc3)

    tl.store(output_ptr + offsets, acc0 + acc1 + acc2 + acc3, mask=mask)


KERNEL = sfu_exp2_kernel


COMPUTE_ITERATIONS = 32
BENCHMARK_SIZES = (2**18, 2**20, 2**22)


def init_args(device):
    return make_args(device, size=2**18, iterations=COMPUTE_ITERATIONS, block_size=256)


def iter_args(device):
    for size in BENCHMARK_SIZES:
        yield make_args(device, size=size, iterations=COMPUTE_ITERATIONS, block_size=256)


def make_args(device, size, iterations, block_size):
    input_tensor = torch.rand(size, device=device, dtype=torch.float32)
    output = torch.empty_like(input_tensor)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    args = (input_tensor, output, size)
    kwargs = {
        "ITERATIONS": iterations,
        "BLOCK_SIZE": block_size,
        "num_warps": 8,
    }
    return args, kwargs, grid

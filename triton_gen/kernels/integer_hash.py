import torch
import triton
import triton.language as tl


@triton.jit
def integer_hash_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    ITERATIONS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.uint32)

    # Two independent avalanche chains hide integer multiply latency. They
    # perform hundreds of integer operations for one load and one store.
    hash0 = value
    hash1 = value ^ 0x9E3779B9
    for _ in range(ITERATIONS):
        hash0 ^= hash0 >> 16
        hash0 *= 0x7FEB352D
        hash0 ^= hash0 >> 15
        hash0 *= 0x846CA68B
        hash0 ^= hash0 >> 16

        hash1 ^= hash1 >> 16
        hash1 *= 0x7FEB352D
        hash1 ^= hash1 >> 15
        hash1 *= 0x846CA68B
        hash1 ^= hash1 >> 16

    tl.store(output_ptr + offsets, hash0 ^ hash1, mask=mask)


KERNEL = integer_hash_kernel


COMPUTE_ITERATIONS = 64
BENCHMARK_SIZES = (2**18, 2**20, 2**22)


def init_args(device):
    return make_args(device, size=2**18, iterations=COMPUTE_ITERATIONS, block_size=256)


def iter_args(device):
    for size in BENCHMARK_SIZES:
        yield make_args(device, size=size, iterations=COMPUTE_ITERATIONS, block_size=256)


def make_args(device, size, iterations, block_size):
    input_tensor = torch.randint(0, 2**31, (size,), device=device, dtype=torch.int32)
    output = torch.empty_like(input_tensor)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    args = (input_tensor, output, size)
    kwargs = {
        "ITERATIONS": iterations,
        "BLOCK_SIZE": block_size,
        "num_warps": 8,
    }
    return args, kwargs, grid

import torch
import triton
import triton.language as tl


@triton.jit
def seeded_dropout_kernel(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


KERNEL = seeded_dropout_kernel


def init_args(device):
    return make_args(device, size=98432, p=0.5, seed=123, block_size=1024)


def iter_args(device):
    for size in (1024, 4096, 65536, 98432, 262144):
        for p in (0.1, 0.5):
            for block_size in (256, 1024):
                yield make_args(device, size=size, p=p, seed=123, block_size=block_size)


def make_args(device, size, p, seed, block_size):
    x = torch.randn(size, device=device)
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    args = (x, output, n_elements, p, seed)
    kwargs = {"BLOCK_SIZE": block_size}
    return args, kwargs, grid

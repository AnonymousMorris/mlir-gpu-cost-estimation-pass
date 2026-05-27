import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def asin_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = libdevice.asin(x)
    tl.store(y_ptr + offsets, y, mask=mask)


KERNEL = asin_kernel


def init_args(device):
    return make_args(device, size=98432, block_size=1024)


def iter_args(device):
    for size in (1024, 4096, 65536, 98432, 262144):
        for block_size in (256, 1024):
            yield make_args(device, size, block_size=block_size)


def make_args(device, size, block_size):
    x = torch.rand(size, device=device)
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    args = (x, y, n_elements)
    kwargs = {"BLOCK_SIZE": block_size}
    return args, kwargs, grid

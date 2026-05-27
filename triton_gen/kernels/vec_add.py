import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


KERNEL = add_kernel


def init_args(device):
    return make_args(device, size=98432, block_size=1024)


def iter_args(device):
    for size in (1024, 4096, 65536, 98432, 262144):
        for block_size in (128, 256, 512, 1024):
            yield make_args(device, size, block_size)


def make_args(device, size, block_size):
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    args = (x, y, output, n_elements)
    kwargs = {"BLOCK_SIZE": block_size}
    return args, kwargs, grid

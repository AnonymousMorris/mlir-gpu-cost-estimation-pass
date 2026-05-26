import torch
import triton
import triton.language as tl


@triton.jit
def vec_add(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n
    a = tl.load(a_ptr + offset, mask)
    b = tl.load(b_ptr + offset, mask)
    output = a + b
    tl.store(out_ptr + offset, output, mask)


KERNEL = vec_add


def init_args(device):
    return make_args(device, size=98432, block_size=1024)


def iter_args(device):
    for size in (1024, 98432):
        for block_size in (256, 1024):
            yield make_args(device, size, block_size)


def make_args(device, size, block_size):
    a = torch.rand(size, device=device)
    b = torch.rand(size, device=device)

    output = torch.empty_like(a)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    args = (a, b, output, n_elements)
    kwargs = {"BLOCK_SIZE": block_size}
    return args, kwargs, grid

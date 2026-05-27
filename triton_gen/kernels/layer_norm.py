import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    mean = 0.0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + cols, y, mask=mask)


KERNEL = _layer_norm_fwd_fused


def init_args(device):
    return make_args(device, M=128, N=1024, eps=1e-5, block_size=1024)


def iter_args(device):
    for M, N in ((64, 256), (128, 512), (128, 1024), (256, 1024), (128, 2048)):
        for eps in (1e-5, 1e-6):
            yield make_args(device, M, N, eps=eps, block_size=triton.next_power_of_2(N))


def make_args(device, M, N, eps, block_size):
    x = torch.randn((M, N), device=device, dtype=torch.float16)
    y = torch.empty_like(x)
    w = torch.ones((N,), device=device, dtype=torch.float16)
    b = torch.zeros((N,), device=device, dtype=torch.float16)
    mean = torch.empty((M,), device=device, dtype=torch.float32)
    rstd = torch.empty((M,), device=device, dtype=torch.float32)
    grid = lambda meta: (M,)
    args = (x, y, w, b, mean, rstd, x.stride(0), N, eps)
    kwargs = {"BLOCK_SIZE": block_size, "num_warps": 4}
    return args, kwargs, grid

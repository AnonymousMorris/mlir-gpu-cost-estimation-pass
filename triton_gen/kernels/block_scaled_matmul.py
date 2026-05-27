import torch
import triton
import triton.language as tl


def is_hip_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


def supports_block_scaling():
    target = triton.runtime.driver.active.get_current_target()
    is_blackwell_cuda = target is not None and target.backend == "cuda" and torch.cuda.get_device_capability()[0] in [10, 11]
    return is_blackwell_cuda or is_hip_cdna4()


@triton.jit
def block_scaled_matmul_kernel_cdna4(a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr, M, N, K, stride_am, stride_ak,
                                     stride_bk, stride_bn, stride_ck, stride_cm, stride_cn, stride_asm, stride_ask,
                                     stride_bsn, stride_bsk,
                                     # Meta-parameters
                                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                     mfma_nonkdim: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32
    num_k_iter = tl.cdiv(K, BLOCK_K // 2)
    # Create pointers for first block of A and B input matrices
    # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_k_split = offs_k
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Create pointers for the first block of A and B scales
    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, (BLOCK_N // 32))) % N
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)

    # B scales are N x K even though B operand is K x N.
    b_scale_ptrs = (b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk)
    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, (BLOCK_M // 32))) % M
    a_scale_ptrs = (a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        # Here we "undo" the shuffle done in global memory (shuffle_scales_cdna4 function).
        if mfma_nonkdim == 32:
            a_scales = tl.load(a_scale_ptrs).reshape(BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4,
                                                     1).permute(0, 3, 1, 4, 2,
                                                                5).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            b_scales = tl.load(b_scale_ptrs).reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4,
                                                     1).permute(0, 3, 1, 4, 2,
                                                                5).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
        elif mfma_nonkdim == 16:
            a_scales = tl.load(a_scale_ptrs).reshape(BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2,
                                                     1).permute(0, 5, 3, 1, 4, 2,
                                                                6).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            b_scales = tl.load(b_scale_ptrs).reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2,
                                                     1).permute(0, 5, 3, 1, 4, 2,
                                                                6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, cache_modifier=None)

        accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        # Advance the ptrs to the next K block.
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk

        a_scale_ptrs += BLOCK_K * stride_ask
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptrs = (c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


KERNEL = block_scaled_matmul_kernel_cdna4


def init_args(device):
    return make_args(device)


def iter_args(device):
    if supports_block_scaling():
        for M, N, K in ((128, 128, 128), (256, 128, 128), (128, 256, 128), (256, 256, 256)):
            for block_m, block_n, block_k, mfma_nonkdim in ((64, 64, 64, 16), (128, 64, 64, 16)):
                yield make_args(device, M=M, N=N, K=K, block_m=block_m, block_n=block_n, block_k=block_k,
                                mfma_nonkdim=mfma_nonkdim)


def make_args(device, M=128, N=128, K=128, block_m=64, block_n=64, block_k=64, mfma_nonkdim=16):
    a = torch.empty((M, K // 2), device=device, dtype=torch.uint8)
    b = torch.empty((K // 2, N), device=device, dtype=torch.uint8)
    c = torch.empty((M, N), device=device, dtype=torch.float16)
    a_scales = torch.empty((M // 32, K), device=device, dtype=torch.uint8)
    b_scales = torch.empty((N // 32, K), device=device, dtype=torch.uint8)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    args = (
        a,
        b,
        c,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        0,
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
    )
    kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "mfma_nonkdim": mfma_nonkdim,
        "num_warps": 4,
        "num_stages": 3,
    }
    return args, kwargs, grid

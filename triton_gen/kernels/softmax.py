import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=PIPELINE_STAGES):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


KERNEL = softmax_kernel


def init_args(device):
    return make_args(device, n_rows=1823, n_cols=781, num_warps=8, pipeline_stages=4)


def iter_args(device):
    for n_rows, n_cols in (
        (64, 128),
        (256, 256),
        (1024, 512),
        (1823, 781),
        (4096, 1024),
    ):
        for num_warps in (4, 8):
            for pipeline_stages in (2, 4):
                yield make_args(device, n_rows, n_cols, num_warps, pipeline_stages=pipeline_stages)


def make_args(device, n_rows, n_cols, num_warps, pipeline_stages):
    x = torch.randn((n_rows, n_cols), device=device)
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)
    grid = lambda meta: (n_rows,)
    args = (y, x, x.stride(0), y.stride(0), n_rows, n_cols)
    kwargs = {
        "BLOCK_SIZE": block_size,
        "PIPELINE_STAGES": pipeline_stages,
        "num_warps": num_warps,
    }
    return args, kwargs, grid

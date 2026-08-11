from . import attention
from . import block_scaled_matmul
from . import dropout
from . import fp32_fma
from . import fp64_fma
from . import grouped_gemm
from . import integer_hash
from . import layer_norm
from . import libdevice_asin
from . import matmul
from . import persistent_matmul
from . import sfu_exp2
from . import softmax
from . import vec_add


KERNEL_MODULES = [
    vec_add,
    softmax,
    matmul,
    fp32_fma,
    fp64_fma,
    sfu_exp2,
    integer_hash,
    dropout,
    layer_norm,
    attention,
    libdevice_asin,
    grouped_gemm,
    persistent_matmul,
    block_scaled_matmul,
]

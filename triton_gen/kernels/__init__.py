from . import attention
from . import block_scaled_matmul
from . import dropout
from . import grouped_gemm
from . import layer_norm
from . import libdevice_asin
from . import matmul
from . import persistent_matmul
from . import softmax
from . import vec_add


KERNEL_MODULES = [
    vec_add,
    softmax,
    matmul,
    dropout,
    layer_norm,
    attention,
    libdevice_asin,
    grouped_gemm,
    persistent_matmul,
    block_scaled_matmul,
]

import atexit

import pycuda.autoinit
from pycuda.tools import clear_context_caches, make_default_context
from skcuda.cublas import cublasCreate, cublasDestroy, cublasDgeam
from .cusparse import cusparseCreate, cusparseDestroy

ctx = make_default_context()
cusparse_handle = cusparseCreate()
cublas_handle = cublasCreate()

BLOCKSIZE = 256

def _grid(size):
    return ((int(size) + BLOCKSIZE-1)//BLOCKSIZE, 1)
def _block():
    return (BLOCKSIZE,1,1)

# Note: skcuda.linalg has a transpose function that only takes one argument,
# but it uses exactly this method, so it's not in-place. We rather track
# the extra buffer, to minimize memory allocation.
def transpose_gpuarray(in_gpu, out_gpu):
    L,M = in_gpu.shape
    if out_gpu.shape != (M,L):
        raise ValueError
    cublasDgeam(cublas_handle, 'T', 0,
                L, M, 1., in_gpu.gpudata,
                M, 0., 0, L, out_gpu.gpudata, L)

from . import Integration

atexit.register(cublasDestroy, cublas_handle)
atexit.register(cusparseDestroy, cusparse_handle)
atexit.register(ctx.pop)
atexit.register(clear_context_caches)

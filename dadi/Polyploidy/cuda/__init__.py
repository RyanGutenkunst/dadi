# Reuse the context and handles from dadi.cuda instead of creating new ones
# to avoid CUDA context conflicts that cause "invalid resource handle" errors
from dadi.cuda import (ctx, cusparse_handle, cublas_handle,
                       BLOCKSIZE, _grid, _block, transpose_gpuarray)

from . import Integration

# Don't register cleanup here - dadi.cuda already handles it

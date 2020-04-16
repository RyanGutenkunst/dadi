import atexit

from pycuda.tools import clear_context_caches, make_default_context
from skcuda.cusparse import cusparseCreate, cusparseDestroy

ctx = make_default_context()
cusparse_handle = cusparseCreate()

from . import Integration

atexit.register(cusparseDestroy, cusparse_handle)
atexit.register(ctx.pop)
atexit.register(clear_context_caches)
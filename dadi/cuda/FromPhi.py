import numpy as np
from scipy.special import betainc

#import pycuda.autoinit

from dadi.Spectrum_mod import _dbeta_cache
from dadi.cuda import kernels, _block, _grid

import pycuda.gpuarray as gpuarray
from skcuda import linalg, misc
linalg.init()
    
def diff_iny_2D(input):
    """
    Calculate input[:,1:] - input[:,:-1]
    """
    L,M = input.shape
    output = gpuarray.empty((L, M-1), dtype=np.float64)
    kernels.diff_iny_2D(input, output, np.int32(L), np.int32(M),
                        grid=_grid(L*(M-1)), block=_block())
    return output

def drop_last_col_2D(input):
    """
    Create input[:,:-1] on GPU
    """
    L,M = input.shape
    output = gpuarray.empty((L,M-1), dtype=np.float64)
    kernels.drop_last_col_2D(input, output, np.int32(L), np.int32(M),
                             grid=_grid(L*M), block=_block())
    return output

def _from_phi_2D_cuda(nx, ny, xx, yy, phi):
    L,M = phi.shape

    xx = np.minimum(np.maximum(xx, 0), 1.0)
    yy = np.minimum(np.maximum(yy, 0), 1.0)

    key = nx, tuple(xx)
    if key not in _dbeta_cache:
        dbeta1_xx = np.empty((nx+1,L-1))
        dbeta2_xx = np.empty((nx+1,L-1))
        for ii in range(0, nx+1):
            b = betainc(ii+1,nx-ii+1,xx)
            dbeta1_xx[ii] = b[1:]-b[:-1]
            b = betainc(ii+2,nx-ii+1,xx)
            dbeta2_xx[ii] = b[1:]-b[:-1]
        _dbeta_cache[key] = dbeta1_xx, dbeta2_xx
    dbeta1_xx, dbeta2_xx = _dbeta_cache[key]
    dbeta1_xx_gpu = gpuarray.to_gpu(dbeta1_xx)
    dbeta2_xx_gpu = gpuarray.to_gpu(dbeta2_xx)

    key = ny, tuple(yy)
    if key not in _dbeta_cache:
        dbeta1_yy = np.empty((ny+1,M-1))
        dbeta2_yy = np.empty((ny+1,M-1))
        for ii in range(0, ny+1):
            b = betainc(ii+1,ny-ii+1,yy)
            dbeta1_yy[ii] = b[1:]-b[:-1]
            b = betainc(ii+2,ny-ii+1,yy)
            dbeta2_yy[ii] = b[1:]-b[:-1]
        _dbeta_cache[key] = dbeta1_yy, dbeta2_yy
    dbeta1_yy, dbeta2_yy = _dbeta_cache[key]
    dbeta1_yy_gpu = gpuarray.to_gpu(dbeta1_yy)
    dbeta2_yy_gpu = gpuarray.to_gpu(dbeta2_yy)
    
    phi_gpu = gpuarray.to_gpu(phi)
    xx_gpu, yy_gpu = gpuarray.to_gpu(xx), gpuarray.to_gpu(yy)
    dx_gpu = misc.subtract(xx_gpu[1:], xx_gpu[:-1])
    dy_gpu = misc.subtract(yy_gpu[1:], yy_gpu[:-1])

    phi_dy_gpu = diff_iny_2D(phi_gpu)

    s_yy_gpu = misc.divide(phi_dy_gpu, dy_gpu)
    c1_yy_gpu = drop_last_col_2D(phi_gpu) - misc.multiply(s_yy_gpu, yy_gpu[:-1])
    c1_yy_gpu /= ny+1

    term1_yy_gpu = linalg.dot(dbeta1_yy_gpu, c1_yy_gpu, transb='T')

    term2_yy_gpu = linalg.dot(dbeta2_yy_gpu, s_yy_gpu, transb='T')
    _ = gpuarray.to_gpu(np.arange(1, ny+2).reshape(ny+1,1)/((ny+1)*(ny+2)))
    term2_yy_gpu = misc.multiply(term2_yy_gpu, _)

    over_y_all_gpu = term1_yy_gpu + term2_yy_gpu

    _ = diff_iny_2D(over_y_all_gpu)
    s_xx_all_gpu = misc.divide(_, dx_gpu)

    _2 = misc.multiply(s_xx_all_gpu, xx_gpu[:-1])
    c1_xx_all_gpu = (drop_last_col_2D(over_y_all_gpu) - _2)/(nx+1.)

    term1_all_gpu = linalg.dot(dbeta1_xx_gpu, c1_xx_all_gpu, transb='T')
    term2_all_gpu = linalg.dot(dbeta2_xx_gpu, s_xx_all_gpu, transb='T')

    _ = gpuarray.to_gpu(np.arange(1, nx+2).reshape(nx+1,1)/((nx+1)*(nx+2)))
    _ = misc.multiply(term2_all_gpu, _)
    data = term1_all_gpu + _

    return data.get()
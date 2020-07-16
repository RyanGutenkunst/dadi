import numpy
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

from dadi.Spectrum_mod import cached_dbeta
#def _from_phi_2D_cuda(nx, ny, xx, yy, phi):
#    L,M = phi.shape
#
#    # Test for xx == yy, since we can make that assumption in the code
#
#    # Can I just leave these on the GPU, to save transfer costs?
#    dbeta1_xx, dbeta2_xx = cached_dbeta(nx, xx)
#    dbeta1_yy, dbeta2_yy = cached_dbeta(ny, xx)
#
#    # Can I just leave these on the GPU in between evaluations?
#    dbeta1_xx_gpu = gpuarray.to_gpu(dbeta1_xx)
#    dbeta2_xx_gpu = gpuarray.to_gpu(dbeta2_xx)
#    dbeta1_yy_gpu = gpuarray.to_gpu(dbeta1_yy)
#    dbeta2_yy_gpu = gpuarray.to_gpu(dbeta2_yy)
#    
#    phi_gpu = gpuarray.to_gpu(phi)
#
#    xx_gpu = yy_gpu = gpuarray.to_gpu(xx)
#    dy_gpu = dx_gpu = misc.subtract(xx_gpu[1:], xx_gpu[:-1])
#
#    phi_dy_gpu = diff_iny_2D(phi_gpu)
#    s_yy_gpu = misc.divide(phi_dy_gpu, dy_gpu)
#
#    c1_yy_gpu = drop_last_col_2D(phi_gpu) - misc.multiply(s_yy_gpu, yy_gpu[:-1])
#    c1_yy_gpu /= ny+1
#
#    term1_yy_gpu = linalg.dot(dbeta1_yy_gpu, c1_yy_gpu, transb='T')
#
#    term2_yy_gpu = linalg.dot(dbeta2_yy_gpu, s_yy_gpu, transb='T')
#    _ = gpuarray.to_gpu(np.arange(1, ny+2).reshape(ny+1,1)/((ny+1)*(ny+2)))
#    term2_yy_gpu = misc.multiply(term2_yy_gpu, _)
#
#    over_y_all_gpu = term1_yy_gpu + term2_yy_gpu
#
#    _ = diff_iny_2D(over_y_all_gpu)
#    s_xx_all_gpu = misc.divide(_, dx_gpu)
#
#    _2 = misc.multiply(s_xx_all_gpu, xx_gpu[:-1])
#    c1_xx_all_gpu = (drop_last_col_2D(over_y_all_gpu) - _2)/(nx+1.)
#
#    term1_all_gpu = linalg.dot(dbeta1_xx_gpu, c1_xx_all_gpu, transb='T')
#    term2_all_gpu = linalg.dot(dbeta2_xx_gpu, s_xx_all_gpu, transb='T')
#
#    _ = gpuarray.to_gpu(np.arange(1, nx+2).reshape(nx+1,1)/((nx+1)*(nx+2)))
#    _ = misc.multiply(term2_all_gpu, _)
#    data = term1_all_gpu + _
#
#    return data.get()

_cache = {'key':None}

def _from_phi_2D_cuda(nx, ny, xx, yy, phi):
    pts,nx,ny = np.int32(phi.shape[0]), np.int32(nx), np.int32(ny)
    if (nx,ny) != _cache['key'] or not np.all(_cache['xx'] == xx):
        dbeta1_xx, dbeta2_xx = cached_dbeta(nx, xx)
        dbeta1_yy, dbeta2_yy = cached_dbeta(ny, xx)
        # Can I leave these on the GPU in between evaluations?
        # Typical use case is many evaluations with constant xx, nx, ny
        # That will work even at higher dimensionality
        dbeta1_xx_gpu = gpuarray.to_gpu(dbeta1_xx)
        dbeta2_xx_gpu = gpuarray.to_gpu(dbeta2_xx)
        dbeta1_yy_gpu = gpuarray.to_gpu(dbeta1_yy)
        dbeta2_yy_gpu = gpuarray.to_gpu(dbeta2_yy)
        xx_gpu = yy_gpu = gpuarray.to_gpu(xx)
        s_yy_gpu = gpuarray.empty((pts, pts-1), dtype=np.float64)
        c1_yy_gpu = gpuarray.empty((pts, pts-1), dtype=np.float64)
        s_xx_all_gpu = gpuarray.empty((ny+1,pts-1), dtype=np.float64)
        c1_xx_all_gpu = gpuarray.empty((ny+1,pts-1), dtype=np.float64)
        _cache['mem'] = dbeta1_xx_gpu, dbeta2_xx_gpu, dbeta1_yy_gpu, dbeta2_yy_gpu, xx_gpu, xx_gpu, s_yy_gpu, c1_yy_gpu, s_xx_all_gpu, c1_xx_all_gpu 
        _cache['key'] = (nx,ny)
        _cache['xx'] = xx
    dbeta1_xx_gpu, dbeta2_xx_gpu, dbeta1_yy_gpu, dbeta2_yy_gpu, xx_gpu, yy_gpu, s_yy_gpu, c1_yy_gpu, s_xx_all_gpu, c1_xx_all_gpu = _cache['mem']

    phi_gpu = gpuarray.to_gpu(phi)

    kernels.calc_s(phi_gpu, yy_gpu, pts, pts, s_yy_gpu,
                    grid=_grid(pts*(pts-1)), block=_block())
    kernels.calc_c1(phi_gpu, yy_gpu, s_yy_gpu, ny, pts, pts, c1_yy_gpu,
                    grid=_grid(pts*(pts-1)), block=_block())

    term1_yy_gpu = linalg.dot(dbeta1_yy_gpu, c1_yy_gpu, transb='T')
    term2_yy_gpu = linalg.dot(dbeta2_yy_gpu, s_yy_gpu, transb='T')

    # To save a memory allocation, modify term1_yy_gpu in place to hold over_y_all_gpu
    kernels.combine_terms(term1_yy_gpu, term2_yy_gpu, ny, np.int32(ny+1), pts, term1_yy_gpu,
                    grid=_grid((ny+1)*pts), block=_block())
    over_y_all_gpu = term1_yy_gpu

    kernels.calc_s(over_y_all_gpu, xx_gpu, np.int32(ny+1), pts, s_xx_all_gpu,
                    grid=_grid((pts-1)*(ny+1)), block=_block())
    kernels.calc_c1(over_y_all_gpu, xx_gpu, s_xx_all_gpu, nx, np.int32(ny+1), pts, c1_xx_all_gpu,
                    grid=_grid((pts-1)*(ny+1)), block=_block())

    term1_all_gpu = linalg.dot(dbeta1_xx_gpu, c1_xx_all_gpu, transb='T')
    term2_all_gpu = linalg.dot(dbeta2_xx_gpu, s_xx_all_gpu, transb='T')

    # To save a memory allocation, modify term1_all_gpu in place to hold data
    kernels.combine_terms(term1_all_gpu, term2_all_gpu, nx, np.int32(nx+1), np.int32(ny+1), term1_all_gpu,
                    grid=_grid((nx+1)*(ny+1)), block=_block())
    gpu = term1_all_gpu.get()
    return gpu

def _from_phi_3D_cuda(nx, ny, nz, xx, yy, zz, phi, mask_corners=True, raw=False):
    data = numpy.zeros((nx+1,ny+1,nz+1))

    dbeta1_zz, dbeta2_zz = cached_dbeta(nz, zz)

    # Quick testing suggests that doing the x direction first for better
    # memory alignment isn't worth much.
    s_zz = (phi[:,:,1:]-phi[:,:,:-1])/(zz[nuax,nuax,1:]-zz[nuax,nuax,:-1])
    c1_zz = (phi[:,:,:-1] - s_zz*zz[nuax,nuax,:-1])/(nz+1)
    # These calculations can be done without this for loop, but the
    # four-dimensional intermediate results consume massive amounts of RAM,
    # which makes the for loop faster for large systems.
    for kk in range(0, nz+1):
        # In testing, these two np.dot lines occupy 2/3 the time, so further
        # speedup will be difficult
        term1 = np.dot(c1_zz, dbeta1_zz[kk])
        term2 = np.dot(s_zz, dbeta2_zz[kk])
        term2 *= (kk+1)/((nz+1)*(nz+2))
        over_z = term1 + term2

        sub_fs = _from_phi_2D_cuda(nx, ny, xx, yy, over_z)
        data[:,:,kk] = sub_fs

    if raw:
        return data
    else:
        return dadi.Spectrum(data, mask_corners=mask_corners)


import pycuda
import dadi
import numpy as np
from numpy import newaxis as nuax
import time
if __name__ == "__main__":
    print('2D test')
    pts = 400
    nx,ny = 20,20
    phi = np.random.uniform(size=(pts,pts))
    xx = np.linspace(0,1,pts)

    start = time.time()
    cpu = dadi.Spectrum._from_phi_2D_linalg(nx,ny,xx,xx,phi, raw=True)
    print("CPU: {0:.3f}".format(time.time()-start))
    start = time.time()
    cpu = dadi.Spectrum._from_phi_2D_linalg(nx,ny,xx,xx,phi, raw=True)
    print("CPU: {0:.3f}".format(time.time()-start))
    start = time.time()
    gpu = _from_phi_2D_cuda(nx,ny,xx,xx,phi)
    print("GPU: {0:.3f}".format(time.time()-start))
    start = time.time()
    gpu = _from_phi_2D_cuda(nx,ny,xx,xx,phi)
    print("GPU: {0:.3f}".format(time.time()-start))

    print('3D test')

    pts = 400
    nx,ny,nz = 20,20,20
    phi = np.random.uniform(size=(pts,pts,pts))
    xx = np.linspace(0,1,pts)

    start = time.time()
    cpu = dadi.Spectrum._from_phi_3D_linalg(nx,ny,nz,xx,xx,xx,phi, raw=True)
    print("CPU: {0:.3f}".format(time.time()-start))
    start = time.time()
    cpu = dadi.Spectrum._from_phi_3D_linalg(nx,ny,nz,xx,xx,xx,phi, raw=True)
    print("CPU: {0:.3f}".format(time.time()-start))
    start = time.time()
    gpu = _from_phi_3D_cuda(nx,ny,nz,xx,xx,xx,phi)
    print("GPU: {0:.3f}".format(time.time()-start))
    start = time.time()
    gpu = _from_phi_3D_cuda(nx,ny,nz,xx,xx,xx,phi)
    print("GPU: {0:.3f}".format(time.time()-start))
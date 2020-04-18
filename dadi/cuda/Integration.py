import numpy as np

def _inject_mutations_2D_valcalc(dt, xx, yy, theta0, frozen1, frozen2,
                                 nomut1, nomut2):
    """
    Inject novel mutations for a timestep.
    """
    val10, val01 = 0, 0
    # Population 1
    if not frozen1 and not nomut1:
        val10 = dt/xx[1] * theta0/2 * 4/((xx[2] - xx[0]) * yy[1])
    # Population 2
    if not frozen2 and not nomut2:
        val01= dt/yy[1] * theta0/2 * 4/((yy[2] - yy[0]) * xx[1])
    return np.float64(val10),np.float64(val01)

import pycuda
import pycuda.gpuarray as gpuarray
from skcuda.cusparse import cusparseDgtsvInterleavedBatch_bufferSizeExt,\
    cusparseDgtsv2StridedBatch_bufferSizeExt, cusparseDgtsvInterleavedBatch, cusparseDgtsv2StridedBatch

import dadi.cuda
from dadi.cuda import cusparse_handle
from . import kernels

def _two_pops_const_params(phi, xx, yy,
                theta0, frozen1, frozen2, nomut1, nomut2,
                ax, bx, cx, ay, by, cy, 
                current_t, dt, T):
    ax_gpu = gpuarray.to_gpu(ax)
    bx_gpu = gpuarray.to_gpu(bx)
    cx_gpu = gpuarray.to_gpu(cx)
    ay_gpu = gpuarray.to_gpu(ay.transpose().copy())
    by_gpu = gpuarray.to_gpu(by.transpose().copy())
    cy_gpu = gpuarray.to_gpu(cy.transpose().copy())

    # du is modified when InterleavedBatch runs.
    # Save a copy on the GPU to refill after each call.
    cx_saved_gpu = gpuarray.to_gpu(cx)
    cy_saved_gpu = gpuarray.to_gpu(cy.transpose().copy())

    phi_gpu = gpuarray.to_gpu(phi)
    phiT_gpu = gpuarray.empty(phi.shape[::-1], phi.dtype)

    # Calculate necessary buffer size
    bsize_int = cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparse_handle, 0, len(xx), ax_gpu.gpudata, bx_gpu.gpudata,
        cx_gpu.gpudata, phi_gpu.gpudata, len(xx))
    bsize_str = cusparseDgtsv2StridedBatch_bufferSizeExt(
        cusparse_handle, len(yy), ay_gpu.gpudata, by_gpu.gpudata,
        cy_gpu.gpudata, phi_gpu.gpudata, len(yy), len(yy))
    pBuffer = pycuda.driver.mem_alloc(max(bsize_int,bsize_str))

    last_dt = np.inf
    while current_t < T:
        this_dt = min(dt, T - current_t)

        val10, val01 = _inject_mutations_2D_valcalc(this_dt, xx, yy, theta0, frozen1, frozen2,
                                                    nomut1, nomut2)
        # Use a single thread to update the two necessary values in phi
        kernels._inject_mutations_2D_vals(phi_gpu, np.int32(len(xx)),
                                  val01, val10, block=(1,1,1))

        if not frozen1:
            # Restore from saved version of dux
            pycuda.driver.memcpy_dtod(cx_gpu.gpudata, cx_saved_gpu.gpudata,
                                      cx_saved_gpu.nbytes)

            # Prepare linear system
            bx_gpu += (1./this_dt - 1./last_dt)
            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, len(xx),
                ax_gpu.gpudata, bx_gpu.gpudata, cx_gpu.gpudata, phi_gpu.gpudata,
                len(xx), pBuffer)

        if not frozen2:
            # Restore from saved version of dux
            pycuda.driver.memcpy_dtod(cy_gpu.gpudata, cy_saved_gpu.gpudata,
                                      cy_saved_gpu.nbytes)
            by_gpu += (1./this_dt - 1./last_dt)
            
            dadi.cuda.transpose_gpuarray(phi_gpu, phiT_gpu)
            phiT_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, len(yy),
                ay_gpu.gpudata, by_gpu.gpudata, cy_gpu.gpudata, phiT_gpu.gpudata,
                len(yy), pBuffer)
            dadi.cuda.transpose_gpuarray(phiT_gpu, phi_gpu)

        last_dt = this_dt
        current_t += this_dt

    phi = phi_gpu.get().reshape((len(xx),len(yy)))
    return phi
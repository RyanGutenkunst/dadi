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
        val01 = dt/yy[1] * theta0/2 * 4/((yy[2] - yy[0]) * xx[1])
    return np.float64(val10),np.float64(val01)

def _inject_mutations_3D_valcalc(dt, xx, yy, zz, theta0, frozen1, frozen2, frozen3):
    """
    Inject novel mutations for a timestep.
    """
    # Population 1
    # Normalization based on the multi-dimensional trapezoid rule is 
    # implemented                      ************** here ***************
    val100, val010, val001 = 0, 0, 0
    if not frozen1:
        val100 = dt/xx[1] * theta0/2 * 8/((xx[2] - xx[0]) * yy[1] * zz[1])
    # Population 2
    if not frozen2:
        val010 = dt/yy[1] * theta0/2 * 8/((yy[2] - yy[0]) * xx[1] * zz[1])
    # Population 3
    if not frozen3:
        val001 = dt/zz[1] * theta0/2 * 8/((zz[2] - zz[0]) * xx[1] * yy[1])
    return np.float64(val100), np.float64(val010), np.float64(val001)

import pycuda
import pycuda.gpuarray as gpuarray
from skcuda.cusparse import cusparseDgtsvInterleavedBatch_bufferSizeExt, cusparseDgtsvInterleavedBatch

import dadi.cuda
from dadi.cuda import cusparse_handle, _grid, _block, transpose_gpuarray
from . import kernels

def _two_pops_const_params(phi, xx, 
                theta0, frozen1, frozen2, nomut1, nomut2,
                ax, bx, cx, ay, by, cy, 
                current_t, dt, T):
    L = np.int32(len(xx))

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
        cusparse_handle, 0, L, ax_gpu.gpudata, bx_gpu.gpudata,
        cx_gpu.gpudata, phi_gpu.gpudata, L)
    pBuffer = pycuda.driver.mem_alloc(bsize_int)

    last_dt = np.inf
    while current_t < T:
        this_dt = min(dt, T - current_t)

        val10, val01 = _inject_mutations_2D_valcalc(this_dt, xx, xx, theta0, frozen1, frozen2,
                                                    nomut1, nomut2)
        # Use a single thread to update the two necessary values in phi
        kernels._inject_mutations_2D_vals(phi_gpu, L,
                                  val01, val10, block=(1,1,1))

        if not frozen1:
            # Restore from saved version of dux
            pycuda.driver.memcpy_dtod(cx_gpu.gpudata, cx_saved_gpu.gpudata,
                                      cx_saved_gpu.nbytes)

            # Prepare linear system
            bx_gpu += (1./this_dt - 1./last_dt)
            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ax_gpu.gpudata, bx_gpu.gpudata, cx_gpu.gpudata, phi_gpu.gpudata,
                L, pBuffer)

        if not frozen2:
            # Restore from saved version of dux
            pycuda.driver.memcpy_dtod(cy_gpu.gpudata, cy_saved_gpu.gpudata,
                                      cy_saved_gpu.nbytes)
            by_gpu += (1./this_dt - 1./last_dt)
            
            transpose_gpuarray(phi_gpu, phiT_gpu)
            phiT_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ay_gpu.gpudata, by_gpu.gpudata, cy_gpu.gpudata, phiT_gpu.gpudata,
                L, pBuffer)
            transpose_gpuarray(phiT_gpu, phi_gpu)

        last_dt = this_dt
        current_t += this_dt

    phi = phi_gpu.get().reshape(L,L)
    return phi

def _two_pops_temporal_params(phi, xx, T, initial_t, nu1_f, nu2_f, m12_f, m21_f, gamma1_f,
            gamma2_f, h1_f, h2_f, theta0_f, frozen1, frozen2, nomut1, nomut2):
    current_t = initial_t
    nu1,nu2 = nu1_f(current_t), nu2_f(current_t)
    m12,m21 = m12_f(current_t), m21_f(current_t)
    gamma1,gamma2 = gamma1_f(current_t), gamma2_f(current_t)
    h1,h2 = h1_f(current_t), h2_f(current_t)

    if dadi.Integration.use_delj_trick:
        raise ValueError("delj trick not currently supported in CUDA execution")

    yy = xx
    dx = dy = np.diff(xx)
    dfactor = dadi.Integration._compute_dfactor(dx)
    xInt = (xx[:-1] + xx[1:])*0.5

    L = M = np.int32(len(xx))

    xx_gpu = yy_gpu = gpuarray.to_gpu(xx)
    dx_gpu = gpuarray.to_gpu(dx)
    dfactor_gpu = gpuarray.to_gpu(dfactor)
    xInt_gpu = gpuarray.to_gpu(xInt)

    phi_gpu = gpuarray.to_gpu(phi)
    phiT_gpu = gpuarray.empty(phi.shape[::-1], phi.dtype)

    Vx_gpu = gpuarray.empty(L, np.float64)
    VIntx_gpu = gpuarray.empty(L-1, np.float64)
    MIntx_gpu = gpuarray.empty((L-1,M), np.float64)

    ax_gpu = gpuarray.zeros((L,M), np.float64)
    bx_gpu = gpuarray.empty((L,M), np.float64)
    cx_gpu = gpuarray.zeros((L,M), np.float64)

    bsize_int = cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparse_handle, 0, L, ax_gpu.gpudata, bx_gpu.gpudata,
        cx_gpu.gpudata, phi_gpu.gpudata, L)
    pBuffer = pycuda.driver.mem_alloc(bsize_int)

    while current_t < T:
        dt = min(dadi.Integration._compute_dt(dx,nu1,[m12],gamma1,h1),
                 dadi.Integration._compute_dt(dy,nu2,[m21],gamma2,h2))
        this_dt = np.float64(min(dt, T - current_t))

        next_t = current_t + this_dt

        nu1,nu2 = nu1_f(next_t), nu2_f(next_t)
        m12,m21 = m12_f(next_t), m21_f(next_t)
        gamma1,gamma2 = gamma1_f(next_t), gamma2_f(next_t)
        h1,h2 = h1_f(next_t), h2_f(next_t)
        theta0 = theta0_f(next_t)

        val10, val01 = dadi.cuda.Integration._inject_mutations_2D_valcalc(this_dt, xx, yy, theta0, frozen1, frozen2,
                                                                          nomut1, nomut2)
        kernels._inject_mutations_2D_vals(phi_gpu, L, np.float64(val01), np.float64(val10), 
                                          block=(1, 1, 1))

        if not frozen1:
            kernels._Vfunc(xx_gpu, nu1, L, Vx_gpu, 
                           grid=_grid(L), block=_block())
            kernels._Vfunc(xInt_gpu, nu1, L-1, VIntx_gpu, 
                           grid=_grid(L-1), block=_block())
            kernels._Mfunc2D(xInt_gpu, yy_gpu, m12, gamma1, h1,
                             L-1, M, MIntx_gpu,
                             grid=_grid((L-1)*M), block=_block())

            kernels._cx0(cx_gpu, L, M, grid=_grid(M), block=_block())
            bx_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MIntx_gpu, Vx_gpu, this_dt, L, M,
                ax_gpu, bx_gpu,
                grid=_grid((L-1)*M), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MIntx_gpu, Vx_gpu, this_dt, L, M,
                bx_gpu, cx_gpu,
                grid=_grid((L-1)*M), block=_block())
            kernels._include_bc(dx_gpu, nu1, m12, gamma1, h1, L, M,
                bx_gpu, block=(1,1,1))

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ax_gpu.gpudata, bx_gpu.gpudata, cx_gpu.gpudata, phi_gpu.gpudata,
                L, pBuffer)
        if not frozen2:
            kernels._Vfunc(xx_gpu, nu2, L, Vx_gpu, 
                           grid=_grid(L), block=_block())
            kernels._Vfunc(xInt_gpu, nu2, L-1, VIntx_gpu, 
                           grid=_grid(L-1), block=_block())
            kernels._Mfunc2D(xInt_gpu, yy_gpu, m21, gamma2, h2,
                             L-1, M, MIntx_gpu,
                             grid=_grid((L-1)*M), block=_block())
            kernels._cx0(cx_gpu, L, M, grid=_grid(M), block=_block())
            bx_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MIntx_gpu, Vx_gpu, this_dt, L, M,
                ax_gpu, bx_gpu,
                grid=_grid((L-1)*M), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MIntx_gpu, Vx_gpu, this_dt, L, M,
                bx_gpu, cx_gpu,
                grid=_grid((L-1)*M), block=_block())
            kernels._include_bc(dx_gpu, nu2, m21, gamma2, h2, L, M,
                bx_gpu, block=(1,1,1))

            transpose_gpuarray(phi_gpu, phiT_gpu)
            phiT_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ax_gpu.gpudata, bx_gpu.gpudata, cx_gpu.gpudata, phiT_gpu.gpudata,
                L, pBuffer)
            transpose_gpuarray(phiT_gpu, phi_gpu)

        current_t = next_t

    return phi_gpu.get()

def _three_pops_const_params(phi, xx,
                theta0, frozen1, frozen2, frozen3, 
                ax, bx, cx, ay, by, cy, az, bz, cz,
                current_t, dt, T):

    L = np.int32(len(xx))

    ax_gpu = gpuarray.to_gpu(ax.reshape(L,L**2))
    bx_gpu = gpuarray.to_gpu(bx.reshape(L,L**2))
    cx_saved_gpu = gpuarray.to_gpu(cx.reshape(L,L**2))
    c_gpu = gpuarray.to_gpu(cx_saved_gpu)

    ay_gpu = gpuarray.to_gpu(ay.reshape(L,L**2).transpose().copy())
    by_gpu = gpuarray.to_gpu(by.reshape(L,L**2).transpose().copy())
    cy_saved_gpu = gpuarray.to_gpu(cy.reshape(L,L**2).transpose().copy())

    az_gpu = gpuarray.to_gpu(az.reshape(L,L**2).transpose().reshape(L,L**2).transpose().copy())
    bz_gpu = gpuarray.to_gpu(bz.reshape(L,L**2).transpose().reshape(L,L**2).transpose().copy())
    cz_saved_gpu = gpuarray.to_gpu(cz.reshape(L,L**2).transpose().reshape(L,L**2).transpose().copy())

    phi_gpu = gpuarray.to_gpu(phi.reshape(L,L**2))
    phi_buf_gpu = gpuarray.to_gpu(phi.reshape(L**2,L))

    bsize_int = cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparse_handle, 0, L, ax_gpu.gpudata, bx_gpu.gpudata,
        cx_gpu.gpudata, phi_gpu.gpudata, L*L)
    pBuffer = pycuda.driver.mem_alloc(bsize_int)

    last_dt = np.inf
    while current_t < T:    
        this_dt = min(dt, T - current_t)

        val100, val010, val001 = \
            _inject_mutations_3D_valcalc(this_dt, xx, xx, xx, theta0, 
                                         frozen1, frozen2, frozen3)
        kernels._inject_mutations_3D_vals(phi_gpu, L,
                                          val001, val010, val100, block=(1,1,1))
        if not frozen1:
            pycuda.driver.memcpy_dtod(c_gpu.gpudata, cx_saved_gpu.gpudata,
                                      cx_saved_gpu.nbytes)

            bx_gpu += (1./this_dt - 1./last_dt)
            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ax_gpu.gpudata, bx_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L**2, pBuffer)

        transpose_gpuarray(phi_gpu, phi_buf_gpu)
        phi_gpu, phi_buf_gpu = phi_buf_gpu.reshape(L,L**2), phi_gpu.reshape(L**2,L)
        if not frozen2:
            pycuda.driver.memcpy_dtod(c_gpu.gpudata, cy_saved_gpu.gpudata,
                                      cy_saved_gpu.nbytes)
            by_gpu += (1./this_dt - 1./last_dt)
            phi_gpu /= this_dt
            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ay_gpu.gpudata, by_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L**2, pBuffer)

        transpose_gpuarray(phi_gpu, phi_buf_gpu)
        phi_gpu, phi_buf_gpu = phi_buf_gpu.reshape(L,L**2), phi_gpu.reshape(L**2,L)
        if not frozen3:
            pycuda.driver.memcpy_dtod(c_gpu.gpudata, cz_saved_gpu.gpudata,
                                      cz_saved_gpu.nbytes)
            bz_gpu += (1./this_dt - 1./last_dt)
            phi_gpu /= this_dt
            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                az_gpu.gpudata, bz_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L**2, pBuffer)

        transpose_gpuarray(phi_gpu, phi_buf_gpu)
        phi_gpu, phi_buf_gpu = phi_buf_gpu.reshape(L,L**2), phi_gpu.reshape(L**2,L)

        last_dt = this_dt
        current_t += this_dt

    return phi_gpu.get().reshape(L,L,L)
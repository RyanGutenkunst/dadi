import numpy as np

def _inject_mutations_2D_valcalc(dt, xx, yy, theta0, frozen1, frozen2,
                                 nomut1, nomut2):
    """
    Calculate mutations that need to be injected.
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
    Calculate mutations that need to be injected.
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

def _inject_mutations_4D_valcalc(dt, xx, yy, zz, aa, theta0, frozen1, frozen2, frozen3, frozen4):
    """
    Calculate mutations that need to be injected.
    """
    # Population 1
    # Normalization based on the multi-dimensional trapezoid rule is 
    # implemented                      ************** here ***************
    val1000, val0100, val0010, val0001 = 0, 0, 0, 0
    if not frozen1:
        val1000 = dt/xx[1] * theta0/2 * 16/((xx[2] - xx[0]) * yy[1] * zz[1] * aa[1])
    # Population 2
    if not frozen2:
        val0100 = dt/yy[1] * theta0/2 * 16/((yy[2] - yy[0]) * xx[1] * zz[1] * aa[1])
    # Population 3
    if not frozen3:
        val0010 = dt/zz[1] * theta0/2 * 16/((zz[2] - zz[0]) * xx[1] * yy[1] * aa[1])
    # Population 4
    if not frozen4:
        val0001 = dt/aa[1] * theta0/2 * 16/((aa[2] - aa[0]) * xx[1] * yy[1] * zz[1])
    return np.float64(val1000), np.float64(val0100), np.float64(val0010), np.float64(val0001)

def _inject_mutations_5D_valcalc(dt, xx, yy, zz, aa, bb, theta0, frozen1, frozen2, frozen3, frozen4, frozen5):
    """
    Calculate mutations that need to be injected.
    """
    # Population 1
    # Normalization based on the multi-dimensional trapezoid rule is 
    # implemented                      ************** here ***************
    val10000, val01000, val00100, val00010, val00001 = 0, 0, 0, 0, 0
    if not frozen1:
        val10000 = dt/xx[1] * theta0/2 * 32/((xx[2] - xx[0]) * yy[1] * zz[1] * aa[1] * bb[1])
    # Population 2
    if not frozen2:
        val01000 = dt/yy[1] * theta0/2 * 32/((yy[2] - yy[0]) * xx[1] * zz[1] * aa[1] * bb[1])
    # Population 3
    if not frozen3:
        val00100 = dt/zz[1] * theta0/2 * 32/((zz[2] - zz[0]) * xx[1] * yy[1] * aa[1] * bb[1])
    # Population 4
    if not frozen4:
        val00010 = dt/aa[1] * theta0/2 * 32/((aa[2] - aa[0]) * xx[1] * yy[1] * zz[1] * bb[1])
    if not frozen5:
        val00001 = dt/bb[1] * theta0/2 * 32/((aa[2] - aa[0]) * xx[1] * yy[1] * zz[1] * aa[1])
    return np.float64(val10000), np.float64(val01000), np.float64(val00100), np.float64(val00010), np.float64(val00001)

import pycuda
import pycuda.gpuarray as gpuarray

import dadi.cuda
from dadi.cuda import cusparse_handle, _grid, _block, transpose_gpuarray
from dadi.cuda.cusparse import cusparseDgtsvInterleavedBatch_bufferSizeExt, cusparseDgtsvInterleavedBatch
from . import kernels

def _two_pops_const_params(phi, xx, 
                theta0, frozen1, frozen2, nomut1, nomut2,
                ax, bx, cx, ay, by, cy, 
                current_t, dt, T):
    L = np.int32(len(xx))

    ax_gpu = gpuarray.to_gpu(ax)
    bx_gpu = gpuarray.to_gpu(bx)
    ay_gpu = gpuarray.to_gpu(ay.transpose().copy())
    by_gpu = gpuarray.to_gpu(by.transpose().copy())

    # du is modified when InterleavedBatch runs.
    # Save a copy on the GPU to refill after each call.
    cx_saved_gpu = gpuarray.to_gpu(cx)
    cy_saved_gpu = gpuarray.to_gpu(cy.transpose().copy())

    phi_gpu = gpuarray.to_gpu(phi)

    # Use this memory buffer for temporary c arrays and transposition
    buff_gpu = gpuarray.empty_like(cx_saved_gpu)

    # Calculate necessary buffer size
    bsize_int = cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparse_handle, 0, L, ax_gpu.gpudata, bx_gpu.gpudata,
        buff_gpu.gpudata, phi_gpu.gpudata, L)
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
            pycuda.driver.memcpy_dtod(buff_gpu.gpudata, cx_saved_gpu.gpudata,
                                      cx_saved_gpu.nbytes)

            # Prepare linear system
            bx_gpu += (1./this_dt - 1./last_dt)
            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ax_gpu.gpudata, bx_gpu.gpudata, buff_gpu.gpudata, phi_gpu.gpudata,
                L, pBuffer)

        if not frozen2:
            # We tranpose here and use InterleavedBatch (rather than StridedBatch)
            # because it is notably faster.
            transpose_gpuarray(phi_gpu, buff_gpu)
            # Switch which memory location is treated as buffer and phiT
            phiT_gpu, buff_gpu = buff_gpu, phi_gpu
            phiT_gpu /= this_dt

            pycuda.driver.memcpy_dtod(buff_gpu.gpudata, cy_saved_gpu.gpudata,
                                      cy_saved_gpu.nbytes)
            by_gpu += (1./this_dt - 1./last_dt)

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ay_gpu.gpudata, by_gpu.gpudata, buff_gpu.gpudata, phiT_gpu.gpudata,
                L, pBuffer)
            transpose_gpuarray(phiT_gpu, buff_gpu)
            phi_gpu, buff_gpu = buff_gpu, phiT_gpu

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

    # By transposing phi, we can use the same functions to generate
    # the a,b,c matrices for both x and y.
    phi_gpu = gpuarray.to_gpu(phi)

    Vx_gpu = gpuarray.empty(L, np.float64)
    VIntx_gpu = gpuarray.empty(L-1, np.float64)

    ax_gpu = gpuarray.empty((L,M), np.float64)
    bx_gpu = gpuarray.empty((L,M), np.float64)
    # Buffer for MInt arrays, c arrays, and transposition
    buff_gpu = gpuarray.empty((L,M), np.float64)

    bsize_int = cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparse_handle, 0, L, ax_gpu.gpudata, bx_gpu.gpudata,
        buff_gpu.gpudata, phi_gpu.gpudata, L)
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
            # Note, if L is an int32, L-1 can be int64. This can
            # break the code. So we wrap the -1 expressions in np.int32.
            kernels._Vfunc(xx_gpu, nu1, L, Vx_gpu, 
                           grid=_grid(L), block=_block())
            kernels._Vfunc(xInt_gpu, nu1, np.int32(L-1), VIntx_gpu, 
                           grid=_grid(L-1), block=_block())
            # Fill buff_gpu with MInt
            kernels._Mfunc2D(xInt_gpu, yy_gpu, m12, gamma1, h1,
                             np.int32(L-1), M, buff_gpu,
                             grid=_grid((L-1)*M), block=_block())

            bx_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                buff_gpu, Vx_gpu, this_dt, L, M,
                ax_gpu, bx_gpu,
                grid=_grid((L-1)*M), block=_block())
            # Note that this transforms buff_gpu from MInt to cx
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                buff_gpu, Vx_gpu, this_dt, L, M,
                bx_gpu, buff_gpu,
                grid=_grid((L-1)*M), block=_block())
            kernels._include_bc(dx_gpu, nu1, gamma1, h1, L, M,
                bx_gpu, block=(1,1,1))
            kernels._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ax_gpu.gpudata, bx_gpu.gpudata, buff_gpu.gpudata, phi_gpu.gpudata,
                L, pBuffer)
        if not frozen2:
            # Use the buffer as destination for transpose
            transpose_gpuarray(phi_gpu, buff_gpu)
            # Use previous phi memory as buffer for this step, and vice versa
            phiT_gpu, buff_gpu = buff_gpu, phi_gpu

            phiT_gpu /= this_dt

            kernels._Vfunc(xx_gpu, nu2, L, Vx_gpu, 
                           grid=_grid(L), block=_block())
            kernels._Vfunc(xInt_gpu, nu2, np.int32(L-1), VIntx_gpu, 
                           grid=_grid(L-1), block=_block())
            kernels._Mfunc2D(xInt_gpu, yy_gpu, m21, gamma2, h2,
                             np.int32(L-1), M, buff_gpu,
                             grid=_grid((L-1)*M), block=_block())

            bx_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                buff_gpu, Vx_gpu, this_dt, L, M,
                ax_gpu, bx_gpu,
                grid=_grid((L-1)*M), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                buff_gpu, Vx_gpu, this_dt, L, M,
                bx_gpu, buff_gpu,
                grid=_grid((L-1)*M), block=_block())
            kernels._include_bc(dx_gpu, nu2, gamma2, h2, L, M,
                bx_gpu, block=(1,1,1))
            kernels._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ax_gpu.gpudata, bx_gpu.gpudata, buff_gpu.gpudata, phiT_gpu.gpudata,
                L, pBuffer)
            transpose_gpuarray(phiT_gpu, buff_gpu)
            phi_gpu, buff_gpu = buff_gpu, phiT_gpu

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

    # There is probably a cleaner way to write this, but this matches
    # the transformations we'll be making to phi.
    az_gpu = gpuarray.to_gpu(az.reshape(L,L**2).transpose().reshape(L,L**2).transpose().copy())
    bz_gpu = gpuarray.to_gpu(bz.reshape(L,L**2).transpose().reshape(L,L**2).transpose().copy())
    cz_saved_gpu = gpuarray.to_gpu(cz.reshape(L,L**2).transpose().reshape(L,L**2).transpose().copy())

    # To save memory, in this 3D method we'll be using the c_gpu matrix
    # as the buffer for transposing.
    phi_gpu = gpuarray.to_gpu(phi.reshape(L,L**2))

    bsize_int = cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparse_handle, 0, L, ax_gpu.gpudata, bx_gpu.gpudata,
        cx_saved_gpu.gpudata, phi_gpu.gpudata, L*L)
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

        # We transpose into the c_gpu memory buffer.
        # Then we switch the names of the phi_gpu and c_gpu buffers.
        transpose_gpuarray(phi_gpu, c_gpu.reshape(L**2,L))
        phi_gpu, c_gpu = c_gpu.reshape(L,L**2), phi_gpu
        if not frozen2:
            pycuda.driver.memcpy_dtod(c_gpu.gpudata, cy_saved_gpu.gpudata,
                                      cy_saved_gpu.nbytes)
            by_gpu += (1./this_dt - 1./last_dt)
            phi_gpu /= this_dt
            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ay_gpu.gpudata, by_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L**2, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L**2,L))
        phi_gpu, c_gpu = c_gpu.reshape(L,L**2), phi_gpu
        if not frozen3:
            pycuda.driver.memcpy_dtod(c_gpu.gpudata, cz_saved_gpu.gpudata,
                                      cz_saved_gpu.nbytes)
            bz_gpu += (1./this_dt - 1./last_dt)
            phi_gpu /= this_dt
            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                az_gpu.gpudata, bz_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L**2, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L**2,L))
        phi_gpu, c_gpu = c_gpu.reshape(L,L**2), phi_gpu

        last_dt = this_dt
        current_t += this_dt

    return phi_gpu.get().reshape(L,L,L)

def _three_pops_temporal_params(phi, xx, T, initial_t, nu1_f, nu2_f, nu3_f, 
            m12_f, m13_f, m21_f, m23_f, m31_f, m32_f, 
            gamma1_f, gamma2_f, gamma3_f, h1_f, h2_f, h3_f,
            theta0_f, frozen1, frozen2, frozen3):
    if dadi.Integration.use_delj_trick:
        raise ValueError("delj trick not currently supported in CUDA execution")

    t = current_t = initial_t
    nu1,nu2,nu3 = nu1_f(t), nu2_f(t), nu3_f(t)
    m12,m13,m21,m23,m31,m32 = m12_f(t), m13_f(t), m21_f(t), m23_f(t), m31_f(t), m32_f(t)
    gamma1,gamma2,gamma3 = gamma1_f(t), gamma2_f(t), gamma3_f(t)
    h1,h2,h3 = h1_f(t), h2_f(t), h3_f(t)

    L = M = N = np.int32(len(xx))

    phi_gpu = gpuarray.to_gpu(phi.reshape(L,M*N))

    yy = zz = xx
    dx = dy = dz = np.diff(xx)
    dfactor = dadi.Integration._compute_dfactor(dx)
    xInt = (xx[:-1] + xx[1:])*0.5

    xx_gpu = gpuarray.to_gpu(xx)
    dx_gpu = gpuarray.to_gpu(dx)
    dfactor_gpu = gpuarray.to_gpu(dfactor)
    xInt_gpu = gpuarray.to_gpu(xInt)

    V_gpu = gpuarray.empty(L, np.float64)
    VInt_gpu = gpuarray.empty(L-1, np.float64)

    a_gpu = gpuarray.empty((L,L*L), np.float64)
    b_gpu = gpuarray.empty((L,L*L), np.float64)
    c_gpu = gpuarray.empty((L,L*L), np.float64)

    bsize_int = cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparse_handle, 0, L, a_gpu.gpudata, b_gpu.gpudata,
        c_gpu.gpudata, phi_gpu.gpudata, L**2)
    pBuffer = pycuda.driver.mem_alloc(bsize_int)

    while current_t < T:
        dt = min(dadi.Integration._compute_dt(dx, nu1, [m12, m13], gamma1, h1),
                 dadi.Integration._compute_dt(dy, nu2, [m21, m23], gamma2, h2),
                 dadi.Integration._compute_dt(dz, nu3, [m31, m32], gamma3, h3))
        this_dt = np.float64(min(dt, T - current_t))

        next_t = current_t + this_dt

        nu1,nu2,nu3 = nu1_f(next_t), nu2_f(next_t), nu3_f(next_t)
        m12,m13 = m12_f(next_t), m13_f(next_t)
        m21,m23 = m21_f(next_t), m23_f(next_t)
        m31,m32 = m31_f(next_t), m32_f(next_t)
        gamma1,gamma2 = gamma1_f(next_t), gamma2_f(next_t)
        gamma3 = gamma3_f(next_t)
        h1,h2,h3 = h1_f(next_t), h2_f(next_t), h3_f(next_t)
        theta0 = theta0_f(next_t)

        if np.any(np.less([T,nu1,nu2,nu3,m12,m13,m21,m23,m31,m32,theta0], 0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if np.any(np.equal([nu1,nu2,nu3], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        val100, val010, val001 = \
            _inject_mutations_3D_valcalc(this_dt, xx, yy, zz, theta0, 
                                         frozen1, frozen2, frozen3)
        kernels._inject_mutations_3D_vals(phi_gpu, L,
                                          val001, val010, val100, block=(1,1,1))
        # I can use the c_gpu buffer for the MInt_gpu buffer, to save GPU memory.
        # Note that I have to reassign this after each transpose operation I do.
        MInt_gpu = c_gpu
        if not frozen1:
            kernels._Vfunc(xx_gpu, nu1, L, V_gpu, 
                           grid=_grid(L), block=_block())
            kernels._Vfunc(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                           grid=_grid(L-1), block=_block())
            kernels._Mfunc3D(xInt_gpu, xx_gpu, xx_gpu, m12, m13, gamma1, h1,
                             np.int32(L-1), M, N, MInt_gpu,
                             grid=_grid((L-1)*M*N), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, L, M*N,
                a_gpu, b_gpu,
                grid=_grid((L-1)*M*N), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, L, M*N,
                b_gpu, c_gpu,
                grid=_grid((L-1)*M*N), block=_block())
            kernels._include_bc(dx_gpu, nu1, gamma1, h1, L, M*N,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, L, M*N, grid=_grid(M*N), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                M*N, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(M*N,L))
        phi_gpu, c_gpu = c_gpu.reshape(M,L*N), phi_gpu.reshape(M,L*N)
        MInt_gpu = c_gpu
        if not frozen2:
            kernels._Vfunc(xx_gpu, nu2, M, V_gpu, 
                           grid=_grid(M), block=_block())
            kernels._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                           grid=_grid(M-1), block=_block())
            # Note the order of the m23, m21 arguments here.
            kernels._Mfunc3D(xInt_gpu, xx_gpu, xx_gpu, m23, m21, gamma2, h2,
                             np.int32(M-1), N, L, MInt_gpu,
                             grid=_grid((M-1)*L*N), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, M, L*N,
                a_gpu, b_gpu,
                grid=_grid((M-1)*L*N), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, M, L*N,
                b_gpu, c_gpu,
                grid=_grid((M-1)*L*N), block=_block())
            kernels._include_bc(dx_gpu, nu2, gamma2, h2, M, L*N,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, M, L*N, grid=_grid(L*N), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, M,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*N, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*N,M))
        phi_gpu, c_gpu = c_gpu.reshape(N,L*M), phi_gpu.reshape(N,L*M)
        MInt_gpu = c_gpu
        if not frozen3:
            kernels._Vfunc(xx_gpu, nu3, N, V_gpu, 
                           grid=_grid(N), block=_block())
            kernels._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                           grid=_grid(N-1), block=_block())
            kernels._Mfunc3D(xInt_gpu, xx_gpu, xx_gpu, m31, m32, gamma3, h3,
                             np.int32(N-1), L, M, MInt_gpu,
                             grid=_grid((N-1)*M*L), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, N, L*M,
                a_gpu, b_gpu,
                grid=_grid((N-1)*L*M), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, N, L*M,
                b_gpu, c_gpu,
                grid=_grid((N-1)*L*M), block=_block())
            kernels._include_bc(dx_gpu, nu3, gamma3, h3, N, L*M,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, N, L*M, grid=_grid(L*M), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, N,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(M*N,L))
        phi_gpu, c_gpu = c_gpu.reshape(L,M*N), phi_gpu.reshape(L,M*N)

        current_t += this_dt

    return phi_gpu.get().reshape(L,M,N)

def _four_pops_temporal_params(phi, xx, T, initial_t, nu1_f, nu2_f, nu3_f, nu4_f,
            m12_f, m13_f, m14_f, m21_f, m23_f, m24_f, m31_f, m32_f, m34_f,
            m41_f, m42_f, m43_f, gamma1_f, gamma2_f, gamma3_f, gamma4_f,
            h1_f, h2_f, h3_f, h4_f, theta0_f, frozen1, frozen2, frozen3, frozen4):
    if dadi.Integration.use_delj_trick:
        raise ValueError("delj trick not currently supported in CUDA execution")

    current_t = initial_t
    nu1, nu2, nu3, nu4 = nu1_f(current_t), nu2_f(current_t), nu3_f(current_t), nu4_f(current_t)
    gamma1, gamma2, gamma3, gamma4 = gamma1_f(current_t), gamma2_f(current_t), gamma3_f(current_t), gamma4_f(current_t)
    h1, h2, h3, h4 = h1_f(current_t), h2_f(current_t), h3_f(current_t), h4_f(current_t)
    m12, m13, m14 = m12_f(current_t), m13_f(current_t), m14_f(current_t)
    m21, m23, m24 = m21_f(current_t), m23_f(current_t), m24_f(current_t)
    m31, m32, m34 = m31_f(current_t), m32_f(current_t), m34_f(current_t)
    m41, m42, m43 = m41_f(current_t), m42_f(current_t), m43_f(current_t)

    L = M = N = O = np.int32(len(xx))

    phi_gpu = gpuarray.to_gpu(phi.reshape(L,M*N*O))

    aa = yy = zz = xx
    da = dx = dy = dz = np.diff(xx)
    dfactor = dadi.Integration._compute_dfactor(dx)
    xInt = (xx[:-1] + xx[1:])*0.5

    xx_gpu = gpuarray.to_gpu(xx)
    dx_gpu = gpuarray.to_gpu(dx)
    dfactor_gpu = gpuarray.to_gpu(dfactor)
    xInt_gpu = gpuarray.to_gpu(xInt)

    V_gpu = gpuarray.empty(L, np.float64)
    VInt_gpu = gpuarray.empty(L-1, np.float64)

    a_gpu = gpuarray.empty((L,M*N*O), np.float64)
    b_gpu = gpuarray.empty((L,M*N*O), np.float64)
    c_gpu = gpuarray.empty((L,M*N*O), np.float64)

    bsize_int = cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparse_handle, 0, L, a_gpu.gpudata, b_gpu.gpudata,
        c_gpu.gpudata, phi_gpu.gpudata, M*N*O)
    pBuffer = pycuda.driver.mem_alloc(bsize_int)

    while current_t < T:
        dt = min(dadi.Integration._compute_dt(dx, nu1, [m12, m13, m14], gamma1, h1),
                 dadi.Integration._compute_dt(dy, nu2, [m21, m23, m24], gamma2, h2),
                 dadi.Integration._compute_dt(dz, nu3, [m31, m32, m34], gamma3, h3),
                 dadi.Integration._compute_dt(da, nu4, [m41, m42, m43], gamma4, h4))
        this_dt = np.float64(min(dt, T - current_t))

        next_t = current_t + this_dt

        nu1, nu2, nu3, nu4 = nu1_f(next_t), nu2_f(next_t), nu3_f(next_t), nu4_f(next_t)
        gamma1, gamma2, gamma3, gamma4 = gamma1_f(next_t), gamma2_f(next_t), gamma3_f(next_t), gamma4_f(next_t)
        h1, h2, h3, h4 = h1_f(next_t), h2_f(next_t), h3_f(next_t), h4_f(next_t)
        m12, m13, m14 = m12_f(next_t), m13_f(next_t), m14_f(next_t)
        m21, m23, m24 = m21_f(next_t), m23_f(next_t), m24_f(next_t)
        m31, m32, m34 = m31_f(next_t), m32_f(next_t), m34_f(next_t)
        m41, m42, m43 = m41_f(next_t), m42_f(next_t), m43_f(next_t)
        theta0 = theta0_f(next_t)

        if np.any(np.less([T,nu1,nu2,nu3,nu4,m12,m13,m14,m21,m23,m24,m31,m32,m34,m41,m42,m43,theta0], 0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if np.any(np.equal([nu1,nu2,nu3,nu4], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        val1000, val0100, val0010, val0001 = \
            _inject_mutations_4D_valcalc(this_dt, xx, yy, zz, aa, theta0, 
                                         frozen1, frozen2, frozen3, frozen4)
        kernels._inject_mutations_4D_vals(phi_gpu, L,
                                          val0001, val0010, val0100, val1000, block=(1,1,1))
        # I can use the c_gpu buffer for the MInt_gpu buffer, to save GPU memory.
        # Note that I have to reassign this after each transpose operation I do.
        MInt_gpu = c_gpu
        if not frozen1:
            kernels._Vfunc(xx_gpu, nu1, L, V_gpu, 
                           grid=_grid(L), block=_block())
            kernels._Vfunc(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                           grid=_grid(L-1), block=_block())
            kernels._Mfunc4D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, gamma1, h1,
                             np.int32(L-1), M, N, O, MInt_gpu,
                             grid=_grid((L-1)*M*N*O), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, L, M*N*O,
                a_gpu, b_gpu,
                grid=_grid((L-1)*M*N*O), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, L, M*N*O,
                b_gpu, c_gpu,
                grid=_grid((L-1)*M*N*O), block=_block())
            kernels._include_bc(dx_gpu, nu1, gamma1, h1, L, M*N*O,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, L, M*N*O, grid=_grid(M*N*O), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                M*N*O, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(M*N*O,L))
        phi_gpu, c_gpu = c_gpu.reshape(M,L*N*O), phi_gpu.reshape(M,L*N*O)
        MInt_gpu = c_gpu
        if not frozen2:
            kernels._Vfunc(xx_gpu, nu2, M, V_gpu, 
                           grid=_grid(M), block=_block())
            kernels._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                           grid=_grid(M-1), block=_block())
            # Note the order of the m arguments here.
            kernels._Mfunc4D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m21, gamma2, h2,
                             np.int32(M-1), N, O, L, MInt_gpu,
                             grid=_grid((M-1)*L*N*O), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, M, L*N*O,
                a_gpu, b_gpu,
                grid=_grid((M-1)*L*N*O), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, M, L*N*O,
                b_gpu, c_gpu,
                grid=_grid((M-1)*L*N*O), block=_block())
            kernels._include_bc(dx_gpu, nu2, gamma2, h2, M, L*N*O,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, M, L*N*O, grid=_grid(L*N*O), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, M,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*N*O, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*N*O,M))
        phi_gpu, c_gpu = c_gpu.reshape(N,L*M*O), phi_gpu.reshape(N,L*M*O)
        MInt_gpu = c_gpu
        if not frozen3:
            kernels._Vfunc(xx_gpu, nu3, N, V_gpu, 
                           grid=_grid(N), block=_block())
            kernels._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                           grid=_grid(N-1), block=_block())
            kernels._Mfunc4D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m31, m32, gamma3, h3,
                             np.int32(N-1), O, L, M, MInt_gpu,
                             grid=_grid((N-1)*M*L*O), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, N, L*M*O,
                a_gpu, b_gpu,
                grid=_grid((N-1)*L*M*O), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, N, L*M*O,
                b_gpu, c_gpu,
                grid=_grid((N-1)*L*M*O), block=_block())
            kernels._include_bc(dx_gpu, nu3, gamma3, h3, N, L*M*O,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, N, L*M*O, grid=_grid(L*M*O), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, N,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M*O, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*M*O,N))
        phi_gpu, c_gpu = c_gpu.reshape(O,L*M*N), phi_gpu.reshape(O,L*M*N)
        MInt_gpu = c_gpu
        if not frozen4:
            kernels._Vfunc(xx_gpu, nu4, O, V_gpu, 
                           grid=_grid(O), block=_block())
            kernels._Vfunc(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                           grid=_grid(O-1), block=_block())
            kernels._Mfunc4D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m41, m42, m43, gamma4, h4,
                             np.int32(O-1), L, M, N, MInt_gpu,
                             grid=_grid((O-1)*M*L*N), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, O, L*M*N,
                a_gpu, b_gpu,
                grid=_grid((O-1)*L*M*N), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, O, L*M*N,
                b_gpu, c_gpu,
                grid=_grid((O-1)*L*M*N), block=_block())
            kernels._include_bc(dx_gpu, nu4, gamma4, h4, O, L*M*N,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, O, L*M*N, grid=_grid(L*M*N), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, O,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M*N, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*M*N,O))
        phi_gpu, c_gpu = c_gpu.reshape(L,M*N*O), phi_gpu.reshape(L,M*N*O)

        current_t += this_dt

    return phi_gpu.get().reshape(L,M,N,O)

def _five_pops_temporal_params(phi, xx, T, initial_t, nu1_f, nu2_f, nu3_f, nu4_f, nu5_f,
            m12_f, m13_f, m14_f, m15_f, m21_f, m23_f, m24_f, m25_f, m31_f, m32_f, m34_f, m35_f,
            m41_f, m42_f, m43_f, m45_f, m51_f, m52_f, m53_f, m54_f, 
            gamma1_f, gamma2_f, gamma3_f, gamma4_f, gamma5_f,
            h1_f, h2_f, h3_f, h4_f, h5_f, theta0_f, frozen1, frozen2, frozen3, frozen4, frozen5):
    if dadi.Integration.use_delj_trick:
        raise ValueError("delj trick not currently supported in CUDA execution")

    current_t = initial_t
    nu1, nu2, nu3, nu4, nu5 = nu1_f(current_t), nu2_f(current_t), nu3_f(current_t), nu4_f(current_t), nu5_f(current_t)
    gamma1, gamma2, gamma3, gamma4, gamma5 = gamma1_f(current_t), gamma2_f(current_t), gamma3_f(current_t), gamma4_f(current_t), gamma5_f(current_t)
    h1, h2, h3, h4, h5 = h1_f(current_t), h2_f(current_t), h3_f(current_t), h4_f(current_t), h5_f(current_t)
    m12, m13, m14, m15 = m12_f(current_t), m13_f(current_t), m14_f(current_t), m15_f(current_t)
    m21, m23, m24, m25 = m21_f(current_t), m23_f(current_t), m24_f(current_t), m25_f(current_t)
    m31, m32, m34, m35 = m31_f(current_t), m32_f(current_t), m34_f(current_t), m35_f(current_t)
    m41, m42, m43, m45 = m41_f(current_t), m42_f(current_t), m43_f(current_t), m45_f(current_t)
    m51, m52, m53, m54 = m51_f(current_t), m52_f(current_t), m53_f(current_t), m54_f(current_t)
    nu1, nu2, nu3, nu4 = nu1_f(current_t), nu2_f(current_t), nu3_f(current_t), nu4_f(current_t)

    L = M = N = O = P = np.int32(len(xx))

    phi_gpu = gpuarray.to_gpu(phi.reshape(L,M*N*O*P))

    bb = aa = yy = zz = xx
    db = da = dx = dy = dz = np.diff(xx)
    dfactor = dadi.Integration._compute_dfactor(dx)
    xInt = (xx[:-1] + xx[1:])*0.5

    xx_gpu = gpuarray.to_gpu(xx)
    dx_gpu = gpuarray.to_gpu(dx)
    dfactor_gpu = gpuarray.to_gpu(dfactor)
    xInt_gpu = gpuarray.to_gpu(xInt)

    V_gpu = gpuarray.empty(L, np.float64)
    VInt_gpu = gpuarray.empty(L-1, np.float64)

    a_gpu = gpuarray.empty((L,M*N*O*P), np.float64)
    b_gpu = gpuarray.empty((L,M*N*O*P), np.float64)
    c_gpu = gpuarray.empty((L,M*N*O*P), np.float64)

    bsize_int = cusparseDgtsvInterleavedBatch_bufferSizeExt(
        cusparse_handle, 0, L, a_gpu.gpudata, b_gpu.gpudata,
        c_gpu.gpudata, phi_gpu.gpudata, M*N*O*P)
    pBuffer = pycuda.driver.mem_alloc(bsize_int)

    while current_t < T:
        dt = min(dadi.Integration._compute_dt(dx, nu1, [m12,m13,m14,m15], gamma1, h1),
                 dadi.Integration._compute_dt(dy, nu2, [m21,m23,m24,m25], gamma2, h2),
                 dadi.Integration._compute_dt(dz, nu3, [m31,m32,m34,m35], gamma3, h3),
                 dadi.Integration._compute_dt(da, nu4, [m41,m42,m43,m45], gamma4, h4),
                 dadi.Integration._compute_dt(db, nu5, [m51,m52,m53,m54],gamma5,h5))
        this_dt = np.float64(min(dt, T - current_t))

        next_t = current_t + this_dt

        nu1, nu2, nu3, nu4, nu5 = nu1_f(next_t), nu2_f(next_t), nu3_f(next_t), nu4_f(next_t), nu5_f(next_t)
        gamma1, gamma2, gamma3, gamma4, gamma5 = gamma1_f(next_t), gamma2_f(next_t), gamma3_f(next_t), gamma4_f(next_t), gamma5_f(next_t)
        h1, h2, h3, h4, h5 = h1_f(next_t), h2_f(next_t), h3_f(next_t), h4_f(next_t), h5_f(next_t)
        m12, m13, m14, m15 = m12_f(next_t), m13_f(next_t), m14_f(next_t), m15_f(next_t)
        m21, m23, m24, m25 = m21_f(next_t), m23_f(next_t), m24_f(next_t), m25_f(next_t)
        m31, m32, m34, m35 = m31_f(next_t), m32_f(next_t), m34_f(next_t), m35_f(next_t)
        m41, m42, m43, m45 = m41_f(next_t), m42_f(next_t), m43_f(next_t), m45_f(next_t)
        m51, m52, m53, m54 = m51_f(next_t), m52_f(next_t), m53_f(next_t), m54_f(next_t)
        theta0 = theta0_f(next_t)

        if np.any(np.less([T,nu1,nu2,nu3,nu4,nu5,m12,m13,m14,m15,m21,
                                 m23,m24,m25, m31,m32,m34,m35, m41,m42,m43,m45,
                                 m51,m52,m53,m54, theta0],
                                0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if np.any(np.equal([nu1,nu2,nu3,nu4,nu5], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        val10000, val01000, val00100, val00010, val00001 = \
            _inject_mutations_5D_valcalc(this_dt, xx, yy, zz, aa, bb, theta0, 
                                         frozen1, frozen2, frozen3, frozen4, frozen5)
        kernels._inject_mutations_5D_vals(phi_gpu, L,
                                          val00001, val00010, val00100, val01000, val10000, block=(1,1,1))
        # I can use the c_gpu buffer for the MInt_gpu buffer, to save GPU memory.
        # Note that I have to reassign this after each transpose operation I do.
        MInt_gpu = c_gpu
        if not frozen1:
            kernels._Vfunc(xx_gpu, nu1, L, V_gpu, 
                           grid=_grid(L), block=_block())
            kernels._Vfunc(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                           grid=_grid(L-1), block=_block())
            kernels._Mfunc5D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, m15, gamma1, h1,
                             np.int32(L-1), M, N, O, P, MInt_gpu,
                             grid=_grid((L-1)*M*N*O*P), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                a_gpu, b_gpu,
                grid=_grid((L-1)*M*N*O*P), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                b_gpu, c_gpu,
                grid=_grid((L-1)*M*N*O*P), block=_block())
            kernels._include_bc(dx_gpu, nu1, gamma1, h1, L, M*N*O*P,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, L, M*N*O*P, grid=_grid(M*N*O*P), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                M*N*O*P, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(M*N*O*P,L))
        phi_gpu, c_gpu = c_gpu.reshape(M,L*N*O*P), phi_gpu.reshape(M,L*N*O*P)
        MInt_gpu = c_gpu
        if not frozen2:
            kernels._Vfunc(xx_gpu, nu2, M, V_gpu, 
                           grid=_grid(M), block=_block())
            kernels._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                           grid=_grid(M-1), block=_block())
            # Note the order of the m arguments here.
            kernels._Mfunc5D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m25, m21, gamma2, h2,
                             np.int32(M-1), N, O, P, L, MInt_gpu,
                             grid=_grid((M-1)*L*N*O*P), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                a_gpu, b_gpu,
                grid=_grid((M-1)*L*N*O*P), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                b_gpu, c_gpu,
                grid=_grid((M-1)*L*N*O*P), block=_block())
            kernels._include_bc(dx_gpu, nu2, gamma2, h2, M, L*N*O*P,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, M, L*N*O*P, grid=_grid(L*N*O*P), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, M,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*N*O*P, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*N*O*P,M))
        phi_gpu, c_gpu = c_gpu.reshape(N,L*M*O*P), phi_gpu.reshape(N,L*M*O*P)
        MInt_gpu = c_gpu
        if not frozen3:
            kernels._Vfunc(xx_gpu, nu3, N, V_gpu, 
                           grid=_grid(N), block=_block())
            kernels._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                           grid=_grid(N-1), block=_block())
            kernels._Mfunc5D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m35, m31, m32, gamma3, h3,
                             np.int32(N-1), O, P, L, M, MInt_gpu,
                             grid=_grid((N-1)*M*L*O*P), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, N, L*M*O*P,
                a_gpu, b_gpu,
                grid=_grid((N-1)*L*M*O*P), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, N, L*M*O*P,
                b_gpu, c_gpu,
                grid=_grid((N-1)*L*M*O*P), block=_block())
            kernels._include_bc(dx_gpu, nu3, gamma3, h3, N, L*M*O*P,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, N, L*M*O*P, grid=_grid(L*M*O*P), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, N,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M*O*P, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*M*O*P,N))
        phi_gpu, c_gpu = c_gpu.reshape(O,L*M*N*P), phi_gpu.reshape(O,L*M*N*P)
        MInt_gpu = c_gpu
        if not frozen4:
            kernels._Vfunc(xx_gpu, nu4, O, V_gpu, 
                           grid=_grid(O), block=_block())
            kernels._Vfunc(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                           grid=_grid(O-1), block=_block())
            kernels._Mfunc5D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m45, m41, m42, m43, gamma4, h4,
                             np.int32(O-1), P, L, M, N, MInt_gpu,
                             grid=_grid((O-1)*M*L*N*P), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                a_gpu, b_gpu,
                grid=_grid((O-1)*L*M*N*P), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                b_gpu, c_gpu,
                grid=_grid((O-1)*L*M*N*P), block=_block())
            kernels._include_bc(dx_gpu, nu4, gamma4, h4, O, L*M*N*P,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, O, L*M*N*P, grid=_grid(L*M*N*P), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, O,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M*N*P, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*M*N*P,O))
        phi_gpu, c_gpu = c_gpu.reshape(P,L*M*N*O), phi_gpu.reshape(P,L*M*N*O)
        MInt_gpu = c_gpu
        if not frozen5:
            kernels._Vfunc(xx_gpu, nu5, P, V_gpu, 
                           grid=_grid(P), block=_block())
            kernels._Vfunc(xInt_gpu, nu5, np.int32(P-1), VInt_gpu, 
                           grid=_grid(P-1), block=_block())
            kernels._Mfunc5D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m51, m52, m53, m54, gamma5, h5,
                             np.int32(P-1), L, M, N, O, MInt_gpu,
                             grid=_grid((P-1)*L*M*N*O), block=_block())

            b_gpu.fill(1./this_dt)
            kernels._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                a_gpu, b_gpu,
                grid=_grid((P-1)*L*M*N*O), block=_block())
            kernels._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                b_gpu, c_gpu,
                grid=_grid((P-1)*L*M*N*O), block=_block())
            kernels._include_bc(dx_gpu, nu5, gamma5, h5, P, L*M*N*O,
                b_gpu, block=(1,1,1))
            kernels._cx0(c_gpu, P, L*M*N*O, grid=_grid(L*M*N*O), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, P,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M*N*O, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*M*N*O,P))
        phi_gpu, c_gpu = c_gpu.reshape(L,M*N*O*P), phi_gpu.reshape(L,M*N*O*P)

        current_t += this_dt

    return phi_gpu.get().reshape(L,M,N,O,P)

import numpy as np

from dadi import Demes

from .. import Integration as PolyInt

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
from . import kernels_poly


def _two_pops_temporal_params(phi, xx, T, initial_t, nu1_f, nu2_f, m12_f, m21_f, sel1_f,
            sel2_f, theta0_f, frozen1, frozen2, nomut1, nomut2, deme_ids, ploidy1, ploidy2):
    current_t = initial_t
    nu1,nu2 = nu1_f(current_t), nu2_f(current_t)
    m12,m21 = m12_f(current_t), m21_f(current_t)
    s1,s2 = sel1_f(current_t), sel2_f(current_t)
   
    if PolyInt.use_delj_trick:
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

    demes_hist = [[0, [nu1,nu2], [m12,m21]]]
    while current_t < T:
        dt = min(PolyInt._compute_dt(dx,nu1,[m12],s1,ploidy1),
                 PolyInt._compute_dt(dy,nu2,[m21],s2,ploidy2))
        this_dt = np.float64(min(dt, T - current_t))

        next_t = current_t + this_dt

        nu1,nu2 = nu1_f(next_t), nu2_f(next_t)
        m12,m21 = m12_f(next_t), m21_f(next_t)
        s1,s2 = sel1_f(next_t), sel2_f(next_t)
        theta0 = theta0_f(next_t)
        demes_hist.append([next_t, [nu1,nu2], [m12,m21]])

        # TODO: edit this to use the inject_mut function in this file? Or maybe this is fine... 
        #       ask Ryan about this... but I think I just need to add an __init__ file for the Polyploidy/cuda directory
        val10, val01 = dadi.cuda.Integration._inject_mutations_2D_valcalc(this_dt, xx, yy, theta0, frozen1, frozen2,
                                                                          nomut1, nomut2)
        kernels_poly._inject_mutations_2D_vals(phi_gpu, L, np.float64(val01), np.float64(val10), 
                                          block=(1, 1, 1))

        ### Everything up until here is not ploidy dependent, so it should all be able to stay as is

        if not frozen1:
            if ploidy1[0]: # if diploid
                # Note, if L is an int32, L-1 can be int64. This can
                # break the code. So we wrap the -1 expressions in np.int32.
                kernels_poly._Vfunc(xx_gpu, nu1, L, Vx_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu1, np.int32(L-1), VIntx_gpu, 
                                grid=_grid(L-1), block=_block())
                # Fill buff_gpu with MInt
                kernels_poly._Mfunc2D(xInt_gpu, yy_gpu, m12, s1[0], s1[1],
                                np.int32(L-1), M, buff_gpu,
                                grid=_grid((L-1)*M), block=_block())

                bx_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    ax_gpu, bx_gpu,
                    grid=_grid((L-1)*M), block=_block())
                # Note that this transforms buff_gpu from MInt to cx
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    bx_gpu, buff_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._include_bc(dx_gpu, nu1, s1[0], s1[1], L, M,
                    bx_gpu, block=(1,1,1))
                kernels_poly._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())
            elif ploidy1[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu1, L, Vx_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu1, np.int32(L-1), VIntx_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc2D_auto(xInt_gpu, yy_gpu, m12, s1[0], s1[1], s1[2], s1[3],
                                np.int32(L-1), M, buff_gpu,
                                grid=_grid((L-1)*M), block=_block())
                bx_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    ax_gpu, bx_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    bx_gpu, buff_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu1, s1[0], s1[1], s1[2], s1[3], L, M,
                    bx_gpu, block=(1,1,1))
                kernels_poly._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())
            elif ploidy1[2]: # if allotetraploid subgenome a
                kernels_poly._Vfunc(xx_gpu, nu1, L, Vx_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu1, np.int32(L-1), VIntx_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc2D_allo_a(xInt_gpu, yy_gpu, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7], 
                                np.int32(L-1), M, buff_gpu,
                                grid=_grid((L-1)*M), block=_block())
                bx_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    ax_gpu, bx_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    bx_gpu, buff_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._include_bc_allo_a(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7], L, M,
                    bx_gpu, block=(1,1,1))
                kernels_poly._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())
            # note that allotetraploid subgenome b cannot be the first dimension here,
            # so to write as minimally and cleanly as possible, we skip that here
            elif ploidy1[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu1, L, Vx_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu1, np.int32(L-1), VIntx_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc2D_autohex(xInt_gpu, yy_gpu, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],
                                np.int32(L-1), M, buff_gpu,
                                grid=_grid((L-1)*M), block=_block())
                bx_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    ax_gpu, bx_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    bx_gpu, buff_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5], L, M,
                    bx_gpu, block=(1,1,1))
                kernels_poly._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())
            elif ploidy1[5]: # if 4+2 hexaploid, tetraploid subgenome
                kernels_poly._Vfunc_tetra(xx_gpu, nu1, L, Vx_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu1, np.int32(L-1), VIntx_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc2D_hex_tetra(xInt_gpu, yy_gpu, m12, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13],
                                np.int32(L-1), M, buff_gpu,
                                grid=_grid((L-1)*M), block=_block())
                bx_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    ax_gpu, bx_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    bx_gpu, buff_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._include_bc_hex_tetra(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13],
                    L, M, bx_gpu, block=(1,1,1))
                kernels_poly._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())
            # similar to allotetraploids, we don't support the diploid subgenome of a 4+2 hexaploid for this dimensions
            # also note that the 2+2+2 hexaploids are only supported in 3D+

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


            if ploidy2[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu2, L, Vx_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(L-1), VIntx_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc2D(xInt_gpu, yy_gpu, m21, s2[0], s2[1],
                                np.int32(L-1), M, buff_gpu,
                                grid=_grid((L-1)*M), block=_block())

                bx_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    ax_gpu, bx_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    bx_gpu, buff_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._include_bc(dx_gpu, nu2, s2[0], s2[1], L, M,
                    bx_gpu, block=(1,1,1))
                kernels_poly._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())
            elif ploidy2[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu2, L, Vx_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu2, np.int32(L-1), VIntx_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc2D_auto(xInt_gpu, yy_gpu, m21, s2[0],s2[1],s2[2],s2[3],
                                np.int32(L-1), M, buff_gpu,
                                grid=_grid((L-1)*M), block=_block())

                bx_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    ax_gpu, bx_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    bx_gpu, buff_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3], L, M,
                    bx_gpu, block=(1,1,1))
                kernels_poly._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())
            # as a counterpoint to the above, we don't support allotet subgenome a being the second dimension
            elif ploidy2[3]: # if allotetraploid subgenome b
                kernels_poly._Vfunc(xx_gpu, nu2, L, Vx_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(L-1), VIntx_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc2D_allo_b(xInt_gpu, yy_gpu, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],
                                np.int32(L-1), M, buff_gpu,
                                grid=_grid((L-1)*M), block=_block())

                bx_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    ax_gpu, bx_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    bx_gpu, buff_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._include_bc_allo_b(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7], L, M,
                    bx_gpu, block=(1,1,1))
                kernels_poly._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())
            elif ploidy2[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu2, L, Vx_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu2, np.int32(L-1), VIntx_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc2D_autohex(xInt_gpu, yy_gpu, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],
                                np.int32(L-1), M, buff_gpu,
                                grid=_grid((L-1)*M), block=_block())

                bx_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    ax_gpu, bx_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    bx_gpu, buff_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5], L, M,
                    bx_gpu, block=(1,1,1))
                kernels_poly._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())
            elif ploidy2[6]: # if 4+2 hexaploid, diploid subgenome
                kernels_poly._Vfunc(xx_gpu, nu2, L, Vx_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(L-1), VIntx_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc2D_hex_dip(xInt_gpu, yy_gpu, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13],
                                np.int32(L-1), M, buff_gpu,
                                grid=_grid((L-1)*M), block=_block())

                bx_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    ax_gpu, bx_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    buff_gpu, Vx_gpu, this_dt, L, M,
                    bx_gpu, buff_gpu,
                    grid=_grid((L-1)*M), block=_block())
                kernels_poly._include_bc_hex_dip(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13], L, M,
                    bx_gpu, block=(1,1,1))
                kernels_poly._cx0(buff_gpu, L, M, grid=_grid(M), block=_block())


            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                ax_gpu.gpudata, bx_gpu.gpudata, buff_gpu.gpudata, phiT_gpu.gpudata,
                L, pBuffer)
            transpose_gpuarray(phiT_gpu, buff_gpu)
            phi_gpu, buff_gpu = buff_gpu, phiT_gpu

        current_t = next_t
    Demes.cache.append(Demes.IntegrationNonConst(history = demes_hist, deme_ids=deme_ids))
    return phi_gpu.get()

def _three_pops_temporal_params(phi, xx, T, initial_t, nu1_f, nu2_f, nu3_f, 
            m12_f, m13_f, m21_f, m23_f, m31_f, m32_f, 
            sel1_f, sel2_f, sel3_f, 
            theta0_f, frozen1, frozen2, frozen3, deme_ids,
            ploidy1, ploidy2, ploidy3):
    if PolyInt.use_delj_trick:
        raise ValueError("delj trick not currently supported in CUDA execution")

    t = current_t = initial_t
    nu1,nu2,nu3 = nu1_f(t), nu2_f(t), nu3_f(t)
    m12,m13,m21,m23,m31,m32 = m12_f(t), m13_f(t), m21_f(t), m23_f(t), m31_f(t), m32_f(t)
    s1,s2,s3 = sel1_f(t), sel2_f(t), sel3_f(t)

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

    demes_hist = [[0, [nu1,nu2,nu3], [m12,m13,m21,m23,m31,m32]]]
    while current_t < T:
        dt = min(PolyInt._compute_dt(dx, nu1, [m12, m13], s1, ploidy1),
                 PolyInt._compute_dt(dy, nu2, [m21, m23], s2, ploidy2),
                 PolyInt._compute_dt(dz, nu3, [m31, m32], s3, ploidy3))
        this_dt = np.float64(min(dt, T - current_t))

        next_t = current_t + this_dt

        nu1,nu2,nu3 = nu1_f(next_t), nu2_f(next_t), nu3_f(next_t)
        m12,m13 = m12_f(next_t), m13_f(next_t)
        m21,m23 = m21_f(next_t), m23_f(next_t)
        m31,m32 = m31_f(next_t), m32_f(next_t)
        s1,s2,s3 = sel1_f(next_t), sel2_f(next_t), sel3_f(next_t)
        theta0 = theta0_f(next_t)
        demes_hist.append([next_t, [nu1,nu2,nu3], [m12,m13,m21,m23,m31,m32]])

        if np.any(np.less([T,nu1,nu2,nu3,m12,m13,m21,m23,m31,m32,theta0], 0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if np.any(np.equal([nu1,nu2,nu3], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        val100, val010, val001 = \
            _inject_mutations_3D_valcalc(this_dt, xx, yy, zz, theta0, 
                                         frozen1, frozen2, frozen3)
        kernels_poly._inject_mutations_3D_vals(phi_gpu, L,
                                          val001, val010, val100, block=(1,1,1))
        # I can use the c_gpu buffer for the MInt_gpu buffer, to save GPU memory.
        # Note that I have to reassign this after each transpose operation I do.
        MInt_gpu = c_gpu
        if not frozen1:
            if ploidy1[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc3D(xInt_gpu, xx_gpu, xx_gpu, m12, m13, s1[0], s1[1],
                                np.int32(L-1), M, N, MInt_gpu,
                                grid=_grid((L-1)*M*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N), block=_block())
                kernels_poly._include_bc(dx_gpu, nu1, s1[0], s1[1], L, M*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N, grid=_grid(M*N), block=_block())
            elif ploidy1[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc3D_auto(xInt_gpu, xx_gpu, xx_gpu, m12, m13, s1[0],s1[1],s1[2],s1[3],
                                np.int32(L-1), M, N, MInt_gpu,
                                grid=_grid((L-1)*M*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3], L, M*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N, grid=_grid(M*N), block=_block())
            # note that here we only support the allotetraploids with two subgenomes 
            # as being passed as the last two dimensions/populations. 
            # So, no allotetraploid or 4+2 hexaploid functions here!
            elif ploidy1[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc3D_autohex(xInt_gpu, xx_gpu, xx_gpu, m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],
                                np.int32(L-1), M, N, MInt_gpu,
                                grid=_grid((L-1)*M*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5], L, M*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N, grid=_grid(M*N), block=_block())
            elif ploidy1[7]: # if 2+2+2 hexaploid - subgenome a
                kernels_poly._Vfunc(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc3D_hex_a(xInt_gpu, xx_gpu, xx_gpu, m12, m13, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],
                                                s1[12],s1[13],s1[14],s1[15],s1[16],s1[17],s1[18],s1[19],s1[20],s1[21],s1[22],s1[23],s1[24],s1[25],
                                                np.int32(L-1), M, N, MInt_gpu,
                                                grid=_grid((L-1)*M*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N), block=_block())
                kernels_poly._include_bc_hex_a(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],
                                                   s1[12],s1[13],s1[14],s1[15],s1[16],s1[17],s1[18],s1[19],s1[20],s1[21],s1[22],s1[23],s1[24],s1[25], 
                                                   L, M*N, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N, grid=_grid(M*N), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                M*N, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(M*N,L))
        phi_gpu, c_gpu = c_gpu.reshape(M,L*N), phi_gpu.reshape(M,L*N)
        MInt_gpu = c_gpu
        if not frozen2:
            if ploidy2[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                # Note the order of the m23, m21 arguments here.
                kernels_poly._Mfunc3D(xInt_gpu, xx_gpu, xx_gpu, m23, m21, s2[0], s2[1],
                                np.int32(M-1), N, L, MInt_gpu,
                                grid=_grid((M-1)*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._include_bc(dx_gpu, nu2, s2[0], s2[1], M, L*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N, grid=_grid(L*N), block=_block())
            elif ploidy2[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc3D_auto(xInt_gpu, xx_gpu, xx_gpu, m23, m21, s2[0],s2[1],s2[2],s2[3],
                                np.int32(M-1), N, L, MInt_gpu,
                                grid=_grid((M-1)*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3], M, L*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N, grid=_grid(L*N), block=_block())
            elif ploidy2[2]: # if allotetraploid subgenome a
                kernels_poly._Vfunc(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc3D_allo_a(xInt_gpu, xx_gpu, xx_gpu, m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],
                                np.int32(M-1), N, L, MInt_gpu,
                                grid=_grid((M-1)*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._include_bc_allo_a(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7], M, L*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N, grid=_grid(L*N), block=_block())
            elif ploidy2[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc3D_autohex(xInt_gpu, xx_gpu, xx_gpu, m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],
                                np.int32(M-1), N, L, MInt_gpu,
                                grid=_grid((M-1)*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5], M, L*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N, grid=_grid(L*N), block=_block())
            elif ploidy2[5]: # if 4+2 hexaploid - tetraploid subgenome
                kernels_poly._Vfunc_tetra(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc3D_hex_tetra(xInt_gpu, xx_gpu, xx_gpu, m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13],
                                np.int32(M-1), N, L, MInt_gpu,
                                grid=_grid((M-1)*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._include_bc_hex_tetra(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13], M, L*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N, grid=_grid(L*N), block=_block())
            elif ploidy2[8]: # if 2+2+2 hexaploid - subgenome b
                kernels_poly._Vfunc(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc3D_hex_b(xInt_gpu, xx_gpu, xx_gpu, m23, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],
                                            s2[12],s2[13],s2[14],s2[15],s2[16],s2[17],s2[18],s2[19],s2[20],s2[21],s2[22],s2[23],s2[24],s2[25],
                                            np.int32(M-1), N, L, MInt_gpu,
                                            grid=_grid((M-1)*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N), block=_block())
                kernels_poly._include_bc_hex_b(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],
                                               s2[12],s2[13],s2[14],s2[15],s2[16],s2[17],s2[18],s2[19],s2[20],s2[21],s2[22],s2[23],s2[24],s2[25],
                                               M, L*N, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N, grid=_grid(L*N), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, M,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*N, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*N,M))
        phi_gpu, c_gpu = c_gpu.reshape(N,L*M), phi_gpu.reshape(N,L*M)
        MInt_gpu = c_gpu
        if not frozen3:
            if ploidy3[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc3D(xInt_gpu, xx_gpu, xx_gpu, m31, m32, s3[0], s3[1],
                                np.int32(N-1), L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._include_bc(dx_gpu, nu3, s3[0], s3[1], N, L*M,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M, grid=_grid(L*M), block=_block())
            elif ploidy3[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc3D_auto(xInt_gpu, xx_gpu, xx_gpu, m31, m32, s3[0],s3[1],s3[2],s3[3],
                                np.int32(N-1), L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3], N, L*M,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M, grid=_grid(L*M), block=_block())
            elif ploidy3[3]: # if allotetraploid subgenome b
                kernels_poly._Vfunc(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc3D_allo_b(xInt_gpu, xx_gpu, xx_gpu, m31, m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],
                                np.int32(N-1), L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._include_bc_allo_b(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7], N, L*M,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M, grid=_grid(L*M), block=_block())
            elif ploidy3[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc3D_autohex(xInt_gpu, xx_gpu, xx_gpu, m31, m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],
                                np.int32(N-1), L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5], N, L*M,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M, grid=_grid(L*M), block=_block())
            elif ploidy3[6]: # if 4+2 hexaploid - diploid subgenome
                kernels_poly._Vfunc(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc3D_hex_dip(xInt_gpu, xx_gpu, xx_gpu, m31, m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13],
                                np.int32(N-1), L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._include_bc_hex_dip(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13], N, L*M,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M, grid=_grid(L*M), block=_block())
            elif ploidy3[9]: # if 2+2+2 hexaploid - subgenome c
                kernels_poly._Vfunc(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc3D_hex_c(xInt_gpu, xx_gpu, xx_gpu, m31, m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],
                                            s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25],
                                            np.int32(N-1), L, M, MInt_gpu,
                                            grid=_grid((N-1)*M*L), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M), block=_block())
                kernels_poly._include_bc_hex_c(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],
                                               s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25], 
                                               N, L*M, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M, grid=_grid(L*M), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, N,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(M*N,L))
        phi_gpu, c_gpu = c_gpu.reshape(L,M*N), phi_gpu.reshape(L,M*N)

        current_t += this_dt
    Demes.cache.append(Demes.IntegrationNonConst(history = demes_hist, deme_ids=deme_ids))
    return phi_gpu.get().reshape(L,M,N)

def _four_pops_temporal_params(phi, xx, T, initial_t, nu1_f, nu2_f, nu3_f, nu4_f,
            m12_f, m13_f, m14_f, m21_f, m23_f, m24_f, m31_f, m32_f, m34_f,
            m41_f, m42_f, m43_f, sel1_f, sel2_f, sel3_f, sel4_f,
            theta0_f, frozen1, frozen2, frozen3, frozen4, deme_ids,
            ploidy1, ploidy2, ploidy3, ploidy4):
    if PolyInt.use_delj_trick:
        raise ValueError("delj trick not currently supported in CUDA execution")

    current_t = initial_t
    nu1, nu2, nu3, nu4 = nu1_f(current_t), nu2_f(current_t), nu3_f(current_t), nu4_f(current_t)
    s1, s2, s3, s4 = sel1_f(current_t), sel2_f(current_t), sel3_f(current_t), sel4_f(current_t)
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

    demes_hist = [[0, [nu1,nu2,nu3,nu4], [m12,m13,m14,m21,m23,m24,m31,m32,m34,m41,m42,m43]]]
    while current_t < T:
        dt = min(PolyInt._compute_dt(dx, nu1, [m12, m13, m14], s1, ploidy1),
                 PolyInt._compute_dt(dy, nu2, [m21, m23, m24], s2, ploidy2),
                 PolyInt._compute_dt(dz, nu3, [m31, m32, m34], s3, ploidy3),
                 PolyInt._compute_dt(da, nu4, [m41, m42, m43], s4, ploidy4))
        this_dt = np.float64(min(dt, T - current_t))

        next_t = current_t + this_dt

        nu1, nu2, nu3, nu4 = nu1_f(next_t), nu2_f(next_t), nu3_f(next_t), nu4_f(next_t)
        s1, s2, s3, s4 = sel1_f(next_t), sel2_f(next_t), sel3_f(next_t), sel4_f(next_t)
        m12, m13, m14 = m12_f(next_t), m13_f(next_t), m14_f(next_t)
        m21, m23, m24 = m21_f(next_t), m23_f(next_t), m24_f(next_t)
        m31, m32, m34 = m31_f(next_t), m32_f(next_t), m34_f(next_t)
        m41, m42, m43 = m41_f(next_t), m42_f(next_t), m43_f(next_t)
        theta0 = theta0_f(next_t)
        demes_hist.append([next_t, [nu1,nu2,nu3,nu4], [m12,m13,m14,m21,m23,m24,m31,m32,m34,m41,m42,m43]])

        if np.any(np.less([T,nu1,nu2,nu3,nu4,m12,m13,m14,m21,m23,m24,m31,m32,m34,m41,m42,m43,theta0], 0)):
            raise ValueError('A time, population size, migration rate, or '
                             'theta0 is < 0. Has the model been mis-specified?')
        if np.any(np.equal([nu1,nu2,nu3,nu4], 0)):
            raise ValueError('A population size is 0. Has the model been '
                             'mis-specified?')

        val1000, val0100, val0010, val0001 = \
            _inject_mutations_4D_valcalc(this_dt, xx, yy, zz, aa, theta0, 
                                         frozen1, frozen2, frozen3, frozen4)
        kernels_poly._inject_mutations_4D_vals(phi_gpu, L,
                                          val0001, val0010, val0100, val1000, block=(1,1,1))
        # I can use the c_gpu buffer for the MInt_gpu buffer, to save GPU memory.
        # Note that I have to reassign this after each transpose operation I do.
        MInt_gpu = c_gpu
        if not frozen1:
            if ploidy1[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc4D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, s1[0], s1[1],
                                np.int32(L-1), M, N, O, MInt_gpu,
                                grid=_grid((L-1)*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N*O), block=_block())
                kernels_poly._include_bc(dx_gpu, nu1, s1[0], s1[1], L, M*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N*O, grid=_grid(M*N*O), block=_block())
            elif ploidy1[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc4D_auto(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, s1[0],s1[1],s1[2],s1[3],
                                np.int32(L-1), M, N, O, MInt_gpu,
                                grid=_grid((L-1)*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N*O), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3], L, M*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N*O, grid=_grid(M*N*O), block=_block())
            elif ploidy1[2]: # if allotetraploid subgenome a
                kernels_poly._Vfunc(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc4D_allo_a(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],
                                np.int32(L-1), M, N, O, MInt_gpu,
                                grid=_grid((L-1)*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N*O), block=_block())
                kernels_poly._include_bc_allo_a(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7], L, M*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N*O, grid=_grid(M*N*O), block=_block())
            elif ploidy1[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc4D_autohex(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],
                                np.int32(L-1), M, N, O, MInt_gpu,
                                grid=_grid((L-1)*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N*O), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5], L, M*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N*O, grid=_grid(M*N*O), block=_block())
            elif ploidy1[5]: # if 4+2 hexaploid - tetraploid subgenome
                kernels_poly._Vfunc_tetra(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc4D_hex_tetra(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13],
                                np.int32(L-1), M, N, O, MInt_gpu,
                                grid=_grid((L-1)*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N*O), block=_block())
                kernels_poly._include_bc_hex_tetra(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13], L, M*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N*O, grid=_grid(M*N*O), block=_block())


            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                M*N*O, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(M*N*O,L))
        phi_gpu, c_gpu = c_gpu.reshape(M,L*N*O), phi_gpu.reshape(M,L*N*O)
        MInt_gpu = c_gpu
        if not frozen2:
            if ploidy2[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                # Note the order of the m arguments here.
                kernels_poly._Mfunc4D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m21, s2[0], s2[1],
                                np.int32(M-1), N, O, L, MInt_gpu,
                                grid=_grid((M-1)*L*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._include_bc(dx_gpu, nu2, s2[0], s2[1], M, L*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O, grid=_grid(L*N*O), block=_block())
            elif ploidy2[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc4D_auto(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m21, s2[0],s2[1],s2[2],s2[3],
                                np.int32(M-1), N, O, L, MInt_gpu,
                                grid=_grid((M-1)*L*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3], M, L*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O, grid=_grid(L*N*O), block=_block())
            elif ploidy2[3]: # if allotetraploid subgenome b
                kernels_poly._Vfunc(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc4D_allo_b(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],
                                np.int32(M-1), N, O, L, MInt_gpu,
                                grid=_grid((M-1)*L*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._include_bc_allo_b(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7], M, L*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O, grid=_grid(L*N*O), block=_block())
            elif ploidy2[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc4D_autohex(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],
                                np.int32(M-1), N, O, L, MInt_gpu,
                                grid=_grid((M-1)*L*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5], M, L*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O, grid=_grid(L*N*O), block=_block())
            elif ploidy2[6]: # if 4+2 hexaploid - diploid subgenome
                kernels_poly._Vfunc(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc4D_hex_dip(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13],
                                np.int32(M-1), N, O, L, MInt_gpu,
                                grid=_grid((M-1)*L*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._include_bc_hex_dip(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13], M, L*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O, grid=_grid(L*N*O), block=_block())
            elif ploidy2[7]: # if 2+2+2 hexaploid - subgenome a
                kernels_poly._Vfunc(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc4D_hex_a(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],
                                            s2[12],s2[13],s2[14],s2[15],s2[16],s2[17],s2[18],s2[19],s2[20],s2[21],s2[22],s2[23],s2[24],s2[25],
                                            grid=_grid((M-1)*L*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O), block=_block())
                kernels_poly._include_bc_hex_a(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],
                                                s2[12],s2[13],s2[14],s2[15],s2[16],s2[17],s2[18],s2[19],s2[20],s2[21],s2[22],s2[23],s2[24],s2[25], 
                                                M, L*N*O,b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O, grid=_grid(L*N*O), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, M,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*N*O, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*N*O,M))
        phi_gpu, c_gpu = c_gpu.reshape(N,L*M*O), phi_gpu.reshape(N,L*M*O)
        MInt_gpu = c_gpu
        if not frozen3:
            if ploidy3[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc4D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m31, m32, s3[0], s3[1],
                                np.int32(N-1), O, L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._include_bc(dx_gpu, nu3, s3[0], s3[1], N, L*M*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M*O, grid=_grid(L*M*O), block=_block())
            elif ploidy3[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc4D_auto(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m31, m32, s3[0],s3[1],s3[2],s3[3],
                                np.int32(N-1), O, L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3], N, L*M*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M*O, grid=_grid(L*M*O), block=_block())
            elif ploidy3[2]: # if allotetraploid subgenome a
                kernels_poly._Vfunc(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc4D_allo_a(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m31, m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],
                                np.int32(N-1), O, L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._include_bc_allo_a(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7], N, L*M*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M*O, grid=_grid(L*M*O), block=_block())
            elif ploidy3[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc4D_autohex(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m31, m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],
                                np.int32(N-1), O, L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5], N, L*M*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M*O, grid=_grid(L*M*O), block=_block())
            elif ploidy3[5]: # if 4+2 hexaploid - tetraploid subgenome
                kernels_poly._Vfunc_tetra(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc4D_hex_tetra(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m31, m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13],
                                np.int32(N-1), O, L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._include_bc_hex_tetra(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],s3[12],s3[13], N, L*M*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M*O, grid=_grid(L*M*O), block=_block())
            elif ploidy3[8]: # if 2+2+2 hexaploid - subgenome b
                kernels_poly._Vfunc(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc4D_hex_b(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m31, m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],
                                            s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25],
                                            np.int32(N-1), O, L, M, MInt_gpu,
                                            grid=_grid((N-1)*M*L*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M*O), block=_block())
                kernels_poly._include_bc_hex_b(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],
                                                s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25],
                                                N, L*M*O, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M*O, grid=_grid(L*M*O), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, N,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M*O, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*M*O,N))
        phi_gpu, c_gpu = c_gpu.reshape(O,L*M*N), phi_gpu.reshape(O,L*M*N)
        MInt_gpu = c_gpu
        if not frozen4:
            if ploidy4[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc4D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m41, m42, m43, s4[0], s4[1],
                                np.int32(O-1), L, M, N, MInt_gpu,
                                grid=_grid((O-1)*M*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._include_bc(dx_gpu, nu4, s4[0], s4[1], O, L*M*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N, grid=_grid(L*M*N), block=_block())
            elif ploidy4[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc4D_auto(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m41, m42, m43, s4[0],s4[1],s4[2],s4[3],
                                np.int32(O-1), L, M, N, MInt_gpu,
                                grid=_grid((O-1)*M*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu4, s4[0],s4[1],s4[2],s4[3], O, L*M*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N, grid=_grid(L*M*N), block=_block())
            elif ploidy4[3]: # if allotetraploid subgenome b
                kernels_poly._Vfunc(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc4D_allo_b(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m41, m42, m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],
                                np.int32(O-1), L, M, N, MInt_gpu,
                                grid=_grid((O-1)*M*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._include_bc_allo_b(dx_gpu, nu4, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7], O, L*M*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N, grid=_grid(L*M*N), block=_block())
            elif ploidy4[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc4D_autohex(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m41, m42, m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],
                                np.int32(O-1), L, M, N, MInt_gpu,
                                grid=_grid((O-1)*M*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu4, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5], O, L*M*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N, grid=_grid(L*M*N), block=_block())
            elif ploidy4[6]: # if 4+2 hexaploid - diploid subgenome
                kernels_poly._Vfunc(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc4D_hex_dip(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m41, m42, m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13],
                                np.int32(O-1), L, M, N, MInt_gpu,
                                grid=_grid((O-1)*M*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._include_bc_hex_dip(dx_gpu, nu4, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13], O, L*M*N,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N, grid=_grid(L*M*N), block=_block())
            elif ploidy4[9]: # if 2+2+2 hexaploid - subgenome c
                kernels_poly._Vfunc(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc4D_hex_c(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, m41, m42, m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],
                                            s4[12],s4[13],s4[14],s4[15],s4[16],s4[17],s4[18],s4[19],s4[20],s4[21],s4[22],s4[23],s4[24],s4[25],
                                            np.int32(O-1), L, M, N, MInt_gpu,
                                            grid=_grid((O-1)*M*L*N), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N), block=_block())
                kernels_poly._include_bc_hex_c(dx_gpu, nu4, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],
                                                s4[12],s4[13],s4[14],s4[15],s4[16],s4[17],s4[18],s4[19],s4[20],s4[21],s4[22],s4[23],s4[24],s4[25], 
                                                O, L*M*N, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N, grid=_grid(L*M*N), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, O,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M*N, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*M*N,O))
        phi_gpu, c_gpu = c_gpu.reshape(L,M*N*O), phi_gpu.reshape(L,M*N*O)

        current_t += this_dt
    Demes.cache.append(Demes.IntegrationNonConst(history = demes_hist, deme_ids=deme_ids))
    return phi_gpu.get().reshape(L,M,N,O)

def _five_pops_temporal_params(phi, xx, T, initial_t, nu1_f, nu2_f, nu3_f, nu4_f, nu5_f,
            m12_f, m13_f, m14_f, m15_f, m21_f, m23_f, m24_f, m25_f, m31_f, m32_f, m34_f, m35_f,
            m41_f, m42_f, m43_f, m45_f, m51_f, m52_f, m53_f, m54_f, 
            sel1_f, sel2_f, sel3_f, sel4_f, sel5_f,
            theta0_f, frozen1, frozen2, frozen3, frozen4, frozen5, deme_ids,
            ploidy1, ploidy2, ploidy3, ploidy4, ploidy5):
    if PolyInt.use_delj_trick:
        raise ValueError("delj trick not currently supported in CUDA execution")

    current_t = initial_t
    nu1, nu2, nu3, nu4, nu5 = nu1_f(current_t), nu2_f(current_t), nu3_f(current_t), nu4_f(current_t), nu5_f(current_t)
    s1, s2, s3, s4, s5 = sel1_f(current_t), sel2_f(current_t), sel3_f(current_t), sel4_f(current_t), sel5_f(current_t)
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
    demes_hist = [[0, [nu1,nu2,nu3,nu4,nu5], [m12,m13,m14,m15,m21,m23,m24,m25,m31,m32,m34,m35,m41,m42,m43,m45,m51,m52,m53,m54]]]
    while current_t < T:
        dt = min(PolyInt._compute_dt(dx, nu1, [m12,m13,m14,m15], s1, ploidy1),
                 PolyInt._compute_dt(dy, nu2, [m21,m23,m24,m25], s2, ploidy2),
                 PolyInt._compute_dt(dz, nu3, [m31,m32,m34,m35], s3, ploidy3),
                 PolyInt._compute_dt(da, nu4, [m41,m42,m43,m45], s4, ploidy4),
                 PolyInt._compute_dt(db, nu5, [m51,m52,m53,m54], s5, ploidy5))
        this_dt = np.float64(min(dt, T - current_t))

        next_t = current_t + this_dt

        nu1, nu2, nu3, nu4, nu5 = nu1_f(next_t), nu2_f(next_t), nu3_f(next_t), nu4_f(next_t), nu5_f(next_t)
        s1, s2, s3, s4, s5 = sel1_f(next_t), sel2_f(next_t), sel3_f(next_t), sel4_f(next_t), sel5_f(next_t)
        m12, m13, m14, m15 = m12_f(next_t), m13_f(next_t), m14_f(next_t), m15_f(next_t)
        m21, m23, m24, m25 = m21_f(next_t), m23_f(next_t), m24_f(next_t), m25_f(next_t)
        m31, m32, m34, m35 = m31_f(next_t), m32_f(next_t), m34_f(next_t), m35_f(next_t)
        m41, m42, m43, m45 = m41_f(next_t), m42_f(next_t), m43_f(next_t), m45_f(next_t)
        m51, m52, m53, m54 = m51_f(next_t), m52_f(next_t), m53_f(next_t), m54_f(next_t)
        theta0 = theta0_f(next_t)
        demes_hist.append([next_t, [nu1,nu2,nu3,nu4,nu5], [m12,m13,m14,m15,m21,m23,m24,m25,m31,m32,m34,m35,m41,m42,m43,m45,m51,m52,m53,m54]])

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
        kernels_poly._inject_mutations_5D_vals(phi_gpu, L,
                                          val00001, val00010, val00100, val01000, val10000, block=(1,1,1))
        # I can use the c_gpu buffer for the MInt_gpu buffer, to save GPU memory.
        # Note that I have to reassign this after each transpose operation I do.
        MInt_gpu = c_gpu
        if not frozen1:
            if ploidy1[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc5D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, m15, s1[0], s1[1],
                                np.int32(L-1), M, N, O, P, MInt_gpu,
                                grid=_grid((L-1)*M*N*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N*O*P), block=_block())
                kernels_poly._include_bc(dx_gpu, nu1, s1[0], s1[1], L, M*N*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N*O*P, grid=_grid(M*N*O*P), block=_block())
            elif ploidy1[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc5D_auto(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, m15, s1[0],s1[1],s1[2],s1[3],
                                np.int32(L-1), M, N, O, P, MInt_gpu,
                                grid=_grid((L-1)*M*N*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N*O*P), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3], L, M*N*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N*O*P, grid=_grid(M*N*O*P), block=_block())
            elif ploidy1[2]: # if allotetraploid subgenome a
                kernels_poly._Vfunc(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc5D_allo_a(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],
                                np.int32(L-1), M, N, O, P, MInt_gpu,
                                grid=_grid((L-1)*M*N*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N*O*P), block=_block())
                kernels_poly._include_bc_allo_a(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7], L, M*N*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N*O*P, grid=_grid(M*N*O*P), block=_block())
            elif ploidy1[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc5D_autohex(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],
                                np.int32(L-1), M, N, O, P, MInt_gpu,
                                grid=_grid((L-1)*M*N*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N*O*P), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5], L, M*N*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N*O*P, grid=_grid(M*N*O*P), block=_block())
            elif ploidy1[5]: # if 4+2 hexaploid - tetraploid subgenome
                kernels_poly._Vfunc_tetra(xx_gpu, nu1, L, V_gpu, 
                                grid=_grid(L), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu1, np.int32(L-1), VInt_gpu, 
                                grid=_grid(L-1), block=_block())
                kernels_poly._Mfunc5D_hex_tetra(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m12, m13, m14, m15, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13],
                                np.int32(L-1), M, N, O, P, MInt_gpu,
                                grid=_grid((L-1)*M*N*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((L-1)*M*N*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, L, M*N*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((L-1)*M*N*O*P), block=_block())
                kernels_poly._include_bc_hex_tetra(dx_gpu, nu1, s1[0],s1[1],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13], 
                                                   L, M*N*O*P, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, L, M*N*O*P, grid=_grid(M*N*O*P), block=_block())
            # note: no 2+2+2 hexaploid for the first or second dimension 
            # since we require these to be the last three dimensions of phi!

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, L,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                M*N*O*P, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(M*N*O*P,L))
        phi_gpu, c_gpu = c_gpu.reshape(M,L*N*O*P), phi_gpu.reshape(M,L*N*O*P)
        MInt_gpu = c_gpu
        if not frozen2:
            if ploidy2[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                # Note the order of the m arguments here.
                kernels_poly._Mfunc5D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m25, m21, s2[0], s2[1],
                                np.int32(M-1), N, O, P, L, MInt_gpu,
                                grid=_grid((M-1)*L*N*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O*P), block=_block())
                kernels_poly._include_bc(dx_gpu, nu2, s2[0], s2[1], M, L*N*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O*P, grid=_grid(L*N*O*P), block=_block())
            elif ploidy2[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc5D_auto(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m25, m21, s2[0],s2[1],s2[2],s2[3],
                                np.int32(M-1), N, O, P, L, MInt_gpu,
                                grid=_grid((M-1)*L*N*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O*P), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3], M, L*N*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O*P, grid=_grid(L*N*O*P), block=_block())
            elif ploidy2[3]: # if allotetraploid subgenome b
                kernels_poly._Vfunc(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc5D_allo_b(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m25, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],
                                np.int32(M-1), N, O, P, L, MInt_gpu,
                                grid=_grid((M-1)*L*N*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O*P), block=_block())
                kernels_poly._include_bc_allo_b(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7], M, L*N*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O*P, grid=_grid(L*N*O*P), block=_block())
            elif ploidy2[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc5D_autohex(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m25, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],
                                np.int32(M-1), N, O, P, L, MInt_gpu,
                                grid=_grid((M-1)*L*N*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O*P), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5], M, L*N*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O*P, grid=_grid(L*N*O*P), block=_block())
            elif ploidy2[6]: # if 4+2 hexaploid - diploid subgenome
                kernels_poly._Vfunc_hex(xx_gpu, nu2, M, V_gpu, 
                                grid=_grid(M), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu2, np.int32(M-1), VInt_gpu, 
                                grid=_grid(M-1), block=_block())
                kernels_poly._Mfunc5D_autohex(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m23, m24, m25, m21, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13],
                                np.int32(M-1), N, O, P, L, MInt_gpu,
                                grid=_grid((M-1)*L*N*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((M-1)*L*N*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, M, L*N*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((M-1)*L*N*O*P), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu2, s2[0],s2[1],s2[2],s2[3],s2[4],s2[5],s2[6],s2[7],s2[8],s2[9],s2[10],s2[11],s2[12],s2[13], 
                                                 M, L*N*O*P, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, M, L*N*O*P, grid=_grid(L*N*O*P), block=_block())
            # similar to the above, we don't support any subgenome 
            # of a 2+2+2 hexaploid for the first two dimensions of phi

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, M,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*N*O*P, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*N*O*P,M))
        phi_gpu, c_gpu = c_gpu.reshape(N,L*M*O*P), phi_gpu.reshape(N,L*M*O*P)
        MInt_gpu = c_gpu
        if not frozen3:
            if ploidy3[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc5D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m35, m31, m32, s3[0],s3[1],
                                np.int32(N-1), O, P, L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M*O*P), block=_block())
                kernels_poly._include_bc(dx_gpu, nu3, s3[0],s3[1], N, L*M*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M*O*P, grid=_grid(L*M*O*P), block=_block())
            elif ploidy3[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc5D_auto(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m35, m31, m32, s3[0],s3[1],s3[2],s3[3],
                                np.int32(N-1), O, P, L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M*O*P), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3], N, L*M*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M*O*P, grid=_grid(L*M*O*P), block=_block())
            # note that we only support allotetraploids as the first pair or last pair of dimensions of phi
            # so, no code for them or 4+2 hexaploids here
            elif ploidy3[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc5D_autohex(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m35, m31, m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],
                                np.int32(N-1), O, P, L, M, MInt_gpu,
                                grid=_grid((N-1)*M*L*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M*O*P), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5], N, L*M*O*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M*O*P, grid=_grid(L*M*O*P), block=_block())
            elif ploidy3[7]: # if 2+2+2 hexaploid - subgenome a
                kernels_poly._Vfunc(xx_gpu, nu3, N, V_gpu, 
                                grid=_grid(N), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu3, np.int32(N-1), VInt_gpu, 
                                grid=_grid(N-1), block=_block())
                kernels_poly._Mfunc5D_hex_a(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m34, m35, m31, m32, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],
                                            s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25],
                                            np.int32(N-1), O, P, L, M, MInt_gpu,
                                            grid=_grid((N-1)*M*L*O*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O*P,
                    a_gpu, b_gpu,
                    grid=_grid((N-1)*L*M*O*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, N, L*M*O*P,
                    b_gpu, c_gpu,
                    grid=_grid((N-1)*L*M*O*P), block=_block())
                kernels_poly._include_bc_hex_a(dx_gpu, nu3, s3[0],s3[1],s3[2],s3[3],s3[4],s3[5],s3[6],s3[7],s3[8],s3[9],s3[10],s3[11],
                                                s3[12],s3[13],s3[14],s3[15],s3[16],s3[17],s3[18],s3[19],s3[20],s3[21],s3[22],s3[23],s3[24],s3[25], 
                                                N, L*M*O*P, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, N, L*M*O*P, grid=_grid(L*M*O*P), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, N,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M*O*P, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*M*O*P,N))
        phi_gpu, c_gpu = c_gpu.reshape(O,L*M*N*P), phi_gpu.reshape(O,L*M*N*P)
        MInt_gpu = c_gpu
        if not frozen4:
            if ploidy4[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc5D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m45, m41, m42, m43, s4[0],s4[1],
                                np.int32(O-1), P, L, M, N, MInt_gpu,
                                grid=_grid((O-1)*M*L*N*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._include_bc(dx_gpu, nu4, s4[0],s4[1], O, L*M*N*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N*P, grid=_grid(L*M*N*P), block=_block())
            elif ploidy4[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc5D_auto(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m45, m41, m42, m43, s4[0],s4[1],s4[2],s4[3],
                                np.int32(O-1), P, L, M, N, MInt_gpu,
                                grid=_grid((O-1)*M*L*N*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu4, s4[0],s4[1],s4[2],s4[3], O, L*M*N*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N*P, grid=_grid(L*M*N*P), block=_block())
            elif ploidy4[2]: # if allotetraploid subgenome a
                kernels_poly._Vfunc(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc5D_allo_a(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m45, m41, m42, m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],
                                np.int32(O-1), P, L, M, N, MInt_gpu,
                                grid=_grid((O-1)*M*L*N*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._include_bc_allo_a(dx_gpu, nu4, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7], O, L*M*N*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N*P, grid=_grid(L*M*N*P), block=_block())
            elif ploidy4[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc5D_autohex(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m45, m41, m42, m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],
                                np.int32(O-1), P, L, M, N, MInt_gpu,
                                grid=_grid((O-1)*M*L*N*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu4, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5], O, L*M*N*P,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N*P, grid=_grid(L*M*N*P), block=_block())
            elif ploidy4[5]: # if 4+2 hexaploid - tetraploid subgenome
                kernels_poly._Vfunc_tetra(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc5D_hex_tetra(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m45, m41, m42, m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13],
                                np.int32(O-1), P, L, M, N, MInt_gpu,
                                grid=_grid((O-1)*M*L*N*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._include_bc_hex_tetra(dx_gpu, nu4, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],s4[12],s4[13], 
                                                   O, L*M*N*P, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N*P, grid=_grid(L*M*N*P), block=_block())
            elif ploidy4[8]: # if 2+2+2 hexaploid - subgenome b
                kernels_poly._Vfunc(xx_gpu, nu4, O, V_gpu, 
                                grid=_grid(O), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu4, np.int32(O-1), VInt_gpu, 
                                grid=_grid(O-1), block=_block())
                kernels_poly._Mfunc5D_hex_b(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m45, m41, m42, m43, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],
                                            s4[12],s4[13],s4[14],s4[15],s4[16],s4[17],s4[18],s4[19],s4[20],s4[21],s4[22],s4[23],s4[24],s4[25],
                                            np.int32(O-1), P, L, M, N, MInt_gpu,
                                            grid=_grid((O-1)*M*L*N*P), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    a_gpu, b_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, O, L*M*N*P,
                    b_gpu, c_gpu,
                    grid=_grid((O-1)*L*M*N*P), block=_block())
                kernels_poly._include_bc_hex_b(dx_gpu, nu4, s4[0],s4[1],s4[2],s4[3],s4[4],s4[5],s4[6],s4[7],s4[8],s4[9],s4[10],s4[11],
                                                s4[12],s4[13],s4[14],s4[15],s4[16],s4[17],s4[18],s4[19],s4[20],s4[21],s4[22],s4[23],s4[24],s4[25],
                                                O, L*M*N*P, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, O, L*M*N*P, grid=_grid(L*M*N*P), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, O,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M*N*P, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*M*N*P,O))
        phi_gpu, c_gpu = c_gpu.reshape(P,L*M*N*O), phi_gpu.reshape(P,L*M*N*O)
        MInt_gpu = c_gpu
        if not frozen5:
            if ploidy5[0]: # if diploid
                kernels_poly._Vfunc(xx_gpu, nu5, P, V_gpu, 
                                grid=_grid(P), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu5, np.int32(P-1), VInt_gpu, 
                                grid=_grid(P-1), block=_block())
                kernels_poly._Mfunc5D(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m51, m52, m53, m54, s5[0], s5[1],
                                np.int32(P-1), L, M, N, O, MInt_gpu,
                                grid=_grid((P-1)*L*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._include_bc(dx_gpu, nu5, s5[0], s5[1], P, L*M*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, P, L*M*N*O, grid=_grid(L*M*N*O), block=_block())
            elif ploidy5[1]: # if autotetraploid
                kernels_poly._Vfunc_tetra(xx_gpu, nu5, P, V_gpu, 
                                grid=_grid(P), block=_block())
                kernels_poly._Vfunc_tetra(xInt_gpu, nu5, np.int32(P-1), VInt_gpu, 
                                grid=_grid(P-1), block=_block())
                kernels_poly._Mfunc5D_auto(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m51, m52, m53, m54, s5[0],s5[1],s5[2],s5[3],
                                np.int32(P-1), L, M, N, O, MInt_gpu,
                                grid=_grid((P-1)*L*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._include_bc_auto(dx_gpu, nu5, s5[0],s5[1],s5[2],s5[3], P, L*M*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, P, L*M*N*O, grid=_grid(L*M*N*O), block=_block())
            elif ploidy5[3]: # if allotetraploid subgenome b
                kernels_poly._Vfunc(xx_gpu, nu5, P, V_gpu, 
                                grid=_grid(P), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu5, np.int32(P-1), VInt_gpu, 
                                grid=_grid(P-1), block=_block())
                kernels_poly._Mfunc5D_allo_b(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m51, m52, m53, m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],
                                np.int32(P-1), L, M, N, O, MInt_gpu,
                                grid=_grid((P-1)*L*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._include_bc_allo_b(dx_gpu, nu5, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7], P, L*M*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, P, L*M*N*O, grid=_grid(L*M*N*O), block=_block())
            elif ploidy5[4]: # if autohexaploid
                kernels_poly._Vfunc_hex(xx_gpu, nu5, P, V_gpu, 
                                grid=_grid(P), block=_block())
                kernels_poly._Vfunc_hex(xInt_gpu, nu5, np.int32(P-1), VInt_gpu, 
                                grid=_grid(P-1), block=_block())
                kernels_poly._Mfunc5D_autohex(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m51, m52, m53, m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],
                                np.int32(P-1), L, M, N, O, MInt_gpu,
                                grid=_grid((P-1)*L*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._include_bc_autohex(dx_gpu, nu5, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5], P, L*M*N*O,
                    b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, P, L*M*N*O, grid=_grid(L*M*N*O), block=_block())
            elif ploidy5[6]: # if 4+2 hexaploid - diploid subgenome
                kernels_poly._Vfunc(xx_gpu, nu5, P, V_gpu, 
                                grid=_grid(P), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu5, np.int32(P-1), VInt_gpu, 
                                grid=_grid(P-1), block=_block())
                kernels_poly._Mfunc5D_hex_dip(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m51, m52, m53, m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],s5[11],s5[12],s5[13],
                                np.int32(P-1), L, M, N, O, MInt_gpu,
                                grid=_grid((P-1)*L*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._include_bc_hex_dip(dx_gpu, nu5, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],s5[11],s5[12],s5[13],
                                                  P, L*M*N*O, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, P, L*M*N*O, grid=_grid(L*M*N*O), block=_block())
            elif ploidy5[9]: # if 2+2+2 hexaploid - subgenome c
                kernels_poly._Vfunc(xx_gpu, nu5, P, V_gpu, 
                                grid=_grid(P), block=_block())
                kernels_poly._Vfunc(xInt_gpu, nu5, np.int32(P-1), VInt_gpu, 
                                grid=_grid(P-1), block=_block())
                kernels_poly._Mfunc5D_hex_c(xInt_gpu, xx_gpu, xx_gpu, xx_gpu, xx_gpu, m51, m52, m53, m54, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],s5[11],
                                            s5[12],s5[13],s5[14],s5[15],s5[16],s5[17],s5[18],s5[19],s5[20],s5[21],s5[22],s5[23],s5[24],s5[25],
                                            np.int32(P-1), L, M, N, O, MInt_gpu,
                                            grid=_grid((P-1)*L*M*N*O), block=_block())

                b_gpu.fill(1./this_dt)
                kernels_poly._compute_ab_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    a_gpu, b_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._compute_bc_nobc(dx_gpu, dfactor_gpu, 
                    MInt_gpu, V_gpu, this_dt, P, L*M*N*O,
                    b_gpu, c_gpu,
                    grid=_grid((P-1)*L*M*N*O), block=_block())
                kernels_poly._include_bc_hex_c(dx_gpu, nu5, s5[0],s5[1],s5[2],s5[3],s5[4],s5[5],s5[6],s5[7],s5[8],s5[9],s5[10],s5[11],
                                                s5[12],s5[13],s5[14],s5[15],s5[16],s5[17],s5[18],s5[19],s5[20],s5[21],s5[22],s5[23],s5[24],s5[25],
                                                P, L*M*N*O, b_gpu, block=(1,1,1))
                kernels_poly._cx0(c_gpu, P, L*M*N*O, grid=_grid(L*M*N*O), block=_block())

            phi_gpu /= this_dt

            cusparseDgtsvInterleavedBatch(cusparse_handle, 0, P,
                a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, phi_gpu.gpudata,
                L*M*N*O, pBuffer)

        transpose_gpuarray(phi_gpu, c_gpu.reshape(L*M*N*O,P))
        phi_gpu, c_gpu = c_gpu.reshape(L,M*N*O*P), phi_gpu.reshape(L,M*N*O*P)

        current_t += this_dt
    Demes.cache.append(Demes.IntegrationNonConst(history = demes_hist, deme_ids=deme_ids))
    return phi_gpu.get().reshape(L,M,N,O,P)

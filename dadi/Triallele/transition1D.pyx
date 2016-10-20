## cythonize transition1D
import numpy as np
cimport numpy as np

def transition1D(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] dx,sig):
    cdef int ii
    cdef np.ndarray[np.float64_t, ndim=2] PV = np.zeros((len(x),len(x)))
    cdef np.ndarray[np.float64_t, ndim=2] PM = np.zeros((len(x),len(x)))
    for ii in range(len(x)):
        if ii == 0:
            PV[ii,ii] = 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii+1] - x[ii])
            PV[ii,ii+1] = -1./2 * 1./dx[ii] * (x[ii+1]*(1-x[ii+1]))/(x[ii+1] - x[ii])
            PM[ii,ii+1] = sig / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
        elif ii == len(x) - 1:
            PV[ii,ii-1] = - 1./2 * 1./dx[ii] * (x[ii-1]*(1-x[ii-1]))/(x[ii] - x[ii-1])
            PM[ii,ii-1] = sig / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            PV[ii,ii] = 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii] - x[ii-1])
        else:
            PV[ii,ii-1] = - 1./2 * 1./dx[ii] * (x[ii-1]*(1-x[ii-1]))/(x[ii] - x[ii-1])
            PM[ii,ii-1] = - sig / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            PV[ii,ii] = 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii] - x[ii-1]) + 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii+1] - x[ii])
            PV[ii,ii+1] = - 1./2 * 1./dx[ii] * (x[ii+1]*(1-x[ii+1]))/(x[ii+1] - x[ii])
            PM[ii,ii+1] = sig / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
    return PV,PM

## cythonize transition1D
import numpy as np
cimport numpy as np

def transition1D(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] dx, dt, gamma, nu):
    cdef int ii
    cdef np.ndarray[np.float64_t, ndim=2] P = np.zeros((len(x),len(x)))
    for ii in range(len(x)):
        if ii == 0:
            P[ii,ii] = 1 + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii+1] - x[ii])
            P[ii,ii+1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii+1]*(1-x[ii+1]))/(x[ii+1] - x[ii]) + gamma * dt / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
        elif ii == len(x) - 1:
            P[ii,ii-1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii-1]*(1-x[ii-1]))/(x[ii] - x[ii-1]) - gamma * dt / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            P[ii,ii] = 1 + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii] - x[ii-1])
        else:
            P[ii,ii-1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii-1]*(1-x[ii-1]))/(x[ii] - x[ii-1]) - gamma * dt / 2 / dx[ii] * x[ii-1] * (1-x[ii-1])
            P[ii,ii] = 1 + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii] - x[ii-1]) + dt/nu * 1./2 * 1./dx[ii] * (x[ii]*(1-x[ii]))/(x[ii+1] - x[ii])
            P[ii,ii+1] = - dt/nu * 1./2 * 1./dx[ii] * (x[ii+1]*(1-x[ii+1]))/(x[ii+1] - x[ii]) + gamma * dt / 2 / dx[ii] * x[ii+1] * (1-x[ii+1])
    return P


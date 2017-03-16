import numpy as np
cimport numpy as np

def surface_interaction(np.ndarray[np.float64_t, ndim=3] phi, np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=3] Psurf):
    cdef int ii
    cdef int jj
    cdef int kk
    cdef np.float64_t amnt
    cdef int dist
    cdef int rmdr
    
    for ii in range(len(x))[:len(x)-1]:
        for jj in range(len(x))[:len(x)-1-ii]:
            for kk in range(len(x))[:len(x)-1-ii-jj]:
                if ii == 0 and jj == 0:
                    continue
                if ii == 0 and kk == 0:
                    continue
                if jj == 0 and kk == 0:
                    continue
                if ii == 0:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (jj+kk))/2
                    rmdr = (len(x)-1 - (jj+kk))%2
                    if rmdr == 0:
                        phi[ii,jj+dist,kk+dist] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii,jj+dist+1,kk+dist] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj+dist,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj,kk] *= (1-amnt)
                elif jj == 0:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (ii+kk))/2
                    rmdr = (len(x)-1 - (ii+kk))%2
                    if rmdr == 0:
                        phi[ii+dist,jj,kk+dist] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii+dist+1,jj,kk+dist] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii+dist,jj,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj,kk] *= (1-amnt)
                elif kk == 0:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (ii+jj))/2
                    rmdr = (len(x)-1 - (ii+jj))%2
                    if rmdr == 0:
                        phi[ii+dist,jj+dist,kk] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii+dist+1,jj+dist,kk] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii+dist,jj+dist+1,kk] += amnt * phi[ii,jj,kk] * 2/2.
                        phi[ii,jj,kk] *= (1-amnt)
                else:
                    amnt = Psurf[ii,jj,kk]
                    dist = (len(x)-1 - (ii+jj+kk))/3
                    rmdr = (len(x)-1 - (ii+jj+kk))%3
                    if rmdr == 0: # in line with boundary grid point
                        phi[ii+dist,jj+dist,kk+dist] += amnt * phi[ii,jj,kk] * 2
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 1:
                        phi[ii+dist+1,jj+dist,kk+dist] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist,jj+dist+1,kk+dist] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist,jj+dist,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii,jj,kk] *= (1-amnt)
                    elif rmdr == 2:
                        phi[ii+dist+1,jj+dist+1,kk+dist] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist+1,jj+dist,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii+dist,jj+dist+1,kk+dist+1] += amnt * phi[ii,jj,kk] * 2/3.
                        phi[ii,jj,kk] *= (1-amnt)

    return phi

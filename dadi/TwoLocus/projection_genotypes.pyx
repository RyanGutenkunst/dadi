import numpy as np
cimport numpy as np
import math

def ln_binomial(int n,int k):
    return math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1) 

def projection_genotypes(int n_from, int n_to, hits):
    cdef int g1
    cdef int g2
    cdef int g3
    cdef int g4
    cdef int g5
    cdef int g6
    cdef int g7
    cdef int g8
    cdef int g9
    cdef int o1
    cdef int o2
    cdef int o3
    cdef int o4
    cdef int o5
    cdef int o6
    cdef int o7
    cdef int o8
    cdef int o9

    g1,g2,g3,g4,g5,g6,g7,g8 = hits
    g9 = n_from - g1 - g2 - g3 - g4 - g5 - g6 - g7 - g8
    weights_to = {}
    for o1 in range(0,g1+1): # AABB
        for o2 in range(0,g2+1): # AABb
            for o3 in range(0,g3+1): # AAbb
                for o4 in range(0,g4+1): # AaBB
                    for o5 in range(0,g5+1): # AaBb
                        for o6 in range(0,g6+1): # Aabb
                            for o7 in range(0,g7+1): # aaBB
                                for o8 in range(0,g8+1): # aaBb
                                    o9 = n_to-o1-o2-o3-o4-o5-o6-o7-o8 # aabb
                                    if o9 < 0 or o9 > g9:
                                        continue
                                    else:
                                        p = 2*o1 + 2*o2 + 2*o3 + o4 + o5 + o6
                                        q = 2*o1 + o2 + 2*o4 + o5 + 2*o7 + o8
                                        if p == 0 or q == 0 or p == n_to or q == n_to:
                                            continue
                                        else:
                                            weight = np.exp(ln_binomial(g1,o1) + ln_binomial(g2,o2) + ln_binomial(g3,o3) + ln_binomial(g4,o4) + ln_binomial(g5,o5) + ln_binomial(g6,o6) + ln_binomial(g7,o7) + ln_binomial(g8,o8) + ln_binomial(g9,o9) - ln_binomial(n_from,n_to))
                                            weights_to[(o1,o2,o3,o4,o5,o6,o7,o8)] = weight

    return weights_to

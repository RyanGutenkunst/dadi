import numpy
from numpy import newaxis as nuax

from scipy import comb
from scipy.special import gammaln

import Numerics
from Numerics import reverse_array, trapz
from scipy.integrate import trapz

def sfs_from_phi_1D(n, xx, phi):
    sfs = numpy.zeros(n+1)
    for ii in range(0,n+1):
        factorx = comb(n,ii) * xx**ii * (1-xx)**(n-ii)
        sfs[ii] = trapz(factorx * phi, xx)

    return sfs

def sfs_from_phi_2D(nx, ny, xx, yy, phi):
    # Calculate the 2D sfs from phi using the trapezoid rule for integration.
    sfs = numpy.zeros((nx+1, ny+1))
    
    # Cache to avoid duplicated work.
    factorx_cache = {}
    for ii in range(0, nx+1):
        factorx = comb(nx, ii) * xx**ii * (1-xx)**(nx-ii)
        factorx_cache[nx,ii] = factorx

    dx, dy = numpy.diff(xx), numpy.diff(yy)
    for jj in range(0,ny+1):
        factory = comb(ny, jj) * yy**jj * (1-yy)**(ny-jj)
        integrated_over_y = trapz(factory[numpy.newaxis,:]*phi, dx=dy)
        for ii in range(0, nx+1):
            factorx = factorx_cache[nx,ii]
            sfs[ii,jj] = trapz(factorx*integrated_over_y, dx=dx)

    return sfs

def sfs_from_phi_3D(nx, ny, nz, xx, yy, zz, phi):
    sfs = numpy.zeros((nx+1, ny+1, nz+1))

    dx, dy, dz = numpy.diff(xx), numpy.diff(yy), numpy.diff(zz)
    half_dx = dx/2.0

    # We cache these calculations...
    factorx_cache, factory_cache = {}, {}
    for ii in range(0, nx+1):
        factorx = comb(nx, ii) * xx**ii * (1-xx)**(nx-ii)
        factorx_cache[nx,ii] = factorx
    for jj in range(0, ny+1):
        factory = comb(ny, jj) * yy**jj * (1-yy)**(ny-jj)
        factory_cache[ny,jj] = factory[nuax,:]

    for kk in range(0, nz+1):
        factorz = comb(nz, kk) * zz**kk * (1-zz)**(nz-kk)
        over_z = trapz(factorz[nuax, nuax,:] * phi, dx=dz)
        for jj in range(0, ny+1):
            factory = factory_cache[ny,jj]
            over_y = trapz(factory * over_z, dx=dy)
            for ii in range(0, nx+1):
                factorx = factorx_cache[nx,ii]
                # It's faster here to do the trapezoid rule explicitly rather
                #  than using SciPy's more general routine.
                integrand = factorx * over_y
                ans = numpy.sum(half_dx * (integrand[1:]+integrand[:-1]))
                sfs[ii,jj,kk] = ans

    return sfs

def optimally_scaled_sfs(model, data):
    """
    Optimially scale model sfs to data sfs.

    Returns a new scaled model sfs.
    """
    return optimal_sfs_scaling(model,data) * model

def optimal_sfs_scaling(model, data):
    """
    Optimal multiplicative scaling factor between model and data.

    This scaling is based on only those entries that are masked in neither
    model nor data.
    """
    model, data = Numerics.intersect_masks(model, data)
    return data.sum()/model.sum()

def randomly_resampled_2D(sfs):
    """
    Randomly scramble individuals among the populations. 
    
    This is useful for measuring divergence. Essentially, this method pools all
    the individuals represented in the sfs and generates two new populations of
    random individuals (without replacement) from that pool.
    """
    n1,n2 = numpy.asarray(sfs.shape)-1
    ntot = n1+n2

    # First generate a 1d sfs for the pooled population.
    sfs_combined = numpy.zeros(ntot+1)
    for ii, row in enumerate(sfs):
        for jj, num_snps in enumerate(row):
            derived_total = ii+jj
            # This isscalar check deals with masking
            if numpy.isscalar(num_snps):
                sfs_combined[derived_total] += num_snps

    # Now resample
    sfs_resamp = numpy.zeros(sfs.shape)
    for derived1 in range(n1+1):
        for derived_total, num_snps in enumerate(sfs_combined):
            derived2 = derived_total - derived1
            ancestral_total = ntot - derived_total
            ancestral1 = n1 - derived1

            prob = numpy.exp(lncomb(derived_total, derived1)
                             + lncomb(ancestral_total, ancestral1)
                             - lncomb(ntot, n1))
            if prob > 0:
                sfs_resamp[derived1, derived2] += prob*num_snps
    return sfs_resamp

def mask_corners(sfs):
    """ 
    Return a masked SFS in which the 'absent in all pops' and 'fixed in all
    pops' entries are masked. These entries are often unobservable.
    """
    mask = numpy.ma.make_mask_none(sfs.shape)
    mask.flat[0] = mask.flat[-1] = True
    sfs = numpy.ma.masked_array(sfs, mask=mask)

    return sfs

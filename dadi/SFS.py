import numpy
from numpy import newaxis as nuax

from scipy import comb

import Numerics
from Numerics import reverse_array, trapz
from scipy.integrate import trapz

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

import numpy
from numpy import newaxis as nuax

from scipy import comb
from scipy.special import gammaln

import Numerics
from Numerics import reverse_array, trapz
from scipy.integrate import trapz

projection_cache = {}
def cached_projection(proj_to, proj_from, hits):
    key = (proj_to, proj_from, hits)
    try:
        return projection_cache[key]
    except KeyError:
        proj_hits = numpy.arange(proj_to+1)
        contrib = comb(proj_to,proj_hits)*comb(proj_from-proj_to,hits-proj_hits)
        contrib /= comb(proj_from, hits)
        projection_cache[key] = contrib
        return contrib

def project_sfs_3D(sfs, n1, n2, n3):
    if (n1 > sfs.shape[0]-1) or (n2 > sfs.shape[1]-1) or (n3 > sfs.shape[2]-1):
        raise ValueError('Cannot project to a sample size greater than '
                         'original. Called sizes were from %s to %s.' 
                         % (tuple(numpy.asarray(sfs.shape)-1), (n1,n2,n3)))
    projected_sfs = numpy.zeros((n1+1, n2+1, n3+1))

    do_masking = numpy.ma.isMaskedArray(sfs)
    if do_masking:
        mask = numpy.zeros(projected_sfs.shape)
        projected_sfs = numpy.ma.masked_array(projected_sfs, mask=mask)

    from1, from2, from3 = numpy.asarray(sfs.shape) - 1
    for hits1 in range(from1+1):
        # These are the least and most possible hits we could have in the
        #  projected sfs.
        least1, most1 = max(n1 - (from1 - hits1), 0), min(hits1,n1)
        proj1 = cached_projection(n1, from1, hits1)[least1:most1+1]

        for hits2 in range(from2+1):
            least2, most2 = max(n2 - (from2 - hits2), 0), min(hits2,n2)
            proj2 = cached_projection(n2, from2, hits2)[least2:most2+1]

            temp = proj1[:,nuax,nuax] * proj2[nuax,:,nuax]
            for hits3 in range(from3+1):
                least3, most3 = max(n3 - (from3 - hits3), 0), min(hits3,n3)
                if do_masking:
                    if  sfs.mask[hits1,hits2,hits3]:
                        projected_sfs.mask[least1:most1+1,least2:most2+1,
                                           least3:most3+1] = True
                        continue
                    current_mask = projected_sfs.mask[least1:most1+1,
                                                      least2:most2+1,
                                                      least3:most3+1]
                    if  numpy.all(current_mask):
                        continue
                if sfs[hits1,hits2,hits3] == 0:
                    continue
                proj3 = cached_projection(n3, from3, hits3)[least3:most3+1]
                projected_sfs[least1:most1+1,least2:most2+1,least3:most3+1] \
                        += sfs[hits1,hits2,hits3] * temp * proj3[nuax,nuax,:]

    return projected_sfs

def project_sfs_2D(sfs, n1, n2):
    projected_sfs = numpy.zeros((n1+1, n2+1))

    do_masking = numpy.ma.isMaskedArray(sfs)
    if do_masking:
        mask = numpy.zeros(projected_sfs.shape)
        projected_sfs = numpy.ma.masked_array(projected_sfs, mask=mask)

    from1, from2 = numpy.asarray(sfs.shape) - 1
    for hits1 in range(from1+1):
        # These are the least and most possible hits we could have in the
        #  projected sfs.
        least1, most1 = max(n1 - (from1 - hits1), 0), min(hits1,n1)
        proj1 = cached_projection(n1, from1, hits1)[least1:most1+1]

        for hits2 in range(from2+1):
            least2, most2 = max(n2 - (from2 - hits2), 0), min(hits2,n2)
            if do_masking:
                if  sfs.mask[hits1,hits2]:
                    projected_sfs.mask[least1:most1+1,least2:most2+1] = True
                    continue
                current_mask = projected_sfs.mask[least1:most1+1,least2:most2+1]
                if numpy.all(current_mask):
                    # This is a work-around for a numpy.ma bug.
                    continue
            if sfs[hits1,hits2] == 0:
                continue
            proj2 = cached_projection(n2, from2, hits2)[least2:most2+1]

            projected_sfs[least1:most1+1,least2:most2+1] \
                    += sfs[hits1,hits2] * proj1[:,nuax] * proj2[nuax,:]

    return projected_sfs

def project_sfs_1D(sfs, n):
    projected_sfs = numpy.zeros(n+1)

    do_masking = numpy.ma.isMaskedArray(sfs)
    if do_masking:
        mask = numpy.zeros(projected_sfs.shape)
        projected_sfs = numpy.ma.masked_array(projected_sfs, mask=mask)

    proj_from = sfs.shape[0] - 1
    for hits in range(proj_from+1):
        proj = cached_projection(n, proj_from, hits)
        # These are the least and most possible hits we could have in the
        #  projstrapped sfs.
        least, most = max(n - (proj_from - hits), 0), min(hits,n)
        if do_masking:
            if  sfs.mask[hits]:
                projected_sfs.mask[least:most+1] = True
                continue
            current_mask = projected_sfs.mask[least:most+1]
            if numpy.all(current_mask):
                continue
        if sfs[hits] == 0:
                continue
        projected_sfs[least:most+1] += sfs[hits] * proj[least:most+1]

    return projected_sfs

def project_sfs(sfs, ns):
    if len(sfs.shape) != len(ns):
        raise ValueError('Requested projection ns and sfs have different '
                         'numbers of dimensions. (%i vs. %i)'
                         % (len(ns), len(sfs.shape)))
    if len(sfs.shape) == 1:
        return project_sfs_1D(sfs, ns[0])
    elif len(sfs.shape) == 2:
        return project_sfs_2D(sfs, ns[0], ns[1])
    elif len(sfs.shape) == 3:
        return project_sfs_3D(sfs, ns[0], ns[1], ns[2])
    else:
        raise ValueError('Projection only supported for spectra of dimension '
                         '3 or fewer at this time.')

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

def fold_sfs(sfs):
    """
    The folded site-frequency spectrum for an n-dimensional sfs.

    The folded SFS assumes that information on which allele is ancestral or
    derived is unavailable. Thus the sfs is in terms of minor allele frequency.
    Note that this makes the sfs into a "triangular" array, hence we use a
    masked array to represent it.

    Returns a masked array containing the folded sfs.

    Note that if a masked cell is folded into non-masked cell, the destination
    cell is masked as well.
    """
    # How many samples total do we have? The folded sfs can only contain
    # entries up to total_samples/2 (rounded down).
    total_samples = numpy.sum(sfs.shape) - sfs.ndim

    # This next chunk of vodoo creates an array 'total_per_cell' in which each
    # element is just the total number of segregating alleles in that cell. This
    # is used to indicate which entries we need to 'fold out'. It looks like
    # voodoo so it can handle arrays of any dimension.
    indices = [range(ni) for ni in sfs.shape]
    new_shapes = [[1] for ni in sfs.shape]
    for ii,ni in enumerate(sfs.shape):
        new_shape = numpy.ones(sfs.ndim)
        new_shape[ii] = ni
        indices[ii] = numpy.reshape(indices[ii], new_shape)
    import operator
    total_per_cell = reduce(operator.add, indices)

    # Here's where we calculate which entries are nonsense in the folded SFS.
    where_folded_out = total_per_cell > int(total_samples/2)

    # This is a fancy way of creating a ::-1,::-1 slice with as many dimensions
    #  as I need.
    reverse_slice = [slice(None, None, -1) for ii in sfs.shape]

    if numpy.ma.isMaskedArray(sfs):
        original_mask = sfs.mask
        # Here we create a mask that masks any values that were masked in the
        # original SFS (or folded onto by a masked value).
        final_mask = numpy.logical_or(original_mask, 
                                      reverse_array(original_mask))
        sfs = sfs.data
    else:
        final_mask = numpy.zeros(sfs.shape, numpy.bool_)
    
    # To do the actual folding, we take those entries that would be folded out,
    # reverse the array along all axes, and add them back to the original sfs.
    reversed = reverse_array(numpy.where(where_folded_out, sfs, 0))
    folded = sfs + reversed

    # Here's where we calculate which entries are nonsense in the folded SFS.
    where_ambiguous = total_per_cell == int(total_samples/2)
    ambiguous = numpy.where(where_ambiguous, sfs, 0)
    folded += -0.5*ambiguous + 0.5*reverse_array(ambiguous)

    # Mask out the remains of the folding operation.
    final_mask = numpy.logical_or(final_mask, where_folded_out)
    folded = numpy.ma.masked_array(folded, mask=final_mask)

    return folded

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

# Create a version of the gamma function that will work with masked arrays.
gammaln_m = numpy.ma.masked_unary_operation(gammaln)
def minus_ll(model, data):
    """
    The negative of the log-likelihood of the data given the model sfs.

    Return a double that is -(log-likelihood)
    """
    return -ll(model, data)

def ll(model, data):
    """
    The log-likelihood of the data given the model sfs.

    Evaluate the log-likelihood of the data given the model. This is based on
    Poisson statistics, where the probability of observing k entries in a cell
    given that the mean number is given by the model is 
    P(k) = exp(-model) * model**k / k!

    Note: If either the model or the data is a masked array, the return ll will
          ignore any elements that are masked in *either* the model or the data.
    """
    ll_arr = ll_per_bin(model, data)
    return ll_arr.sum()

def ll_per_bin(model, data):
    """
    The Poisson log-likelihood of each entry in the data given the model sfs.
    """
    return -model + data*numpy.log(model) - gammaln_m(data + 1)

def ll_multinom_per_bin(model, data):
    """
    Mutlinomial log-likelihood of each entry in the data given the model.

    Scales the model sfs to have the optimal theta for comparison with the data.
    """
    theta_opt = optimal_sfs_scaling(model, data)
    return ll_per_bin(theta_opt*model, data)

def ll_multinom(model, data):
    """
    Log-likelihood of the data given the model, with optimal rescaling.

    Evaluate the log-likelihood of the data given the model. This is based on
    Poisson statistics, where the probability of observing k entries in a cell
    given that the mean number is given by the model is 
    P(k) = exp(-model) * model**k / k!

    model is optimally scaled to maximize ll before calculation.

    Note: If either the model or the data is a masked array, the return ll will
          ignore any elements that are masked in *either* the model or the data.
    """
    ll_arr = ll_multinom_per_bin(model, data)
    return ll_arr.sum()

def minus_ll_multinom(model, data):
    """
    The negative of the log-likelihood of the data given the model sfs.

    Return a double that is -(log-likelihood)
    """
    return -ll_multinom(model, data)

def linear_Poisson_residual(model, data, mask=0):
    """
    Return the Poisson residuals, (data - model)/sqrt(model), of model and data.

    mask sets the level in model below which the returned residual array is
    masked. The default of 0 excludes values where the residuals are not 
    defined.

    In the limit that the mean of the Poisson distribution is large, these
    residuals are normally distributed. (If the mean is small, the Anscombe
    residuals are better.)
    """
    resid = (data - model)/numpy.sqrt(model)
    if numpy.isscalar(mask):
        resid = numpy.ma.masked_where(model <= mask, resid)
    return resid

def Anscombe_Poisson_residual(model, data, mask=1e-2):
    """
    Return the Anscombe Poisson residuals between model and data.

    mask sets the level in model below which the returned residual array is
    masked. This excludes very small values where the residuals are not normal.
    1e-2 seems to be a good default for the NIEHS human data. (model = 1e-2,
    data = 0, yields a residual of ~1.5.)

    Residuals defined in this manner are more normally distributed than the
    linear residuals when the mean is small. See this reference below for
    justification: Pierce DA and Schafer DW, "Residuals in generalized linear
    models" Journal of the American Statistical Association, 81(396)977-986
    (1986).

    Note that I tried implementing the "adjusted deviance" residuals, but they
    always looked like crap for the cases where the data was 0.
    """
    resid = 1.5*(data**(2./3) - (model**(2./3)-model**(-1./3)/9))/model**(1./6)
    if numpy.isscalar(mask):
        resid = numpy.ma.masked_where(model <= mask, resid)
    # XXX... It makes more sense to me to have a minus sign here... So when the
    # model is high, the residual is positive.
    return -resid

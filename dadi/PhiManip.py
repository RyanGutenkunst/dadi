import numpy
from numpy import newaxis as nuax

from dadi import Numerics

def phi_1D(xx, theta=1.0, gamma=0):
    """
    One-dimensional phi for a constant-sized population with selection.

    xx: one-dimensional grid of frequencies upon which phi is defined
    theta: scaled mutation rate, equal to 4*Nc * u, where u is the mutation 
           event rate per generation for the simulated locus.

    Returns a new phi array.
    """
    if gamma == 0:
        return phi_1D_snm(xx, theta)

    exp = numpy.exp
    phi = theta/(xx*(1-xx)) * (1-exp(-2*gamma*(1-xx)))/(1-exp(-2*gamma))
    if xx[0] == 0:
        phi[0] = phi[1]
    if xx[-1] == 1:
        limit = 2*gamma * exp(2*gamma)/(exp(2*gamma)-1)
        phi[-1] = limit
    return phi

def phi_1D_snm(xx, theta=1.0):
    """
    Standard neutral one-dimensional probability density.

    xx: one-dimensional grid of frequencies upon which phi is defined
    theta: scaled mutation rate, equal to 4*Nc * u, where u is the mutation 
           event rate per generation for the simulated locus.

    Returns a new phi array.
    """
    phi = theta/xx
    if xx[0] == 0:
        phi[0] = phi[1]
    return phi

def phi_1D_to_2D(xx, phi_1D):
    """
    Implement a one-to-two population split.

    xx: one-dimensional grid of frequencies upon which phi is defined
    phi1D: initial probability density

    Returns a new two-dimensional phi array.
    """
    pts = len(xx)
    phi_2D = numpy.zeros((pts, pts))
    for ii in range(1, pts-1):
        phi_2D[ii,ii] = phi_1D[ii] * 2/(xx[ii+1]-xx[ii-1])
    return phi_2D

def phi_2D_to_3D_split_2(xx, phi_2D):
    """
    Split population 2 into populations 2 and 3.

    xx: one-dimensional grid of frequencies upon which phi is defined
    phi2D: initial probability density

    Returns a new three-dimensional phi array.
    """
    pts = len(xx)
    phi_3D = numpy.zeros((pts, pts, pts))
    for jj in range(1,pts-1):
        phi_3D[:,jj,jj] = phi_2D[:,jj]*2/(xx[jj+1]-xx[jj-1])
    phi_3D[:,0,0] = phi_2D[:,0]*2/(xx[1] - xx[0])
    phi_3D[:,-1,-1] = phi_2D[:,-1]*2/(xx[-1] - xx[-2])
    return phi_3D

def phi_2D_to_3D_split_1(xx, phi_2D):
    """
    Split population 1 into populations 1 and 3.

    xx: one-dimensional grid of frequencies upon which phi is defined
    phi2D: initial probability density

    Returns a new three-dimensional phi array.
    """
    pts = len(xx)
    phi_3D = numpy.zeros((pts, pts, pts))
    for jj in range(1,pts-1):
        phi_3D[jj,:,jj] = phi_2D[jj,:]*2/(xx[jj+1]-xx[jj-1])
    phi_3D[0,:,0] = phi_2D[0,:]*2/(xx[1] - xx[0])
    phi_3D[-1,:,-1] = phi_2D[-1,:]*2/(xx[-1] - xx[-2])
    return phi_3D

def _admixture_intermediates(phi, ad_z, zz):
    # Find where those z values map to in the zz array.
    # Note that zz[upper_z[ii,jj]] >= ad_z[ii,jj]
    upper_z_index = numpy.searchsorted(zz, ad_z)
    # I occasionally seem to get values > 1 (floating pt error) in ad_z. This
    # corrects them.
    upper_z_index = numpy.minimum(upper_z_index, len(zz)-1)
    lower_z_index = upper_z_index - 1
    upper_z = zz[upper_z_index]
    lower_z = zz[lower_z_index]
    
    # These are the spacings between points. To special-case the endpoints, we
    # use where to set the appropriate spacings to zero. This turns out to be
    # what we want to do for proper normalization via the trapezoid rule.
    delz0 = zz[lower_z_index] - zz[lower_z_index-1]
    delz0 = numpy.where(lower_z_index == 0, 0, delz0)
    
    delz1 = zz[upper_z_index] - zz[lower_z_index]
    delz1 = numpy.where(upper_z_index == 0, 0, delz1)
    
    delz2 = zz[(upper_z_index+1)%len(zz)] - zz[upper_z_index]
    delz2 = numpy.where(upper_z_index == len(zz)-1, 0, delz2)
    
    # frac_lower is the fraction that gets assigned to the lower z index
    frac_lower = (upper_z - ad_z)/(upper_z - lower_z)
    # frac_upper is the fraction that gets assigned to the upper z index
    frac_upper = (ad_z-lower_z)/(upper_z - lower_z)
    # This is the overall normalization for the entry
    norm = 2*phi/(frac_lower*delz0 + delz1 + frac_upper*delz2)

    return lower_z_index, upper_z_index, frac_lower, frac_upper, norm

def _two_pop_admixture_intermediates(phi_2D, f, xx,yy,zz):
    """
    Intermediate results used when splitting a new population out from a 2D phi.
    """
    # For each point x,y in phi, this is the corresponding frequency z that SNPs
    # with frequency x and y in populations 1 and 2 would map to.
    ad_z = f*xx[:,nuax] + (1-f)*yy[nuax,:]

    lower_z_index, upper_z_index, frac_lower, frac_upper, norm \
            = _admixture_intermediates(phi_2D, ad_z, zz)

    return lower_z_index, upper_z_index, frac_lower, frac_upper, norm

def _three_pop_admixture_intermediates(phi_3D, f1,f2, xx,yy,zz,ww):
    """
    Intermediate results used when splitting a new population out from a 3D phi.

    ww: frequency mapping for the new population.
    """
    # For each point x,y,z in phi, this is the corresponding frequency w that
    # SNPs with frequency x,y,z in populations 1,2,3 would map to.
    if f1 + f2 > 1:
        raise ValueError('Admixture proportions (f1=%f, f2 = %f) are '
                         'non-sensible.' % (f1, f2))
    ad_w = f1*xx[:,nuax,nuax] + f2*yy[nuax,:,nuax] + (1-f1-f2)*zz[nuax,nuax,:]

    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _admixture_intermediates(phi_3D, ad_w, ww)

    return lower_w_index, upper_w_index, frac_lower, frac_upper, norm

def phi_2D_to_3D_admix(phi, f, xx,yy,zz):
    """
    Create population 3 admixed from populations 1 and 2.

    Returns a 3D sfs of shape (len(xx),len(yy),len(zz))

    phi:   phi corresponding to original 2 populations
    f:     Fraction of population 3 derived from population 1. (A fraction 1-f
             will be derived from population 2.)
    xx,yy: Mapping of points in phi to frequencies in populations 1 and 2.
    zz:    Frequency mapping that will be used along population 3 axis.
    """
    lower_z_index, upper_z_index, frac_lower, frac_upper, norm \
            = _two_pop_admixture_intermediates(phi, f, xx,yy,zz)


    # Assemble our result.
    # This uses numpy's fancy indexing. It is much, much faster than an
    # explicit loop.
    # See the numpy-discussion post "Numpy Advanced Indexing Question" by
    # Robert Kern on July 16, 2008
    # http://projects.scipy.org/pipermail/numpy-discussion/2008-July/035776.html
    idx_i = numpy.arange(len(xx))[:,numpy.newaxis]
    idx_j = numpy.arange(len(yy))[numpy.newaxis,:]

    phi_3D = numpy.zeros((len(xx), len(yy), len(zz)))
    phi_3D[idx_i, idx_j, lower_z_index] = frac_lower*norm
    phi_3D[idx_i, idx_j, upper_z_index] += frac_upper*norm

    return phi_3D

def phi_2D_admix_1_into_2(phi, f, xx,yy):
    """
    Admix population 1 into population 2.

    Alters phi in place and returns the new version.

    phi:   phi corresponding to original 2 populations
    f:     Fraction of updated population 2 to be derived from population 1. 
             (A fraction 1-f will be derived from the original population 2.)
    xx,yy: Mapping of points in phi to frequencies in populations 1 and 2.
    """
    # This is just like the the split_admix situation, but we're splitting into
    # a population with zz=yy. We could do this by creating a xx by yy by yy
    # array, then integrating out the second population. That's a big waste of
    # memory, however.
    lower_z_index, upper_z_index, frac_lower, frac_upper, norm \
            = _two_pop_admixture_intermediates(phi, f, xx,yy,yy)

    # Basically, we're splitting into a third zz population, then integrating
    # over yy to be left with the two populations we care about.
    lower_cont = frac_lower*norm
    upper_cont = frac_upper*norm
    idx_j = numpy.arange(phi.shape[1])
    for ii in range(phi.shape[0]):
        phi_int = numpy.zeros((phi.shape[1], phi.shape[1]))
        # Use fancy indexing to avoid the commented out loop.
        #for jj in xrange(len(yy)):
        #    phi_int[jj, upper_z_index[ii,jj]] += frac_upper[ii,jj]*norm[ii,jj]
        #    phi_int[jj, lower_z_index[ii,jj]] += frac_lower[ii,jj]*norm[ii,jj]
        phi_int[idx_j, lower_z_index[ii]] = lower_cont[ii]
        phi_int[idx_j, upper_z_index[ii]] += upper_cont[ii]
        phi[ii] = Numerics.trapz(phi_int, yy, axis=0)

    return phi

def phi_2D_admix_2_into_1(phi, f, xx,yy):
    """
    Admix population 2 into population 1.

    Alters phi in place and returns the new version.

    phi:   phi corresponding to original 2 populations
    f:     Fraction of updated population 1 to be derived from population 2. 
             (A fraction 1-f will be derived from the original population 1.)
    xx,yy: Mapping of points in phi to frequencies in populations 1 and 2.
    """
    # Note that it's 1-f here since f now denotes the fraction coming from
    # population 2.
    lower_z_index, upper_z_index, frac_lower, frac_upper, norm \
            = _two_pop_admixture_intermediates(phi, 1-f, xx,yy,xx)

    idx_i = numpy.arange(phi.shape[0])
    lower_cont = frac_lower*norm
    upper_cont = frac_upper*norm
    for jj in xrange(len(yy)):
        phi_int = numpy.zeros((len(xx), len(xx)))
        phi_int[idx_i, lower_z_index[:,jj]] = lower_cont[:,jj]
        phi_int[idx_i, upper_z_index[:,jj]] = upper_cont[:,jj]
        phi[:,jj] = Numerics.trapz(phi_int, xx, axis=0)

    return phi

def phi_3D_admix_1_and_2_into_3(phi, f1,f2, xx,yy,zz):
    """
    Admix populations 1 and 2 into population 3.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 3 populations.
    f1:       Fraction of updated population 3 to be derived from population 1. 
    f2:       Fraction of updated population 3 to be derived from population 2. 
              A fraction (1-f1-f2) will be derived from the original pop 3.
    xx,yy,zz: Mapping of points in phi to frequencies in populations 1,2 and 3.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _three_pop_admixture_intermediates(phi, f1,f2, xx,yy,zz, zz)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    # Basically, we're splitting into a fourth ww population, then integrating
    # over zz to be left with the two populations we care about.
    idx_k = numpy.arange(phi.shape[2])
    for ii in xrange(phi.shape[0]):
        for jj in xrange(phi.shape[1]):
            phi_int = numpy.zeros((phi.shape[2], phi.shape[2]))
            phi_int[idx_k, lower_w_index[ii,jj]] = lower_cont[ii,jj]
            phi_int[idx_k, upper_w_index[ii,jj]] = upper_cont[ii,jj]
            phi[ii,jj] = Numerics.trapz(phi_int, zz, axis=0)

    return phi

def phi_3D_admix_1_and_3_into_2(phi, f1,f3, xx,yy,zz):
    """
    Admix populations 1 and 3 into population 2.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 3 populations.
    f1:       Fraction of updated population 2 to be derived from population 1. 
    f3:       Fraction of updated population 2 to be derived from population 3. 
              A fraction (1-f1-f3) will be derived from the original pop 2.
    xx,yy,zz: Mapping of points in phi to frequencies in populations 1,2 and 3.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _three_pop_admixture_intermediates(phi, f1,1-f1-f3, xx,yy,zz, yy)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    # Basically, we're splitting into a fourth ww population, then integrating
    # over yy to be left with the two populations we care about.
    idx_j = numpy.arange(phi.shape[1])
    for ii in xrange(phi.shape[0]):
        for kk in xrange(phi.shape[2]):
            phi_int = numpy.zeros((phi.shape[1], phi.shape[1]))
            phi_int[idx_j, lower_w_index[ii,:,kk]] = lower_cont[ii,:,kk]
            phi_int[idx_j, upper_w_index[ii,:,kk]] = upper_cont[ii,:,kk]
            phi[ii,:,kk] = Numerics.trapz(phi_int, yy, axis=0)

    return phi

def phi_3D_admix_2_and_3_into_1(phi, f2,f3, xx,yy,zz):
    """
    Admix populations 2 and 3 into population 1.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 3 populations.
    f2:       Fraction of updated population 1 to be derived from population 2. 
    f3:       Fraction of updated population 1 to be derived from population 3. 
              A fraction (1-f2-f3) will be derived from the original pop 1.
    xx,yy,zz: Mapping of points in phi to frequencies in populations 1,2 and 3.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _three_pop_admixture_intermediates(phi, 1-f2-f3,f2, xx,yy,zz, xx)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    # Basically, we're splitting into a fourth ww population, then integrating
    # over yy to be left with the two populations we care about.
    idx_i = numpy.arange(phi.shape[0])
    for jj in xrange(phi.shape[1]):
        for kk in xrange(phi.shape[2]):
            phi_int = numpy.zeros((phi.shape[0], phi.shape[0]))
            phi_int[idx_i, lower_w_index[:,jj,kk]] = lower_cont[:,jj,kk]
            phi_int[idx_i, upper_w_index[:,jj,kk]] = upper_cont[:,jj,kk]
            phi[:,jj,kk] = Numerics.trapz(phi_int, xx, axis=0)

    return phi

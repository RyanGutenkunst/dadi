"""
Manipulating population frequency spectra phi. e.g. population splittings and
admixture
"""
import numpy
from numpy import newaxis as nuax
import scipy.integrate

from dadi import Numerics

def phi_1D(xx, nu=1.0, theta0=1.0, gamma=0, h=0.5, theta=None, beta=1):
    """
    One-dimensional phi for a constant-sized population with genic selection.

    xx: one-dimensional grid of frequencies upon which phi is defined
    nu: size of this population, relative to the reference population size Nref.
    theta0: scaled mutation rate, equal to 4*Nref * u, where u is the mutation 
            event rate per generation for the simulated locus and Nref is the 
            reference population size.
    gamma: scaled selection coefficient, equal to 2*Nref * s, where s is the
           selective advantage.
    h: Dominance coefficient. If A is the selected allele, the aa has fitness 1,
       aA has fitness 1+2sh and AA has fitness 1+2s. h = 0.5 corresonds to
       genic selection.
    theta: deprecated in favor of distinct nu and theta0 arguments, for 
           consistency with Integration functions.

    Returns a new phi array.
    """

    if theta is not None:
        raise ValueError('The parameter theta has been deprecated in favor of '
                         'parameters nu and theta0, for consistency with the '
                         'Integration functions.')

    if h == 0.5:
        return phi_1D_genic(xx, nu, theta0, gamma, beta=beta)

    # Eqn 1 from Williamson, Fledel-Alon, Bustamante _Genetics_ 168:463 (2004).

    # Modified to incorporate fact that for beta != 1, we get a term of 
    # 4*beta/(beta+1)^2 in V. This can be implemented by rescaling gamma
    # and rescaling the final phi.
    gamma = gamma * 4.*beta/(beta+1.)**2

    # Our final result is of the form 
    # exp(Q) * int_0_x exp(-Q) / int_0_1 exp(-Q)

    # For large negative gamma, exp(-Q) becomes numerically infinite.
    # To work around this, we can adjust Q in both the top and bottom
    # integrals by the same factor. We choose to make that factor the 
    # maximum of -Q, which is -2*gamma.
    Qadjust = 0
    # For negative gamma, the maximum of -Q is -2*gamma.
    if gamma < 0 and numpy.isinf(numpy.exp(-2*gamma)):
        Qadjust = -2*gamma

    # For large positive gamma, the prefactor exp(Q) becomes numerically
    # infinite, while the numerator becomes very small. To work around this,
    # we'll can pull the prefactor into the numerator integral.

    # Evaluate the denominator integral.
    integrand = lambda xi: numpy.exp(-4*gamma*h*xi - 2*gamma*(1-2*h)*xi**2
                                     - Qadjust)
    int0, eps = scipy.integrate.quad(integrand, 0, 1, epsabs=0,
                                     points=numpy.linspace(0,1,41))

    ints = numpy.empty(len(xx))
    # Evaluate the numerator integrals
    if gamma < 0:
        # In this case, the prefactor is not divergent, so we can evaluate
        # the numerator as before, using the Qadjust if necessary.
        for ii,q in enumerate(xx):
            val, eps = scipy.integrate.quad(integrand, q, 1, epsabs=0,
                                            points=numpy.linspace(q,1,41))
            ints[ii] = val
        phi = numpy.exp(4*gamma*h*xx + 2*gamma*(1-2*h)*xx**2)*ints/int0
    else:
        # In this case, the prefactor may be divergent, so we do the integral
        # with the prefactor pulled inside
        integrand = lambda xi, q: numpy.exp(-4*gamma*h*(xi-q) -
                                            2*gamma*(1-2*h)*(xi**2-q**2))
        for ii,q in enumerate(xx):
            val, eps = scipy.integrate.quad(integrand, q, 1, args=(q,))
            ints[ii] = val
        phi = ints/int0

    # Protect from division by zero errors
    phi[1:-1] *= 1./(xx[1:-1]*(1-xx[1:-1]))
    # Technically, phi diverges at 0. This kludge lets us do numerics
    # sensibly.
    phi[0] = phi[1]
    # I used Mathematica to calculate the proper limit for x goes to 1.
    # But if we've adjusted the denominator integrand, then that limit doesn't
    # hold. We only need to do that in cases of strong negative selection,
    # when phi near 1 should be almost zero anyways. So we'll just ensure
    # that it is at least monotonically decreasing.
    if Qadjust == 0:
        phi[-1] = 1./int0
    else:
        phi[-1] = min(phi[-1], phi[-2])

    return phi * nu*theta0 * 4.*beta/(beta+1.)**2

def phi_1D_genic(xx, nu=1.0, theta0=1.0, gamma=0, theta=None, beta=1):
    """
    One-dimensional phi for a constant-sized population with genic selection.

    xx: one-dimensional grid of frequencies upon which phi is defined
    nu: size of this population, relative to the reference population size Nref.
    theta0: scaled mutation rate, equal to 4*Nref * u, where u is the mutation 
            event rate per generation for the simulated locus and Nref is the 
            reference population size.
    gamma: scaled selection coefficient, equal to 2*Nref * s, where s is the
           selective advantage.
    theta: deprecated in favor of distinct nu and theta0 arguments, for 
           consistency with Integration functions.

    Returns a new phi array.
    """

    if theta is not None:
        raise ValueError('The parameter theta has been deprecated in favor of '
                         'parameters nu and theta0, for consistency with the '
                         'Integration functions.')
    if gamma == 0:
        return phi_1D_snm(xx, nu, theta0, beta=beta)

    # Beta effectively re-scales gamma.
    gamma = gamma * 4.*beta/(beta+1.)**2

    exp = numpy.exp
    # Protect from warnings on division by zero
    if xx[0] == 0 and xx[-1] == 1:
        phi = 0*xx
        if gamma > -300:
            phi[1:-1] = 1./(xx[1:-1]*(1-xx[1:-1]))\
                    * (1-exp(-2*gamma*(1-xx[1:-1])))/(1-exp(-2*gamma))
        else:
            # Avoid overflow issues for very negative gammas
            phi[1:-1] = 1./(xx[1:-1]*(1-xx[1:-1])) * exp(2*gamma*xx[1:-1])
    else:
        if gamma > -300:
            phi = 1./(xx*(1-xx)) * (1-exp(-2*gamma*(1-xx)))/(1-exp(-2*gamma))
        else:
            phi = 1./(xx*(1-xx)) * exp(2*gamma*xx)

    if xx[0] == 0:
        phi[0] = phi[1]
    if xx[-1] == 1:
        if gamma < 300:
            limit = 2*gamma * exp(2*gamma)/(exp(2*gamma)-1)
        else:
            limit = 2*gamma
        phi[-1] = limit
    return phi * nu*theta0 * 4.*beta/(beta+1.)**2

def phi_1D_snm(xx, nu=1.0, theta0=1.0, theta=None, beta=1):
    """
    Standard neutral one-dimensional probability density.

    xx: one-dimensional grid of frequencies upon which phi is defined
    nu: size of this population, relative to the reference population size Nref.
    theta0: scaled mutation rate, equal to 4*Nref * u, where u is the mutation 
            event rate per generation for the simulated locus and Nref is the 
            reference population size.
    theta: deprecated in favor of distinct nu and theta0 arguments, for 
           consistency with Integration functions.

    Returns a new phi array.
    """

    if theta is not None:
        raise ValueError('The parameter theta has been deprecated in favor of '
                         'parameters nu and theta0, for consistency with the '
                         'Integration functions.')
    # Protect from division by zero errors
    if xx[0] == 0:
        phi = 0*xx
        phi[1:] = nu*theta0/xx[1:]
        phi[0] = phi[1]
    else:
        phi = nu*theta0/xx
    return phi * 4.*beta/(beta+1.)**2

def check_xx(xx):
    """
    Check whether xx is monotonically increasing from 0 to 1. 
    """
    if not xx[0] == 0 and xx[1] == 1:
        raise ValueError('Input xx argument does not run from 0 to 1.'
                         'Have you passed in an incorrect argument?')
    if not numpy.all(numpy.diff(xx) >= 0):
        raise ValueError('Input xx argument is not monotonically increasing. '
                         'Have you passed in an incorrect argument?')

def phi_1D_to_2D(xx, phi_1D):
    """
    Implement a one-to-two population split.

    xx: one-dimensional grid of frequencies upon which phi is defined
    phi1D: initial probability density

    Returns a new two-dimensional phi array.
    """
    check_xx(xx)

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
    check_xx(xx)

    return phi_2D_to_3D_admix(phi_2D,0,xx,xx,xx)

def phi_2D_to_3D_split_1(xx, phi_2D):
    """
    Split population 1 into populations 1 and 3.

    xx: one-dimensional grid of frequencies upon which phi is defined
    phi2D: initial probability density

    Returns a new three-dimensional phi array.
    """
    check_xx(xx)

    return phi_2D_to_3D_admix(phi_2D,1,xx,xx,xx)

def _admixture_intermediates(phi, ad_z, zz):
    # Find where those z values map to in the zz array.
    # Note that zz[upper_z[ii,jj]] >= ad_z[ii,jj]
    upper_z_index = numpy.searchsorted(zz, ad_z)
    # I occasionally seem to get values > 1 (floating pt error) in ad_z. This
    # corrects them.
    upper_z_index = numpy.minimum(upper_z_index, len(zz)-1)
    # Also ensure that upper_z_index >= 1, so lower_z_index is never < 0.
    upper_z_index = numpy.maximum(upper_z_index, 1)
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

def _four_pop_admixture_intermediates(phi_4D, f1,f2,f3, xx,yy,zz,aa,bb):
    """
    Intermediate results used when splitting a new population out from a 4D phi.

    ww: frequency mapping for the new population.
    """
    # For each point x,y,z,a in phi, this is the corresponding frequency b that
    # SNPs with frequency x,y,z,a in populations 1,2,3,4 would map to.
    if f1 + f2 + f3> 1:
        raise ValueError('Admixture proportions (f1=%f, f2 = %f, f3=%f) are '
                         'non-sensible.' % (f1, f2, f3))
    ad_w = f1*xx[:,nuax,nuax,nuax] + f2*yy[nuax,:,nuax,nuax] + f3*zz[nuax,nuax,:,nuax]\
        + (1-f1-f2-f3)*aa[nuax,nuax,nuax,:]

    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _admixture_intermediates(phi_4D, ad_w, bb)

    return lower_w_index, upper_w_index, frac_lower, frac_upper, norm

def _five_pop_admixture_intermediates(phi_5D, f1,f2,f3,f4, xx,yy,zz,aa,bb,cc):
    """
    Intermediate results used when splitting a new population out from a 4D phi.

    ww: frequency mapping for the new population.
    """
    # For each point x,y,z,a,b in phi, this is the corresponding frequency c that
    # SNPs with frequency x,y,z,a,b in populations 1,2,3,4,5 would map to.
    if f1 + f2 + f3 + f4 > 1:
        raise ValueError('Admixture proportions (f1=%f, f2 = %f, f3=%f, f4=%f) are '
                         'non-sensible.' % (f1, f2, f3,  f4))
    ad_w = f1*xx[:,nuax,nuax,nuax,nuax] + f2*yy[nuax,:,nuax,nuax,nuax] + f3*zz[nuax,nuax,:,nuax,nuax]\
        + f4*aa[nuax,nuax,nuax,:,nuax] + (1-f1-f2-f3-f4)*bb[nuax,nuax,nuax,nuax,:]

    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _admixture_intermediates(phi_5D, ad_w, cc)

    return lower_w_index, upper_w_index, frac_lower, frac_upper, norm


def phi_2D_to_3D_admix(phi, f1, xx,yy,zz):
    """
    Create population 3 admixed from populations 1 and 2.

    Returns a 3D sfs of shape (len(xx),len(yy),len(zz))

    phi:   phi corresponding to original 2 populations
    f1:     Fraction of population 3 derived from population 1. (A fraction 1-f1
             will be derived from population 2.)
    xx,yy: Mapping of points in phi to frequencies in populations 1 and 2.
    zz:    Frequency mapping that will be used along population 3 axis.
    """
    lower_z_index, upper_z_index, frac_lower, frac_upper, norm \
            = _two_pop_admixture_intermediates(phi, f1, xx,yy,zz)


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
phi_2D_to_3D = phi_2D_to_3D_admix

def phi_3D_to_4D(phi, f1,f2, xx,yy,zz,aa):
    """
    Create population 4 from populations 1, 2, and 3.

    Returns a 4D sfs of shape (len(xx),len(yy),len(zz),len(aa))

    phi:   phi corresponding to original 3 populations
    f1:    Fraction of population 4 derived from population 1.
    f2:    Fraction of population 4 derived from population 2.
           (A fraction 1-f1-f2 will be derived from population 3.)
    xx,yy,zz: Mapping of points in phi to frequencies in populations 1, 2, and 3.
    aa:    Frequency mapping that will be used along population 4 axis.
    """
    lower_z_index, upper_z_index, frac_lower, frac_upper, norm \
            = _three_pop_admixture_intermediates(phi, f1, f2, xx,yy,zz, aa)

    # Assemble our result.
    # This uses numpy's fancy indexing. It is much, much faster than an
    # explicit loop.
    # See the numpy-discussion post "Numpy Advanced Indexing Question" by
    # Robert Kern on July 16, 2008
    # http://projects.scipy.org/pipermail/numpy-discussion/2008-July/035776.html
    idx_i = numpy.arange(len(xx))[:,nuax,nuax]
    idx_j = numpy.arange(len(yy))[nuax,:,nuax]
    idx_k = numpy.arange(len(zz))[nuax,nuax,:]

    phi_4D = numpy.zeros((len(xx), len(yy), len(zz), len(aa)))
    phi_4D[idx_i, idx_j, idx_k, lower_z_index] = frac_lower*norm
    phi_4D[idx_i, idx_j, idx_k, upper_z_index] += frac_upper*norm

    return phi_4D

def phi_4D_to_5D(phi, f1,f2,f3, xx,yy,zz,aa,bb):
    """
    Create population 5 from populations 1, 2, 3, and 4.

    Returns a 5D sfs of shape (len(xx),len(yy),len(zz),len(aa),len(bb))

    phi:   phi corresponding to original 3 populations
    f1:    Fraction of population 4 derived from population 1.
    f2:    Fraction of population 4 derived from population 2.
    f4:    Fraction of population 4 derived from population 3.
           (A fraction 1-f1-f2-f3 will be derived from population 4.)
    xx,yy,zz,aa: Mapping of points in phi to frequencies in populations 1, 2, 3, 4.
    bb:    Frequency mapping that will be used along population bb axis.
    """
    lower_z_index, upper_z_index, frac_lower, frac_upper, norm \
            = _four_pop_admixture_intermediates(phi, f1,f2,f3, xx,yy,zz,aa, bb)

    # Assemble our result.
    # This uses numpy's fancy indexing. It is much, much faster than an
    # explicit loop.
    # See the numpy-discussion post "Numpy Advanced Indexing Question" by
    # Robert Kern on July 16, 2008
    # http://projects.scipy.org/pipermail/numpy-discussion/2008-July/035776.html
    idx_i = numpy.arange(len(xx))[:,nuax,nuax,nuax]
    idx_j = numpy.arange(len(yy))[nuax,:,nuax,nuax]
    idx_k = numpy.arange(len(zz))[nuax,nuax,:,nuax]
    idx_l = numpy.arange(len(aa))[nuax,nuax,nuax,:]

    phi_5D = numpy.zeros((len(xx), len(yy), len(zz), len(aa), len(bb)))
    phi_5D[idx_i, idx_j, idx_k, idx_l, lower_z_index] = frac_lower*norm
    phi_5D[idx_i, idx_j, idx_k, idx_l, upper_z_index] += frac_upper*norm

    return phi_5D

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
        #for jj in range(len(yy)):
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
    for jj in range(len(yy)):
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
    for ii in range(phi.shape[0]):
        for jj in range(phi.shape[1]):
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
    for ii in range(phi.shape[0]):
        for kk in range(phi.shape[2]):
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
    for jj in range(phi.shape[1]):
        for kk in range(phi.shape[2]):
            phi_int = numpy.zeros((phi.shape[0], phi.shape[0]))
            phi_int[idx_i, lower_w_index[:,jj,kk]] = lower_cont[:,jj,kk]
            phi_int[idx_i, upper_w_index[:,jj,kk]] = upper_cont[:,jj,kk]
            phi[:,jj,kk] = Numerics.trapz(phi_int, xx, axis=0)

    return phi

def phi_4D_admix_into_1(phi, f2,f3,f4, xx,yy,zz,aa):
    """
    Admix populations 2, 3, and 4 into population 1.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 4 populations.
    f2:       Fraction of updated population 1 to be derived from population 2. 
    f3:       Fraction of updated population 1 to be derived from population 3. 
    f4:       Fraction of updated population 1 to be derived from population 4. 
              A fraction (1-f2-f3-f4) will be derived from the original pop 1.
    xx,yy,zz,aa: Mapping of points in phi to frequencies in populations 1,2,3, and 4.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _four_pop_admixture_intermediates(phi, 1-f2-f3-f4,f2,f3, xx,yy,zz,aa, xx)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    # Basically, we're splitting into a fifth bb population, then integrating
    # over xx to be left with the two populations we care about.
    idx_i = numpy.arange(phi.shape[0])
    for jj in range(phi.shape[1]):
        for kk in range(phi.shape[2]):
            for ll in range(phi.shape[3]):
                phi_int = numpy.zeros((phi.shape[0], phi.shape[0]))
                phi_int[idx_i, lower_w_index[:,jj,kk,ll]] = lower_cont[:,jj,kk,ll]
                phi_int[idx_i, upper_w_index[:,jj,kk,ll]] = upper_cont[:,jj,kk,ll]
                phi[:,jj,kk,ll] = Numerics.trapz(phi_int, xx, axis=0)

    return phi

def phi_4D_admix_into_4(phi, f1,f2,f3, xx,yy,zz,aa):
    """
    Admix populations 1,2, and 3 into population 4.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 4 populations.
    f1:       Fraction of updated population 4 to be derived from population 1. 
    f2:       Fraction of updated population 4 to be derived from population 2. 
    f3:       Fraction of updated population 4 to be derived from population 3. 
              A fraction (1-f1-f2-f3) will be derived from the original pop 4.
    xx,yy,zz,aa: Mapping of points in phi to frequencies in populations 1,2,3 and 4.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _four_pop_admixture_intermediates(phi, f1,f2,f3, xx,yy,zz,aa, yy)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    idx_l = numpy.arange(phi.shape[3])
    for ii in range(phi.shape[0]):
        for jj in range(phi.shape[1]):
            for kk in range(phi.shape[2]):
                phi_int = numpy.zeros((phi.shape[3], phi.shape[3]))
                phi_int[idx_l, lower_w_index[ii,jj,kk,:]] = lower_cont[ii,jj,kk,:]
                phi_int[idx_l, upper_w_index[ii,jj,kk,:]] = upper_cont[ii,jj,kk,:]
                phi[ii,jj,kk,:] = Numerics.trapz(phi_int, aa, axis=0)

    return phi

def phi_4D_admix_into_3(phi, f1,f2,f4, xx,yy,zz,aa):
    """
    Admix populations 1,2, and 4 into population 3.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 4 populations.
    f1:       Fraction of updated population 3 to be derived from population 1. 
    f2:       Fraction of updated population 3 to be derived from population 2. 
    f4:       Fraction of updated population 3 to be derived from population 4. 
              A fraction (1-f1-f2-f4) will be derived from the original pop 3.
    xx,yy,zz,aa: Mapping of points in phi to frequencies in populations 1,2,3 and 4.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _four_pop_admixture_intermediates(phi, f1,f2,1-f1-f2-f4, xx,yy,zz,aa, yy)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    idx_k = numpy.arange(phi.shape[2])
    for ii in range(phi.shape[0]):
        for jj in range(phi.shape[1]):
            for ll in range(phi.shape[3]):
                phi_int = numpy.zeros((phi.shape[2], phi.shape[2]))
                phi_int[idx_k, lower_w_index[ii,jj,:,ll]] = lower_cont[ii,jj,:,ll]
                phi_int[idx_k, upper_w_index[ii,jj,:,ll]] = upper_cont[ii,jj,:,ll]
                phi[ii,jj,:,ll] = Numerics.trapz(phi_int, zz, axis=0)

    return phi

def phi_4D_admix_into_2(phi, f1,f3,f4, xx,yy,zz,aa):
    """
    Admix populations 1,3, and 4 into population 2.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 4 populations.
    f1:       Fraction of updated population 2 to be derived from population 1. 
    f3:       Fraction of updated population 2 to be derived from population 3. 
    f4:       Fraction of updated population 2 to be derived from population 4. 
              A fraction (1-f1-f3-f4) will be derived from the original pop 2.
    xx,yy,zz,aa: Mapping of points in phi to frequencies in populations 1,2,3 and 4.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _four_pop_admixture_intermediates(phi, f1,1-f1-f3-f4,f3, xx,yy,zz,aa, yy)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    idx_j = numpy.arange(phi.shape[1])
    for ii in range(phi.shape[0]):
        for kk in range(phi.shape[2]):
            for ll in range(phi.shape[3]):
                phi_int = numpy.zeros((phi.shape[1], phi.shape[1]))
                phi_int[idx_j, lower_w_index[ii,:,kk,ll]] = lower_cont[ii,:,kk,ll]
                phi_int[idx_j, upper_w_index[ii,:,kk,ll]] = upper_cont[ii,:,kk,ll]
                phi[ii,:,kk,ll] = Numerics.trapz(phi_int, yy, axis=0)

    return phi

def phi_5D_admix_into_1(phi, f2,f3,f4,f5, xx,yy,zz,aa,bb):
    """
    Admix populations 2, 3, 4, and 5 into population 1.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 5 populations.
    f2:       Fraction of updated population 1 to be derived from population 2. 
    f3:       Fraction of updated population 1 to be derived from population 3. 
    f4:       Fraction of updated population 1 to be derived from population 4. 
    f5:       Fraction of updated population 1 to be derived from population 5. 
              A fraction (1-f2-f3-f4-f5) will be derived from the original pop 1.
    xx,yy,zz,aa,bb: Mapping of points in phi to frequencies in populations 1,2,3,4, and 5.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _five_pop_admixture_intermediates(phi, 1-f2-f3-f4-f5,f2,f3,f4, xx,yy,zz,aa,bb, xx)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    idx_i = numpy.arange(phi.shape[0])
    for jj in range(phi.shape[1]):
        for kk in range(phi.shape[2]):
            for ll in range(phi.shape[3]):
                for mm in range(phi.shape[4]):
                    phi_int = numpy.zeros((phi.shape[0], phi.shape[0]))
                    phi_int[idx_i, lower_w_index[:,jj,kk,ll,mm]] = lower_cont[:,jj,kk,ll,mm]
                    phi_int[idx_i, upper_w_index[:,jj,kk,ll,mm]] = upper_cont[:,jj,kk,ll,mm]
                    phi[:,jj,kk,ll,mm] = Numerics.trapz(phi_int, xx, axis=0)

    return phi

def phi_5D_admix_into_2(phi, f1,f3,f4,f5, xx,yy,zz,aa,bb):
    """
    Admix populations 1, 3, 4, and 5 into population 2.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 5 populations.
    f1:       Fraction of updated population 2 to be derived from population 1. 
    f3:       Fraction of updated population 2 to be derived from population 3. 
    f4:       Fraction of updated population 2 to be derived from population 4. 
    f5:       Fraction of updated population 2 to be derived from population 5. 
              A fraction (1-f1-f3-f4-f5) will be derived from the original pop 2.
    xx,yy,zz,aa,bb: Mapping of points in phi to frequencies in populations 1,2,3,4, and 5.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _five_pop_admixture_intermediates(phi, f1, 1-f1-f3-f4-f5,f3,f4, xx,yy,zz,aa,bb, xx)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    idx_j = numpy.arange(phi.shape[1])
    for ii in range(phi.shape[0]):
        for kk in range(phi.shape[2]):
            for ll in range(phi.shape[3]):
                for mm in range(phi.shape[4]):
                    phi_int = numpy.zeros((phi.shape[1], phi.shape[1]))
                    phi_int[idx_j, lower_w_index[ii,:,kk,ll,mm]] = lower_cont[ii,:,kk,ll,mm]
                    phi_int[idx_j, upper_w_index[ii,:,kk,ll,mm]] = upper_cont[ii,:,kk,ll,mm]
                    phi[ii,:,kk,ll,mm] = Numerics.trapz(phi_int, xx, axis=0)

    return phi

def phi_5D_admix_into_3(phi, f1,f2,f4,f5, xx,yy,zz,aa,bb):
    """
    Admix populations 1, 2, 4, and 5 into population 3.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 5 populations.
    f1:       Fraction of updated population 3 to be derived from population 1. 
    f2:       Fraction of updated population 3 to be derived from population 2. 
    f4:       Fraction of updated population 3 to be derived from population 4. 
    f5:       Fraction of updated population 3 to be derived from population 5. 
              A fraction (1-f1-f2-f4-f5) will be derived from the original pop 3.
    xx,yy,zz,aa,bb: Mapping of points in phi to frequencies in populations 1,2,3,4, and 5.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _five_pop_admixture_intermediates(phi, f1, f2, 1-f1-f2-f4-f5,f4, xx,yy,zz,aa,bb, xx)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    idx_k = numpy.arange(phi.shape[2])
    for ii in range(phi.shape[0]):
        for jj in range(phi.shape[1]):
            for ll in range(phi.shape[3]):
                for mm in range(phi.shape[4]):
                    phi_int = numpy.zeros((phi.shape[2], phi.shape[2]))
                    phi_int[idx_k, lower_w_index[ii,jj,:,ll,mm]] = lower_cont[ii,jj,:,ll,mm]
                    phi_int[idx_k, upper_w_index[ii,jj,:,ll,mm]] = upper_cont[ii,jj,:,ll,mm]
                    phi[ii,jj,:,ll,mm] = Numerics.trapz(phi_int, xx, axis=0)

    return phi

def phi_5D_admix_into_4(phi, f1,f2,f3,f5, xx,yy,zz,aa,bb):
    """
    Admix populations 1, 2, 3, and 5 into population 4.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 5 populations.
    f1:       Fraction of updated population 4 to be derived from population 1. 
    f2:       Fraction of updated population 4 to be derived from population 2. 
    f3:       Fraction of updated population 4 to be derived from population 3. 
    f5:       Fraction of updated population 4 to be derived from population 5. 
              A fraction (1-f1-f2-f3-f5) will be derived from the original pop 4.
    xx,yy,zz,aa,bb: Mapping of points in phi to frequencies in populations 1,2,3,4, and 5.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _five_pop_admixture_intermediates(phi, f1, f2, f3, 1-f1-f2-f3-f5, xx,yy,zz,aa,bb, xx)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    idx_l = numpy.arange(phi.shape[3])
    for ii in range(phi.shape[0]):
        for jj in range(phi.shape[1]):
            for kk in range(phi.shape[2]):
                for mm in range(phi.shape[4]):
                    phi_int = numpy.zeros((phi.shape[3], phi.shape[3]))
                    phi_int[idx_l, lower_w_index[ii,jj,kk,:,mm]] = lower_cont[ii,jj,kk,:,mm]
                    phi_int[idx_l, upper_w_index[ii,jj,kk,:,mm]] = upper_cont[ii,jj,kk,:,mm]
                    phi[ii,jj,kk,:,mm] = Numerics.trapz(phi_int, xx, axis=0)

    return phi

def phi_5D_admix_into_5(phi, f1,f2,f3,f4, xx,yy,zz,aa,bb):
    """
    Admix populations 1, 2, 3, and 4 into population 5.

    Alters phi in place and returns the new version.

    phi:      phi corresponding to original 5 populations.
    f1:       Fraction of updated population 5 to be derived from population 1. 
    f2:       Fraction of updated population 5 to be derived from population 2. 
    f3:       Fraction of updated population 5 to be derived from population 3. 
    f4:       Fraction of updated population 5 to be derived from population 3. 
              A fraction (1-f1-f2-f3-f4) will be derived from the original pop 5.
    xx,yy,zz,aa,bb: Mapping of points in phi to frequencies in populations 1,2,3,4, and 5.
    """
    lower_w_index, upper_w_index, frac_lower, frac_upper, norm \
            = _five_pop_admixture_intermediates(phi, f1, f2, f3, f4, xx,yy,zz,aa,bb, xx)

    lower_cont = frac_lower * norm
    upper_cont = frac_upper * norm

    idx_m = numpy.arange(phi.shape[4])
    for ii in range(phi.shape[0]):
        for jj in range(phi.shape[1]):
            for kk in range(phi.shape[2]):
                for ll in range(phi.shape[3]):
                    phi_int = numpy.zeros((phi.shape[4], phi.shape[4]))
                    phi_int[idx_m, lower_w_index[ii,jj,kk,ll,:]] = lower_cont[ii,jj,kk,ll,:]
                    phi_int[idx_m, upper_w_index[ii,jj,kk,ll,:]] = upper_cont[ii,jj,kk,ll,:]
                    phi[ii,jj,kk,ll,:] = Numerics.trapz(phi_int, xx, axis=0)

    return phi

def remove_pop(phi, xx, popnum):
    """
    Remove a population from phi.

    Returns new phi with one fewer population.

    phi: phi corresponding to original populations
    xx: Mapping of points in phi to frequencies in population to be removed
    popnum: Population number to remove, numbering from 1.
    """
    return Numerics.trapz(phi, xx, axis=popnum-1)

def filter_pops(phi, xx, tokeep):
    """
    Filter phi to keep only certain populations.

    Returns new phi with len(tokeep) populations.

    phi: phi corresponding to original populations
    xx: Mapping of points in phi to frequencies in population to be removed
    tokeep: List of population numbers to keep, numbering from 1.
    """
    toremove = list(range(1, phi.ndim+1))
    for pop_ii in tokeep:
        toremove.remove(pop_ii)
    for pop_ii in sorted(toremove)[::-1]:
        phi = remove_pop(phi, xx, pop_ii)
    return phi

def phi_1D_X(xx, nu=1.0, theta0=1.0, gamma=0, h=0.5, beta=1, alpha=1):
    """
    One-dimensional phi for a constant-sized population with genic selection.

    xx: one-dimensional grid of frequencies upon which phi is defined
    nu: size of this population, relative to the reference population size Nref.
    theta0: scaled mutation rate, equal to 4*Nref * u, where u is the mutation 
            event rate per generation for the simulated locus and Nref is the 
            reference population size.
    gamma: scaled selection coefficient, equal to 2*Nref * s, where s is the
           selective advantage.
    h: Dominance coefficient. If A is the selected allele, the aa has fitness 1,
       aA has fitness 1+2sh and AA has fitness 1+2s. Male carriers have fitness
       1+2s. h = 0.5 corresonds to genic selection.

    Returns a new phi array.
    """
    Kv = (2.*beta+4.)*(beta+1.)/(9.*beta)
    Km1 = 4./3. * gamma*(0.5+h)
    Km2 = 4./3.*gamma*(1.-2.*h)
    g1 = Km1/Kv
    g2 = Km2/Kv

    # First we evaluate the relevant integrals.
    ints = numpy.empty(len(xx))
    integrand = lambda xi: numpy.exp(-2*g1*xi - g2*xi**2)
    val, eps = scipy.integrate.quad(integrand, 0, 1)
    int0 = val
    for ii,q in enumerate(xx):
        val, eps = scipy.integrate.quad(integrand, q, 1)
        ints[ii] = val

    phi = numpy.exp(2*g1*xx + g2*xx**2)*ints/int0
    # Protect from division by zero errors
    if xx[0] == 0 and xx[-1] == 1:
        phi[1:-1] *= 1./(xx[1:-1]*(1-xx[1:-1]))
    else:
        phi *= 1./(xx*(1-xx))

    if xx[0] == 0:
        # Technically, phi diverges at 0. This fixes lets us do numerics
        # sensibly.
        phi[0] = phi[1]
    if xx[-1] == 1:
        # I used Mathematica to check that this was the proper limit.
        phi[-1] = 1./int0
    return phi * nu*theta0 * 1./Kv * 2./(1.+2.*beta)*(1./(1.+alpha) + beta)


def reorder_pops(phi, neworder):
    """
    Get Spectrum with populations in new order

    Returns new Spectrum with same number of populations, but in a different order

    neworder: Integer list defining new order of populations, indexing the orginal
              populations from 1. Must contain all integers from 1 to number of pops.
    """
    if sorted(neworder) != [_+1 for _ in range(phi.ndim)]:
        raise(ValueError("neworder argument misspecified"))
    newaxes = [_-1 for _ in neworder]
    phi = phi.transpose(newaxes)
    
    return phi





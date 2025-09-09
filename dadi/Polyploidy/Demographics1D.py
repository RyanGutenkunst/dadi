"""
One dimensional demographic models (with polyploids!).
"""

import numpy
from dadi import Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum
from . import Integration as PolyInt
import scipy


# this is a partially developed function from the PhiManip.phi_1D function modified 
# for autotetraploids
def phi_1D_autotet(xx, nu=1.0, theta0=1.0, gamma1=0, gamma2=0, gamma3=0, gamma4=0, 
           theta=None, beta=1, deme_ids=None):
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
    deme_ids: sequence of strings representing the names of demes

    Returns a new phi array.
    """
    #Demes.cache = [Demes.Initiation(nu, deme_ids=deme_ids)]

    if theta is not None:
        raise ValueError('The parameter theta has been deprecated in favor of '
                         'parameters nu and theta0, for consistency with the '
                         'Integration functions.')

    ### Comment out the genic selection case, because 
    ### I don't want to write a separate function for that case at this point :)
    #if h == 0.5:
    #   return phi_1D_genic(xx, nu, theta0, gamma, beta=beta)

    # Eqn 1 from Williamson, Fledel-Alon, Bustamante _Genetics_ 168:463 (2004).

    # Modified to incorporate fact that for beta != 1, we get a term of 
    # 4*beta/(beta+1)^2 in V. This can be implemented by rescaling gamma
    # and rescaling the final phi.
    ### Here, I ignore beta terms for testing... can come back and scale gammas later if needed
    # gamma = gamma * 4.*beta/(beta+1.)**2

    # Our final result is of the form 
    # exp(Q) * int_0_x exp(-Q) / int_0_1 exp(-Q)

    # For large negative gamma, exp(-Q) becomes numerically infinite.
    # To work around this, we can adjust Q in both the top and bottom
    # integrals by the same factor. We choose to make that factor the 
    # maximum of -Q, which is -2*gamma.
    Qadjust = 0
    # For negative gamma, the maximum of -Q is -2*gamma.
    if gamma4 < 0 and numpy.isinf(numpy.exp(-2*gamma4)):
        Qadjust = -2*gamma4

    # For large positive gamma, the prefactor exp(Q) becomes numerically
    # infinite, while the numerator becomes very small. To work around this,
    # we'll can pull the prefactor into the numerator integral.

    # Evaluate the denominator integral.
    integrand = lambda xi: numpy.exp(-2*(4*gamma1*xi - 12*gamma1*xi**2 + 6*gamma2*xi**2
                                         + 12*gamma1*xi**3 - 4*gamma1*xi**4 - 12*gamma2*xi**3
                                         + 6*gamma2*xi**4 + 4*gamma3*xi**3 - 4*gamma3*xi**4
                                         + gamma4*xi**4)
                                        - Qadjust)
    int0, eps = scipy.integrate.quad(integrand, 0, 1, epsabs=0,
                                     points=numpy.linspace(0,1,41))

    ints = numpy.empty(len(xx))
    # Evaluate the numerator integrals
    if gamma4 < 0:
        # In this case, the prefactor is not divergent, so we can evaluate
        # the numerator as before, using the Qadjust if necessary.
        for ii,q in enumerate(xx):
            val, eps = scipy.integrate.quad(integrand, q, 1, epsabs=0,
                                            points=numpy.linspace(q,1,41))
            ints[ii] = val
        phi = numpy.exp(2*(4*gamma1*xx - 12*gamma1*xx**2 + 6*gamma2*xx**2 + 12*gamma1*xx**3 
                           - 4*gamma1*xx**4 - 12*gamma2*xx**3 + 6*gamma2*xx**4 
                           + 4*gamma3*xx**3 - 4*gamma3*xx**4 + gamma4*xx**4))*ints/int0
    else:
        # In this case, the prefactor may be divergent, so we do the integral
        # with the prefactor pulled inside
        integrand = lambda xi, q: numpy.exp(-2*(4*gamma1*(xi-q) - 12*gamma1*(xi**2-q**2) 
                                            + 6*gamma2*(xi**2 - q**2) + 12*gamma1*(xi**3 - q**3) 
                                            - 4*gamma1*(xi**4-q**4) - 12*gamma2*(xi**3-q**3)
                                            + 6*gamma2*(xi**4-q**4) + 4*gamma3*(xi**3-q**3) 
                                            - 4*gamma3*(xi**4-q**4) + gamma4*(xi**4-q**4)))
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

    return phi * nu*theta0 


def snm_dips(notused, ns, pts, theta0=1):
    """
    ns = (n1,)

    Standard neutral model for a single diploid population.
    """
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta0)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
snm_dips.__param_names__ = []

def snm_autos(notused, ns, pts, theta0=1):
    """
    ns = (n1,)

    Standard neutral model for a single autotetraploid population.
    """
    xx = Numerics.default_grid(pts)
    phi = phi_1D_autotet(xx, theta0=theta0)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
snm_autos.__param_names__ = []

def two_epoch_dips(params, ns, pts, theta0=1):
    """
    Instantaneous size change some time ago (diploids).
    
    ns = (n1,)
    params = (T, nu)

    n1: size of the resulting spectrum
    nu: ratio of contemporary to ancestral population sizes
    T: time ago of the instantaneous size change (in units of 2*Na generations)
    """
    T, nu = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta0)

    phi = PolyInt.one_pop(phi, xx, T, nu, theta0=theta0)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def two_epoch_autotets(params, ns, pts, theta0=1):
    """
    Instantaneous size change some time ago (autotetraploids).
    
    ns = (n1,)
    params = (T, nu)

    n1: size of the resulting spectrum
    nu: ratio of contemporary to ancestral population sizes
    T: time ago of the instantaneous size change (in units of 2*Na generations)
    """
    T, nu = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta0)

    phi = PolyInt.one_pop(phi, xx, T, nu, ploidyflag=PolyInt.PloidyType.AUTO, theta0=theta0)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
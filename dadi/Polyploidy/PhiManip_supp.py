# additional functions for phi manipulation with polyploids
# primarily, we add support for the autotetraploid and autohexaploid equilibrium phis

import numpy
import scipy
import dadi.Demes as Demes
from .Integration import PloidyType

# notes: 
# 1. in the case where there is no selection, we should instead scale the equilibrium 
# phi from phi_1D in phi_manip.py by 2 for autotetraploids and by 3 for autohexaploids
# 2. in the case where there is genic selection, we should instead scale both the gammas 
# and the equilibrium phy by factors of 2 for autotetraploids and by 3 for autohexaploids, respectively
def phi_1D_autotet(xx, nu=1.0, theta0=1.0, sel_dict={'gamma':0}, deme_ids=None):
    """
    Compute a one-dimensional phi for a constant-sized autotetraploid population 
    with arbitrary selection.

    Args:
        xx (array): One-dimensional grid of frequencies upon which phi is defined.
        nu (float): Size of this population, relative to the reference population size Nref.
        theta0 (float): Scaled mutation rate, equal to 4*Nref * u, where u is the mutation 
            event rate per generation for the simulated locus in a diploid population
            and Nref is the reference population size. 
        sel_dict (dictionary): Dictionary of selection parameters 
            (see PloidyType class for details).
        deme_ids (list, optional): Sequence of strings representing the names of demes.

    Returns:
        phi (array): A new phi array.
    """
    Demes.cache = [Demes.Initiation(nu, deme_ids=deme_ids)]

    # convert selection dictionary to gamma values
    [gamma1, gamma2, gamma3, gamma4] = PloidyType.AUTO.pack_sel_params(sel_dict)[:4]

    ### Here, we choose not to develop a separate function for the genic selection case
    ### Technically, genic selection for autotetraploids ia s special rescaling (by 2) of 
    ### the diploid equilibrium phi, so we should be able to use that if needed.

    # Based on Eqn 1 from Williamson, Fledel-Alon, Bustamante _Genetics_ 168:463 (2004)
    # or Eqn 9.20 from Kimura _J. Appl. Prob._ (1964).
    # with modifications to reflect the correct polynomial in the exponent for selection
    # in the autotetraploid case

    # Our final result is of the form 
    # exp(Q) * int_0_x exp(-Q) / int_0_1 exp(-Q)

    # For large negative gamma, exp(-Q) becomes numerically infinite.
    # To work around this, we can adjust Q in both the top and bottom
    # integrals by the same factor. We choose to make that factor the 
    # maximum of -Q. 
    # In the autotetraploid case, this max(-Q) is -2*gamma4, 
    # presuming that gamma1,2,3,4 are monotonic.
    Qadjust = 0
    # For negative gamma, the maximum of -Q is -2*gamma4.
    if gamma4 < 0 and numpy.isinf(numpy.exp(-2*gamma4)):
        Qadjust = -2*gamma4

    # For large positive gamma, the prefactor exp(Q) becomes numerically
    # infinite, while the numerator becomes very small. To work around this,
    # we can pull the prefactor into the numerator integral.

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

    # note the 2 below which accounts for the fact that theta0 is the *diploid* scaled mutation rate 
    # which is half the autotetraploid scaled mutation rate
    return phi * 2*nu*theta0 

def phi_1D_autohex(xx, nu=1.0, theta0=1.0, sel_dict={'gamma':0}, deme_ids=None):
    """
    Compute a one-dimensional phi for a constant-sized autohexaploid population 
    with arbitrary selection.

    Args:
        xx (array): One-dimensional grid of frequencies upon which phi is defined.
        nu (float): Size of this population, relative to the reference population size Nref.
        theta0 (float): Scaled mutation rate, equal to 4*Nref * u, where u is the mutation 
            event rate per generation for the simulated locus in a diploid population
            and Nref is the reference population size. 
        sel_dict (dictionary): Dictionary of selection parameters 
            (see PloidyType class for details).
        deme_ids (list, optional): Sequence of strings representing the names of demes.

    Returns:
        phi (array): A new phi array.
    """
    Demes.cache = [Demes.Initiation(nu, deme_ids=deme_ids)]

    # convert selection dictionary to gamma values
    [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6] = PloidyType.AUTOHEX.pack_sel_params(sel_dict)[:6]

    ### Here, we choose not to develop a separate function for the genic selection case.
    ### Technically, genic selection for autohexaploids ia s special rescaling (by 3) of 
    ### the diploid equilibrium phi, so we should be able to use that if needed.

    # Based on Eqn 1 from Williamson, Fledel-Alon, Bustamante _Genetics_ 168:463 (2004)
    # or Eqn 9.20 from Kimura _J. Appl. Prob._ (1964).
    # with modifications to reflect the correct polynomial in the exponent for selection
    # in the autohexaploid case

    # Our final result is of the form 
    # exp(Q) * int_0_x exp(-Q) / int_0_1 exp(-Q)

    # For large negative gamma, exp(-Q) becomes numerically infinite.
    # To work around this, we can adjust Q in both the top and bottom
    # integrals by the same factor. We choose to make that factor the 
    # maximum of -Q. 
    # In the autohexaploid case, this max(-Q) is -2*gamma6, 
    # presuming that gamma1,2,3,4,5,6 are monotonic.
    Qadjust = 0
    # For negative gamma, the maximum of -Q is -2*gamma6.
    if gamma4 < 0 and numpy.isinf(numpy.exp(-2*gamma6)):
        Qadjust = -2*gamma4


    # For large positive gamma, the prefactor exp(Q) becomes numerically
    # infinite, while the numerator becomes very small. To work around this,
    # we can pull the prefactor into the numerator integral.

    # Evaluate the denominator integral.
    integrand = lambda xi: numpy.exp(-2*(   6*gamma1*xi 
                                         - 30*gamma1*xi**2 + 15*gamma2*xi**2
                                         + 60*gamma1*xi**3 - 60*gamma2*xi**3 + 20*gamma3*xi**3
                                         - 60*gamma1*xi**4 + 90*gamma2*xi**4 - 60*gamma3*xi**4 + 15*gamma4*xi**4
                                         + 30*gamma1*xi**5 - 60*gamma2*xi**5 + 60*gamma3*xi**5 - 30*gamma4*xi**5 + 6*gamma5*xi**5
                                         -  6*gamma1*xi**6 + 15*gamma2*xi**6 - 20*gamma3*xi**6 + 15*gamma4*xi**6 - 6*gamma5*xi**6 + gamma6*xi**6)
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
        phi = numpy.exp(2*(6*gamma1*xx 
                        - 30*gamma1*xx**2 + 15*gamma2*xx**2
                        + 60*gamma1*xx**3 - 60*gamma2*xx**3 + 20*gamma3*xx**3
                        - 60*gamma1*xx**4 + 90*gamma2*xx**4 - 60*gamma3*xx**4 + 15*gamma4*xx**4
                        + 30*gamma1*xx**5 - 60*gamma2*xx**5 + 60*gamma3*xx**5 - 30*gamma4*xx**5 + 6*gamma5*xx**5
                        -  6*gamma1*xx**6 + 15*gamma2*xx**6 - 20*gamma3*xx**6 + 15*gamma4*xx**6 - 6*gamma5*xx**6 + gamma6*xx**6)
                        )*ints/int0
    else:
        # In this case, the prefactor may be divergent, so we do the integral
        # with the prefactor pulled inside
        integrand = lambda xi, q: numpy.exp(-2*(6*gamma1*(xi-q) 
                                             - 30*gamma1*(xi**2-q**2) + 15*gamma2*(xi**2-q**2)
                                             + 60*gamma1*(xi**3-q**3) - 60*gamma2*(xi**3-q**3) + 20*gamma3*(xi**3-q**3)
                                             - 60*gamma1*(xi**4-q**4) + 90*gamma2*(xi**4-q**4) - 60*gamma3*(xi**4-q**4) + 15*gamma4*(xi**4-q**4)
                                             + 30*gamma1*(xi**5-q**5) - 60*gamma2*(xi**5-q**5) + 60*gamma3*(xi**5-q**5) - 30*gamma4*(xi**5-q**5) + 6*gamma5*(xi**5-q**5)
                                             -  6*gamma1*(xi**6-q**6) + 15*gamma2*(xi**6-q**6) - 20*gamma3*(xi**6-q**6) + 15*gamma4*(xi**6-q**6) - 6*gamma5*(xi**6-q**6) + gamma6*(xi**6-q**6)))
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

    # note the 3 below which accounts for the fact that theta0 is the *diploid* scaled mutation rate 
    # which is a third the autohexaploid scaled mutation rate
    return phi * 3*nu*theta0 

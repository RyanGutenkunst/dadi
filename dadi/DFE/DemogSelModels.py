"""
Input models of demography + selection.
"""
from dadi import Numerics, Integration, PhiManip, Spectrum

def equil(params, ns, pts):
    """
    Equilibrium demography, plus selection.

    params: [gamma]
    ns: Sample sizes
    pts: Grid point settings for integration

    Note that DFE methods internally apply make_extrap_func,
    so there is no need to make it extrapolate again.
    """
    gamma = params[0]

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)

    return Spectrum.from_phi(phi, ns, (xx,))

def two_epoch(params, ns, pts):
    """
    Instantaneous population size change, plus selection.

    params: [nu,T,gamma]
    ns: Sample sizes
    pts: Grid point settings for integration

    Note that DFE methods internally apply make_extrap_func,
    So there is no need to make it extrapolate again.

    nu: Final population size
    T: Time of size change
    """
    nu, T, gamma = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)
    phi = Integration.one_pop(phi, xx, T, nu, gamma=gamma)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
    
def three_epoch(params, ns, pts):
    """
    params = (nuB,nuF,TB,TF,gamma)
    ns = (n1,)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations) 
    TF: Time since bottleneck recovery (in units of 2*Na generations) 
    gamma: Scaled selection coefficient

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nuB,nuF,TB,TF,gamma = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)

    phi = Integration.one_pop(phi, xx, TB, nuB, gamma=gamma)
    phi = Integration.one_pop(phi, xx, TF, nuF, gamma=gamma)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def IM_pre(params, ns, pts):
    """
    Isolation-with-migration model with exponential pop growth, a size change
    prior to split, and selection.

    params: [nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma1,gamma2]
    ns: Sample sizes
    pts: Grid point settings for integration

    Note that DFE methods internally apply make_extrap_func,
    So there is no need to make it extrapolate again.

    Note also: Selection in contemporary population 1 is assumed to equil
               that in the ancestral population.

    nuPre: Size after first size change
    TPre: Time before split of first size change.
    s: Fraction of nuPre that goes to pop1. (Pop 2 has size nuPre*(1-s).)
    nu1: Final size of pop 1.
    nu2: Final size of pop 2.
    T: Time in the past of split (in units of 2*Na generations) 
    m12: Migration from pop 2 to pop 1 (2*Na*m12)
    m21: Migration from pop 1 to pop 2
    gamma1: Scaled selection coefficient in pop 1 *and* ancestral pop.
    gamma2: Scaled selection coefficient in pop 2
    """
    nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma1,gamma2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = Integration.one_pop(phi, xx, TPre, nu=nuPre, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    nu1_0 = nuPre*s
    nu2_0 = nuPre*(1-s)
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T)
    nu2_func = lambda t: nu2_0 * (nu2/nu2_0)**(t/T)
    phi = Integration.two_pops(phi, xx, T, nu1_func, nu2_func,
                               m12=m12, m21=m21, 
                               gamma1=gamma1, gamma2=gamma2)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs

def IM_pre_single_gamma(params, ns, pts):
    """
    IM_pre model with selection assumed to be equal in all populations.

    See IM_pre for argument definitions, but only a single gamma in params.
    """
    nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma = params
    return IM_pre((nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma,gamma), ns, pts)

def IM(params, ns, pts):
    """
    Isolation-with-migration model with exponential pop growth and selection.

    params: [s,nu1,nu2,T,m12,m21,gamma1,gamma2]
    ns: Sample sizes
    pts: Grid point settings for integration

    Note that this function is defined using a decorator with make_extrap_func.
    So there is no need to make it extrapolate again.

    Note also: Selection in contemporary population 1 is assumed to equal
               that in the ancestral population.

    s: Fraction of nuPre that goes to pop1. (Pop 2 has size nuPre*(1-s).)
    nu1: Final size of pop 1.
    nu2: Final size of pop 2.
    T: Time in the past of split (in units of 2*Na generations) 
    m12: Migration from pop 2 to pop 1 (2*Na*m12)
    m21: Migration from pop 1 to pop 2
    gamma1: Scaled selection coefficient in pop 1 *and* ancestral pop.
    gamma2: Scaled selection coefficient in pop 2
    """
    s,nu1,nu2,T,m12,m21,gamma1,gamma2 = params
    return IM_pre((1,0,s,nu1,nu2,T,m12,m21,gamma1,gamma2), ns, pts)

def IM_single_gamma(params, ns, pts):
    """
    IM model with selection assumed to be equal in all populations.

    See IM for argument definitions, but only a single gamma in params.
    """
    s,nu1,nu2,T,m12,m21,gamma = params
    return IM((s,nu1,nu2,T,m12,m21,gamma,gamma), ns, pts)

def split_mig(params, ns, pts):
    """
    Instantaneous split into two populations of specified size, with symmetric migration.
    params = [nu1,nu2,T,m]

    ns: Sample sizes
    pts: Grid point settings for integration

    Note that DFE methods internally apply make_extrap_func,
    So there is no need to make it extrapolate again.

    Note also: Selection in contemporary population 1 is assumed to equal
               that in the ancestral population.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations) 
    m: Migration rate between populations (2*Na*m)
    gamma1: Scaled selection coefficient in pop 1 *and* ancestral pop.
    gamma2: Scaled selection coefficient in pop 2
    """
    nu1,nu2,T,m,gamma1,gamma2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T, nu1, nu2, m12=m, m21=m, gamma1=gamma1,
                               gamma2=gamma2)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs

def split_mig_single_gamma(params, ns, pts):
    """
    split_mig model with selection assumed to be equal in all populations.

    See split_mig for argument definitions, but only a single gamma in params.
    """
    nu1,nu2,T,m,gamma = params
    return split_mig([nu1,nu2,T,m,gamma,gamma], ns, pts)
    
def split_asym_mig(params, ns, pts):
    """
    Instantaneous split into two populations of specified size, with asymmetric migration.
    params = [nu1,nu2,T,m]

    ns: Sample sizes
    pts: Grid point settings for integration

    Note that DFE methods internally apply make_extrap_func,
    So there is no need to make it extrapolate again.

    Note also: Selection in contemporary population 1 is assumed to equal
               that in the ancestral population.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations)
    m12: Migration rate from population 2 to population 1 (2*Na*m12)
    m21: Migration rate from population 1 to population 2 (2*Na*m21)
    gamma1: Scaled selection coefficient in pop 1 *and* ancestral pop.
    gamma2: Scaled selection coefficient in pop 2
    """
    nu1,nu2,T,m12,m21,gamma1,gamma2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T, nu1, nu2, m12=m12, m21=m21, gamma1=gamma1,
                               gamma2=gamma2)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs

def split_asym_mig_single_gamma(params, ns, pts):
    """
    split_asym_mig model with selection assumed to be equal in all populations.

    See split_asym_mig for argument definitions, but only a single gamma in params.
    """
    nu1,nu2,T,m12,m21,gamma = params
    return split_asym_mig([nu1,nu2,T,m12,m21,gamma,gamma], ns, pts)
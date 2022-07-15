"""
Input models of demography + selection.
"""
from dadi import Numerics, Integration, PhiManip, Spectrum
import numpy

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
equil.__param_names__ = ['gamma']

def two_epoch_sel(params, ns, pts):
    """
    Instantaneous population size change, plus selection.

    params: [nu,T,gamma]
    ns: Sample sizes
    pts: Grid point settings for integration

    Note that DFE methods internally apply make_extrap_func,
    So there is no need to make it extrapolate again.

    nu: Final population size
    T: Time of size change
    gamma: Scaled selection coefficient
    """
    nu, T, gamma = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)
    phi = Integration.one_pop(phi, xx, T, nu, gamma=gamma)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
two_epoch_sel.__param_names__ = ['nu', 'T', 'gamma']
two_epoch = two_epoch_sel

def IM_pre_sel(params, ns, pts):
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
IM_pre_sel.__param_names__ = ['nuPre', 'TPre', 's', 'nu1', 'nu2', 'T', 'm12', 'm21', 'gamma1', 'gamma2']
IM_pre = IM_pre_sel

def IM_pre_sel_single_gamma(params, ns, pts):
    """
    IM_pre_sel model with selection assumed to be equal in all populations.

    See IM_pre_sel for argument definitions, but only a single gamma in params.
    """
    nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma = params
    return IM_pre_sel((nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma,gamma), ns, pts)
IM_pre_sel_single_gamma.__param_names__ = ['nuPre', 'TPre', 's', 'nu1', 'nu2', 'T', 'm12', 'm21', 'gamma']
IM_pre_single_gamma = IM_pre_sel_single_gamma

def IM_sel(params, ns, pts):
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
    return IM_pre_sel((1,0,s,nu1,nu2,T,m12,m21,gamma1,gamma2), ns, pts)
IM_sel.__param_names__ = ['s', 'nu1', 'nu2', 'T', 'm12', 'm21', 'gamma1', 'gamma2']
IM = IM_sel

def IM_sel_single_gamma(params, ns, pts):
    """
    IM_sel model with selection assumed to be equal in all populations.

    See IM_sel for argument definitions, but only a single gamma in params.
    """
    s,nu1,nu2,T,m12,m21,gamma = params
    return IM_sel((s,nu1,nu2,T,m12,m21,gamma,gamma), ns, pts)
IM_sel_single_gamma.__param_names__ = ['s', 'nu1', 'nu2', 'T', 'm12', 'm21', 'gamma']
IM_single_gamma = IM_sel_single_gamma

def split_mig_sel(params, ns, pts):
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
split_mig_sel.__param_names__ = ['nu1', 'nu2', 'T', 'm', 'gamma1', 'gamma2']
split_mig = split_mig_sel

def split_mig_sel_single_gamma(params, ns, pts):
    """
    split_mig_sel model with selection assumed to be equal in all populations.

    See split_mig_sel for argument definitions, but only a single gamma in params.
    """
    nu1,nu2,T,m,gamma = params
    return split_mig_sel([nu1,nu2,T,m,gamma,gamma], ns, pts)
split_mig_sel_single_gamma.__param_names__ = ['nu1', 'nu2', 'T', 'm', 'gamma']
split_mig_single_gamma = split_mig_sel_single_gamma

def split_asym_mig_sel(params, ns, pts):
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
split_asym_mig_sel.__param_names__ = ['nu1', 'nu2', 'T', 'm12', 'm21', 'gamma1', 'gamma2']
split_asym_mig = split_asym_mig_sel

def split_asym_mig_sel_single_gamma(params, ns, pts):
    """
    split_asym_mig_sel model with selection assumed to be equal in all populations.

    See split_asym_mig_sel for argument definitions, but only a single gamma in params.
    """
    nu1,nu2,T,m12,m21,gamma = params
    return split_asym_mig_sel([nu1,nu2,T,m12,m21,gamma,gamma], ns, pts)
split_asym_mig_sel_single_gamma.__param_names__ = ['nu1', 'nu2', 'T', 'm12', 'm21', 'gamma']
split_asym_mig_single_gamma = split_asym_mig_sel_single_gamma

def split_delay_mig_sel(params, ns, pts):
    """
    params = (nu1,nu2,Tpre,Tmig,m12,m21)
    ns = (n1,n2)

    Split into two populations of specifed size, with migration after some time has passed post split.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    Tpre: Time in the past after split but before migration (in units of 2*Na generations) 
    Tmig: Time in the past after migration starts (in units of 2*Na generations) 
    m12: Migration from pop 2 to pop 1 (2*Na*m12)
    m21: Migration from pop 1 to pop 2 (2*Na*m21)
    n1,n2: Sample sizes of resulting Spectrum
    pts: Number of grid points to use in integration.
    gamma1: Scaled selection coefficient in pop 1 *and* ancestral pop.
    gamma2: Scaled selection coefficient in pop 2
    """
    nu1,nu2,Tpre,Tmig,m12,m21,gamma1,gamma2 = params

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, Tpre, nu1, nu2, m12=0, m21=0, 
                               gamma1=gamma1, gamma2=gamma2)
    phi = Integration.two_pops(phi, xx, Tmig, nu1, nu2, m12=m12, m21=m21, 
                               gamma1=gamma1, gamma2=gamma2)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
split_delay_mig_sel.__param_names__ = ['nu1', 'nu2', 'Tpre', 'Tmig', 'm12', 'm21', 'gamma1', 'gamma2']

def split_delay_mig_sel_single_gamma(params, ns, pts):
    """
    split_delay_mig_sel model with selection assumed to be equal in all populations.

    See split_delay_mig_sel for argument definitions, but only a single gamma in params.
    """
    nu1,nu2,Tpre,Tmig,m12,m21,gamma = params
    return split_delay_mig_sel([nu1,nu2,Tpre,Tmig,m12,m21,gamma,gamma], ns, pts)
split_delay_mig_sel_single_gamma.__param_names__ = ['nu1', 'nu2', 'Tpre', 'Tmig', 'm12', 'm21', 'gamma']

def three_epoch_sel(params, ns, pts):
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
three_epoch_sel.__param_names__ = ['nuB', 'nuF', 'TB', 'TF', 'gamma']
three_epoch = three_epoch_sel

def bottlegrowth_2d_sel(params, ns, pts):
    """
    params = (nuB,nuF,T,gamma1,gamma2)
    ns = (n1,n2)

    Instantanous size change followed by exponential growth with no population
    split.

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contempoary to ancient population size
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    gamma1: Scaled selection coefficient in pop 1 *and* ancestral pop.
    gamma2: Scaled selection coefficient in pop 2
    n1,n2: Sample sizes of resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nuB,nuF,T,gamma1,gamma2 = params
    return bottlegrowth_split_mig_sel((nuB,nuF,0,T,0,gamma1,gamma2), ns, pts)
bottlegrowth_2d_sel.__param_names__ = ['nuB', 'nuF', 'T', 'gamma1', 'gamma2']

def bottlegrowth_2d_sel_single_gamma(params, ns, pts):
    """
    bottlegrowth_2d_sel model with selection assumed to be equal in all populations.

    See bottlegrowth_2d_sel for argument definitions, but only a single gamma in params.
    """
    nuB,nuF,T,gamma = params
    return bottlegrowth_split_mig_sel((nuB,nuF,0,T,0,gamma,gamma), ns, pts)
bottlegrowth_2d_sel_single_gamma.__param_names__ = ['nuB', 'nuF', 'T', 'gamma']

def bottlegrowth_split_sel(params, ns, pts):
    """
    params = (nuB,nuF,T,Ts,gamma1,gamma2)
    ns = (n1,n2)

    Instantanous size change followed by exponential growth then split.

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contempoary to ancient population size
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    Ts: Time in the past at which the two populations split.
    gamma1: Scaled selection coefficient in pop 1 *and* ancestral pop.
    gamma2: Scaled selection coefficient in pop 2
    n1,n2: Sample sizes of resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nuB,nuF,T,Ts,gamma1,gamma2 = params
    return bottlegrowth_split_mig_sel((nuB,nuF,0,T,Ts,gamma1,gamma2), ns, pts)
bottlegrowth_split_sel.__param_names__ = ['nuB', 'nuF', 'T', 'Ts', 'gamma1', 'gamma2']

def bottlegrowth_split_sel_single_gamma(params, ns, pts):
    """
    bottlegrowth_split_sel model with selection assumed to be equal in all populations.

    See bottlegrowth_split_sel for argument definitions, but only a single gamma in params.
    """
    nuB,nuF,T,Ts,gamma = params
    return bottlegrowth_split_mig_sel((nuB,nuF,0,T,Ts,gamma,gamma), ns, pts)
bottlegrowth_split_sel_single_gamma.__param_names__ = ['nuB', 'nuF', 'T', 'Ts', 'gamma']

def bottlegrowth_split_mig_sel(params, ns, pts):
    """
    params = (nuB,nuF,m,T,Ts,gamma1,gamma2)
    ns = (n1,n2)

    Instantanous size change followed by exponential growth then split with
    migration.

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contempoary to ancient population size
    m: Migration rate between the two populations (2*Na*m).
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    Ts: Time in the past at which the two populations split.
    gamma1: Scaled selection coefficient in pop 1 *and* ancestral pop.
    gamma2: Scaled selection coefficient in pop 2
    n1,n2: Sample sizes of resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nuB,nuF,m,T,Ts,gamma1,gamma2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma1)

    if T >= Ts:
        nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
        phi = Integration.one_pop(phi, xx, T-Ts, nu_func, gamma=gamma1)

        phi = PhiManip.phi_1D_to_2D(xx, phi)
        nu0 = nu_func(T-Ts)
        nu_func = lambda t: nu0*numpy.exp(numpy.log(nuF/nu0) * t/Ts)
        phi = Integration.two_pops(phi, xx, Ts, nu_func, nu_func, m12=m, m21=m,
                                   gamma1=gamma1, gamma2=gamma2)
    else:
        phi = PhiManip.phi_1D_to_2D(xx, phi)
        phi = Integration.two_pops(phi, xx, Ts-T, 1, 1, m12=m, m21=m,
                                   gamma1=gamma1, gamma2=gamma2)
        nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
        phi = Integration.two_pops(phi, xx, T, nu_func, nu_func, m12=m, m21=m,
                                   gamma1=gamma1, gamma2=gamma2)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
bottlegrowth_split_mig_sel.__param_names__ = ['nuB', 'nuF', 'm', 'T', 'Ts', 'gamma1', 'gamma2']

def bottlegrowth_split_mig_sel_single_gamma(params, ns, pts):
    """
    bottlegrowth_split_mig_sel model with selection assumed to be equal in all populations.

    See bottlegrowth_split_mig_sel for argument definitions, but only a single gamma in params.
    """
    nuB,nuF,m,T,Ts,gamma = params
    return bottlegrowth_split_mig_sel((nuB,nuF,m,T,Ts,gamma,gamma), ns, pts)
bottlegrowth_split_mig_sel_single_gamma.__param_names__ = ['nuB', 'nuF', 'm', 'T', 'Ts', 'gamma']

def growth_sel(params, ns, pts):
    """
    Exponential growth beginning some time ago.

    params = (nu,T,gamma)
    ns = (n1,)

    nu: Ratio of contemporary to ancient population size
    T: Time in the past at which growth began (in units of 2*Na 
       generations) 
    gamma: Scaled selection coefficient
    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nu,T,gamma = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)

    nu_func = lambda t: numpy.exp(numpy.log(nu) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func, gamma=gamma)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
growth_sel.__param_names__ = ['nu', 'T', 'gamma']

def bottlegrowth_1d_sel(params, ns, pts):
    """
    Instantanous size change followed by exponential growth.

    params = (nuB,nuF,T,gamma)
    ns = (n1,)

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contemporary to ancient population size
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    gamma: Scaled selection coefficient
    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nuB,nuF,T,gamma = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)

    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func, gamma=gamma)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
bottlegrowth_1d_sel.__param_names__ = ['nuB', 'nuF', 'T', 'gamma']
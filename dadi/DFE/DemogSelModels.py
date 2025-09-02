"""
Input models of demography + selection.
"""
from dadi import Numerics, Integration, PhiManip, Spectrum
import numpy

def equil(params, ns, pts):
    """
    Equilibrium demography, plus selection.

    Args:
        params (list): [gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    Note:
        DFE methods internally apply make_extrap_func, so there is no need to make it extrapolate again.
    """
    gamma = params[0]

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)

    return Spectrum.from_phi(phi, ns, (xx,))
equil.__param_names__ = ['gamma']

def two_epoch_sel(params, ns, pts):
    """
    Instantaneous population size change, plus selection.

    Args:
        params (list): [nu, T, gamma]
           
            - nu (float): Final population size
            
            - T (float): Time of size change
            
            - gamma (float): Scaled selection coefficient
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    Note:
        DFE methods internally apply make_extrap_func, so there is no need to make it extrapolate again.
    """
    nu, T, gamma = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)
    phi = Integration.one_pop(phi, xx, T, nu, gamma=gamma)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
two_epoch_sel.__param_names__ = ['nu', 'T', 'gamma']
# two_epoch = two_epoch_sel

def IM_pre_sel(params, ns, pts):
    """
    Isolation-with-migration model with exponential population growth, 
    a size change prior to split, and selection.

    Args:
        params (list): [nuPre, TPre, s, nu1, nu2, T, m12, m21, gamma1, gamma2]
            - nuPre (float): Size after first size change
            - TPre (float): Time before split of first size change
            - s (float): Fraction of nuPre that goes to pop1 (Pop 2 has size nuPre*(1-s))
            - nu1 (float): Final size of pop 1
            - nu2 (float): Final size of pop 2
            - T (float): Time in the past of split (in units of 2*Na generations)
            - m12 (float): Migration from pop 2 to pop 1 (2*Na*m12)
            - m21 (float): Migration from pop 1 to pop 2
            - gamma1 (float): Scaled selection coefficient in pop 1 and ancestral pop
            - gamma2 (float): Scaled selection coefficient in pop 2
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    Note:
        - DFE methods internally apply make_extrap_func, so there is no need to make it extrapolate again.
        - Selection in contemporary population 1 is assumed to equal that in the ancestral population.
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
# IM_pre = IM_pre_sel

def IM_pre_sel_single_gamma(params, ns, pts):
    """
    IM_pre_sel model with selection assumed to be equal in all populations.

    Args:
        params (list): [nuPre, TPre, s, nu1, nu2, T, m12, m21, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    See Also:
        [IM_pre_sel][dadi.DFE.DemogSelModels.IM_pre_sel] for argument definitions, but only a single gamma in params.
    """
    nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma = params
    return IM_pre_sel((nuPre,TPre,s,nu1,nu2,T,m12,m21,gamma,gamma), ns, pts)
IM_pre_sel_single_gamma.__param_names__ = ['nuPre', 'TPre', 's', 'nu1', 'nu2', 'T', 'm12', 'm21', 'gamma']
# IM_pre_single_gamma = IM_pre_sel_single_gamma

def IM_sel(params, ns, pts):
    """
    Isolation-with-migration model with exponential population growth and selection.

    Args:
        params (list): [s, nu1, nu2, T, m12, m21, gamma1, gamma2]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    params:
        s (float): Fraction of the ancestral population size (Na) that goes to pop1 (Pop 2 has size Na*(1-s))
        nu1 (float): Final size of pop 1
        nu2 (float): Final size of pop 2
        T (float): Time in the past of split (in units of 2*Na generations)
        m12 (float): Migration from pop 2 to pop 1 (2*Na*m12)
        m21 (float): Migration from pop 1 to pop 2
        gamma1 (float): Scaled selection coefficient in pop 1 and ancestral pop
        gamma2 (float): Scaled selection coefficient in pop 2

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    Note:
        - This function is defined using a decorator with make_extrap_func, so there is no need to make it extrapolate again.
        - Selection in contemporary population 1 is assumed to equal that in the ancestral population.
    """
    s,nu1,nu2,T,m12,m21,gamma1,gamma2 = params
    return IM_pre_sel((1,0,s,nu1,nu2,T,m12,m21,gamma1,gamma2), ns, pts)
IM_sel.__param_names__ = ['s', 'nu1', 'nu2', 'T', 'm12', 'm21', 'gamma1', 'gamma2']
# IM = IM_sel

def IM_sel_single_gamma(params, ns, pts):
    """
    IM_sel model with selection assumed to be equal in all populations.

    Args:
        params (list): [s, nu1, nu2, T, m12, m21, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    See Also:
        IM_sel for argument definitions, but only a single gamma in params.
    """
    s,nu1,nu2,T,m12,m21,gamma = params
    return IM_sel((s,nu1,nu2,T,m12,m21,gamma,gamma), ns, pts)
IM_sel_single_gamma.__param_names__ = ['s', 'nu1', 'nu2', 'T', 'm12', 'm21', 'gamma']
# IM_single_gamma = IM_sel_single_gamma

def split_mig_sel(params, ns, pts):
    """
    Instantaneous split into two populations of specified size, with symmetric migration.

    Args:
        params (list): [nu1, nu2, T, m, gamma1, gamma2]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    params:
        nu1 (float): Size of population 1 after split
        nu2 (float): Size of population 2 after split
        T (float): Time in the past of split (in units of 2*Na generations)
        m (float): Migration rate between populations (2*Na*m)
        gamma1 (float): Scaled selection coefficient in pop 1 and ancestral pop
        gamma2 (float): Scaled selection coefficient in pop 2

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    Note:
        - DFE methods internally apply make_extrap_func, so there is no need to make it extrapolate again.
        - Selection in contemporary population 1 is assumed to equal that in the ancestral population.
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
# split_mig = split_mig_sel

def split_mig_sel_single_gamma(params, ns, pts):
    """
    split_mig_sel model with selection assumed to be equal in all populations.

    Args:
        params (list): [nu1, nu2, T, m, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    See Also:
        split_mig_sel for argument definitions, but only a single gamma in params.
    """
    nu1,nu2,T,m,gamma = params
    return split_mig_sel([nu1,nu2,T,m,gamma,gamma], ns, pts)
split_mig_sel_single_gamma.__param_names__ = ['nu1', 'nu2', 'T', 'm', 'gamma']
# split_mig_single_gamma = split_mig_sel_single_gamma

def split_asym_mig_sel(params, ns, pts):
    """
    Instantaneous split into two populations of specified size, with asymmetric migration.

    Args:
        params (list): [nu1, nu2, T, m12, m21, gamma1, gamma2]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    params:
        nu1 (float): Size of population 1 after split
        nu2 (float): Size of population 2 after split
        T (float): Time in the past of split (in units of 2*Na generations)
        m12 (float): Migration rate from population 2 to population 1 (2*Na*m12)
        m21 (float): Migration rate from population 1 to population 2 (2*Na*m21)
        gamma1 (float): Scaled selection coefficient in pop 1 and ancestral pop
        gamma2 (float): Scaled selection coefficient in pop 2

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    Note:
        - DFE methods internally apply make_extrap_func, so there is no need to make it extrapolate again.
        - Selection in contemporary population 1 is assumed to equal that in the ancestral population.
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
# split_asym_mig = split_asym_mig_sel

def split_asym_mig_sel_single_gamma(params, ns, pts):
    """
    split_asym_mig_sel model with selection assumed to be equal in all populations.

    Args:
        params (list): [nu1, nu2, T, m12, m21, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    See Also:
        split_asym_mig_sel for argument definitions, but only a single gamma in params.
    """
    nu1,nu2,T,m12,m21,gamma = params
    return split_asym_mig_sel([nu1,nu2,T,m12,m21,gamma,gamma], ns, pts)
split_asym_mig_sel_single_gamma.__param_names__ = ['nu1', 'nu2', 'T', 'm12', 'm21', 'gamma']
# split_asym_mig_single_gamma = split_asym_mig_sel_single_gamma

def split_delay_mig_sel(params, ns, pts):
    """
    Split into two populations of specified size, with migration after some time has passed post split.

    Args:
        params (list): [nu1, nu2, Tpre, Tmig, m12, m21, gamma1, gamma2]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    params:
        nu1 (float): Size of population 1 after split
        nu2 (float): Size of population 2 after split
        Tpre (float): Time in the past after split but before migration (in units of 2*Na generations)
        Tmig (float): Time in the past after migration starts (in units of 2*Na generations)
        m12 (float): Migration from pop 2 to pop 1 (2*Na*m12)
        m21 (float): Migration from pop 1 to pop 2 (2*Na*m21)
        gamma1 (float): Scaled selection coefficient in pop 1 and ancestral pop
        gamma2 (float): Scaled selection coefficient in pop 2

    Returns:
        fs (Spectrum): Resulting frequency spectrum
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

    Args:
        params (list): [nu1, nu2, Tpre, Tmig, m12, m21, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    See Also:
        split_delay_mig_sel for argument definitions, but only a single gamma in params.
    """
    nu1,nu2,Tpre,Tmig,m12,m21,gamma = params
    return split_delay_mig_sel([nu1,nu2,Tpre,Tmig,m12,m21,gamma,gamma], ns, pts)
split_delay_mig_sel_single_gamma.__param_names__ = ['nu1', 'nu2', 'Tpre', 'Tmig', 'm12', 'm21', 'gamma']

def three_epoch_sel(params, ns, pts):
    """
    Three epoch model with selection.

    Args:
        params (list): [nuB, nuF, TB, TF, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    params:
        nuB (float): Ratio of bottleneck population size to ancient pop size
        nuF (float): Ratio of contemporary to ancient pop size
        TB (float): Length of bottleneck (in units of 2*Na generations)
        TF (float): Time since bottleneck recovery (in units of 2*Na generations)
        gamma (float): Scaled selection coefficient

    Returns:
        fs (Spectrum): Resulting frequency spectrum
    """
    nuB,nuF,TB,TF,gamma = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)

    phi = Integration.one_pop(phi, xx, TB, nuB, gamma=gamma)
    phi = Integration.one_pop(phi, xx, TF, nuF, gamma=gamma)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
three_epoch_sel.__param_names__ = ['nuB', 'nuF', 'TB', 'TF', 'gamma']
# three_epoch = three_epoch_sel

def bottlegrowth_2d_sel(params, ns, pts):
    """
    Instantaneous size change followed by exponential growth with no population split.

    Args:
        params (list): [nuB, nuF, T, gamma1, gamma2]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    params:
        nuB (float): Ratio of population size after instantaneous change to ancient population size
        nuF (float): Ratio of contemporary to ancient population size
        T (float): Time in the past at which instantaneous change happened and growth began (in units of 2*Na generations)
        gamma1 (float): Scaled selection coefficient in pop 1 and ancestral pop
        gamma2 (float): Scaled selection coefficient in pop 2

    Returns:
        fs (Spectrum): Resulting frequency spectrum
    """
    nuB,nuF,T,gamma1,gamma2 = params
    return bottlegrowth_split_mig_sel((nuB,nuF,0,T,0,gamma1,gamma2), ns, pts)
bottlegrowth_2d_sel.__param_names__ = ['nuB', 'nuF', 'T', 'gamma1', 'gamma2']

def bottlegrowth_2d_sel_single_gamma(params, ns, pts):
    """
    bottlegrowth_2d_sel model with selection assumed to be equal in all populations.

    Args:
        params (list): [nuB, nuF, T, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    See Also:
        bottlegrowth_2d_sel for argument definitions, but only a single gamma in params.
    """
    nuB,nuF,T,gamma = params
    return bottlegrowth_split_mig_sel((nuB,nuF,0,T,0,gamma,gamma), ns, pts)
bottlegrowth_2d_sel_single_gamma.__param_names__ = ['nuB', 'nuF', 'T', 'gamma']

def bottlegrowth_split_sel(params, ns, pts):
    """
    Instantaneous size change followed by exponential growth then split.

    Args:
        params (list): [nuB, nuF, T, Ts, gamma1, gamma2]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    params:
        nuB (float): Ratio of population size after instantaneous change to ancient population size
        nuF (float): Ratio of contemporary to ancient population size
        T (float): Time in the past at which instantaneous change happened and growth began (in units of 2*Na generations)
        Ts (float): Time in the past at which the two populations split
        gamma1 (float): Scaled selection coefficient in pop 1 and ancestral pop
        gamma2 (float): Scaled selection coefficient in pop 2

    Returns:
        fs (Spectrum): Resulting frequency spectrum
    """
    nuB,nuF,T,Ts,gamma1,gamma2 = params
    return bottlegrowth_split_mig_sel((nuB,nuF,0,T,Ts,gamma1,gamma2), ns, pts)
bottlegrowth_split_sel.__param_names__ = ['nuB', 'nuF', 'T', 'Ts', 'gamma1', 'gamma2']

def bottlegrowth_split_sel_single_gamma(params, ns, pts):
    """
    bottlegrowth_split_sel model with selection assumed to be equal in all populations.

    Args:
        params (list): [nuB, nuF, T, Ts, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    See Also:
        bottlegrowth_split_sel for argument definitions, but only a single gamma in params.
    """
    nuB,nuF,T,Ts,gamma = params
    return bottlegrowth_split_mig_sel((nuB,nuF,0,T,Ts,gamma,gamma), ns, pts)
bottlegrowth_split_sel_single_gamma.__param_names__ = ['nuB', 'nuF', 'T', 'Ts', 'gamma']

def bottlegrowth_split_mig_sel(params, ns, pts):
    """
    Instantaneous size change followed by exponential growth then split with migration.

    Args:
        params (list): [nuB, nuF, m, T, Ts, gamma1, gamma2]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    params:
        nuB (float): Ratio of population size after instantaneous change to ancient population size
        nuF (float): Ratio of contemporary to ancient population size
        m (float): Migration rate between the two populations (2*Na*m)
        T (float): Time in the past at which instantaneous change happened and growth began (in units of 2*Na generations)
        Ts (float): Time in the past at which the two populations split
        gamma1 (float): Scaled selection coefficient in pop 1 and ancestral pop
        gamma2 (float): Scaled selection coefficient in pop 2

    Returns:
        fs (Spectrum): Resulting frequency spectrum
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

    Args:
        params (list): [nuB, nuF, m, T, Ts, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    Returns:
        fs (Spectrum): Resulting frequency spectrum

    See Also:
        bottlegrowth_split_mig_sel for argument definitions, but only a single gamma in params.
    """
    nuB,nuF,m,T,Ts,gamma = params
    return bottlegrowth_split_mig_sel((nuB,nuF,m,T,Ts,gamma,gamma), ns, pts)
bottlegrowth_split_mig_sel_single_gamma.__param_names__ = ['nuB', 'nuF', 'm', 'T', 'Ts', 'gamma']

def growth_sel(params, ns, pts):
    """
    Exponential growth beginning some time ago.

    Args:
        params (list): [nu, T, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    params:
        nu (float): Ratio of contemporary to ancient population size
        T (float): Time in the past at which growth began (in units of 2*Na generations)
        gamma (float): Scaled selection coefficient

    Returns:
        fs (Spectrum): Resulting frequency spectrum
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
    Instantaneous size change followed by exponential growth.

    Args:
        params (list): [nuB, nuF, T, gamma]
        ns (list): Sample sizes
        pts (int): Grid point settings for integration

    params:
        nuB (float): Ratio of population size after instantaneous change to ancient population size
        nuF (float): Ratio of contemporary to ancient population size
        T (float): Time in the past at which instantaneous change happened and growth began (in units of 2*Na generations)
        gamma (float): Scaled selection coefficient

    Returns:
        fs (Spectrum): Resulting frequency spectrum
    """
    nuB,nuF,T,gamma = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)

    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func, gamma=gamma)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
bottlegrowth_1d_sel.__param_names__ = ['nuB', 'nuF', 'T', 'gamma']
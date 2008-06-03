"""
Two population demographic models.
"""
import numpy

from dadi import Numerics, PhiManip, SFS, Integration

def snm(n1,n2, pts):
    """
    Standard neutral model, populations never diverge.
    """
    xx = Numerics.other_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    sfs = SFS.sfs_from_phi_2D(n1, n2, xx, xx, phi)
    return sfs

def bottlegrowth((nuB, nuF, T), n1, n2, pts):
    """
    Instantanous size change followed by exponential growth with no population
    split.

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contempoary to ancient population size
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    n1,n2: Shape of resulting SFS
    pts: Number of grid points to use in integration.
    """
    xx = Numerics.other_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func)

    phi = PhiManip.phi_1D_to_2D(xx, phi)

    sfs = SFS.sfs_from_phi_2D(n1, n2, xx, xx, phi)
    return sfs

def bottlegrowth_split((nuB, nuF, T, Ts), n1, n2, pts):
    """
    Instantanous size change followed by exponential growth then split.

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contempoary to ancient population size
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    Ts: Time in the past at which the two populations split.
    n1,n2: Shape of resulting SFS
    pts: Number of grid points to use in integration.
    """
    xx = Numerics.other_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
    phi = Integration.one_pop(phi, xx, T-Ts, nu_func)

    phi = PhiManip.phi_1D_to_2D(xx, phi)
    nu0 = nu_func(T-Ts)
    nu_func = lambda t: nu0*numpy.exp(numpy.log(nuF/nu0) * t/Ts)
    phi = Integration.two_pops(phi, xx, Ts, nu_func, nu_func)

    sfs = SFS.sfs_from_phi_2D(n1, n2, xx, xx, phi)
    return sfs

def bottlegrowth_split_mig((nuB, nuF, m, T, Ts), n1, n2, pts):
    """
    Instantanous size change followed by exponential growth then split with
    migration.

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contempoary to ancient population size
    m: Migration rate between the two populations (2*Na*m).
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    Ts: Time in the past at which the two populations split.
    n1,n2: Shape of resulting SFS
    pts: Number of grid points to use in integration.
    """
    xx = Numerics.other_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
    phi = Integration.one_pop(phi, xx, T-Ts, nu_func)

    phi = PhiManip.phi_1D_to_2D(xx, phi)
    nu0 = nu_func(T-Ts)
    nu_func = lambda t: nu0*numpy.exp(numpy.log(nuF/nu0) * t/Ts)
    phi = Integration.two_pops(phi, xx, Ts, nu_func, nu_func, m12=m, m21=m)

    sfs = SFS.sfs_from_phi_2D(n1, n2, xx, xx, phi)
    return sfs

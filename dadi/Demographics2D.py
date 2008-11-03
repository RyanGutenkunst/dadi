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

def bottlegrowth((nuB, nuF, T), n1,n2, pts):
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
    return bottlegrowth_split_mig((nuB,nuF,0,T,0), n1,n2, pts)

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
    return bottlegrowth_split_mig((nuB,nuF,0,T,Ts), n1,n2, pts)

def bottlegrowth_split_mig((nuB, nuF, m, T, Ts), n1,n2, pts):
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

def split_mig((nu1, nu2, T, m), n1,n2, pts):
    """
    Split into two populations of specifed size, with migration.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations) 
    m: Migration rate between populations (2*Na*m)
    n1,n2: Shape of resulting SFS
    pts: Number of grid points to use in integration.
    """
    xx = Numerics.other_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T, nu1, nu2, m12=m, m21=m)

    sfs = SFS.sfs_from_phi_2D(n1, n2, xx, xx, phi)
    return sfs

def split_mig_mscore((nu1, nu2, T, m)):
    """
    ms core command for split_mig.
    """
    command = "-n 1 %(nu1)f -n 2 %(nu2)f "\
            "-ma x %(m12)f %(m21)f x "\
            "-ej %(T)f 2 1 -en %(T)f 1 1"

    sub_dict = {'nu1':nu1, 'nu2':nu2, 'm12':2*m, 'm21':2*m, 'T': T/2}

    return command % sub_dict

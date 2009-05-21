"""
Single population demographic models.
"""
import numpy

from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum

def snm(notused, (n1,), pts):
    """
    Standard neutral model.

    n1: Number of samples in resulting SFS
    pts: Number of grid points to use in integration.
    """
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    sfs = Spectrum.from_phi(phi, (n1,), (xx,))
    return sfs

def two_epoch((nu, T), (n1,), pts):
    """
    Instantaneous size change some time ago.

    nu: Ratio of contemporary to ancient population size
    T: Time in the past at which size change happened (in units of 2*Na 
       generations) 
    n1: Number of samples in resulting SFS
    pts: Number of grid points to use in integration.
    """
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    
    phi = Integration.one_pop(phi, xx, T, nu)

    sfs = Spectrum.from_phi(phi, (n1,), (xx,))
    return sfs

def growth((nu, T), (n1,), pts):
    """
    Exponential growth beginning some time ago.

    nu: Ratio of contemporary to ancient population size
    T: Time in the past at which growth began (in units of 2*Na 
       generations) 
    n1: Number of samples in resulting SFS
    pts: Number of grid points to use in integration.
    """
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: numpy.exp(numpy.log(nu) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func)

    sfs = Spectrum.from_phi(phi, (n1,), (xx,))
    return sfs

def bottlegrowth((nuB, nuF, T), (n1,), pts):
    """
    Instantanous size change followed by exponential growth.

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contempoary to ancient population size
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    n1: Number of samples in resulting SFS
    pts: Number of grid points to use in integration.
    """
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func)

    sfs = Spectrum.from_phi(phi, (n1,), (xx,))
    return sfs

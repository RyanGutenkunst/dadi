"""
Single population demographic models.
"""
import numpy

from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum

def snm(notused, ns, pts):
    """
    Standard neutral model.

    ns = (n1,)

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def two_epoch(params, ns, pts):
    """
    Instantaneous size change some time ago.

    params = (nu,T)
    ns = (n1,)

    nu: Ratio of contemporary to ancient population size
    T: Time in the past at which size change happened (in units of 2*Na 
       generations) 
    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nu,T = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    
    phi = Integration.one_pop(phi, xx, T, nu)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def growth(params, ns, pts):
    """
    Exponential growth beginning some time ago.

    params = (nu,T)
    ns = (n1,)

    nu: Ratio of contemporary to ancient population size
    T: Time in the past at which growth began (in units of 2*Na 
       generations) 
    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nu,T = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: numpy.exp(numpy.log(nu) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def bottlegrowth(params, ns, pts):
    """
    Instantanous size change followed by exponential growth.

    params = (nuB,nuF,T)
    ns = (n1,)

    nuB: Ratio of population size after instantanous change to ancient
         population size
    nuF: Ratio of contemporary to ancient population size
    T: Time in the past at which instantaneous change happened and growth began
       (in units of 2*Na generations) 
    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nuB,nuF,T = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

def three_epoch(params, ns, pts):
    """
    params = (nuB,nuF,TB,TF)
    ns = (n1,)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations) 
    TF: Time since bottleneck recovery (in units of 2*Na generations) 

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nuB,nuF,TB,TF = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    phi = Integration.one_pop(phi, xx, TB, nuB)
    phi = Integration.one_pop(phi, xx, TF, nuF)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs

"""
Single population demographic models.
"""
import numpy

from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum

def snm_1d(notused, ns, pts):
    """
    Standard neutral model.

    Parameters:
        notused (None): Placeholder parameter, not used in the function.
        ns (tuple[int]): Number of samples in resulting Spectrum (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
snm_1d.__param_names__ = []
snm = snm_1d

def two_epoch(params, ns, pts):
    """
    Instantaneous size change some time ago.

    Parameters:
        params (tuple[float, float]): (nu, T) where:
            
            - nu (float): Ratio of contemporary to ancient population size.
            
            - T (float): Time in the past at which size change happened 
              (in units of 2*Na generations).
        ns (tuple[int]): Number of samples in resulting Spectrum (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    nu,T = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    
    phi = Integration.one_pop(phi, xx, T, nu)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
two_epoch.__param_names__ = ['nu', 'T']

def growth(params, ns, pts):
    """
    Exponential growth beginning some time ago.

    Parameters:
        params (tuple[float, float]): (nu, T) where:
            
            - nu (float): Ratio of contemporary to ancient population size.
            
            - T (float): Time in the past at which growth began
              (in units of 2*Na generations).
        ns (tuple[int]): Number of samples in resulting Spectrum (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    nu,T = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: numpy.exp(numpy.log(nu) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
growth.__param_names__ = ['nu', 'T']

def bottlegrowth_1d(params, ns, pts):
    """
    Instantaneous size change followed by exponential growth.

    Parameters:
        params (tuple[float, float, float]): (nuB, nuF, T) where:
            
            - nuB (float): Ratio of population size after instantaneous change 
              to ancient population size.
            
            - nuF (float): Ratio of contemporary to ancient population size.
            
            - T (float): Time in the past at which instantaneous change happened 
              and growth began (in units of 2*Na generations).
        ns (tuple[int]): Number of samples in resulting Spectrum (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    nuB,nuF,T = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB) * t/T)
    phi = Integration.one_pop(phi, xx, T, nu_func)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
bottlegrowth_1d.__param_names__ = ['nuB', 'nuF', 'T']
bottlegrowth = bottlegrowth_1d

def three_epoch(params, ns, pts):
    """
    Three-epoch demographic model.

    Parameters:
        params (tuple[float, float, float, float]): (nuB, nuF, TB, TF) where:
            
            - nuB (float): Ratio of bottleneck population size to ancient pop size.
            
            - nuF (float): Ratio of contemporary to ancient pop size.
            
            - TB (float): Length of bottleneck (in units of 2*Na generations).
            
            - TF (float): Time since bottleneck recovery (in units of 2*Na generations).
        ns (tuple[int]): Number of samples in resulting Spectrum (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    nuB,nuF,TB,TF = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    phi = Integration.one_pop(phi, xx, TB, nuB)
    phi = Integration.one_pop(phi, xx, TF, nuF)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
three_epoch.__param_names__ = ['nuB', 'nuF', 'TB', 'TF']

def three_epoch_inbreeding(params, ns, pts):
    """
    Three-epoch demographic model with inbreeding.

    Parameters:
        params (tuple[float, float, float, float, float]): (nuB, nuF, TB, TF, F) where:
            
            - nuB (float): Ratio of bottleneck population size to ancient pop size.
            
            - nuF (float): Ratio of contemporary to ancient pop size.
            
            - TB (float): Length of bottleneck (in units of 2*Na generations).
            
            - TF (float): Time since bottleneck recovery (in units of 2*Na generations).
            
            - F (float): Inbreeding coefficient.
        ns (tuple[int]): Number of samples in resulting Spectrum (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    nuB,nuF,TB,TF,F = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    phi = Integration.one_pop(phi, xx, TB, nuB)
    phi = Integration.one_pop(phi, xx, TF, nuF)

    fs = Spectrum.from_phi_inbreeding(phi, ns, (xx,), (F,), (2,))
    return fs
three_epoch_inbreeding.__param_names__ = ['nuB', 'nuF', 'TB', 'TF', 'F']
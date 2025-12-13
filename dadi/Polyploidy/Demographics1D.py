"""
One dimensional demographic models (with polyploids!).
"""

from dadi import Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum
from . import Integration as PolyInt
import dadi.Demes as Demes
from . import PhiManip_supp

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
    phi = PhiManip_supp.phi_1D_autotet(xx, theta0=theta0)
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
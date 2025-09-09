"""
Two dimensional demographic models (with polyploids!).
"""

import numpy
from dadi import Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum
from . import Integration as PolyInt


def autotet_formation(params, ns, pts, theta0=1):
    """
    params = (T_WGD, nu_auto)
    ns = (n1, n2)

    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto. 

    T_WGD: Time in the past at which the WGD occurred, creating 
        an autotetraploid population (in units of 2*Na generations)
    nu_auto: Relative population size of the post-WGD autotetraploids
    n1, n2: sample sizes of the resulting spectrum (number of chromosomes)
    pts: Number of grid points to use in the numerical integration
    """
    T_WGD, nu_auto = params

    autoflag = PolyInt.PloidyType.AUTO

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta0)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    # Note: here, we set pop1 = dips and pop2 = autos
    # integrate from the WGD to the present
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu2=nu_auto, 
                           ploidyflag2=autoflag, theta0=theta0)
    
    print(f"phi: {phi}")
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx))

    print(f"fs: {fs}")  

    return fs

def autotet_formation_bottle(params, ns, pts, theta0=1):
    """
    params = (Tb, T_post, nub, nu_post)
    ns = (n1, n2)

    Two population model of autotetraploid formation where the 
    autotetraploid population splits and undergoes a bottleneck.

    Tb: Time in the past at which the WGD occurred, creating 
        an autotetraploid population undergoing a bottleneck 
        (in units of 2*Na generations)
    T_post: Time in the past at which an instantaneous size change occurs
            in the autotetraploid population (in units of 2*Na generations)
    nub: Population size of the post-WGD autotetraploid population during the bottleneck
    nu_post: Population size of the contemporary autotetraploid population 
             after recovering from the bottleneck
    n1, n2: sample sizes of the resulting spectrum (number of chromosomes)
    pts: Number of grid points to use in the numerical integration
    """
    Tb, T_post, nub, nu_post = params

    autoflag = PolyInt.PloidyType.AUTO

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, theta0=theta0)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    # Note: here, we set pop1 = dips and pop2 = autos
    # integrate from the WGD to the end of the bottleneck
    phi = PolyInt.two_pops(phi, xx, Tb, nu2=nub, 
                           ploidyflag2=autoflag, theta0=theta0)
    # integrate from the end of the bottleneck to the present
    phi = PolyInt.two_pops(phi, xx, T_post, nu2=nu_post, 
                           ploidyflag2=autoflag, theta0=theta0)
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
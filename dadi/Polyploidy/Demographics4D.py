"""
Four dimensional demographic models (with polyploids!).
"""

import numpy
from dadi import Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum
from . import Integration as PolyInt

def allotetraploid_bottleneck_asym_mig(params, ns, pts):
    """
    Three population (4D) model of allotetraploid formation where 
    the diploid progenitors diverge and then a WGD event occurs 
    forming the allotetraploid population.

    Parameters:
        params (tuple): (T_div, T_WGD, nu_allo, H, m31, m42)

            - T_div: Time in the past at which the diploid progenitors diverge 
               (in units of 2*Na generations).

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               allotetraploid population (in units of 2*Na generations).

            - nu_allo: Ratio of contemporary allotetraploid to ancient diploid population size 
               (ratio of *census* sizes).

            - H: homoeologous exchange rate between the two subgenomes, = 4*Na*eta
                (similar to a migration rate, see Blishcak et al. 2023 for details).

            - m31: migration rate from pop 1 (diploid progenitor A) into pop 3 (subgenome A)

            - m42: migration rate from pop 1 (diploid progenitor B) into pop 4 (subgenome B)
        ns (tuple): Sample sizes (n1, n2, n3, n4).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_div, T_WGD, nu_allo, H, m31, m42 = params
    
    alloaflag = PolyInt.PloidyType.ALLOa
    allobflag = PolyInt.PloidyType.ALLOb

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    # integrate for T_div-T_WGD to model the diploid progenitors diverging
    phi = PolyInt.two_pops(phi, xx, T_div-T_WGD)

    # create the allotetraploid population from the two diploid progenitors
    # the third dimension will be subgenome A entirely derived from the first diploid pop
    phi = PhiManip.phi_2D_to_3D(phi, 1, xx, xx, xx)
    # the fourth dimension will be subgenome B entirely derived from the second diploid pop
    phi = PhiManip.phi_3D_to_4D(phi, 0, 1, xx, xx, xx, xx)

    # then, integrate forward for a period of T_WGD
    phi = PolyInt.four_pops(phi, xx, T_WGD, m31=m31, m42=m42, m34=H, m43=H, nu1=1, nu2=1, nu3=nu_allo, nu4=nu_allo,
                            ploidyflag3=alloaflag, ploidyflag4=allobflag)
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx,xx,xx))
    return fs
allotetraploid_bottleneck_asym_mig.__param_names__ = ['T_div', 'T_WGD', 'nu_allo', 'H', 'm31', 'm42']

def allotetraploid_bottleneck_mig(params, ns, pts):
    """
    Three population (4D) model of allotetraploid formation where 
    the diploid progenitors diverge and then a WGD event occurs 
    forming the allotetraploid population.

    Parameters:
        params (tuple): (T_div, T_WGD, nu_allo, H, m)

            - T_div: Time in the past at which the diploid progenitors diverge 
               (in units of 2*Na generations).

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               allotetraploid population (in units of 2*Na generations).

            - nu_allo: Ratio of contemporary allotetraploid to ancient diploid population size 
               (ratio of *census* sizes).

            - H: homoeologous exchange rate between the two subgenomes, = 4*Na*eta
                (similar to a migration rate, see Blishcak et al. 2023 for details).

            - m: migration rate from pop 1 (diploid progenitor A) into pop 3 (subgenome A) and 
                 from pop 2 (diploid progenitor B) into pop 4 (subgenome B)
        ns (tuple): Sample sizes (n1, n2, n3, n4).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_div, T_WGD, nu_allo, H, m = params
    return allotetraploid_bottleneck_asym_mig((T_div, T_WGD, nu_allo, H, m, m), ns, pts)
allotetraploid_bottleneck_mig.__param_names__ = ['T_div', 'T_WGD', 'nu_allo', 'H', 'm']

def allotetraploid_bottleneck(params, ns, pts):
    """
    Three population (4D) model of allotetraploid formation where 
    the diploid progenitors diverge and then a WGD event occurs 
    forming the allotetraploid population.

    Parameters:
        params (tuple): (T_div, T_WGD, nu_allo, H)

            - T_div: Time in the past at which the diploid progenitors diverge 
               (in units of 2*Na generations).

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               allotetraploid population (in units of 2*Na generations).

            - nu_allo: Ratio of contemporary allotetraploid to ancient diploid population size 
               (ratio of *census* sizes).

            - H: homoeologous exchange rate between the two subgenomes, = 4*Na*eta
                (similar to a migration rate, see Blishcak et al. 2023 for details).
        ns (tuple): Sample sizes (n1, n2, n3, n4).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_div, T_WGD, nu_allo, H = params
    return allotetraploid_bottleneck_asym_mig((T_div, T_WGD, nu_allo, H, 0, 0), ns, pts)
allotetraploid_bottleneck.__param_names__ = ['T_div', 'T_WGD', 'nu_allo', 'H']

def allotetraploid_bottleneck_noHE(params, ns, pts):
    """
    Three population (4D) model of allotetraploid formation where 
    the diploid progenitors diverge and then a WGD event occurs 
    forming the allotetraploid population without HEs.

    Parameters:
        params (tuple): (T_div, T_WGD, nu_allo)

            - T_div: Time in the past at which the diploid progenitors diverge 
               (in units of 2*Na generations).

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               allotetraploid population (in units of 2*Na generations).

            - nu_allo: Ratio of contemporary allotetraploid to ancient diploid population size 
               (ratio of *census* sizes).
        ns (tuple): Sample sizes (n1, n2, n3, n4).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_div, T_WGD, nu_allo = params
    return allotetraploid_bottleneck_asym_mig((T_div, T_WGD, nu_allo, 0, 0, 0), ns, pts)
allotetraploid_bottleneck.__param_names__ = ['T_div', 'T_WGD', 'nu_allo']


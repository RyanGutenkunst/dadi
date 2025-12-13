"""
Two dimensional demographic models (with polyploids!).
"""

import numpy
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum
from . import Integration as PolyInt


# Models of Autotetraploid Formation following WGD

def autotet_bottleneck(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a bottlenecked size of nu_auto.
    
    Parameters:
        params (tuple): (T_WGD, nu_auto)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_auto: Ratio of contemporary autotetraploid to ancient diploid population size 
               (ratio of *census* sizes).
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nu_auto = params
    return autotet_bottleneck_asym_mig((T_WGD, nu_auto, 0, 0), ns, pts)
autotet_bottleneck.__param_names__ = ['T_WGD', 'nu_auto']

def autotet_bottleneck_mig(params, ns, pts):
    """
    Two population model of autotetraploid formation with subsequent migration where the 
    autotetraploid population splits and maintains a bottlenecked size of nu_auto.
    
    Parameters:
        params (tuple): (T_WGD, nu_auto, m)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_auto: Ratio of contemporary autotetraploid to ancient diploid population size 
               (ratio of *census* sizes).

            - m: symmetric migration rate between the two populations (2*Na*m)
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nu_auto, m = params

    return autotet_bottleneck_asym_mig((T_WGD, nu_auto, m, m), ns, pts)
autotet_bottleneck_mig.__param_names__ = ['T_WGD', 'nu_auto', 'm']

def autotet_bottleneck_asym_mig(params, ns, pts):
    """
    Two population model of autotetraploid formation with subsequent migration where the 
    autotetraploid population splits and maintains a bottlenecked size of nu_auto.
    
    Parameters:
        params (tuple): (T_WGD, nu_auto, m12, m21)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_auto: Ratio of contemporary autotetraploid to ancient diploid population size 
               (ratio of *census* sizes).

            - m12: migration rate from pop 2 (auotetraploids) into pop 1 (diploids) (2*Na*m12)

            - m21: migration rate from pop 1 (diploids) into pop 2 (auotetraploids) (2*Na*m21)
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nu_auto, m12, m21 = params

    autoflag = PolyInt.PloidyType.AUTO

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    # Note: here, we set pop1 = dips and pop2 = autos
    # integrate from the WGD to the present
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu2=nu_auto, m12=m12, m21=m21,
                           ploidyflag2=autoflag)
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx)) 
    return fs
autotet_bottleneck_asym_mig.__param_names__ = ['T_WGD', 'nu_auto', 'm12', 'm21']

def autotet_bottlegrowth(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T, nuB, nuF)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancient diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancient diploid population size 
               (ratio of *census* sizes).

        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nuB, nuF = params

    return autotet_bottlegrowth_asym_mig((T_WGD, nuB, nuF, 0, 0), ns, pts)
autotet_bottlegrowth.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'm12', 'm21']

def autotet_bottlegrowth_mig(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T, nuB, nuF, m)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancient diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancient diploid population size 
               (ratio of *census* sizes).

            - m: symmetric migration rate between the two populations (2*Na*m)
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nuB, nuF, m = params

    return autotet_bottlegrowth_asym_mig((T_WGD, nuB, nuF, m, m), ns, pts)
autotet_bottlegrowth_mig.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'm']

def autotet_bottlegrowth_asym_mig(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T, nuB, nuF, m12, m21)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancient diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancient diploid population size 
               (ratio of *census* sizes).

            - m12: migration rate from pop 2 (auotetraploids) into pop 1 (diploids) (2*Na*m12)

            - m21: migration rate from pop 1 (diploids) into pop 2 (auotetraploids) (2*Na*m21)
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nuB, nuF, m12, m21 = params

    autoflag = PolyInt.PloidyType.AUTO

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    # Note: here, we set pop1 = dips and pop2 = autos
    
    # set up the nu_func for only the auotetraploid population
    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB)* t/T_WGD)
    
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu2=nu_func, m12=m12, m21=m21, ploidyflag2=autoflag)
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
autotet_bottlegrowth_asym_mig.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'm12', 'm21']


# Diploid models to Test msprime against dadi 

def dip_split_mig_poly(params, ns, pts):
    """
    Two population model of diploids one pop splits and maintains a size of nu_derived.
    
    Parameters:
        params (tuple): (T_div, nu_derived, m)

            - T_div: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_div: Ratio of contemporary diploid to ancient diploid population size 
               (ratio of *census* sizes).

            - m: symmetric migration rate between the two populations (2*Na*m)
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_div, nu_div, m = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    # integrate with polyploidy module
    phi = PolyInt.two_pops(phi, xx, T_div, nu2=nu_div, m12=m, m21=m)
    fs = Spectrum.from_phi(phi, ns, (xx,xx))

    return fs

def dip_split_mig_dadi(params, ns, pts):
    """
    Two population model of diploids one pop splits and maintains a size of nu_derived.
    
    Parameters:
        params (tuple): (T_div, nu_derived, m)

            - T_div: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_derived: Ratio of contemporary diploid to ancient diploid population size 
               (ratio of *census* sizes).

            - m: symmetric migration rate between the two populations (2*Na*m)
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_div, nu_derived, m = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    # integrate with original dadi module
    phi = Integration.two_pops(phi, xx, T_div, nu2=nu_derived, m12=m, m21=m)
    fs = Spectrum.from_phi(phi, ns, (xx,xx)) 
    return fs


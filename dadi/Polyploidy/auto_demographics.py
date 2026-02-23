from dadi import Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum
from . import Integration as PolyInt
from . import PhiManip_supp
import numpy


### Single autotetraploid population models 
### starting from the autotetraploid equilibrium phi

def snm(notused, ns, pts, theta0=1):
    """
    ns = (n1,)

    Standard neutral model for a single autotetraploid population.
    """
    xx = Numerics.default_grid(pts)
    phi = PhiManip_supp.phi_1D_autotet(xx, theta0=theta0)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
snm.__param_names__ = []

### Single autotetraploid population models
### starting from the diploid equilibrium phi

def one_epoch(params, ns, pts, theta0=1):
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
one_epoch.__param_names__ = ['T', 'nu']


def two_epoch(params, ns, pts):
    """
    Two epoch model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu.
    
    Parameters:
        params (tuple): (T_WGD, nu)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu: Ratio of contemporary autotetraploid to ancient diploid population size 
               (ratio of *census* sizes).
        ns (tuple): Sample sizes (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, nu = params
    autoflag = PolyInt.PloidyType.AUTO
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PolyInt.one_pop(phi, xx, T_WGD, nu=nu, ploidyflag=autoflag)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
two_epoch.__param_names__ = ['T_WGD', 'nu']

def bottlegrowth(params, ns, pts):
    """
    Bottlegrowth model of autotetraploid formation where the 
    autotetraploid population starts with size nuWGD and 
    grows exponentially to a size of nuF
    
    Parameters:
        params (tuple): (T_WGD, nuWGD, nuF)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuWGD: Ratio of autotetraploid population immediately after WGD
                to ancient diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary autotetraploid population
                to ancient diploid population size (ratio of *census* sizes).
        ns (tuple): Sample sizes (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, nuWGD, nuF = params
    nu_f = lambda t: nuWGD*numpy.exp(numpy.log(nuF/nuWGD) * t/T_WGD)
    autoflag = PolyInt.PloidyType.AUTO
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PolyInt.one_pop(phi, xx, T_WGD, nu=nu_f, ploidyflag=autoflag)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
bottlegrowth.__param_names__ = ['T_WGD', 'nuWGD', 'nuF']


def three_epoch(params, ns, pts):
    """
    Three epoch model of autotetraploid formation where the 
    autotetraploid population splits, maintains a size of nuWGD for T_WGD, 
    and then changes size again to nu_c for a period of TF.
    This is similar to having a bottleneck for some period and then recover after the bottleneck.
    
    Parameters:
        params (tuple): (T_WGD, TF, nuWGD, nuF)

            - T_WGD: Time length between the WGD event and second size change, creating the  
               autotetraploid population (in units of 2*Na generations).

            - TF: Time in the past at which the second epoch begins.

            - nuWGD: Ratio of initial autotetraploid population (during first epoch)
                 to ancient diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary autotetraploid population (during second epoch)
                 to ancient diploid population size (ratio of *census* sizes).
        ns (tuple): Sample sizes (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, TF, nuWGD, nuF  = params
    autoflag = PolyInt.PloidyType.AUTO
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PolyInt.one_pop(phi, xx, T_WGD, nu=nuWGD, ploidyflag=autoflag)
    phi = PolyInt.one_pop(phi, xx, TF, nu=nuF, ploidyflag=autoflag)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
three_epoch.__param_names__ = ['T_WGD', 'TF', 'nuWGD', 'nuF']

### Single autotetraploid population models with the diploid progenitors

def bottleneck_w_dips(params, ns, pts):
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
    return bottleneck_asym_mig_w_dips((T_WGD, nu_auto, 0, 0), ns, pts)
bottleneck_w_dips.__param_names__ = ['T_WGD', 'nu_auto']

def bottleneck_mig_w_dips(params, ns, pts):
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

    return bottleneck_asym_mig_w_dips((T_WGD, nu_auto, m, m), ns, pts)
bottleneck_mig_w_dips.__param_names__ = ['T_WGD', 'nu_auto', 'm']

def bottleneck_asym_mig_w_dips(params, ns, pts):
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
bottleneck_asym_mig_w_dips.__param_names__ = ['T_WGD', 'nu_auto', 'm12', 'm21']

def bottlegrowth_w_dips(params, ns, pts):
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

    return bottlegrowth_asym_mig_w_dips((T_WGD, nuB, nuF, 0, 0), ns, pts)
bottlegrowth_w_dips.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'm12', 'm21']

def bottlegrowth_mig_w_dips(params, ns, pts):
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

    return bottlegrowth_asym_mig_w_dips((T_WGD, nuB, nuF, m, m), ns, pts)
bottlegrowth_mig_w_dips.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'm']

def bottlegrowth_asym_mig_w_dips(params, ns, pts):
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
    
    # set up the nu_func for only the auotetraploid population
    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB)* t/T_WGD)
    
    # Note: here, we set pop1 = dips and pop2 = autos
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu2=nu_func, m12=m12, m21=m21, ploidyflag2=autoflag)
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
bottlegrowth_asym_mig_w_dips.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'm12', 'm21']

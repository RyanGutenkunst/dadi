from dadi import Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum
from . import Integration as PolyInt
from . import PhiManip_supp
import numpy


### Single autotetraploid population models 
### starting from the autotetraploid equilibrium phi

# add models here

### Single autotetraploid population models
### starting from the diploid equilibrium phi

def two_epoch_sel(params, ns, pts):
    """
    Two epoch model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu.
    
    Parameters:
        params (tuple): (T_WGD, nu, gamma)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu: Ratio of contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - gamma: population-scaled selection coefficient (= 2*Na*s)
        ns (tuple): Sample sizes (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, nu, gamma = params
    autoflag = PolyInt.PloidyType.AUTO
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)
    phi = PolyInt.one_pop(phi, xx, T_WGD, nu=nu, ploidyflag=autoflag, sel_dict={"gamma": gamma})
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
two_epoch_sel.__param_names__ = ['T_WGD', 'nu', 'gamma']

def bottlegrowth_sel(params, ns, pts):
    """
    Bottlegrowth model of autotetraploid formation where the 
    autotetraploid population starts with size nuWGD and 
    grows exponentially to a size of nuF
    
    Parameters:
        params (tuple): (T_WGD, nuWGD, nuF, gamma)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuWGD: Ratio of autotetraploid population immediately after WGD
                to ancestral diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary autotetraploid population
                to ancestral diploid population size (ratio of *census* sizes).

            - gamma: population-scaled selection coefficient (= 2*Na*s)
        ns (tuple): Sample sizes (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, nuWGD, nuF, gamma = params
    nu_f = lambda t: nuWGD*numpy.exp(numpy.log(nuF/nuWGD) * t/T_WGD)
    autoflag = PolyInt.PloidyType.AUTO
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)
    phi = PolyInt.one_pop(phi, xx, T_WGD, nu=nu_f,
                          sel_dict={"gamma": gamma}, ploidyflag=autoflag)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
bottlegrowth_sel.__param_names__ = ['T_WGD', 'nuWGD', 'nuF', 'gamma']

def three_epoch_sel(params, ns, pts):
    """
    Three epoch model of autotetraploid formation where the 
    autotetraploid population splits, maintains a size of nuWGD for T_WGD, 
    and then changes size again to nu_c for a period of TF.
    This is similar to having a bottleneck for some period and then recover after the bottleneck.
    
    Parameters:
        params (tuple): (T_WGD, TF, nuWGD, nuF, gamma)

            - T_WGD: Time length between the WGD event and second size change, creating the  
               autotetraploid population (in units of 2*Na generations).

            - TF: Time in the past at which the second epoch begins.

            - nuWGD: Ratio of initial autotetraploid population (during first epoch)
                 to ancestral diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary autotetraploid population (during second epoch)
                 to ancestral diploid population size (ratio of *census* sizes).

            - gamma: population-scaled selection coefficient (= 2*Na*s)
        ns (tuple): Sample sizes (n1,).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, TF, nuWGD, nuF, gamma  = params
    autoflag = PolyInt.PloidyType.AUTO
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)
    phi = PolyInt.one_pop(phi, xx, T_WGD, nu=nuWGD, 
                          sel_dict={"gamma": gamma}, ploidyflag=autoflag)
    phi = PolyInt.one_pop(phi, xx, TF, nu=nuF, 
                          sel_dict={"gamma": gamma}, ploidyflag=autoflag)
    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs
three_epoch_sel.__param_names__ = ['T_WGD', 'TF', 'nuWGD', 'nuF', 'gamma']

### Single autotetraploid population models with the diploid progenitors
### with *different/independent* selection coefficients

def bottleneck_w_dips_sel(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a bottlenecked size of nu_auto.
    
    Parameters:
        params (tuple): (T_WGD, nu_auto, gamma1, gamma2)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_auto: Ratio of contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - gamma1: population-scaled selection coefficient (= 2*Na*s)
                for the diploid population

            - gamma2: population-scaled selection coefficient (= 2*Na*s)
                for the autotetraploid population
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nu_auto, gamma1, gamma2 = params
    return bottleneck_asym_mig_w_dips_sel((T_WGD, nu_auto, 0, 0, gamma1, gamma2), ns, pts)
bottleneck_w_dips_sel.__param_names__ = ['T_WGD', 'nu_auto', 'gamma1', 'gamma2']

def bottleneck_mig_w_dips_sel(params, ns, pts):
    """
    Two population model of autotetraploid formation with subsequent migration where the 
    autotetraploid population splits and maintains a bottlenecked size of nu_auto.
    
    Parameters:
        params (tuple): (T_WGD, nu_auto, m, gamma1, gamma2)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_auto: Ratio of contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - m: symmetric migration rate between the two populations (2*Na*m)

            - gamma1: population-scaled selection coefficient (= 2*Na*s)
                for the diploid population

            - gamma2: population-scaled selection coefficient (= 2*Na*s)
                for the autotetraploid population
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nu_auto, m, gamma1, gamma2 = params

    return bottleneck_asym_mig_w_dips_sel((T_WGD, nu_auto, m, m, gamma1, gamma2), ns, pts)
bottleneck_mig_w_dips_sel.__param_names__ = ['T_WGD', 'nu_auto', 'm', 'gamma1', 'gamma2']

def bottleneck_asym_mig_w_dips_sel(params, ns, pts):
    """
    Two population model of autotetraploid formation with subsequent migration where the 
    autotetraploid population splits and maintains a bottlenecked size of nu_auto.
    
    Parameters:
        params (tuple): (T_WGD, nu_auto, m12, m21, gamma1, gamma2)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_auto: Ratio of contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - m12: migration rate from pop 2 (auotetraploids) into pop 1 (diploids) (2*Na*m12)

            - m21: migration rate from pop 1 (diploids) into pop 2 (auotetraploids) (2*Na*m21)

            - gamma1: population-scaled selection coefficient (= 2*Na*s)
                for the diploid population

            - gamma2: population-scaled selection coefficient (= 2*Na*s)
                for the autotetraploid population
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nu_auto, m12, m21, gamma1, gamma2 = params

    autoflag = PolyInt.PloidyType.AUTO

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    # Note: here, we set pop1 = dips and pop2 = autos
    # integrate from the WGD to the present
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu2=nu_auto, m12=m12, m21=m21,
                           sel_dict1={"gamma": gamma1}, sel_dict2={"gamma": gamma2},
                           ploidyflag2=autoflag)
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx)) 
    return fs
bottleneck_asym_mig_w_dips_sel.__param_names__ = ['T_WGD', 'nu_auto', 'm12', 'm21', 'gamma1', 'gamma2']

def bottlegrowth_w_dips_sel(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, nuB, nuF, gamma1, gamma2)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - gamma1: population-scaled selection coefficient (= 2*Na*s)
                for the diploid population

            - gamma2: population-scaled selection coefficient (= 2*Na*s)
                for the autotetraploid population
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nuB, nuF, gamma1, gamma2 = params

    return bottlegrowth_asym_mig_w_dips_sel((T_WGD, nuB, nuF, 0, 0, gamma1, gamma2), ns, pts)
bottlegrowth_w_dips_sel.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'gamma1', 'gamma2']

def bottlegrowth_mig_w_dips_sel(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, nuB, nuF, m, gamma1, gamma2)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - m: symmetric migration rate between the two populations (2*Na*m)

            - gamma1: population-scaled selection coefficient (= 2*Na*s)
                for the diploid population

            - gamma2: population-scaled selection coefficient (= 2*Na*s)
                for the autotetraploid population
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nuB, nuF, m, gamma1, gamma2 = params

    return bottlegrowth_asym_mig_w_dips_sel((T_WGD, nuB, nuF, m, m, gamma1, gamma2), ns, pts)
bottlegrowth_mig_w_dips_sel.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'm', 'gamma1', 'gamma2']

def bottlegrowth_asym_mig_w_dips_sel(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, nuB, nuF, m12, m21, gamma1, gamma2)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - m12: migration rate from pop 2 (auotetraploids) into pop 1 (diploids) (2*Na*m12)

            - m21: migration rate from pop 1 (diploids) into pop 2 (auotetraploids) (2*Na*m21)

            - gamma1: population-scaled selection coefficient (= 2*Na*s)
                for the diploid population

            - gamma2: population-scaled selection coefficient (= 2*Na*s)
                for the autotetraploid population
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nuB, nuF, m12, m21, gamma1, gamma2 = params

    autoflag = PolyInt.PloidyType.AUTO

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    
    # set up the nu_func for only the auotetraploid population
    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB)* t/T_WGD)
    
    # Note: here, we set pop1 = dips and pop2 = autos
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu2=nu_func, m12=m12, m21=m21, 
                           sel_dict1={"gamma": gamma1}, sel_dict2={"gamma": gamma2},
                           ploidyflag2=autoflag)
    
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
bottlegrowth_asym_mig_w_dips_sel.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'm12', 'm21', 'gamma1', 'gamma2']


def bottlegrowth_dip_size_change_sel(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, T_dip, nuB, nuF, nu_dip, gamma1, gamma2)

            - T_WGD: Time between the WGD and diploid size change (in units of 2*Na generations).

            - T_dip: Time in the past at which the diploid population changes size
                (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - nu_dip: Ratio of contemporary diploid population to ancestral diploid population size

            - gamma1: population-scaled selection coefficient (= 2*Na*s)
                for the diploid population

            - gamma2: population-scaled selection coefficient (= 2*Na*s)
                for the autotetraploid population
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, T_dip, nuB, nuF, nu_dip, gamma1, gamma2 = params

    return bottlegrowth_dip_size_change_asym_mig_sel((T_WGD, T_dip, nuB, nuF, nu_dip, 0, 0, gamma1, gamma2), ns, pts)
bottlegrowth_dip_size_change_sel.__param_names__ = ['T_WGD', 'T_dip', 'nuB', 'nuF', 'nu_dip', 'gamma1', 'gamma2']


def bottlegrowth_dip_size_change_mig_sel(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, T_dip, nuB, nuF, nu_dip, m, gamma1, gamma2)

            - T_WGD: Time between the WGD and diploid size change (in units of 2*Na generations).

            - T_dip: Time in the past at which the diploid population changes size
                (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - nu_dip: Ratio of contemporary diploid population to ancestral diploid population size

            - m: symmetric migration rate between the two populations (2*Na*m)

            - gamma1: population-scaled selection coefficient (= 2*Na*s)
                for the diploid population

            - gamma2: population-scaled selection coefficient (= 2*Na*s)
                for the autotetraploid population
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, T_dip, nuB, nuF, nu_dip, m, gamma1, gamma2 = params

    return bottlegrowth_dip_size_change_asym_mig_sel((T_WGD, T_dip, nuB, nuF, nu_dip, m, m, gamma1, gamma2), ns, pts)
bottlegrowth_dip_size_change_mig_sel.__param_names__ = ['T_WGD', 'T_dip', 'nuB', 'nuF', 'nu_dip', 'm', 'gamma1', 'gamma2']


def bottlegrowth_dip_size_change_asym_mig_sel(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, T_dip, nuB, nuF, nu_dip, m12, m21, gamma1, gamma2)

            - T_WGD: Time between the WGD and diploid size change (in units of 2*Na generations).

            - T_dip: Time in the past at which the diploid population changes size
                (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - nu_dip: Ratio of contemporary diploid population to ancestral diploid population size

            - m12: migration rate from pop 2 (auotetraploids) into pop 1 (diploids) (2*Na*m12)

            - m21: migration rate from pop 1 (diploids) into pop 2 (auotetraploids) (2*Na*m21)
        
            - gamma1: population-scaled selection coefficient (= 2*Na*s)
                for the diploid population

            - gamma2: population-scaled selection coefficient (= 2*Na*s)
                for the autotetraploid population
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, T_dip, nuB, nuF, nu_dip, m12, m21, gamma1, gamma2 = params

    autoflag = PolyInt.PloidyType.AUTO

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma1)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    
    T_total = T_WGD + T_dip
    # set up the nu_func for only the auotetraploid population
    nu_func = lambda t: nuB*numpy.exp(numpy.log(nuF/nuB)* t/T_total)
    
    # Note: here, we set pop1 = dips and pop2 = autos
    # integrate from the WGD to dip size change
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu2=nu_func, m12=m12, m21=m21, 
                           sel_dict1={"gamma": gamma1}, sel_dict2={"gamma": gamma2},
                           ploidyflag2=autoflag)
    # integrate from dip size change to the present
    phi = PolyInt.two_pops(phi, xx, T_total, nu1=nu_dip, nu2=nu_func, m12=m12, m21=m21, 
                           sel_dict1={"gamma": gamma1}, sel_dict2={"gamma": gamma2},
                           ploidyflag2=autoflag, initial_t = T_WGD)

    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
bottlegrowth_dip_size_change_asym_mig_sel.__param_names__ = ['T_WGD', 'T_dip', 'nuB', 'nuF', 'nu_dip', 'm12', 'm21', 'gamma1', 'gamma2']


### Single autotetraploid population models with the diploid progenitors
### with the *same* selection coefficients

def bottleneck_w_dips_sel_single_gamma(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu.
    
    Parameters:
        params (tuple): (T_WGD, nu_auto, gamma)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_auto: Ratio of contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - gamma: population-scaled selection coefficient (= 2*Na*s)
                Note that this is used for both the diploid and autotetraploid populations.
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.    
    """
    T_WGD, nu_auto, gamma = params
    return bottleneck_w_dips_sel((T_WGD, nu_auto, gamma, gamma), ns, pts)
bottleneck_w_dips_sel_single_gamma.__param_names__ = ['T_WGD', 'nu_auto', 'gamma']

def bottleneck_mig_w_dips_sel_single_gamma(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.
    
    Parameters:
        params (tuple): (T_WGD, nu_auto, m, gamma)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_auto: Ratio of contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - m: symmetric migration rate between the two populations (2*Na*m)

            - gamma: population-scaled selection coefficient (= 2*Na*s)
                Note that this is used for both the diploid and autotetraploid populations.
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, nu_auto, m, gamma = params

    return bottleneck_mig_w_dips_sel((T_WGD, nu_auto, m, gamma, gamma), ns, pts)
bottleneck_mig_w_dips_sel_single_gamma.__param_names__ = ['T_WGD', 'nu_auto', 'm', 'gamma']

def bottleneck_asym_mig_w_dips_sel_single_gamma(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a bottlenecked size of nu_auto.
    
    Parameters:
        params (tuple): (T_WGD, nu_auto, m12, m21, gamma)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu_auto: Ratio of contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - m12: migration rate from pop 2 (auotetraploids) into pop 1 (diploids) (2*Na*m12)

            - m21: migration rate from pop 1 (diploids) into pop 2 (auotetraploids) (2*Na*m21)

            - gamma: population-scaled selection coefficient (= 2*Na*s)
                Note that this is used for both the diploid and autotetraploid populations.
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nu_auto, m12, m21, gamma = params

    return bottleneck_asym_mig_w_dips_sel((T_WGD, nu_auto, m12, m21, gamma, gamma), ns, pts)
bottleneck_asym_mig_w_dips_sel_single_gamma.__param_names__ = ['T_WGD', 'nu_auto', 'm12', 'm21', 'gamma']

def bottlegrowth_w_dips_sel_single_gamma(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, nuB, nuF, gamma)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - gamma: population-scaled selection coefficient (= 2*Na*s)
                Note that this is used for both the diploid and autotetraploid populations.

        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_WGD, nuB, nuF, gamma = params
    return bottlegrowth_w_dips_sel((T_WGD, nuB, nuF, gamma, gamma), ns, pts)
bottlegrowth_w_dips_sel_single_gamma.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'gamma']  

def bottlegrowth_mig_w_dips_sel_single_gamma(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, nuB, nuF, m, gamma)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - m: symmetric migration rate between the two populations (2*Na*m)

            - gamma: population-scaled selection coefficient (= 2*Na*s)
                Note that this is used for both the diploid and autotetraploid populations.
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nuB, nuF, m, gamma = params

    return bottlegrowth_mig_w_dips_sel((T_WGD, nuB, nuF, m, gamma, gamma), ns, pts)
bottlegrowth_mig_w_dips_sel_single_gamma.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'm', 'gamma']

def bottlegrowth_asym_mig_w_dips_sel_single_gamma(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, nuB, nuF, m12, m21, gamma)

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - m12: migration rate from pop 2 (auotetraploids) into pop 1 (diploids) (2*Na*m12)

            - m21: migration rate from pop 1 (diploids) into pop 2 (auotetraploids) (2*Na*m21)

            - gamma: population-scaled selection coefficient (= 2*Na*s)
                Note that this is used for both the diploid and autotetraploid populations.
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, nuB, nuF, m12, m21, gamma = params

    return bottlegrowth_asym_mig_w_dips_sel((T_WGD, nuB, nuF, m12, m21, gamma, gamma), ns, pts)
bottlegrowth_asym_mig_w_dips_sel_single_gamma.__param_names__ = ['T_WGD', 'nuB', 'nuF', 'm12', 'm21', 'gamma']


def bottlegrowth_dip_size_change_sel_single_gamma(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, T_dip, nuB, nuF, nu_dip, gamma)

            - T_WGD: Time between the WGD and diploid size change (in units of 2*Na generations).

            - T_dip: Time in the past at which the diploid population changes size
                (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - nu_dip: Ratio of contemporary diploid population to ancestral diploid population size

            - gamma: population-scaled selection coefficient (= 2*Na*s)
                Note that this is used for both the diploid and autotetraploid populations.
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, T_dip, nuB, nuF, nu_dip, gamma = params

    return bottlegrowth_dip_size_change_sel((T_WGD, T_dip, nuB, nuF, nu_dip, gamma, gamma), ns, pts)
bottlegrowth_dip_size_change_sel_single_gamma.__param_names__ = ['T_WGD', 'T_dip', 'nuB', 'nuF', 'nu_dip', 'gamma']


def bottlegrowth_dip_size_change_mig_sel_single_gamma(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, T_dip, nuB, nuF, nu_dip, m, gamma)

            - T_WGD: Time between the WGD and diploid size change (in units of 2*Na generations).

            - T_dip: Time in the past at which the diploid population changes size
                (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - nu_dip: Ratio of contemporary diploid population to ancestral diploid population size

            - m: symmetric migration rate between the two populations (2*Na*m)

            - gamma: population-scaled selection coefficient (= 2*Na*s)
                Note that this is used for both the diploid and autotetraploid populations.
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, T_dip, nuB, nuF, nu_dip, m, gamma = params

    return bottlegrowth_dip_size_change_mig_sel((T_WGD, T_dip, nuB, nuF, nu_dip, m, gamma, gamma), ns, pts)
bottlegrowth_dip_size_change_mig_sel_single_gamma.__param_names__ = ['T_WGD', 'T_dip', 'nuB', 'nuF', 'nu_dip', 'm', 'gamma']


def bottlegrowth_dip_size_change_asym_mig_sel_single_gamma(params, ns, pts):
    """
    Two population model of autotetraploid formation where the 
    autotetraploid population splits and maintains a size of nu_auto.

    Parameters:
        params (tuple): (T_WGD, T_dip, nuB, nuF, nu_dip, m12, m21, gamma)

            - T_WGD: Time between the WGD and diploid size change (in units of 2*Na generations).

            - T_dip: Time in the past at which the diploid population changes size
                (in units of 2*Na generations).

            - nuB: Ratio of bottlenecked autotetraploid to ancestral diploid population size s
               (ratio of *census* sizes).

            - nuF: Ratio of final or contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - nu_dip: Ratio of contemporary diploid population to ancestral diploid population size

            - m12: migration rate from pop 2 (auotetraploids) into pop 1 (diploids) (2*Na*m12)

            - m21: migration rate from pop 1 (diploids) into pop 2 (auotetraploids) (2*Na*m21)

            - gamma: population-scaled selection coefficient (= 2*Na*s)
                Note that this is used for both the diploid and autotetraploid populations.
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_WGD, T_dip, nuB, nuF, nu_dip, m12, m21, gamma = params

    return bottlegrowth_dip_size_change_asym_mig_sel((T_WGD, T_dip, nuB, nuF, nu_dip, m12, m21, gamma, gamma), ns, pts)
bottlegrowth_asym_mig_w_dips_sel_single_gamma.__param_names__ = ['T_WGD', 'T_dip', 'nuB', 'nuF', 'nu_dip', 'm12', 'm21', 'gamma']
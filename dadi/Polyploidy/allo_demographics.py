from dadi import Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum
from . import Integration as PolyInt
import numpy

### Single allotetraploid population models

def two_epoch(params, ns, pts):
    """
    Two epoch model of allotetraploid formation where the 
    diploid progenitors diverge for 1 diffusion unit and then the
    allotetraploid population splits and maintains a size of nu.
    
    Parameters:
        params (tuple): (T_div, T_WGD, nu, H)
            - T_div: Divergence period between the two diploid progenitors 
               (in units of 2*Na generations).

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu: Ratio of contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).

            - H: homoeologous exchange rate (in terms of 2*Na*eta)
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting (collapsed) frequency spectrum.
    """
    T_div, T_WGD, nu, H = params
    alloaflag = PolyInt.PloidyType.ALLOa
    allobflag = PolyInt.PloidyType.ALLOb
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PolyInt.two_pops(phi, xx, T_div)
    # then, integrate for T_WGD with allotetraploids
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu1=nu, nu2=nu, m12=H, m21=H,
                           ploidyflag1=alloaflag, ploidyflag2=allobflag)
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
two_epoch.__param_names__ = ['T_div', 'T_WGD', 'nu', 'H']

def two_epoch_noHE(params, ns, pts):
    """
    Two epoch model of allotetraploid formation where the 
    diploid progenitors diverge for 1 diffusion unit and then the
    allotetraploid population splits and maintains a size of nu.
    
    Parameters:
        params (tuple): (T_div, T_WGD, nu)
            - T_div: Divergence period between the two diploid progenitors 
               (in units of 2*Na generations).

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               autotetraploid population (in units of 2*Na generations).

            - nu: Ratio of contemporary autotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting (collapsed) frequency spectrum.
    """
    T_div, T_WGD, nu = params
    fs = two_epoch((T_div, T_WGD, nu, 0), ns, pts)
    return fs
two_epoch_noHE.__param_names__ = ['T_div', 'T_WGD', 'nu']

def bottlegrowth(params, ns, pts):
    """
    Bottlegrowth model of allotetraploid formation where the 
    allotetraploid population starts with size nuWGD and 
    grows exponentially to a size of nuF
    
    Parameters:
        params (tuple): (T_div, T_WGD, nuWGD, nuF, H)
            - T_div: Divergence period between the two diploid progenitors 
               (in units of 2*Na generations).

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               allotetraploid population (in units of 2*Na generations).

            - nuWGD: Ratio of allotetraploid population immediately after WGD
                to ancestral diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary allotetraploid population
                to ancestral diploid population size (ratio of *census* sizes).

            - H: homoeologous exchange rate (in terms of 2*Na*eta)
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_div, T_WGD, nuWGD, nuF, H = params
    nu_f = lambda t: nuWGD*numpy.exp(numpy.log(nuF/nuWGD) * t/T_WGD)
    alloaflag = PolyInt.PloidyType.ALLOa
    allobflag = PolyInt.PloidyType.ALLOb
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PolyInt.two_pops(phi, xx, T_div)
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu1=nu_f, nu2=nu_f, m12=H, m21=H,
                           ploidyflag1=alloaflag, ploidyflag2=allobflag)
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
bottlegrowth.__param_names__ = ["T_div", "T_WGD", "nuWGD", "nuF", "H"]

def bottlegrowth_noHE(params, ns, pts):
    """
    Bottlegrowth model of allotetraploid formation where the 
    allotetraploid population starts with size nuWGD and 
    grows exponentially to a size of nuF
    
    Parameters:
        params (tuple): (T_div, T_WGD, nuWGD, nuF)
            - T_div: Divergence period between the two diploid progenitors 
               (in units of 2*Na generations).

            - T_WGD: Time in the past at which the WGD occurred, creating the  
               allotetraploid population (in units of 2*Na generations).

            - nuWGD: Ratio of allotetraploid population immediately after WGD
                to ancestral diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary allotetraploid population
                to ancestral diploid population size (ratio of *census* sizes).
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_div, T_WGD, nuWGD, nuF = params
    fs = bottlegrowth((T_div, T_WGD, nuWGD, nuF, 0), ns, pts)
    return fs
bottlegrowth_noHE.__param_names__ = ["T_div", "T_WGD", "nuWGD", "nuF"]

def three_epoch(params, ns, pts):
    """
    Three epoch model of allotetraploid formation where the 
    allotetraploid population splits, maintains a size of nuWGD for T_WGD, 
    and then changes size again to nuF for a period of TF.
    This is similar to having a bottleneck for some period and then recover after the bottleneck.
    
    Parameters:
        params (tuple): (T_WGD, TF, nuWGD, nuF, H)
            - T_div: Divergence period between the two diploid progenitors 
               (in units of 2*Na generations).

            - T_WGD: Time length between the WGD event and second size change, creating the  
               allotetraploid population (in units of 2*Na generations).

            - TF: Time in the past at which the second epoch begins.

            - nuWGD: Ratio of initial allotetraploid population (during first epoch)
                 to ancestral diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary allotetraploid population (during second epoch)
                 to ancestral diploid population size (ratio of *census* sizes).

            - H: homoeologous exchange rate (in terms of 2*Na*eta)
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_div, T_WGD, TF, nuWGD, nuF, H  = params
    alloaflag = PolyInt.PloidyType.ALLOa
    allobflag = PolyInt.PloidyType.ALLOb
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PolyInt.two_pops(phi, xx, T_div)
    # second epoch
    phi = PolyInt.two_pops(phi, xx, T_WGD, nu1=nuWGD, nu2=nuWGD, m12=H, m21=H,
                           ploidyflag1=alloaflag, ploidyflag2=allobflag)
    # third epoch
    phi = PolyInt.two_pops(phi, xx, TF, nu1=nuF, nu2=nuF, m12=H, m21=H,
                           ploidyflag1=alloaflag, ploidyflag2=allobflag)
    fs = Spectrum.from_phi(phi, ns, (xx,xx))
    return fs
three_epoch.__param_names__ = ['T_div', 'T_WGD', 'TF', 'nuWGD', 'nuF', 'H']


def three_epoch_noHE(params, ns, pts):
    """
    Three epoch model of allotetraploid formation where the 
    allotetraploid population splits, maintains a size of nuWGD for T_WGD, 
    and then changes size again to nuF for a period of TF.
    This is similar to having a bottleneck for some period and then recover after the bottleneck.
    
    Parameters:
        params (tuple): (T_div, T_WGD, TF, nuWGD, nuF)
            - T_div: Divergence period between the two diploid progenitors 
               (in units of 2*Na generations).

            - T_WGD: Time length between the WGD event and second size change, creating the  
               allotetraploid population (in units of 2*Na generations).

            - TF: Time in the past at which the second epoch begins.

            - nuWGD: Ratio of initial allotetraploid population (during first epoch)
                 to ancestral diploid population size (ratio of *census* sizes).

            - nuF: Ratio of contemporary allotetraploid population (during second epoch)
                 to ancestral diploid population size (ratio of *census* sizes).
        ns (tuple): Sample sizes (n1,n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    T_div, T_WGD, TF, nuWGD, nuF = params
    fs = three_epoch((T_div, T_WGD, TF, nuWGD, nuF, 0), ns, pts)
    return fs
three_epoch_noHE.__param_names__ = ['T_div', 'T_WGD', 'TF', 'nuWGD', 'nuF']


### Single allotetraploid population models with the diploid progenitors
def bottleneck_asym_mig_w_dips(params, ns, pts):
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

            - nu_allo: Ratio of contemporary allotetraploid to ancestral diploid population size 
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
bottleneck_asym_mig_w_dips.__param_names__ = ['T_div', 'T_WGD', 'nu_allo', 'H', 'm31', 'm42']

def bottleneck_mig_w_dips(params, ns, pts):
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

            - nu_allo: Ratio of contemporary allotetraploid to ancestral diploid population size 
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
    return bottleneck_asym_mig_w_dips((T_div, T_WGD, nu_allo, H, m, m), ns, pts)
bottleneck_mig_w_dips.__param_names__ = ['T_div', 'T_WGD', 'nu_allo', 'H', 'm']

def bottleneck_w_dips(params, ns, pts):
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

            - nu_allo: Ratio of contemporary allotetraploid to ancestral diploid population size 
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
    return bottleneck_asym_mig_w_dips((T_div, T_WGD, nu_allo, H, 0, 0), ns, pts)
bottleneck_w_dips.__param_names__ = ['T_div', 'T_WGD', 'nu_allo', 'H']

def bottleneck_noHE_w_dips(params, ns, pts):
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

            - nu_allo: Ratio of contemporary allotetraploid to ancestral diploid population size 
               (ratio of *census* sizes).
        ns (tuple): Sample sizes (n1, n2, n3, n4).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.

    Raises:
        ValueError: If `params` does not contain the expected number of elements.
    """
    T_div, T_WGD, nu_allo = params
    return bottleneck_asym_mig_w_dips((T_div, T_WGD, nu_allo, 0, 0, 0), ns, pts)
bottleneck_noHE_w_dips.__param_names__ = ['T_div', 'T_WGD', 'nu_allo']

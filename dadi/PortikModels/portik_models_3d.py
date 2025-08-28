'''
Models for testing various three population scenarios.

Adapted from https://github.com/dportik
'''
from dadi import Numerics, PhiManip, Integration
from dadi.Spectrum_mod import Spectrum

###############################################################################
# Basic models of (no gene flow / gene flow)
# between (all / some) population pairs
###############################################################################


def split_nomig(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3.

    Parameters:
        params (list): A list of 6 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 6 parameters
    nu1, nuA, nu2, nu3, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=0, m21=0)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
split_nomig.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'T1', 'T2']


def split_symmig_all(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3, with symmetrical 
    migration between all population pairs.

    Parameters:
        params (list): A list of 10 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - m3 (float): Migration rate between populations 1 and 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        dadi.Spectrum_mod.Spectrum: The resulting frequency spectrum.
    """
    # 10 parameters
    nu1, nuA, nu2, nu3, mA, m1, m2, m3, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=m1, m21=m1, m23=m2, m32=m2, m13=m3, m31=m3)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
split_symmig_all.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'm1', 'm2', 'm3', 'T1', 'T2']


def split_symmig_adjacent(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3. Assumes population 2 
    occurs between populations 1 and 3, which do not come into contact.

    Parameters:
        params (list): A list of 9 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 9 parameters
    nu1, nuA, nu2, nu3, mA, m1, m2, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=m1, m21=m1, m23=m2, m32=m2, m13=0, m31=0)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
split_symmig_adjacent.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'm1', 'm2', 'T1', 'T2']

################################################################################
# Various models based on forest refugia timing,
# all with symmetric gene flow estimates
###############################################################################


def refugia_adj_1(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3, with no gene flow. 
    After all splits, symmetric secondary contact occurs between adjacent 
    populations (1<->2, 2<->3, but not 1<->3).

    Parameters:
        params (list): A list of 9 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
            
            - T3 (float): The scaled time between the secondary contact 
              and the present (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 9 parameters
    nu1, nuA, nu2, nu3, m1, m2, T1, T2, T3 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=0, m21=0)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T3, nu1=nu1, nu2=nu2, nu3=nu3, m12=m1, m21=m1, m23=m2, m32=m2, m13=0, m31=0)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
refugia_adj_1.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'm1', 'm2', 'T1', 'T2', 'T3']


def refugia_adj_2(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3, with gene flow. 
    After the appearance of populations 2 and 3, gene flow also occurs 
    between populations 1 and 2.

    Parameters:
        params (list): A list of 8 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 8 parameters
    nu1, nuA, nu2, nu3, m1, m2, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=0, m21=0)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=m1, m21=m1, m23=m2, m32=m2, m13=0, m31=0)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
refugia_adj_2.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'm1', 'm2', 'T1', 'T2']


def refugia_adj_3(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3. Initially, no gene 
    flow occurs, but secondary contact introduces gene flow between 
    populations 1 and 2, and between populations 2 and 3.

    Parameters:
        params (list): A list of 10 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - T1a (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T1b (float): The scaled time for secondary contact between 
              populations 1 and (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 10 parameters
    nu1, nuA, nu2, nu3, mA, m1, m2, T1a, T1b, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1a, nu1=nu1, nu2=nuA, m12=0, m21=0)
    phi = Integration.two_pops(phi, xx, T1b, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=m1, m21=m1, m23=m2, m32=m2, m13=0, m31=0)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
refugia_adj_3.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'm1', 'm2', 'T1a', 'T1b', 'T2']

###############################################################################
# Various models based on ancient migration and contemporary isolation
###############################################################################


def ancmig_adj_3(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by gene flow, which then stops. A subsequent split occurs 
    between populations 2 and 3, with no gene flow.

    Parameters:
        params (list): A list of 8 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - T1a (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T1b (float): The scaled time for no gene flow between populations 
              1 and (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the cessation of gene flow 
              and the split of populations 2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 8 parameters
    nu1, nuA, nu2, nu3, mA, T1a, T1b, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1a, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = Integration.two_pops(phi, xx, T1b, nu1=nu1, nu2=nuA, m12=0, m21=0)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
ancmig_adj_3.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'T1a', 'T1b', 'T2']


def ancmig_adj_2(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by gene flow. A subsequent split occurs between populations 
    2 and 3, after which all gene flow ceases.

    Parameters:
        params (list): A list of 7 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 7 parameters
    nu1, nuA, nu2, nu3, mA, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
ancmig_adj_2.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'T1', 'T2']


def ancmig_adj_1(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by gene flow. A subsequent split occurs between populations 
    2 and 3 with gene flow, after which all gene flow ceases.

    Parameters:
        params (list): A list of 10 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
            
            - T3 (float): The scaled time between the cessation of gene flow 
              and the present (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 10 parameters
    nu1, nuA, nu2, nu3, mA, m1, m2, T1, T2, T3 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=m1, m21=m1, m23=m2, m32=m2, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T3, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
ancmig_adj_1.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'm1', 'm2', 'T1', 'T2', 'T3']


###############################################################################
# Simultaneous split models (with/without migration/secondary contact
# and size changes)
# Written for Barratt et al. (2018)
###############################################################################

def sim_split_no_mig(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3, with no gene flow occurring between any population pairs.

    Parameters:
        params (list): A list of 4 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - T1 (float): The scaled time between the split and the present 
              (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 4 parameters
    nu1, nu2, nu3, T1 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_no_mig.__param_names__ = ['nu1', 'nu2', 'nu3', 'T1']


def sim_split_no_mig_size(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3, with no gene flow occurring. However, size changes occur 
    after the split.

    Parameters:
        params (list): A list of 8 parameters:
            
            - nu1a (float): Size of population 1 after the split.
            
            - nu2a (float): Size of population 2 after the split.
            
            - nu3a (float): Size of population 3 after the split.
            
            - nu1b (float): Size of population 1 after the size change.
            
            - nu2b (float): Size of population 2 after the size change.
            
            - nu3b (float): Size of population 3 after the size change.
            
            - T1 (float): The scaled time between the split and the size change 
              (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the size change and the present 
              (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 8 parameters
    nu1a, nu2a, nu3a, nu1b, nu2b, nu3b, T1, T2 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1=nu1a, nu2=nu2a, nu3=nu3a,
        m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1b, nu2=nu2b, nu3=nu3b,
        m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_no_mig_size.__param_names__ = ['nu1a', 'nu2a', 'nu3a', 'nu1b', 'nu2b', 'nu3b', 'T1', 'T2']


def sim_split_sym_mig_all(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3, with symmetrical migration occurring between all population pairs.

    Parameters:
        params (list): A list of 7 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - m3 (float): Migration rate between populations 1 and 3.
            
            - T1 (float): The scaled time between the split and the present 
              (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 7 parameters
    nu1, nu2, nu3, m1, m2, m3, T1 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1=nu1, nu2=nu2, nu3=nu3, m12=m1, m21=m1, m23=m2, m32=m2, m13=m3, m31=m3)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_sym_mig_all.__param_names__ = ['nu1', 'nu2', 'nu3', 'm1', 'm2', 'm3', 'T1']


def sim_split_sym_mig_adjacent(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3, with symmetrical migration occurring between adjacent population pairs (ie 1<->2, 2<->3, but not 1<->3).

    Parameters:
        params (list): A list of 6 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - T1 (float): The scaled time between the split and the present 
              (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 6 parameters
    nu1, nu2, nu3, m1, m2, T1 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1=nu1, nu2=nu2, nu3=nu3, m12=m1, m21=m1, m23=m2, m32=m2, m13=0, m31=0)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_sym_mig_adjacent.__param_names__ = ['nu1', 'nu2', 'nu3', 'm1', 'm2', 'T1']


def sim_split_refugia_sym_mig_all(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3, followed by isolation. A period of symmetric secondary contact 
    occurs between all populations after all splits are complete.

    Parameters:
        params (list): A list of 8 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - m3 (float): Migration rate between populations 1 and 3.
            
            - T1 (float): The scaled time between the split and the present 
              (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the migration (secondary contact) 
              and the present (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 8 parameters
    nu1, nu2, nu3, m1, m2, m3, T1, T2 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1=nu1, nu2=nu2, nu3=nu3,
        m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3,
        m12=m1, m21=m1, m23=m2, m32=m2, m13=m3, m31=m3)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_refugia_sym_mig_all.__param_names__ = ['nu1', 'nu2', 'nu3', 'm1', 'm2', 'm3', 'T1', 'T2']


def sim_split_refugia_sym_mig_adjacent(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3, followed by isolation. A period of symmetric secondary contact 
    occurs between adjacent populations (1<->2, 2<->3, but not 1<->3) 
    after all splits are complete.

    Parameters:
        params (list): A list of 7 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - T1 (float): The scaled time between the split and the present 
              (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the migration (secondary contact) 
              and the present (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 7 parameters
    nu1, nu2, nu3, m1, m2, T1, T2 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1=nu1, nu2=nu2, nu3=nu3,
        m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3,
        m12=m1, m21=m1, m23=m2, m32=m2, m13=0, m31=0)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_refugia_sym_mig_adjacent.__param_names__ = ['nu1', 'nu2', 'nu3', 'm1', 'm2', 'T1', 'T2']


###############################################################
# Models with extra size change step (potential human impact)
# Written for Barratt et al. (2018)
###############################################################

def split_nomig_size(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3. No migration occurs 
    between any population pairs, but size changes occur.

    Parameters:
        params (list): A list of 10 parameters:
            
            - nu1a (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2a (float): Size of population 2 after the split.
            
            - nu3a (float): Size of population 3 after the split.
            
            - nu1b (float): Size of population 1 after the size change.
            
            - nu2b (float): Size of population 2 after the size change.
            
            - nu3b (float): Size of population 3 after the size change.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
            
            - T3 (float): The scaled time between the size change and the present 
              (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 10 parameters
    nu1a, nuA, nu2a, nu3a, nu1b, nu2b, nu3b, T1, T2, T3 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1a, nuA, m12=0, m21=0)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1a, nu2a, nu3a, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T3, nu1b, nu2b, nu3b, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
split_nomig_size.__param_names__ = ['nu1a', 'nuA', 'nu2a', 'nu3a', 'nu1b', 'nu2b', 'nu3b', 'T1', 'T2', 'T3']


def ancmig_2_size(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by gene flow. A subsequent split occurs between populations 
    2 and 3, after which all gene flow ceases, and size changes occur.

    Parameters:
        params (list): A list of 11 parameters:
            
            - nu1a (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2a (float): Size of population 2 after the split.
            
            - nu3a (float): Size of population 3 after the split.
            
            - nu1b (float): Size of population 1 after the size change.
            
            - nu2b (float): Size of population 2 after the size change.
            
            - nu3b (float): Size of population 3 after the size change.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
            
            - T3 (float): The scaled time between the size change and the present 
              (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 11 parameters
    nu1a, nuA, nu2a, nu3a, nu1b, nu2b, nu3b, mA, T1, T2, T3 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1a, nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1a, nu2a, nu3a, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T3, nu1b, nu2b, nu3b, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
ancmig_2_size.__param_names__ = ['nu1a', 'nuA', 'nu2a', 'nu3a', 'nu1b', 'nu2b', 'nu3b', 'mA', 'T1', 'T2', 'T3']


def sim_split_refugia_sym_mig_adjacent_size(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3, followed by isolation. A period of symmetric secondary contact 
    occurs between adjacent populations (1<->2, 2<->3, but not 1<->3) 
    after all splits are complete, and size changes occur.

    Parameters:
        params (list): A list of 11 parameters:
            
            - nu1a (float): Size of population 1 after the split.
            
            - nu2a (float): Size of population 2 after the split.
            
            - nu3a (float): Size of population 3 after the split.
            
            - nu1b (float): Size of population 1 after the size change.
            
            - nu2b (float): Size of population 2 after the size change.
            
            - nu3b (float): Size of population 3 after the size change.
            
            - m1 (float): Migration rate between populations 1 and 2 (2*Na*m).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - T1 (float): The scaled time between the split and the present 
              (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the migration (secondary contact) 
              and the present (in units of 2*Na generations).
            
            - T3 (float): The scaled time between the size change and the present 
              (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 11 parameters
    nu1a, nu2a, nu3a, nu1b, nu2b, nu3b, m1, m2, T1, T2, T3 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1a, nu2a, nu3a, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T2, nu1a, nu2a, nu3a, m12=m1, m21=m1, m23=m2, m32=m2, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T3, nu1b, nu2b, nu3b, m12=m1, m21=m1, m23=m2, m32=m2, m13=0, m31=0)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_refugia_sym_mig_adjacent_size.__param_names__ = ['nu1a', 'nu2a', 'nu3a', 'nu1b', 'nu2b', 'nu3b', 'm1', 'm2', 'T1', 'T2', 'T3']


###############################################################
# Variation on divergence, with pop3 geographically between
# pop1 and pop2
# Written for Firneno et al. (2020)
###############################################################

# but with pop 3 treated as 'middle' population
def refugia_adj_2_var_sym(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3, with gene flow. 
    After the appearance of populations 2 and 3, gene flow also occurs 
    between populations 1 and 3.

    Parameters:
        params (list): A list of 8 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - m3 (float): Migration rate between populations 1 and 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 8 parameters
    nu1, nuA, nu2, nu3, m2, m3, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=0, m21=0)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=m2, m32=m2, m13=m3, m31=m3)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
refugia_adj_2_var_sym.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'm2', 'm3', 'T1', 'T2']


# but with pop 3 treated as 'middle' population
def refugia_adj_2_var_uni(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3. Gene flow occurs 
    from population 1 to population 3 (unidirectional), and from 
    population 2 to population 3 (unidirectional).

    Parameters:
        params (list): A list of 8 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m32 (float): Migration rate from population 2 to population 3 (2*Na*m).
            
            - m31 (float): Migration rate from population 1 to population 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 8 parameters
    nu1, nuA, nu2, nu3, m32, m31, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=0, m21=0)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=m32, m13=0, m31=m31)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
refugia_adj_2_var_uni.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'm32', 'm31', 'T1', 'T2']


# but with pop 3 treated as 'middle' population
def refugia_adj_3_var_sym(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3. Initially, no gene 
    flow occurs, but secondary contact introduces gene flow between 
    populations 1 and 3, and between populations 2 and 3.

    Parameters:
        params (list): A list of 10 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - m3 (float): Migration rate between populations 1 and 3.
            
            - T1a (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T1b (float): The scaled time for secondary contact between 
              populations 1 and (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 10 parameters
    nu1, nuA, nu2, nu3, mA, m2, m3, T1a, T1b, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1a, nu1=nu1, nu2=nuA, m12=0, m21=0)
    phi = Integration.two_pops(phi, xx, T1b, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=m2, m32=m2, m13=m3, m31=m3)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
refugia_adj_3_var_sym.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'm2', 'm3', 'T1a', 'T1b', 'T2']


# but with pop 3 treated as 'middle' population
def refugia_adj_3_var_uni(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3. Initially, no gene 
    flow occurs, but secondary contact introduces gene flow from 
    population 1 to population 3 (unidirectional), and from population 
    2 to population 3 (unidirectional).

    Parameters:
        params (list): A list of 10 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - m32 (float): Migration rate from population 2 to population 3 (2*Na*m).
            
            - m31 (float): Migration rate from population 1 to population 3.
            
            - T1a (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T1b (float): The scaled time for secondary contact between 
              populations 1 and (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 10 parameters
    nu1, nuA, nu2, nu3, mA, m32, m31, T1a, T1b, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1a, nu1=nu1, nu2=nuA, m12=0, m21=0)
    phi = Integration.two_pops(phi, xx, T1b, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=m32, m13=0, m31=m31)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
refugia_adj_3_var_uni.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'm32', 'm31', 'T1a', 'T1b', 'T2']


# but with pop 3 treated as 'middle' population
def split_sym_mig_adjacent_var1(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3. Symmetrical migration 
    occurs between adjacent population pairs (1<->3, 2<->3, but not 1<->2).

    Parameters:
        params (list): A list of 9 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - m3 (float): Migration rate between populations 1 and 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 9 parameters
    nu1, nuA, nu2, nu3, mA, m2, m3, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=m2, m32=m2, m13=m3, m31=m3)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
split_sym_mig_adjacent_var1.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'm2', 'm3', 'T1', 'T2']

# but with pop 3 treated as 'middle' population


def split_uni_mig_adjacent_var1(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3. Gene flow occurs 
    from population 1 to population 3 (unidirectional), and from 
    population 2 to population 3 (unidirectional).

    Parameters:
        params (list): A list of 9 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - m32 (float): Migration rate from population 2 to population 3 (2*Na*m).
            
            - m31 (float): Migration rate from population 1 to population 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 9 parameters
    nu1, nuA, nu2, nu3, mA, m32, m31, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=m32, m13=0, m31=m31)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
split_uni_mig_adjacent_var1.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'm32', 'm31', 'T1', 'T2']

# but with pop 3 treated as 'middle' population


def split_sym_mig_adjacent_var2(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3. Symmetrical migration 
    occurs between population 1 and population (2,3), then between 
    population 1 and population 3 only.

    Parameters:
        params (list): A list of 8 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - m3 (float): Migration rate between populations 1 and 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 8 parameters
    nu1, nuA, nu2, nu3, mA, m3, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=m3, m31=m3)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
split_sym_mig_adjacent_var2.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'm3', 'T1', 'T2']


# but with pop 3 treated as 'middle' population
def split_uni_mig_adjacent_var2(params, ns, pts):
    """
    Simulates a model with a split between population 1 and (2,3), 
    followed by a split between populations 2 and 3. Symmetrical migration 
    occurs between population 1 and population (2,3), then from population 
    1 to population 3 (unidirectional).

    Parameters:
        params (list): A list of 8 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nuA (float): Size of population (2,3) after the split from 1.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - mA (float): Migration rate between population 1 and population (2,3).
            
            - m31 (float): Migration rate from population 1 to population 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs (2,3) (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the split of populations 
              2 and 3 (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 8 parameters
    nu1, nuA, nu2, nu3, mA, m31, T1, T2 = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nuA, m12=mA, m21=mA)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=0, m31=m31)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
split_uni_mig_adjacent_var2.__param_names__ = ['nu1', 'nuA', 'nu2', 'nu3', 'mA', 'm31', 'T1', 'T2']


###############################################################
# Variation on simultaneous split models, with pop3
# geographically between pop1 and pop2
# Written for Firneno et al. (2020)
###############################################################

# but with pop 3 treated as 'middle' population
def sim_split_sym_mig_adjacent_var(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3, with symmetrical migration occurring between adjacent population pairs.

    Parameters:
        params (list): A list of 6 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - m3 (float): Migration rate between populations 1 and 3.
            
            - T1 (float): The scaled time between the split and the present 
              (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 6 parameters
    nu1, nu2, nu3, m2, m3, T1 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=m2, m32=m2, m13=m3, m31=m3)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_sym_mig_adjacent_var.__param_names__ = ['nu1', 'nu2', 'nu3', 'm2', 'm3', 'T1']

# but with pop 3 treated as 'middle' population


def sim_split_uni_mig_adjacent_var(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3. Gene flow occurs from population 1 to population 3 (unidirectional), 
    and from population 2 to population 3 (unidirectional).

    Parameters:
        params (list): A list of 6 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m32 (float): Migration rate from population 2 to population 3 (2*Na*m).
            
            - m31 (float): Migration rate from population 1 to population 3.
            
            - T1 (float): The scaled time between the split and the present 
              (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 6 parameters
    nu1, nu2, nu3, m32, m31, T1 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=m32, m13=0, m31=m31)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_uni_mig_adjacent_var.__param_names__ = ['nu1', 'nu2', 'nu3', 'm32', 'm31', 'T1']


# but with pop 3 treated as 'middle' population
def sim_split_refugia_sym_mig_adjacent_var(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3, followed by isolation. A period of symmetric secondary contact 
    occurs between adjacent populations (1<->3, 2<->3, but not 1<->2) 
    after all splits are complete.

    Parameters:
        params (list): A list of 7 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - m3 (float): Migration rate between populations 1 and 3.
            
            - T1 (float): The scaled time between the split and the present 
              (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the migration (secondary contact) 
              and the present (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 7 parameters
    nu1, nu2, nu3, m2, m3, T1, T2 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=m2, m32=m2, m13=m3, m31=m3)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_refugia_sym_mig_adjacent_var.__param_names__ = ['nu1', 'nu2', 'nu3', 'm2', 'm3', 'T1', 'T2']

# but with pop 3 treated as 'middle' population


def sim_split_refugia_uni_mig_adjacent_var(params, ns, pts):
    """
    Simulates a model with a simultaneous split between populations 1, 2, 
    and 3, followed by isolation. After all splits are complete, gene flow 
    occurs from population 1 to population 3 (unidirectional), and from 
    population 2 to population 3 (unidirectional).

    Parameters:
        params (list): A list of 7 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the split.
            
            - m32 (float): Migration rate from population 2 to population 3 (2*Na*m).
            
            - m31 (float): Migration rate from population 1 to population 3.
            
            - T1 (float): The scaled time between the split and the present 
              (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the migration (secondary contact) 
              and the present (in units of 2*Na generations).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 7 parameters
    nu1, nu2, nu3, m32, m31, T1, T2 = params
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = Integration.three_pops(
        phi, xx, T1, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=m32, m13=0, m31=m31)
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
sim_split_refugia_uni_mig_adjacent_var.__param_names__ = ['nu1', 'nu2', 'nu3', 'm32', 'm31', 'T1', 'T2']


###############################################################
# Admixed ("hybrid") origin models
# Written for Firneno et al. (2020)
###############################################################

def admix_origin_no_mig(params, ns, pts):
    """
    Simulates a model with a split between population 1 and 2, 
    followed by the origin of population 3 through admixture from 
    populations 1 and 2. No gene flow occurs after the origin of population 3.

    Parameters:
        params (list): A list of 6 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the origin.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs 2 and the origin of population 3 (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the origin of population 3 
              and the present (in units of 2*Na generations).
            
            - f (float): Fraction of population 3 derived from population 1 
              (with fraction 1-f derived from population 2).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 6 parameters
    nu1, nu2, nu3, T1, T2, f = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nu2, m12=0, m21=0)
    phi = PhiManip.phi_2D_to_3D_admix(phi, f, xx, xx, xx)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=0, m13=0, m31=0)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
admix_origin_no_mig.__param_names__ = ['nu1', 'nu2', 'nu3', 'T1', 'T2', 'f']


def admix_origin_sym_mig_adj(params, ns, pts):
    """
    Simulates a model with a split between population 1 and 2, 
    followed by the origin of population 3 through admixture from 
    populations 1 and 2. Symmetrical gene flow occurs between 
    population 3 and populations 1 and 2.

    Parameters:
        params (list): A list of 8 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the origin.
            
            - m2 (float): Migration rate between populations 2 and 3.
            
            - m3 (float): Migration rate between populations 1 and 3 (2*Na*m).
            
            - T1 (float): The scaled time between the split of populations 
              1 vs 2 and the origin of population 3 (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the origin of population 3 
              and the present (in units of 2*Na generations).
            
            - f (float): Fraction of population 3 derived from population 1 
              (with fraction 1-f derived from population 2).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 8 parameters
    nu1, nu2, nu3, m2, m3, T1, T2, f = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nu2, m12=0, m21=0)
    phi = PhiManip.phi_2D_to_3D_admix(phi, f, xx, xx, xx)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=m2, m32=m2, m13=m3, m31=m3)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
admix_origin_sym_mig_adj.__param_names__ = ['nu1', 'nu2', 'nu3', 'm2', 'm3', 'T1', 'T2', 'f']


def admix_origin_uni_mig_adj(params, ns, pts):
    """
    Simulates a model with a split between population 1 and 2, 
    followed by the origin of population 3 through admixture from 
    populations 1 and 2. Gene flow occurs from population 1 to 
    population 3 (unidirectional), and from population 2 to 
    population 3 (unidirectional).

    Parameters:
        params (list): A list of 8 parameters:
            
            - nu1 (float): Size of population 1 after the split.
            
            - nu2 (float): Size of population 2 after the split.
            
            - nu3 (float): Size of population 3 after the origin.
            
            - m32 (float): Migration rate from population 2 to population 3 (2*Na*m).
            
            - m31 (float): Migration rate from population 1 to population 3.
            
            - T1 (float): The scaled time between the split of populations 
              1 vs 2 and the origin of population 3 (in units of 2*Na generations).
            
            - T2 (float): The scaled time between the origin of population 3 
              and the present (in units of 2*Na generations).
            
            - f (float): Fraction of population 3 derived from population 1 
              (with fraction 1-f derived from population 2).
        ns (list): Sample sizes for the resulting spectrum.
        pts (int): Number of grid points for numerical integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # 8 parameters
    nu1, nu2, nu3, m32, m31, T1, T2, f = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, T1, nu1=nu1, nu2=nu2, m12=0, m21=0)
    phi = PhiManip.phi_2D_to_3D_admix(phi, f, xx, xx, xx)
    phi = Integration.three_pops(
        phi, xx, T2, nu1=nu1, nu2=nu2, nu3=nu3, m12=0, m21=0, m23=0, m32=m32, m13=0, m31=m31)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs
admix_origin_uni_mig_adj.__param_names__ = ['nu1', 'nu2', 'nu3', 'm32', 'm31', 'T1', 'T2', 'f']

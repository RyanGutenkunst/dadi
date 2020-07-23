"""
For examples of dadi's usage, see the examples directory in the source
distribution.

Documentation of all methods can be found in doc/api/index.html of the source
distribution.
"""
import logging
logging.basicConfig()

from . import Demographics1D, Demographics2D, Inference, Integration
from . import Misc, Numerics, PhiManip, Spectrum_mod, tridiag

# We do it this way so it's easier to reload.
Spectrum = Spectrum_mod.Spectrum

# Protect import of Plotting in case matplotlib not installed.
try:
    from . import Plotting
except ImportError:
    pass

# When doing arithmetic with Spectrum objects (which are masked arrays), we
# often have masked values which generate annoying arithmetic warnings. Here
# we tell numpy to ignore such warnings. This puts greater onus on the user to
# check results, but for our use case I think it's the better default.
import numpy
numpy.seterr(all='ignore')

def cuda_enabled(toggle=None):
    """
    Enable or disable cuda execution
    """
    if toggle is None:
        return Integration.cuda_enabled
    elif toggle == True:
        try:
            from . import cuda
            Integration.cuda_enabled = True
            return True
        except:
            print("Failed to import dadi.cuda")
            return False
    elif toggle == False:
        Integration.cuda_enabled = False
        return False
    else:
        raise ValueError("toggle must be True, False, or None")

def pts_to_RAM(pts, P):
    """
    Approximate RAM usage for a given grid points and number of populations 

    pts: Grid points setting
    P: Number of populations
    """
    return 8*4*pts**P / 1024**3

def RAM_to_pts(RAM, P):
    """
    Approximate maximum grid points given the number of populations and available RAM

    pts: Grid points setting
    P: Number of populations
    """
    return int((RAM*1024**3/(8*4))**(1./P))
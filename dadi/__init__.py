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
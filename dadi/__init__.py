"""
For examples of dadi's usage, see the examples directory in the source
distribution.

Documentation of all methods can be found in doc/api/index.html of the source
distribution.
"""
import logging
logging.basicConfig()

import Demographics1D
import Demographics2D
import Inference
import Integration
import Misc
import Numerics
import PhiManip
# Protect import of Plotting in case matplotlib not installed.
try:
    import Plotting
except ImportError:
    pass

# We do it this way so it's easier to reload.
import Spectrum_mod 
Spectrum = Spectrum_mod.Spectrum

try:
    # This is to try and ensure we have a nice __SVNVERSION__ attribute, so
    # when we get bug reports, we know what version they were using. The
    # svnversion file is created by setup.py.
    import os
    _directory = os.path.dirname(Integration.__file__)
    _svn_file = os.path.join(_directory, 'svnversion')
    __SVNVERSION__ = file(_svn_file).read().strip()
except:
    __SVNVERSION__ = 'Unknown'

# When doing arithmetic with Spectrum objects (which are masked arrays), we
# often have masked values which generate annoying arithmetic warnings. Here
# we tell numpy to ignore such warnings. This puts greater onus on the user to
# check results, but for our use case I think it's the better default.
import numpy
numpy.seterr(all='ignore')

from Spectrum_mod import Spectrum
import Inference
import Integration
import PhiManip
import Numerics
# Protect import of Plotting in case matplotlib not installed.
try:
    import Plotting
except ImportError:
    pass

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

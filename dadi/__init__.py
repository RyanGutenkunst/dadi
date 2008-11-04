from Spectrum_mod import Spectrum
import Inference
import Integration
import PhiManip
import Numerics
import ms
try:
    import Plotting
except ImportError:
    pass

try:
    # This is to try and ensure we have a nice __SVNVERSION__ attribute, so
    # when we get bug reports, we know what version they were using. The
    # svnversion file is created by setup.py.
    import os
    __DIRECTORY__ = os.path.dirname(Integration.__file__)
    __svn_file__ = os.path.join(__DIRECTORY__, 'svnversion')
    __SVNVERSION__ = file(__svn_file__).read().strip()
except:
    __SVNVERSION__ = 'Unknown'

import numpy
# This gives a nicer printout for masked arrays.
numpy.ma.default_real_fill_value = numpy.nan

import IO
import Integration
import PhiManip
import SFS
import ms
import Plotting

try:
    import os
    __DIRECTORY__ = os.path.dirname(IO.__file__)
    __svn_file__ = os.path.join(__DIRECTORY__, 'svnversion')
    __SVNVERSION__ = file(__svn_file__).read().strip()
except:
    __SVNVERSION__ = 'Unknown'

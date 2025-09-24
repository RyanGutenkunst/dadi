"""
Modeling polyploid populations.
"""

from . import Integration, wrightfisher, cuda
from . import Demographics1D, Demographics2D    
from dadi.Polyploidy.Integration import PloidyType

# Make Integration available for direct import
__all__ = ['Integration', 'wrightfisher', 'cuda', 'PloidyType', 'Demographics1D', 'Demographics2D']
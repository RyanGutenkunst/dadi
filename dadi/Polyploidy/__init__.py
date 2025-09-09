"""
Modeling polyploid populations.
"""

from . import Integration, wrightfisher
from . import Demographics1D, Demographics2D    
from dadi.Polyploidy.Integration import PloidyType

# Make Integration available for direct import
__all__ = ['Integration', 'wrightfisher', 'PloidyType', 'Demographics1D', 'Demographics2D']
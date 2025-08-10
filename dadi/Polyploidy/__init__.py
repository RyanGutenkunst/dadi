"""
Modeling polyploid populations.
"""

from . import Integration, wrightfisher
from .Integration import PloidyType

# Make Integration available for direct import
__all__ = ['Integration', 'wrightfisher', 'PloidyType']
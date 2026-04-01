"""
Modeling polyploid populations.
"""

from . import Integration, wrightfisher, PhiManip_supp
from . import allo_demographics, auto_demographics
from dadi.Polyploidy.Integration import PloidyType

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
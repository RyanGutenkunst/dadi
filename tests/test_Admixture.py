import os
import pytest
import numpy
import dadi
from dadi.Integration import one_pop, two_pops, three_pops

def test_het_ascertained_and_admix_prop_conflict():
    """
    Test error raised on conflicting options.
    """
    phi = numpy.zeros((2,2))
    xx = numpy.linspace(0,1,2)
    admix_props = [[0.2,0.8],[0.9,0.1]]
    with pytest.raises(NotImplementedError) as e_info:
        dadi.Spectrum.from_phi(phi, [2,2], [xx,xx], admix_props=admix_props, het_ascertained='xx')
    assert(str(e_info.value) ==  str('admix_props and het_ascertained options cannot be used '+
    '\n            simultaneously. Instead, please use the PhiManip methods to '+
    '\n            implement admixture. If this proves inappropriate for your use, '+
    '\n            contact the the dadi developers, as it may be possible to support'+
    '\n            both options simultaneously in the future.'))

def test_het_ascertained_argument():
    """
    Test check for improper het_ascertained argument.
    """
    phi = numpy.zeros((2,2))
    xx = numpy.linspace(0,1,2)
    admix_props = [[0.2,0.8],[0.9,0.1]]
    with pytest.raises(ValueError) as e_info:
        dadi.Spectrum.from_phi(phi, [2,2], [xx,xx], het_ascertained=['xx', 'yy'])
    assert(str(e_info.value) == "If used, het_ascertained must be 'xx', 'yy', or 'zz'.")

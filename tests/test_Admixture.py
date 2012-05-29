import os
import unittest
import numpy
import dadi
from dadi.Integration import one_pop, two_pops, three_pops

class AdmixtureTestCase(unittest.TestCase):
    def test_het_ascertained_and_admix_prop_conflict(self):
        """
        Test error raised on conflicting options.
        """
        phi = numpy.zeros((2,2))
        xx = numpy.linspace(0,1,2)
        admix_props = [[0.2,0.8],[0.9,0.1]]
        with self.assertRaises(NotImplementedError):
            dadi.Spectrum.from_phi(phi, [2,2], [xx,xx], 
                                   admix_props=admix_props, 
                                   het_ascertained='xx')

    def test_het_ascertained_argument(self):
        """
        Test check for improper het_ascertained argument.
        """
        phi = numpy.zeros((2,2))
        xx = numpy.linspace(0,1,2)
        admix_props = [[0.2,0.8],[0.9,0.1]]
        with self.assertRaises(ValueError):
            dadi.Spectrum.from_phi(phi, [2,2], [xx,xx], 
                                   het_ascertained=['xx', 'yy'])

suite = unittest.TestLoader().loadTestsFromTestCase(AdmixtureTestCase)

if __name__ == '__main__':
    unittest.main()

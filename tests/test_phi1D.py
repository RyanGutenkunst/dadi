import unittest
import numpy
import dadi

class phi1DTestCase(unittest.TestCase):
    """
    Test routines for generating input phi's.

    These tests are primarily motivated by the desire to avoid divide by
    zero warnings.
    """
    def setUp(self):
        # Ensures that divide by zero raised an exception
        self.prior_seterr = numpy.seterr(divide='raise')

    def tearDown(self):
        # Restores seterr state for other tests
        numpy.seterr(**self.prior_seterr)
    
    xx = dadi.Numerics.default_grid(20)
    def test_snm(self):
        """
        Test standard neutral model.
        """
        dadi.PhiManip.phi_1D(self.xx)
        
    def test_genic(self):
        """
        Test non-dominant genic selection.
        """
        dadi.PhiManip.phi_1D(self.xx, gamma=1)
        dadi.PhiManip.phi_1D(self.xx, gamma=-500)

    def test_dominance(self):
        """
        Test selection with dominance.
        """
        dadi.PhiManip.phi_1D(self.xx, gamma=1, h=0.3)

suite = unittest.TestLoader().loadTestsFromTestCase(phi1DTestCase)

if __name__ == '__main__':
    unittest.main()

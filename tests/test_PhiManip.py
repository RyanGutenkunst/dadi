import unittest
import numpy as np
import dadi

class PhiManipTestCase(unittest.TestCase):
    def test_filter_pops(self):
        """
        Test filtering of populations
        """
        pts = 5
        nu1,nu2,nu3,T = 0.1, 1, 10, 0.1
        # Compare phi's from this 3D integration
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi_all = dadi.Integration.three_pops(phi, xx, T=T, nu1=nu1, nu2=nu2, nu3=nu3)

        phi = dadi.PhiManip.phi_1D(xx)
        phi1 = dadi.Integration.one_pop(phi, xx, T=T, nu=nu1)
        phi = dadi.PhiManip.phi_1D(xx)
        phi2 = dadi.Integration.one_pop(phi, xx, T=T, nu=nu2)
        # Note that we can't directly compare with nu3=10, because it will have different
        # timesteps in integration, leading to different results. Here we fix timesteps
        # by doing a 2D integration.
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi3 = dadi.Integration.two_pops(phi, xx, T=T, nu1=nu1, nu2=nu3)

        self.assertTrue(np.allclose(phi1[1:-1], dadi.PhiManip.filter_pops(phi_all, xx, [1])[1:-1]))
        self.assertTrue(np.allclose(phi2[1:-1], dadi.PhiManip.filter_pops(phi_all, xx, [2])[1:-1]))
        phi3_comp = dadi.PhiManip.filter_pops(phi_all, xx, [1,3])
        phi3[0,0] = phi3[-1,-1] = phi3_comp[0,0] = phi3_comp[-1,-1] = 0
        self.assertTrue(np.allclose(phi3, phi3_comp))


suite = unittest.TestLoader().loadTestsFromTestCase(PhiManipTestCase)
if __name__ == '__main__':
    unittest.main()
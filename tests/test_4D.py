import unittest
import numpy as np
import dadi

class Int4DTestCase(unittest.TestCase):
    def test_splitting(self):
        """
        Test splitting into 4D.
        """
        # Check that all splits yield SNM

        pts = 50
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        fs1 = dadi.Spectrum.from_phi(phi, [10], (xx,))

        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        fs4 = dadi.Spectrum.from_phi(phi, [10,10,10,10], (xx,xx,xx,xx))

        # Test each marginal spectrum
        for ii in range(4):
            tomarg = list(range(4))
            tomarg.remove(ii)
            fsm = fs4.marginalize(tomarg)
            self.assertTrue(np.allclose(fs1, fsm, rtol=1e-3, atol=1e-3))

    def test_integration_SNM(self):
        """
        Test simple SNM integration.
        """
        pts = 30
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        fs1 = dadi.Spectrum.from_phi(phi, [5], (xx,))

        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D_admix(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        # No demography, so all pops should still be SNM
        phi = dadi.Integration.four_pops(phi, xx, T=0.1)
        fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))

        # Loose tolerance of this comparison is okay. We're looking
        # for gross violation due to bugs.
        for ii in range(4):
            tomarg = list(range(4))
            tomarg.remove(ii)
            fsm = fs4.marginalize(tomarg)
            self.assertTrue(np.allclose(fs1, fsm, rtol=1e-2, atol=1e-2))

    def test_integration_2Dcomp(self):
        """
        Integration tested by comparison to 2D integrations
        """
        pts = 20
        nu1 = lambda t: 0.5 + 5*t
        nu2 = lambda t: 10-50*t
        m12 = lambda t: 2-t
        m21 = lambda t: 0.5+3*t
        gamma1 = lambda t: -2*t
        gamma2 = lambda t: 3*t
        h1 = lambda t: 0.2+t
        h2 = lambda t: 0.9-t

        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=nu1, nu2=nu2, 
                                        m12=m12, m21=m21, gamma1=gamma1, gamma2=gamma2, h1=h1, h2=h2)
        fs2 = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))
        
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu1=nu1, nu4=nu2, 
                                         m14=m12, m41=m21, gamma1=gamma1, gamma4=gamma2, h1=h1, h4=h2)
        fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
        fsm = fs4.marginalize((1,2))
        self.assertTrue(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))
        
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu2=nu1, nu4=nu2, 
                                         m24=m12, m42=m21, gamma2=gamma1, gamma4=gamma2, h2=h1, h4=h2)
        fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
        fsm = fs4.marginalize((0,2))
        self.assertTrue(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))
        
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu3=nu1, nu4=nu2, 
                                         m34=m12, m43=m21, gamma3=gamma1, gamma4=gamma2, h3=h1, h4=h2)
        fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
        fsm = fs4.marginalize((0,1))
        self.assertTrue(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

    def test_admix_props(self):
        """
        Test admix_props in from_phi
        """
        pts = 20
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
        fs2 = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx), admix_props=((0.3,0.7),(0.9,0.1)))
        
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu3=0.5, nu4=10, m34=2, m43=0.5, gamma3=-1, gamma4=1, h3=0.2, h4=0.7)
        fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx),
                                     admix_props=((1,0,0,0),(0,1,0,0),(0,0,0.3,0.7),(0,0,0.9,0.1)))
        fsm = fs4.marginalize((0,1))
        self.assertTrue(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

    def test_admixture(self):
        """
        Test phi_4D_admix_into_4
        """
        pts = 20
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
        phi = dadi.PhiManip.phi_2D_admix_1_into_2(phi,0.8,xx,xx)
        fs2 = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))

        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu3=0.5, nu4=10, m34=2, m43=0.5, gamma3=-1, gamma4=1, h3=0.2, h4=0.7)
        phi = dadi.PhiManip.phi_4D_admix_into_4(phi,0,0,0.8,xx,xx,xx,xx)
        fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
        fsm = fs4.marginalize((0,1))
        self.assertTrue(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.7)
        phi = dadi.PhiManip.phi_4D_admix_into_2(phi,0.8,0,0,xx,xx,xx,xx)
        fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
        fsm = fs4.marginalize((2,3))
        self.assertTrue(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu1=0.5, nu3=10, m13=2, m31=0.5, gamma1=-1, gamma3=1, h1=0.2, h3=0.7)
        phi = dadi.PhiManip.phi_4D_admix_into_3(phi,0.8,0,0,xx,xx,xx,xx)
        fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
        fsm = fs4.marginalize((1,3))
        self.assertTrue(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
        phi = dadi.PhiManip.phi_2D_admix_2_into_1(phi,0.8,xx,xx)
        fs2 = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))

        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        phi = dadi.Integration.four_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.7)
        phi = dadi.PhiManip.phi_4D_admix_into_1(phi,0.8,0,0,xx,xx,xx,xx)
        fs4 = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
        fsm = fs4.marginalize((2,3))
        self.assertTrue(np.allclose(fs2, fsm, rtol=1e-2, atol=1e-2))

suite = unittest.TestLoader().loadTestsFromTestCase(Int4DTestCase)

if __name__ == '__main__':
    unittest.main()
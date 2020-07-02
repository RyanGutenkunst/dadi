import unittest
import numpy as np
import dadi

class Int5DTestCase(unittest.TestCase):
    def test_splitting(self):
        """
        Test splitting into 5D.
        """
        # Check that all splits yield SNM

        pts = 50
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        fs1 = dadi.Spectrum.from_phi(phi, [10], (xx,))

        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_4D_to_5D(phi, 0, 0, 0, xx,xx,xx,xx,xx)
        fs4 = dadi.Spectrum.from_phi(phi, [10,10,10,10,10], (xx,xx,xx,xx,xx))

        # Test each marginal spectrum
        for ii in range(5):
            tomarg = list(range(5))
            tomarg.remove(ii)
            fsm = fs4.marginalize(tomarg)
            self.assertTrue(np.allclose(fs1, fsm, rtol=1e-3, atol=1e-3))

    def test_integration_SNM(self):
        """
        Test simple SNM integration.
        """
        pts = 5
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        ref = dadi.Integration.one_pop(phi, xx, T=0.1)

        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.PhiManip.phi_2D_to_3D_admix(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0,0, xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_4D_to_5D(phi, 0,0,0, xx,xx,xx,xx,xx)
        phi = dadi.Integration.five_pops(phi, xx, T=0.1)

        # Check all marginal phi, which should be identical except for 0,1 entries.
        for ii in range(5):
            toremove = list(range(5))
            toremove.remove(ii)
            # Remove all the populations except the one we care about
            test = phi
            for pop in toremove[::-1]:
                test = dadi.PhiManip.remove_pop(test, xx, pop)
            print(np.allclose(ref[1:-1],test[1:-1]))

    def test_integration_nomig(self):
        nu1 = lambda t: 0.5 + 50*t
        nu2 = lambda t: 10-50*t
        gamma1 = lambda t: -30*t
        gamma2 = lambda t: 30*t
        h1 = lambda t: 0.2+5*t
        h2 = lambda t: 0.9-5*t
        T = 0.1

        pin = (T, nu1, nu2, gamma1, gamma2, h1, h2)

        @dadi.Numerics.make_extrap_func
        def ref_func(params, ns, pts):
            T, nu1, nu2, gamma1, gamma2, h1, h2 = params

            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)
            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
            phi = dadi.Integration.two_pops(phi, xx, T=T, nu1=nu1, gamma1=gamma1, h1=h1,
                                            nu2=nu2, gamma2=gamma2, h2=h2)
            return dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))

        @dadi.Numerics.make_extrap_func
        def test_func_all(params, ns, pts):
            T, nu1, nu2, gamma1, gamma2, h1, h2 = params

            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)
            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
            phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
            phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
            phi = dadi.PhiManip.phi_4D_to_5D(phi, 0,0,0, xx,xx,xx,xx,xx)
            phi = dadi.Integration.five_pops(phi, xx, T=T, nu1=nu1, gamma1=gamma1, h1=h1,
                                             nu2=nu2, gamma2=gamma2, h2=h2,
                                             nu3=nu2, gamma3=gamma2, h3=h2,
                                             nu4=nu2, gamma4=gamma2, h4=h2,
                                             nu5=nu2, gamma5=gamma2, h5=h2)
            fs_all = dadi.Spectrum.from_phi(phi, [5,5,5,5,5], (xx,xx,xx,xx,xx))
            return fs_all

        ref = ref_func(pin, None, [16,18,20])

        fs_all = test_func_all(pin, None, [16,18,20])
        fs2 = fs_all.marginalize([2,3,4])
        fs3 = fs_all.marginalize([1,3,4])
        fs4 = fs_all.marginalize([1,2,4])
        fs5 = fs_all.marginalize([1,2,3])

        self.assertTrue(np.allclose(fs2, ref, atol=1e-2))
        self.assertTrue(np.allclose(fs3, ref, atol=1e-2))
        self.assertTrue(np.allclose(fs4, ref, atol=1e-2))
        self.assertTrue(np.allclose(fs5, ref, atol=1e-2))

    def test_integration_mig(self):
        """
        Integration tested by comparison to 2D integrations
        """
        m12 = lambda t: 2-19*t
        m21 = lambda t: 0.5+30*t
        T = 0.1

        pin = (T,m12,m21)

        @dadi.Numerics.make_extrap_func
        def ref_func(params, ns, pts):
            T, m12, m21 = params

            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)
            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
            phi = dadi.Integration.two_pops(phi, xx, T=T, m12=m12, m21=m21)
            return dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))

        @dadi.Numerics.make_extrap_func
        def test_func_marg(params, ns, popkept, pts):
            T, m12, m21 = params

            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)
            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
            phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
            phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
            phi = dadi.PhiManip.phi_4D_to_5D(phi, 0,0,0, xx,xx,xx,xx,xx)
            kwargs = {'m1{0}'.format(popkept):m12,
                      'm{0}1'.format(popkept):m21}
            phi = dadi.Integration.five_pops(phi, xx, T=T, **kwargs)
            fs_all = dadi.Spectrum.from_phi(phi, [5,5,5,5,5], (xx,xx,xx,xx,xx))
            tomarg = [1,2,3,4]
            tomarg.remove(popkept-1)
            fs = fs_all.marginalize(tomarg)
            return fs

        ref = ref_func(pin, None, [16,18,20])
        for ii in range(2,6):
            test = test_func_marg(pin, None, 2, [16, 18, 20])
            self.assertTrue(np.allclose(ref, test, atol=1e-3))

    def test_admix_into(self):
        """
        Test phi_5D_admix_into methods
        """
        pts = 5
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
        ref = dadi.PhiManip.phi_2D_admix_1_into_2(phi,0.8,xx,xx)

        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0,0, xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_4D_to_5D(phi, 0,0,0, xx,xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_5D_admix_into_2(phi,0.8,0,0,0,xx,xx,xx,xx,xx)

        test = dadi.PhiManip.remove_pop(phi,xx,5)
        test = dadi.PhiManip.remove_pop(test,xx,4)
        test = dadi.PhiManip.remove_pop(test,xx,3)

        self.assertTrue(np.allclose(ref, test))

        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0,0, xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_4D_to_5D(phi, 0,0,0, xx,xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_5D_admix_into_3(phi,0.8,0,0,0,xx,xx,xx,xx,xx)

        test = dadi.PhiManip.remove_pop(phi,xx,5)
        test = dadi.PhiManip.remove_pop(test,xx,4)
        test = dadi.PhiManip.remove_pop(test,xx,2)

        self.assertTrue(np.allclose(ref, test))

        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0,0, xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_4D_to_5D(phi, 0,0,0, xx,xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_5D_admix_into_4(phi,0.8,0,0,0,xx,xx,xx,xx,xx)

        test = dadi.PhiManip.remove_pop(phi,xx,5)
        test = dadi.PhiManip.remove_pop(test,xx,3)
        test = dadi.PhiManip.remove_pop(test,xx,2)

        self.assertTrue(np.allclose(ref, test))

        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=0.1, nu1=0.5, nu2=10, m12=2, m21=0.5, gamma1=-1, gamma2=1, h1=0.2, h2=0.9)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0,0, xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_4D_to_5D(phi, 0,0,0, xx,xx,xx,xx,xx)
        phi = dadi.PhiManip.phi_5D_admix_into_5(phi,0.8,0,0,0,xx,xx,xx,xx,xx)

        test = dadi.PhiManip.remove_pop(phi,xx,4)
        test = dadi.PhiManip.remove_pop(test,xx,3)
        test = dadi.PhiManip.remove_pop(test,xx,2)

        self.assertTrue(np.allclose(ref, test))

    def test_4D_to_5D(self):
        """
        Test splitting with admixture
        """
        # Not a comprhensive test, since it only considers a single scenario
        pts = 4
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=1, nu1=10, nu2=0.1)
        # Create pop 3 as a mixture of 1&2
        ref = dadi.PhiManip.phi_2D_to_3D(phi, 0.3, xx,xx,xx)

        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        phi = dadi.Integration.two_pops(phi, xx, T=1, nu1=10, nu2=0.1)
        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
        phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
        # Create pop 5 as a mixture of 1&2
        phi = dadi.PhiManip.phi_4D_to_5D(phi, 0.3, 0.7, 0, xx,xx,xx,xx,xx)
        # Remove pops 4 and 3
        test = dadi.PhiManip.remove_pop(phi,xx,4)
        test = dadi.PhiManip.remove_pop(test,xx,3)

        self.assertTrue(np.allclose(ref, test))

# Don't run these tests by default, because they are very slow
#suite = unittest.TestLoader().loadTestsFromTestCase(Int5DTestCase)

if __name__ == '__main__':
    unittest.main()
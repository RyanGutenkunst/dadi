import unittest, numpy as np
import dadi.DFE
from dadi.DFE import pdfs
from dadi import Numerics, Integration, PhiManip, Spectrum

@dadi.Numerics.make_extrap_func
def nodemog_sel(params, ns, pts):
    gamma = params[0]

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx, gamma=gamma)

    return Spectrum.from_phi(phi, ns, (xx,))

class DFETestCase(unittest.TestCase):
    def test_Cache1D_generation(self):
        """
        Trivial test that Cache1D generation doesn't crash.
        """
        dadi.DFE.Cache1D([], [10], nodemog_sel, [20,30,40], 
                         gamma_bounds=(1e-4, 20), gamma_pts=2)

    def test_1D_integration(self):
        """
        Trivial test that Cache1D integration doesn't crash.
        """
        s1 = dadi.DFE.Cache1D([], [10], nodemog_sel, [20,30,40], 
                              gamma_bounds=(1e-4, 20), gamma_pts=2)
        s1.integrate([1], None, pdfs.exponential, 1.0, None, True)

    def test_1D_integration_pointpos(self):
        """
        Trivial test that Cache1D integration doesn't crash.
        """
        s1 = dadi.DFE.Cache1D([], [10], nodemog_sel, [20,30,40], 
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[2])
        # Case in which gammapos has been cached
        s1.integrate_point_pos([1,0.1,2], None, pdfs.exponential, 1.0, nodemog_sel, None, True)
        # Case in which gammapos has not been cached
        s1.integrate_point_pos([1,0.1,1.21], None, pdfs.exponential, 1.0, nodemog_sel, None, True)

    def test_1D_optimization(self):
        """
        Trivial test that optimization of 1D integration doesn't crash.
        """
        ns = [10]
        theta = 10.
        data = theta*nodemog_sel([-1], ns, [40,50,60])
        s1 = dadi.DFE.Cache1D([], ns, nodemog_sel, [20,30,40], 
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[2])

        sel_dist = pdfs.exponential
        popt = dadi.Inference.optimize([1], data, s1.integrate, pts=None,
                                       func_args=[sel_dist, theta],
                                       lower_bound=[0], upper_bound=[10],
                                       multinom=False)

    def test_1D_pointpos_optimization(self):
        """
        Trivial test that optimization of 1D integration with point positive mass doesn't crash.
        """
        ns = [10]
        theta = 10.
        data = theta*nodemog_sel([-1], ns, [40,50,60])
        s1 = dadi.DFE.Cache1D([], ns, nodemog_sel, [20,30,40], 
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[2])

        sel_dist = pdfs.exponential
        # Test with gammapos held fixed at cached value
        popt = dadi.Inference.optimize([1,0.2,2], data, s1.integrate_point_pos, pts=None,
                                       func_args=[sel_dist, theta, nodemog_sel],
                                       lower_bound=[0,0,None], upper_bound=[10,1,10],
                                       fixed_params = [None,None,2],
                                       multinom=False)
        # Test with gammapos allowed to vary
        popt = dadi.Inference.optimize([1,0.2,2], data, s1.integrate_point_pos, pts=None,
                                       func_args=[sel_dist, theta, nodemog_sel],
                                       lower_bound=[0,0,None], upper_bound=[10,1,10],
                                       multinom=False)

    def test_1D_pointpos_correctness(self):
        """
        Compare with result from previous verion of code built directly off fitdadi.
        """
        @dadi.Numerics.make_extrap_func
        def IM_sel(params, ns, pts):
            s,nu1,nu2,T,m12,m21,gamma = params

            xx = Numerics.default_grid(pts)

            phi = PhiManip.phi_1D(xx, gamma=gamma)
            phi = PhiManip.phi_1D_to_2D(xx, phi)

            nu1_func = lambda t: s * (nu1/s)**(t/T)
            nu2_func = lambda t: (1-s) * (nu2/(1-s))**(t/T)
            phi = Integration.two_pops(phi, xx, T, nu1_func, nu2_func,
                                       m12=m12, m21=m21, gamma1=gamma,
                                       gamma2=gamma)

            fs = Spectrum.from_phi(phi, ns, (xx,xx))
            return fs

        demo_params = [0.5,2,0.5,0.1,0,0]
        ns = [8, 12]
        pts_l = [60, 80, 100]

        s1 = dadi.DFE.Cache1D(demo_params, ns, IM_sel, pts_l, 
                              gamma_bounds=(1e-2, 10), gamma_pts=100)

        fs = s1.integrate([-0.5, 0.5], None, pdfs.lognormal, 
                                    1e5, None, exterior_int=False)
        comp = dadi.Spectrum.from_file('test_data/fitdadi.IM_no_ext_test.fs')   
        assert(np.allclose(fs, comp))

        fs = s1.integrate([-0.5, 0.5], None, pdfs.lognormal, 
                          1e5, None, exterior_int=True)
        comp = dadi.Spectrum.from_file('test_data/fitdadi.IM_test.fs')   
        assert(np.allclose(fs, comp))

        fs = s1.integrate_point_pos([-0.5, 0.5, 0.1, 4.3], None, pdfs.lognormal, 
                                    1e5, IM_sel, None)
        comp = dadi.Spectrum.from_file('test_data/fitdadi.IM_point_pos_test.fs')   
        assert(np.allclose(fs, comp))

suite = unittest.TestLoader().loadTestsFromTestCase(DFETestCase)

if __name__ == '__main__':
    unittest.main()
    @dadi.Numerics.make_extrap_func
    def IM_sel(params, ns, pts):
        s,nu1,nu2,T,m12,m21,gamma = params

        xx = Numerics.default_grid(pts)

        phi = PhiManip.phi_1D(xx, gamma=gamma)
        phi = PhiManip.phi_1D_to_2D(xx, phi)

        nu1_func = lambda t: s * (nu1/s)**(t/T)
        nu2_func = lambda t: (1-s) * (nu2/(1-s))**(t/T)
        phi = Integration.two_pops(phi, xx, T, nu1_func, nu2_func,
                                   m12=m12, m21=m21, gamma1=gamma,
                                   gamma2=gamma)

        fs = Spectrum.from_phi(phi, ns, (xx,xx))
        return fs

    demo_params = [0.5,2,0.5,0.1,0,0]
    ns = [8, 12]
    pts_l = [60, 80, 100]

    #import Selection
    #s1_old = Selection.spectra(demo_params, ns, IM_sel, pts_l=pts_l,
    #                       Npts=100, int_bounds=(1e-2, 10))

    #fs_old = s1_old.integrate_point_pos([-0.5, 0.5, 0.1, 4.3], Selection.lognormal_dist,
    #                                    1e5, IM_sel)
    #fs_old.to_file('test_data/fitdadi.IM_point_pos_test.fs')

    s1 = dadi.DFE.Cache1D(demo_params, ns, IM_sel, pts_l, 
                          gamma_bounds=(1e-2, 10), gamma_pts=100)

    fs = s1.integrate_point_pos([-0.5, 0.5, 0.1, 4.3], None, pdfs.lognormal, 
                                1e5, IM_sel, None)

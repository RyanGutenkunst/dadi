import unittest, numpy as np
import dadi.DFE
from dadi.DFE import PDFs, DemogSelModels


class DFETestCase(unittest.TestCase):
    def test_Cache1D_generation(self):
        """
        Trivial test that Cache1D generation doesn't crash.
        """
        dadi.DFE.Cache1D([], [10], DemogSelModels.equil, [20,30,40], 
                         gamma_bounds=(1e-4, 20), gamma_pts=2)

    def test_1D_integration(self):
        """
        Trivial test that Cache1D integration doesn't crash.
        """
        s1 = dadi.DFE.Cache1D([], [10], DemogSelModels.equil, [20,30,40], 
                              gamma_bounds=(1e-4, 20), gamma_pts=2)
        s1.integrate([1], None, PDFs.exponential, 1.0, None, True)

    def test_1D_integration_pointpos(self):
        """
        Trivial test that Cache1D integration doesn't crash.
        """
        s1 = dadi.DFE.Cache1D([], [10], DemogSelModels.equil, [20,30,40], 
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[2])
        # Case in which gammapos has been cached
        s1.integrate_point_pos([1,0.1,2], None, PDFs.exponential, 1.0, DemogSelModels.equil, None, True)
        # Case in which gammapos has not been cached
        s1.integrate_point_pos([1,0.1,1.21], None, PDFs.exponential, 1.0, DemogSelModels.equil, None, True)

    def test_1D_optimization(self):
        """
        Trivial test that optimization of 1D integration doesn't crash.
        """
        ns = [10]
        theta = 10.
        data = theta*DemogSelModels.equil([-1], ns, [40,50,60])
        s1 = dadi.DFE.Cache1D([], ns, DemogSelModels.equil, [20,30,40], 
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[2])

        sel_dist = PDFs.exponential
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
        data = theta*DemogSelModels.equil([-1], ns, [40,50,60])
        s1 = dadi.DFE.Cache1D([], ns, DemogSelModels.equil, [20,30,40], 
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[2])

        sel_dist = PDFs.exponential
        # Test with gammapos held fixed at cached value
        popt = dadi.Inference.optimize([1,0.2,2], data, s1.integrate_point_pos, pts=None,
                                       func_args=[sel_dist, theta, DemogSelModels.equil],
                                       lower_bound=[0,0,None], upper_bound=[10,1,10],
                                       fixed_params = [None,None,2],
                                       multinom=False)
        # Test with gammapos allowed to vary
        popt = dadi.Inference.optimize([1,0.2,2], data, s1.integrate_point_pos, pts=None,
                                       func_args=[sel_dist, theta, DemogSelModels.equil],
                                       lower_bound=[0,0,None], upper_bound=[10,1,10],
                                       multinom=False)

    def test_1D_integration_correctness(self):
        """
        Compare with result from previous verion of code built directly off fitdadi.
        """
        demo_params = [0.5,2,0.5,0.1,0,0]
        ns = [8, 12]
        pts_l = [60, 80, 100]

        s1 = dadi.DFE.Cache1D(demo_params, ns, DemogSelModels.IM_single_gamma, pts_l, 
                              gamma_bounds=(1e-2, 10), gamma_pts=100)

        fs = s1.integrate([-0.5, 0.5], None, PDFs.lognormal, 
                                    1e5, None, exterior_int=False)
        comp = dadi.Spectrum.from_file('test_data/fitdadi.IM_no_ext_test.fs')   
        assert(np.allclose(fs, comp))

        fs = s1.integrate([-0.5, 0.5], None, PDFs.lognormal, 
                          1e5, None, exterior_int=True)
        comp = dadi.Spectrum.from_file('test_data/fitdadi.IM_test.fs')   
        assert(np.allclose(fs, comp))

        fs = s1.integrate_point_pos([-0.5, 0.5, 0.1, 4.3], None, PDFs.lognormal, 
                                    1e5, DemogSelModels.IM_single_gamma, None)
        comp = dadi.Spectrum.from_file('test_data/fitdadi.IM_point_pos_test.fs')   
        assert(np.allclose(fs, comp))

    def test_2D_cache_generation(self):
        """
        Trivial test that Cache2D generation doesn't crash.
        """
        demo_params = [0.5,2,0.5,0.01,0,0]
        dadi.DFE.Cache2D(demo_params, [3,3], DemogSelModels.IM, [20],
                         gamma_bounds=(1e-4, 2), gamma_pts=2)

    def test_2D_integration(self):
        """
        Trivial test that Cache2D integration doesn't crash.
        """
        demo_params = [0.5,2,0.5,0.01,0,0]
        dadi.DFE.Cache2D(demo_params, [3,3], DemogSelModels.IM, [20],
                         gamma_bounds=(1e-4, 2), gamma_pts=2)

suite = unittest.TestLoader().loadTestsFromTestCase(DFETestCase)

if __name__ == '__main__':
    unittest.main()

    # For testing against old fitdadi code
    #s1 = dadi.DFE.Cache1D(demo_params, ns, DemogSelModels.IM_single_gamma, pts_l, 
    #                      gamma_bounds=(1e-2, 10), gamma_pts=100)
    #fs = s1.integrate_point_pos([-0.5, 0.5, 0.1, 4.3], None, PDFs.lognormal, 
    #                           #1e5, DemogSelModels.IM_single_gamma, None)
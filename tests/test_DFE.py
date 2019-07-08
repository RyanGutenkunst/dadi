import unittest
import numpy as np
import dadi.DFE
from dadi.DFE import PDFs, DemogSelModels


class DFETestCase(unittest.TestCase):
    def test_Cache1D_generation(self):
        """
        Trivial test that Cache1D generation doesn't crash.
        """
        dadi.DFE.Cache1D([], [10], DemogSelModels.equil, [20, 30, 40],
                         gamma_bounds=(1e-4, 20), gamma_pts=2)
        dadi.DFE.Cache1D([], [10], DemogSelModels.equil, [20, 30, 40],
                         gamma_bounds=(1e-4, 20), gamma_pts=10, mp=True)

    def test_1D_integration(self):
        """
        Trivial test thats Cache1D integration doesn't crash.
        """
        s1 = dadi.DFE.Cache1D([], [10], DemogSelModels.equil, [20, 30, 40],
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[2])
        # Basic integration
        s1.integrate([1], None, PDFs.exponential, 1.0)
        # Case in which gammapos has been cached
        s1.integrate_point_pos(
            [1, 0.1, 2], None, PDFs.exponential, 1.0, DemogSelModels.equil)
        # Case in which gammapos has not been cached
        s1.integrate_point_pos(
            [1, 0.1, 1.21], None, PDFs.exponential, 1.0, DemogSelModels.equil)
        # Case in which one gammapos has not been cached
        s1.integrate_point_pos(
            [1, 0.1, 1.21, 0.3, 4.3], None, PDFs.exponential, 1.0, DemogSelModels.equil, 2)

    def test_1D_optimization(self):
        """
        Trivials test that optimization of 1D integration doesn't crash.
        """
        ns = [10]
        theta = 10.
        data = theta*DemogSelModels.equil([-1], ns, [40, 50, 60])
        s1 = dadi.DFE.Cache1D([], ns, DemogSelModels.equil, [20, 30, 40],
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[2])
        # Test with basic integration
        sel_dist = PDFs.exponential
        popt = dadi.Inference.optimize([1], data, s1.integrate, pts=None,
                                       func_args=[sel_dist, theta],
                                       lower_bound=[0], upper_bound=[10],
                                       multinom=False)
        # Test with gammapos held fixed at cached value
        popt = dadi.Inference.optimize([1, 0.2, 2], data, s1.integrate_point_pos, pts=None,
                                       func_args=[sel_dist, theta,
                                           DemogSelModels.equil],
                                       lower_bound=[0, 0, None], upper_bound=[10, 1, 10],
                                       fixed_params=[None, None, 2],
                                       multinom=False)
        # Test with gammapos allowed to vary
        popt = dadi.Inference.optimize([1, 0.2, 2], data, s1.integrate_point_pos, pts=None,
                                       func_args=[sel_dist, theta,
                                           DemogSelModels.equil],
                                       lower_bound=[0, 0, None], upper_bound=[10, 1, 10],
                                       multinom=False)

    def test_1D_integration_correctness(self):
        """
        Compare with result from previous verion of code built directly off fitdadi.
        """
        demo_params = [0.5, 2, 0.5, 0.1, 0, 0]
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
                                    1e5, DemogSelModels.IM_single_gamma)
        comp = dadi.Spectrum.from_file(
            'test_data/fitdadi.IM_point_pos_test.fs')
        assert(np.allclose(fs, comp))

    def test_2D_cache_generation(self):
        """
        Trivial test that Cache2D generation doesn't crash.
        """
        demo_params = [0.5, 2, 0.5, 0.01, 0, 0]
        dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM, [20],
                         gamma_bounds=(1e-4, 2), gamma_pts=2)
        dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM, [20],
                         gamma_bounds=(1e-4, 2), gamma_pts=4, mp=True)

    def test_2D_integration(self):
        """
        Trivial test that Cache2D integration doesn't crash.
        """
        demo_params = [0.5, 2, 0.5, 0.01, 0, 0]
        s2 = dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM, [20],
                         gamma_bounds=(1e-4, 2), gamma_pts=2,
                         additional_gammas=[0.2])
        s2.integrate([2, 1, 0.4], None, PDFs.biv_lognormal, 1, None)
        # This should pass, because requested gammapos is cached
        s2.integrate_symmetric_point_pos(
            [2, 1, 0.4, 0.1, 0.2], None, PDFs.biv_lognormal, 1, None)
        # This should fail, because requested gammapos is not cached
        with self.assertRaises(IndexError):
            s2.integrate_symmetric_point_pos(
                [2, 1, 0.4, 0.1, 2], None, PDFs.biv_lognormal, 1, None)

    def test_2D_optimization(self):
        """
        Trivial tests that Cache2D optimization doesn't crash.
        """
        demo_params = [0.5, 2, 0.5, 0.03, 1, 2]
        ns, pts_l, theta = [3, 3], [20, 30, 40], 10
        data = theta*DemogSelModels.IM(demo_params+[-2, -3], ns, pts_l)
        s2 = dadi.DFE.Cache2D(demo_params, ns, DemogSelModels.IM, pts_l,
                              gamma_bounds=(1e-4, 2), gamma_pts=2,
                              additional_gammas=[0.2])
        sel_dist = PDFs.biv_lognormal
        # Test with basic integration
        popt = dadi.Inference.optimize([2, 1, 0.5], data, s2.integrate, pts=None,
                                       func_args=[sel_dist, theta],
                                       lower_bound=[None, 0, 0], upper_bound=[None, None, 1],
                                       multinom=False, maxiter=2)
        # Test with point mass of positive selection
        popt = dadi.Inference.optimize([2, 1, 0.5, 0.1, 0.2], data, s2.integrate_symmetric_point_pos, pts=None,
                                       func_args=[sel_dist, theta],
                                       lower_bound=[None, 0, 0], upper_bound=[None, None, 1],
                                       fixed_params=[
                                           None, None, None, None, 0.2],
                                       multinom=False, maxiter=2)

    def test_2D_correctness(self):
        """
        Compare with result from previous verion of code built directly off fitdadi.
        """
        demo_params = [0.5, 2, 0.5, 0.1, 0, 0]
        ns = [8, 12]
        pts_l = [60, 80, 100]

        s2 = dadi.DFE.Cache2D(demo_params, ns, DemogSelModels.IM, pts=pts_l,
                              gamma_pts=5, gamma_bounds=(1e-2, 10),
                              additional_gammas=[4.3])
        fs = s2.integrate_symmetric_point_pos([-0.5, 0.5, 0.5, 0.1, 4.3], None,
                                              PDFs.biv_lognormal, 1e5)
        comp = dadi.Spectrum.from_file(
            'test_data/fitdadi.IM_2D_point_pos_test.fs')
        assert(np.allclose(fs, comp))

    def test_mixture(self):
        """
        Trivial tests that mixture models don't crash
        """
        demo_params = [0.5, 2, 0.5, 0.01, 0, 0]
        ns, pts_l = [3,3], [20]
        s1 = dadi.DFE.Cache1D(demo_params, ns, DemogSelModels.IM_single_gamma, pts_l,
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[4.3])
        s2 = dadi.DFE.Cache2D(demo_params, ns, DemogSelModels.IM, pts_l,
                              gamma_bounds=(1e-4, 2), gamma_pts=2,
                              additional_gammas=[4.3])
        # Basic mixture model
        dadi.DFE.mixture([-0.5,0.5,0.5,0.1],None,s1,s2,PDFs.lognormal,PDFs.biv_lognormal,1,None)
        # Test with positive selection
        dadi.DFE.mixture_symmetric_point_pos([-0.5,0.5,0.5,0.1,4.3,0.1],None,s1,s2,PDFs.lognormal,PDFs.biv_lognormal,1,None)
        # Test for case that should fail
        with self.assertRaises(IndexError):
            dadi.DFE.mixture_symmetric_point_pos([-0.5,0.5,0.5,0.1,4.9,0.1],
                                                 None, s1, s2, PDFs.lognormal, PDFs.biv_lognormal, 1, None)

    #def test_plotting(self):
    #    import matplotlib.pyplot as plt
    #    sel_dist = PDFs.biv_lognormal
    #    # Asymmteric
    #    params = [0.5,-0.5,0.5,1,-0.8]
    #    gammax = -np.logspace(-2, 1, 20)
    #    gammay = -np.logspace(-1, 2, 30)

    #    fig = plt.figure(137, figsize=(4,3), dpi=150)
    #    fig.clear()
    #    ax = fig.add_subplot(1,1,1)
    #    dadi.DFE.Plotting.plot_biv_dfe(gammax, gammay, sel_dist, params, logweight=True, ax=ax)
    #    fig.tight_layout()

    #    # With positive selection
    #    params = [0.5,-0.5,0.5,1,0.0,0.3,3,0.3,4]
    #    fig = dadi.DFE.Plotting.plot_biv_point_pos_dfe(gammax, gammay, sel_dist, params,
    #                                                   fignum=23, rho=params[4])

    #    plt.show()

def generate_old_fitdadi_data():
    import Selection
    import Selection_2d
    demo_params = [0.5, 2, 0.5, 0.1, 0, 0]
    ns = [8, 12]
    pts_l = [60, 80, 100]

    func_ex_single = dadi.Numerics.make_extrap_func(Selection_2d.IM_single_sel)
    # 1D code, as modified by the Gutenkunst group
    s1 = Selection.spectra(demo_params, ns, func_ex_single, pts_l=pts_l,
                           Npts=100, int_bounds=(1e-2, 10))
    fs = s1.integrate_point_pos([-0.5, 0.5, 0.1, 4.3], Selection.lognormal_dist,
                                1e5, func_ex_single)
    fs.to_file('test_data/fitdadi.IM_point_pos_test.fs')

    func_ex = dadi.Numerics.make_extrap_func(Selection_2d.IM_sel)
    # 2D code created by the Gutenkunst group
    s2 = Selection_2d.spectra2d(demo_params, ns, func_ex, pts=pts_l,
                                Npts=5, int_bounds=(1e-2, 10),
                                additional_gammas=[4.3])
    fs = s2.integrate_biv_symmetric_point_pos([-0.5, 0.5, 0.5, 0.1, 4.3], None,
                                              Selection_2d.biv_lognorm_pdf, 1e5)
    fs.to_file('test_data/fitdadi.IM_2D_point_pos_test.fs')

suite=unittest.TestLoader().loadTestsFromTestCase(DFETestCase)

if __name__ == '__main__':
    try:
        generate_old_fitdadi_data()
        print('Generated data for comparison with old fitdadi code.')
    except ImportError:
        print('Failed to import old fitdadi code, using stored comparison results.')
        pass

    unittest.main()

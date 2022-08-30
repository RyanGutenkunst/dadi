import unittest
import numpy as np
import dadi.DFE
from dadi.DFE import PDFs, DemogSelModels
from dadi.DFE import Cache1D, Cache2D, Vourlaki_mixture

def trivial_fs(params, ns, pts): 
    return dadi.Spectrum([[0, 0.5], [0.5, 0]])

class DFETestCase(unittest.TestCase):
    def test_Cache1D_generation(self):
        """
        Trivial test that Cache1D generation doesn't crash.
        """
        dadi.DFE.Cache1D([], [10], DemogSelModels.equil, [20, 30, 40],
                         gamma_bounds=(1e-4, 20), gamma_pts=2)
        s1 = dadi.DFE.Cache1D([], [10], DemogSelModels.equil, [20, 30, 40],
                         gamma_bounds=(1e-4, 20), gamma_pts=10, cpus=2)
        s1.integrate([-0.5, 0.5], None, PDFs.lognormal,
                1e5, None, exterior_int=False)

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
        data = theta*DemogSelModels.equil([-1], ns, 60)
        s1 = dadi.DFE.Cache1D([], ns, DemogSelModels.equil, [20, 30, 40],
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[2])
        # Test with basic integration
        sel_dist = PDFs.exponential
        popt,llopt = dadi.Inference.opt([1], data, s1.integrate, pts=None,
                                  func_args=[sel_dist, theta],
                                  lower_bound=[0], upper_bound=[10],
                                  multinom=False, maxtime=10)
        # Test with gammapos held fixed at cached value
        popt,llopt = dadi.Inference.opt([1, 0.2, 2], data, s1.integrate_point_pos, pts=None,
                                  func_args=[sel_dist, theta, DemogSelModels.equil],
                                  lower_bound=[0, 0, None], upper_bound=[10, 1, 10],
                                  fixed_params=[None, None, 2],
                                  multinom=False, maxtime=10)
        # Test with gammapos allowed to vary
        popt,llopt = dadi.Inference.opt([1, 0.2, 2], data, s1.integrate_point_pos, pts=None,
                                  func_args=[sel_dist, theta, DemogSelModels.equil],
                                  lower_bound=[0, 0, None], upper_bound=[10, 1, 10],
                                  multinom=False, maxtime=10)

    def test_1D_integration_correctness(self):
        """
        Compare with result from previous verion of code built directly off fitdadi.
        """
        demo_params = [0.5, 2, 0.5, 0.1, 0, 0]
        ns = [8, 12]
        pts_l = [60, 80, 100]

        s1 = dadi.DFE.Cache1D(demo_params, ns, DemogSelModels.IM_sel_single_gamma, pts_l,
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
                                    1e5, DemogSelModels.IM_sel_single_gamma)
        comp = dadi.Spectrum.from_file(
            'test_data/fitdadi.IM_point_pos_test.fs')
        assert(np.allclose(fs, comp))

    def test_2D_cache_generation(self):
        """
        Trivial test that Cache2D generation doesn't crash.
        """
        demo_params = [0.5, 2, 0.5, 0.01, 0, 0]
        dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM_sel, [20],
                         gamma_bounds=(1e-4, 2), gamma_pts=2)
        s2 = dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM_sel, [20],
                         gamma_bounds=(1e-4, 2), gamma_pts=4, cpus=2)
        s2.integrate([2, 1, 0.4], None, PDFs.biv_lognormal, 1, None)

        # Merging of separate caches
        s2a = dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM_sel, [20],
                gamma_bounds=(1e-4, 2), gamma_pts=4, cpus=2,
                split_jobs=3, this_job_id=0)
        s2b = dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM_sel, [20],
                gamma_bounds=(1e-4, 2), gamma_pts=4, cpus=2,
                split_jobs=3, this_job_id=1)
        s2c = dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM_sel, [20],
                gamma_bounds=(1e-4, 2), gamma_pts=4, cpus=2,
                split_jobs=3, this_job_id=2)
        # Merge caches
        s2m = dadi.DFE.Cache2D.merge([s2a,s2b,s2c])
        assert(np.allclose(s2.spectra, s2m.spectra))
        # Incomplete merge
        with self.assertRaises(ValueError):
            dadi.DFE.Cache2D.merge([s2a,s2b])

    def test_cache_GPU(self):
        """
        Test that Cache generation with GPUs works
        """
        # Short circuit test if not CUDA
        if not dadi.cuda_enabled():
            return

        demo_params = [0.5, 2, 0.5, 0.01, 0, 0]
        s2 = dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM_sel, [20],
                         gamma_bounds=(1e-4, 2), gamma_pts=4, cpus=2)
        s2_gpu = dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM_sel, [20],
                         gamma_bounds=(1e-4, 2), gamma_pts=4, cpus=0, gpus=1)
        self.assertTrue(np.allclose(s2.spectra, s2_gpu.spectra))

    def test_2D_integration(self):
        """
        Trivial test that Cache2D integration doesn't crash.
        """
        demo_params = [0.5, 2, 0.5, 0.01, 0, 0]
        s2 = dadi.DFE.Cache2D(demo_params, [3, 3], DemogSelModels.IM_sel, [20],
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
        data = theta*DemogSelModels.IM_sel(demo_params+[-2, -3], ns, pts_l[-1])
        s2 = dadi.DFE.Cache2D(demo_params, ns, DemogSelModels.IM_sel, pts_l,
                              gamma_bounds=(1e-4, 2), gamma_pts=2,
                              additional_gammas=[0.2])
        sel_dist = PDFs.biv_lognormal
        # Test with basic integration
        popt,llopt = dadi.Inference.opt([2, 1, 0.5], data, s2.integrate, pts=None,
                                  func_args=[sel_dist, theta],
                                  lower_bound=[None, 0, -0.999], upper_bound=[None, None, 0.999],
                                  multinom=False, maxtime=10)
        # Test with point mass of positive selection
        popt,llopt = dadi.Inference.opt([2, 1, 0.5, 0.1, 0.2], data, s2.integrate_symmetric_point_pos, pts=None,
                                  func_args=[sel_dist, theta],
                                  lower_bound=[None, 0, -0.999, None, None], upper_bound=[None, None, 0.999, None, None],
                                  fixed_params=[None, None, None, None, 0.2],
                                  multinom=False, maxtime=10)

    def test_2D_correctness(self):
        """
        Compare with result from previous verion of code built directly off fitdadi.
        """
        demo_params = [0.5, 2, 0.5, 0.1, 0, 0]
        ns = [8, 12]
        pts_l = [60, 80, 100]

        s2 = dadi.DFE.Cache2D(demo_params, ns, DemogSelModels.IM_sel, pts=pts_l,
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
        s1 = dadi.DFE.Cache1D(demo_params, ns, DemogSelModels.IM_sel_single_gamma, pts_l,
                              gamma_bounds=(1e-4, 20), gamma_pts=2,
                              additional_gammas=[4.3])
        s2 = dadi.DFE.Cache2D(demo_params, ns, DemogSelModels.IM_sel, pts_l,
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

    def test_Vourlaki_normalization(self):
        """
        Tests of normalization with Vourlaki_mixture
        """
        ns, pts_l = [2,2], [1]
        s1 = Cache1D([], ns, trivial_fs, pts=pts_l, gamma_pts=100,
                     gamma_bounds=(1e-4, 2000), additional_gammas=[10])
        s2 = Cache2D([], ns, trivial_fs, pts=pts_l, gamma_pts=100,
                     gamma_bounds=(1e-4, 2000), additional_gammas=[10])

        # No gamma changes, no positive component: ppos_wild=0.0, pchange=0.0
        fs = Vourlaki_mixture([1, 10, 0, 10, 0, 0], None, s1, s2, 1.0, None)
        assert(np.allclose(fs.sum(), 1, atol=0.01))

        # No gamma changes, with positive component: ppos_wild=0.5, pchange=0.0
        fs = Vourlaki_mixture([1, 10, 0.5, 10, 0, 0], None, s1, s2, 1.0, None)
        assert(np.allclose(fs.sum(), 1, atol=0.01))

        # Substantial gamma changes, no positive component: ppos_wild=0.0, pchange=0.5
        fs = Vourlaki_mixture([1, 10, 0.0, 10, 0.5, 0], None, s1, s2, 1.0, None)
        assert(np.allclose(fs.sum(), 1, atol=0.01))

        # Substantial gamma changes, with positive component: ppos_wild=0.5, pchange=0.5
        fs = Vourlaki_mixture([1, 10, 0.5, 10, 0.5, 0], None, s1, s2, 1.0, None)
        assert(np.allclose(fs.sum(), 1, atol=0.01))

        # Substantial gamma changes, with positive component, some changing to positive:
        #  ppos_wild=0.5, pchange=0.5, pchange_pos=0.5
        fs = Vourlaki_mixture([1, 10, 0.5, 10, 1.0, 0.5], None, s1, s2, 1.0, None)
        assert(np.allclose(fs.sum(), 1, atol=0.01))

        # Theta != 1.0
        fs = Vourlaki_mixture([1, 10, 0.5, 10, 1.0, 0.5], None, s1, s2, 2.0, None)
        assert(np.allclose(fs.sum(), 2, atol=0.01))

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

    func_ex_single = dadi.Numerics.make_extrap_func(Selection_2d.IM_sel_single_sel)
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
    #try:
    #    generate_old_fitdadi_data()
    #    print('Generated data for comparison with old fitdadi code.')
    #except ImportError:
    #    print('Failed to import old fitdadi code, using stored comparison results.')
    #    pass

    # Run tests using Windows-style multiprocessing. This is more fragile, so
    # we test against it.
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    unittest.main()
import unittest
import dadi
import numpy as np

#simple model to test splits up to three populations
model = "../examples/demes_test/gutenkunst_ooa.yml"
sampled_demes = ["YRI", "CEU", "CHB"]
sample_sizes = [10, 10, 10]

# #simple model to test admix using all populations to form a new one
# model = "../examples/demes_test/browning_america.yml"
# sampled_demes = ['AFR', 'EUR', 'EAS', 'ADMIX']
# sample_sizes = [10, 10, 10, 10]

pts_l = [15,20,25]
class DemesTestCase(unittest.TestCase):
    def test_basic_loading(self):
        fs_demes = dadi.Spectrum.from_demes(model, sampled_demes=sampled_demes, sample_sizes=sample_sizes, pts=pts_l)
        self.assertTrue(np.allclose(fs_demes[2,5,6], 0.0025822560555528017))
        self.assertTrue(np.allclose(fs_demes[6,3,2], 0.0017156293580770643))
    def test_demes_vs_dadi(self):
        fs_demes = dadi.Spectrum.from_demes(model, sampled_demes=sampled_demes, sample_sizes=sample_sizes, pts=pts_l)
        def OutOfAfrica(params, ns, pts):
            nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs = params
            xx = dadi.Numerics.default_grid(pts)

            phi = dadi.PhiManip.phi_1D(xx)
            phi = dadi.Integration.one_pop(phi, xx, TAf, nu = nuAf)

            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
            phi = dadi.Integration.two_pops(phi, xx, TB, nu1 = nuAf, nu2 = nuB, m12 = mAfB, m21 = mAfB)

            phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)

            nuEu_func = lambda t : nuEu0 * (nuEu/nuEu0) ** (t/TEuAs)
            nuAs_func = lambda t : nuAs0 * (nuAs/nuAs0) ** (t/TEuAs)
            phi = dadi.Integration.three_pops(phi, xx, TEuAs, nu1 = nuAf, nu2 = nuEu_func, nu3 = nuAs_func, 
                m12 = mAfEu, m13 = mAfAs, m21 = mAfEu, m23 = mEuAs, m31 = mAfAs, m32 = mEuAs)

            fs = dadi.Spectrum.from_phi(phi, ns, (xx, xx, xx))
            return fs

        params = [1.6849315068493151, 0.2876712328767123, 
        0.136986301369863, 4.071917808219178, 0.06986301369863014, 7.409589041095891, 
        3.65, 0.438, 0.2774, 1.4016,
        0.2191780821917808, 0.3254794520547945, 0.05808219178082192]

        func_ex = dadi.Numerics.make_extrap_func(OutOfAfrica)

        fs_dadi = func_ex(params, sample_sizes, pts_l)
        # fs_demes.pop_ids=None

        self.assertTrue(np.allclose(fs_demes[2,5,6], fs_dadi[2,5,6]))
        self.assertTrue(np.allclose(fs_demes[6,3,2], fs_demes[6,3,2]))

suite=unittest.TestLoader().loadTestsFromTestCase(DataTestCase)

if __name__ == '__main__':
    unittest.main()









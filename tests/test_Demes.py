import unittest
import dadi
import numpy as np
from dadi.Demes import Demes

#simple model to test splits up to three populations
model = "../examples/demes_test/gutenkunst_ooa.yml"
sampled_demes = ["YRI", "CEU", "CHB"]
sample_sizes = [10, 10, 10]

pts_l = [15,20,25]
pts = pts_l[0]
class TestSplits(unittest.TestCase):
    def test_split(self):
        xx = dadi.Numerics.default_grid(pts)
        phi1D = dadi.PhiManip.phi_1D(xx)

        dadi_phi2D = dadi.PhiManip.phi_1D_to_2D(xx, phi1D)
        dadi_phi3D_split_1 = dadi.PhiManip.phi_2D_to_3D_split_1(xx, dadi_phi2D)
        dadi_phi3D_split_2 = dadi.PhiManip.phi_2D_to_3D_split_2(xx, dadi_phi2D)
        
        pop_ids = ['1']
        parent = '1'
        children=['A', 'B']
        demes_phi2D = Demes._split_phi(phi1D, xx, pop_ids, parent)
        self.assertTrue(np.allclose(demes_phi2D, dadi_phi2D))

        pop_ids = ['A','B']
        for i, parent in zip(range(2),pop_ids):
            phifunc = [dadi.PhiManip.phi_2D_to_3D_split_1,dadi.PhiManip.phi_2D_to_3D_split_2][i]
            dadi_phi3D = phifunc(xx, dadi_phi2D)
            demes_phi3D = Demes._split_phi(demes_phi2D, xx, pop_ids, parent)
            self.assertTrue(np.allclose(demes_phi3D, dadi_phi3D))
        
        pop_ids = ['A','B','C']
        proportions = [[1,0,0], [0,1,0], [0,0,1]]
        for props, parent in zip(proportions, pop_ids):
            dadi_phi4D = dadi.PhiManip.phi_3D_to_4D(dadi_phi3D, props[0], props[1], xx,xx,xx,xx)
            demes_phi4D = Demes._split_phi(dadi_phi3D, xx, pop_ids, parent)
            self.assertTrue(np.allclose(demes_phi4D, dadi_phi4D))

        pop_ids = ['A','B','C','D']
        proportions = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
        for props, parent in zip(proportions, pop_ids):
            dadi_phi5D = dadi.PhiManip.phi_4D_to_5D(dadi_phi4D, props[0], props[1], props[2], xx,xx,xx,xx,xx)
            demes_phi5D = Demes._split_phi(dadi_phi4D, xx, pop_ids, parent)
            self.assertTrue(np.allclose(demes_phi4D, dadi_phi4D))

class DemesDataTestCase(unittest.TestCase):
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

        Ne = 7300
        gens = 25
        nuA = 12300 / Ne
        TA = (220e3 - 140e3) / 2 / Ne / gens
        nuB = 2100 / Ne
        TB = (140e3 - 21.2e3) / 2 / Ne / gens
        nuEu0 = 1000 / Ne
        nuEuF = 29725 / Ne
        nuAs0 = 510 / Ne
        nuAsF = 54090 / Ne
        TF = 21.2e3 / 2 / Ne / gens
        mAfB = 2 * Ne * 25e-5
        mAfEu = 2 * Ne * 3e-5
        mAfAs = 2 * Ne * 1.9e-5
        mEuAs = 2 * Ne * 9.6e-5
        params = [
        nuA, nuB, nuEu0, nuEuF, nuAs0, nuAsF, mAfB, mAfEu, mAfAs, mEuAs, TA, TB, TF
        ]
        # params = [1.6849315068493151, 0.2876712328767123, 
        # 0.136986301369863, 4.071917808219178, 0.06986301369863014, 7.409589041095891, 
        # 3.65, 0.438, 0.2774, 1.4016,
        # 0.2191780821917808, 0.3254794520547945, 0.05808219178082192]

        func_ex = dadi.Numerics.make_extrap_func(OutOfAfrica)

        fs_dadi = func_ex(params, sample_sizes, pts_l)
        # fs_demes.pop_ids=None

        l = []
        for i in range(1,len(fs_demes)-1):
            for ii in range(1,len(fs_demes[i])-1):
                for iii in range(1,len(fs_demes[i][ii])-1):
                    if np.allclose(fs_demes[i,ii,iii], fs_dadi[i,ii,iii]):
                        l.append([i,ii,iii])

        self.assertTrue(np.allclose(fs_demes[2,5,6], fs_dadi[2,5,6]))
        self.assertTrue(np.allclose(fs_demes[6,3,2], fs_demes[6,3,2]))

suite=unittest.TestLoader().loadTestsFromTestCase(TestSplits)

if __name__ == '__main__':
    unittest.main()









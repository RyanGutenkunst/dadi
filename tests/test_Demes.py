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
class DemesTests(unittest.TestCase):
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
            self.assertTrue(np.allclose(demes_phi5D, dadi_phi5D))

    def test_integration(self):
        xx = dadi.Numerics.default_grid(pts)
        phi1D = dadi.PhiManip.phi_1D(xx)

        integration_params = nu, T, M, gamma, h, theta, frozen = [
        [1.685],0.22, [], [0], [0.5], 1, [False]
        ]
        pop_ids = ['1']

        dadi_phi1D = dadi.Integration.one_pop(phi1D, xx, T, nu[0])
        demes_phi1D = Demes._integrate_phi(phi1D, xx, integration_params, pop_ids)

        integration_params = nu, T, M, gamma, h, theta, frozen = [
        [1.685,0.29],0.325, np.array([[0,3.65],[3.65,0]]), [0,0], [0.5,0.5], 1, [False,False]
        ]
        pop_ids = ['1','2']
        phi2D = dadi.PhiManip.phi_1D_to_2D(xx, dadi_phi1D)

        dadi_phi2D = dadi.Integration.two_pops(phi2D, xx, T, nu[0], nu[1], m12=M[0,1], m21=M[1,0])
        demes_phi2D = Demes._integrate_phi(phi2D, xx, integration_params, pop_ids)
        self.assertTrue(np.allclose(demes_phi2D, dadi_phi2D))

        integration_params = nu, T, M, gamma, h, theta, frozen = [
        [2,1,3],1, np.array([[0,1,2],[1,0,2],[2,2,0]]), 
        [0,0,0], [0.5,0.5,0.5], 1, [False,False,False]
        ]
        pop_ids = ['1','2','3']
        phi3D = dadi.PhiManip.phi_2D_to_3D_split_2(xx, dadi_phi2D)

        dadi_phi3D = dadi.Integration.three_pops(phi3D, xx, T, nu[0], nu[1], nu[2], 
        	m12=M[0,1], m13=M[0,2], m21=M[1,0], m23=M[1,2], m31=M[2,0], m32=M[2,1])
        demes_phi3D = Demes._integrate_phi(phi3D, xx, integration_params, pop_ids)
        self.assertTrue(np.allclose(demes_phi3D, dadi_phi3D))

    def test_admix(self):
        xx = dadi.Numerics.default_grid(pts)
        phi1D = dadi.PhiManip.phi_1D(xx)

        dadi_phi2D = dadi.PhiManip.phi_1D_to_2D(xx, phi1D)

        pop_ids = ['1']
        parent = '1'
        children=['A', 'B']
        demes_phi2D = Demes._split_phi(phi1D, xx, pop_ids, parent)

        proportions, pop_ids = [0.7, children]
        for sources, dest in zip(pop_ids, pop_ids[::-1]):
            i = pop_ids.index(dest)
            phifunc = [
            dadi.PhiManip.phi_2D_admix_2_into_1,
            dadi.PhiManip.phi_2D_admix_1_into_2
            ][i]
            dadi_phi2D_admix = phifunc(dadi_phi2D, proportions, xx,xx)
            demes_phi2D_admix = Demes._admix_phi(demes_phi2D, xx, proportions, pop_ids, sources, dest)
            self.assertTrue(np.allclose(demes_phi2D_admix, dadi_phi2D_admix))
    def test_admix_new_pop(self):
        xx = dadi.Numerics.default_grid(pts)
        phi1D = dadi.PhiManip.phi_1D(xx)

        dadi_phi2D = dadi.PhiManip.phi_1D_to_2D(xx, phi1D)

        pop_ids = ['1']
        parent = '1'
        children=['A', 'B']
        demes_phi2D = Demes._split_phi(phi1D, xx, pop_ids, parent)

        proportions, pop_ids, parents = [[0.7, 0.3], children, children]
        dadi_phi3D = dadi.PhiManip.phi_2D_to_3D_admix(dadi_phi2D, proportions[0], xx,xx,xx)
        demes_phi3D = Demes._admix_new_pop_phi(demes_phi2D, xx, proportions, pop_ids, parents)
        self.assertTrue(np.allclose(demes_phi3D, dadi_phi3D))

    def test_basic_loading(self):
        fs_demes = dadi.Spectrum.from_demes(model, sampled_demes=sampled_demes, sample_sizes=sample_sizes, pts=pts_l)
        self.assertTrue(np.allclose(fs_demes[2,5,6], 0.0025822560555528017))
        self.assertTrue(np.allclose(fs_demes[6,3,2], 0.0017156293580770643))

    def test_demes_vs_dadi(self):
        fs_demes = dadi.Spectrum.from_demes(model, sampled_demes=sampled_demes, sample_sizes=sample_sizes, pts=pts_l)
        
        def OutOfAfrica_with_demes_reordering(params, ns, pts):
            nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs = params
            xx = dadi.Numerics.default_grid(pts)

            phi = dadi.PhiManip.phi_1D(xx)
            phi = dadi.Integration.one_pop(phi, xx, TAf, nu = nuAf)

            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
            #Reorder after first split
            phi = dadi.PhiManip.reorder_pops(phi,[2,1])

            # 2D Integration
            # OOA is first pop, so now nu1=nuB and nu2=nuAf
            phi = dadi.Integration.two_pops(phi, xx, TB, nu1 = nuB, nu2 = nuAf, m12 = mAfB, m21 = mAfB)

            # OOA is now the first population, so you use phi_2D_to_3D_split_1
            # instead of phi_2D_to_3D_split_2
            phi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)

            # Because YRI should be the first pop, we need to reorder
            # so that it is the first population
            phi = dadi.PhiManip.reorder_pops(phi,[2,1,3])

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

        func_ex = dadi.Numerics.make_extrap_func(OutOfAfrica_with_demes_reordering)
        fs_dadi = func_ex(params, sample_sizes, pts_l)

        self.assertTrue(np.allclose(fs_demes, fs_dadi))
        #self.assertTrue(np.allclose(fs_demes[6,3,2], fs_demes[6,3,2]))

suite=unittest.TestLoader().loadTestsFromTestCase(DemesTests)

if __name__ == '__main__':
    unittest.main()

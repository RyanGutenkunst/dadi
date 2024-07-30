import pytest
import dadi
import numpy as np

# Check if Demes is installed
try:
    from dadi.Demes import Demes
    dadi.Spectrum.from_demes("tests/demes/gutenkunst_ooa.yaml", 
                            sampled_demes=["YRI", "CEU", "CHB"], 
                            sample_sizes=[2, 3, 4], 
                            pts=[15,20,25])
    skip = False
except:
    print("ImportError: demes is not installed, need to `pip install demes`")
    skip = True

@pytest.fixture
def test_details():
    #simple model to test splits up to three populations
    pytest.model = "tests/demes/gutenkunst_ooa.yaml"
    pytest.sampled_demes = ["YRI", "CEU", "CHB"]
    pytest.sample_sizes = [10, 10, 10]

    pytest.pts_l = [15,20,25]
    pytest.pts = 15

@pytest.mark.skipif(skip, reason="Could not load Demes")
def test_split(test_details):
    xx = dadi.Numerics.default_grid(pytest.pts)
    phi1D = dadi.PhiManip.phi_1D(xx)

    dadi_phi2D = dadi.PhiManip.phi_1D_to_2D(xx, phi1D)
    dadi_phi3D_split_1 = dadi.PhiManip.phi_2D_to_3D_split_1(xx, dadi_phi2D)
    dadi_phi3D_split_2 = dadi.PhiManip.phi_2D_to_3D_split_2(xx, dadi_phi2D)
    
    pop_ids = ['1']
    parent = '1'
    children=['A', 'B']
    demes_phi2D = Demes._split_phi(phi1D, xx, pop_ids, parent, new_pop_ids=children)
    assert(np.allclose(demes_phi2D, dadi_phi2D))

    pop_ids = ['A','B']
    for i, parent in zip(range(2),pop_ids):
        phifunc = [dadi.PhiManip.phi_2D_to_3D_split_1,dadi.PhiManip.phi_2D_to_3D_split_2][i]
        dadi_phi3D = phifunc(xx, dadi_phi2D)
        demes_phi3D = Demes._split_phi(demes_phi2D, xx, pop_ids, parent, new_pop_ids=['A','B','C'])
        assert(np.allclose(demes_phi3D, dadi_phi3D))
    
    pop_ids = ['A','B','C']
    proportions = [[1,0,0], [0,1,0], [0,0,1]]
    for props, parent in zip(proportions, pop_ids):
        dadi_phi4D = dadi.PhiManip.phi_3D_to_4D(dadi_phi3D, props[0], props[1], xx,xx,xx,xx)
        demes_phi4D = Demes._split_phi(dadi_phi3D, xx, pop_ids, parent, new_pop_ids=['A','B','C','D'])
        assert(np.allclose(demes_phi4D, dadi_phi4D))

    pop_ids = ['A','B','C','D']
    proportions = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    for props, parent in zip(proportions, pop_ids):
        dadi_phi5D = dadi.PhiManip.phi_4D_to_5D(dadi_phi4D, props[0], props[1], props[2], xx,xx,xx,xx,xx)
        demes_phi5D = Demes._split_phi(dadi_phi4D, xx, pop_ids, parent, new_pop_ids=['A','B','C','D','E'])
        assert(np.allclose(demes_phi5D, dadi_phi5D))

@pytest.mark.skipif(skip, reason="Could not load Demes")
def test_integration(test_details):
    xx = dadi.Numerics.default_grid(pytest.pts)
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
    assert(np.allclose(demes_phi2D, dadi_phi2D))

    integration_params = nu, T, M, gamma, h, theta, frozen = [
    [2,1,3],1, np.array([[0,1,2],[1,0,2],[2,2,0]]), 
    [0,0,0], [0.5,0.5,0.5], 1, [False,False,False]
    ]
    pop_ids = ['1','2','3']
    phi3D = dadi.PhiManip.phi_2D_to_3D_split_2(xx, dadi_phi2D)

    dadi_phi3D = dadi.Integration.three_pops(phi3D, xx, T, nu[0], nu[1], nu[2], 
    	m12=M[0,1], m13=M[0,2], m21=M[1,0], m23=M[1,2], m31=M[2,0], m32=M[2,1])
    demes_phi3D = Demes._integrate_phi(phi3D, xx, integration_params, pop_ids)
    assert(np.allclose(demes_phi3D, dadi_phi3D))

@pytest.mark.skipif(skip, reason="Could not load Demes")
def test_admix(test_details):
    xx = dadi.Numerics.default_grid(pytest.pts)
    phi1D = dadi.PhiManip.phi_1D(xx)

    dadi_phi2D = dadi.PhiManip.phi_1D_to_2D(xx, phi1D)

    pop_ids = ['1']
    parent = '1'
    children=['A', 'B']
    demes_phi2D = Demes._split_phi(phi1D, xx, pop_ids, parent, new_pop_ids=children)

    proportions, pop_ids = [0.7, children]
    for sources, dest in zip(pop_ids, pop_ids[::-1]):
        i = pop_ids.index(dest)
        phifunc = [
        dadi.PhiManip.phi_2D_admix_2_into_1,
        dadi.PhiManip.phi_2D_admix_1_into_2
        ][i]
        dadi_phi2D_admix = phifunc(dadi_phi2D, proportions, xx,xx)
        demes_phi2D_admix = Demes._admix_phi(demes_phi2D, xx, proportions, pop_ids, sources, dest)
        assert(np.allclose(demes_phi2D_admix, dadi_phi2D_admix))

@pytest.mark.skipif(skip, reason="Could not load Demes")
def test_admix_new_pop(test_details):
    xx = dadi.Numerics.default_grid(pytest.pts)
    phi1D = dadi.PhiManip.phi_1D(xx)

    dadi_phi2D = dadi.PhiManip.phi_1D_to_2D(xx, phi1D)

    pop_ids = ['1']
    parent = '1'
    children=['A', 'B']
    demes_phi2D = Demes._split_phi(phi1D, xx, pop_ids, parent, new_pop_ids=children)

    proportions, pop_ids, parents = [[0.7, 0.3], children, children]
    dadi_phi3D = dadi.PhiManip.phi_2D_to_3D_admix(dadi_phi2D, proportions[0], xx,xx,xx)
    demes_phi3D = Demes._admix_new_pop_phi(demes_phi2D, xx, proportions, pop_ids, parents, new_pop_ids=children)
    assert(np.allclose(demes_phi3D, dadi_phi3D))

@pytest.mark.skipif(skip, reason="Could not load Demes")
def test_basic_loading(test_details):
    fs_demes = dadi.Spectrum.from_demes(pytest.model, sampled_demes=pytest.sampled_demes, sample_sizes=pytest.sample_sizes, pts=pytest.pts_l)
    assert(np.allclose(fs_demes[2,5,6], 0.0025822560555528017))
    assert(np.allclose(fs_demes[6,3,2], 0.0017156293580770643))

@pytest.mark.skipif(skip, reason="Could not load Demes")
def test_demes_vs_dadi(test_details):
    fs_demes = dadi.Spectrum.from_demes(pytest.model, sampled_demes=pytest.sampled_demes, sample_sizes=pytest.sample_sizes, pts=pytest.pts_l)
    
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
    fs_dadi = func_ex(params, pytest.sample_sizes, pytest.pts_l)

    assert(np.allclose(fs_demes, fs_dadi))
    #self.assertTrue(np.allclose(fs_demes[6,3,2], fs_demes[6,3,2]))

def import_export_match(fname, sampled_demes, sample_sizes, pts, Nref, generation_time, deme_mapping=None):
    fsin = dadi.Spectrum.from_demes(fname, sampled_demes, sample_sizes, pts)
    gout = dadi.Demes.output(Nref, deme_mapping, generation_time)
    fsout = dadi.Spectrum.from_demes(gout, sampled_demes, sample_sizes, pts)
    assert(np.allclose(fsin,fsout, rtol=1e-3, atol=1e-4))

@pytest.mark.skipif(skip, reason="Could not load Demes")
def test_demes_export_match():
    import_export_match('tests/demes/bottleneck.yaml', ['our_population'], [5], 10, 1e4, 1)
    import_export_match('tests/demes/browning_america.yaml', ['AFR', 'EAS', 'EUR', 'ADMIX'], [2,3,4,5], 5, 7310, 1)
     # cloning_example.yaml: dadi doesn't implement cloning
    import_export_match('tests/demes/gutenkunst_ooa.yaml', ['YRI','CEU','CHB'], [3,5,8], [10,15,20], 7300, 25)
    # jacobs_papuans.yaml: too many populations for dadi
    import_export_match('tests/demes/linear_size_function_example.yaml', ['pop_1','pop_2'], [4,5], 20, 100, 1)
    import_export_match('tests/demes/offshoots.yaml', ['ancestral','offshoot1','offshoot2'], [3,5,8], 20, 1000, 1)
    # selfing_example.yaml: dadi doesn't implement selfing
    import_export_match('tests/demes/two_epoch.yaml', ['deme0'], [10], 20, 1000, 1)
    import_export_match('tests/demes/zigzag.yaml', ['generic'], [10], 20, 7156, 1)

@pytest.mark.skipif(skip, reason="Could not load Demes")
def test_export_mapping():
    def three_test(params, ns, pts):
        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx, nu=2)

        phi = dadi.Integration.one_pop(phi, xx, T=0.1, nu=2)

        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)

        nu_func = lambda t: 0.1 * (3/0.1)**(t/0.2)
        phi = dadi.Integration.two_pops(phi, xx, 0.2, nu_func, 3, m12=2)
        phi = dadi.Integration.two_pops(phi, xx, 0.2, 0.2, 0.3, m21=0.1)

        phi = dadi.PhiManip.phi_2D_to_3D(phi, 0.3, xx, xx, xx)

        phi = dadi.Integration.three_pops(phi, xx, 0.3, 1, 2, 3, m12=2, m13=4)
        phi = dadi.PhiManip.phi_3D_admix_1_and_3_into_2(phi, 0.2, 0, xx,xx,xx)
        phi = dadi.PhiManip.reorder_pops(phi, [3,2,1])
        phi = dadi.Integration.three_pops(phi, xx, 0.2, 3, 2, 1)

        phi = dadi.PhiManip.filter_pops(phi, xx, [1])
        #phi = dadi.Integration.two_pops(phi, xx, 0.2, 0.4, 0.1, m21=0.1)
        phi = dadi.Integration.one_pop(phi, xx, T=0.1, nu=2)

        return dadi.Spectrum.from_phi(phi, ns, [xx])

    fs = three_test(None, [2], 4)
    g = dadi.Demes.output(deme_mapping={'YRI':['d1_1', 'd1_2', 'd1_3'], 'Bottle':['d2_2'],
                                                'CEU':['d2_3'], 'CHB':['d3_3']})
    # The correctness test here is visual. Here just testing whether method crashes.
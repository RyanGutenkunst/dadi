import numpy as np
import dadi
import demes, demesdraw

def one_test(params, ns, pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx, nu=0.3)

    phi = dadi.Integration.one_pop(phi, xx, T=0.1, nu=2)
    nu_func = lambda t: 0.1 * (3/0.1) ** np.exp(t/0.3)
    phi = dadi.Integration.one_pop(phi, xx, T=0.2, nu=nu_func)
    phi = dadi.Integration.one_pop(phi, xx, T=0.3, nu=1)

    return dadi.Spectrum.from_phi(phi, ns, [xx])

def two_test(params, ns, pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx, nu=20)

    phi = dadi.Integration.one_pop(phi, xx, T=0.1, nu=2)

    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)

    phi = dadi.Integration.two_pops(phi, xx, 0.2, 2, 3, m12=0.00001)
    phi = dadi.PhiManip.phi_2D_admix_1_into_2(phi, 0, xx,xx)
    phi = dadi.Integration.two_pops(phi, xx, 0.2, 0.2, 0.3, m12=20)

    return dadi.Spectrum.from_phi(phi, ns, [xx,xx])

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

def three_test_named(params, ns, pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx, nu=2, deme_ids=['ancestral'])

    phi = dadi.Integration.one_pop(phi, xx, T=0.1, nu=2, deme_ids=['AMH'])

    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi, deme_ids=['YRI', 'OOA'])

    nu_func = lambda t: 0.1 * (3/0.1)**(t/0.2)
    phi = dadi.Integration.two_pops(phi, xx, 0.2, nu_func, 3, m12=2)
    phi = dadi.Integration.two_pops(phi, xx, 0.2, 0.2, 0.3, m21=0.1, deme_ids=['YRI_switch','OOA'])

    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0.3, xx, xx, xx, deme_ids=['YRI_final','CEU','CHB'])

    phi = dadi.Integration.three_pops(phi, xx, 0.3, 1, 2, 3, m12=2, m13=4)
    phi = dadi.PhiManip.phi_3D_admix_1_and_3_into_2(phi, 0.2, 0, xx,xx,xx)
    phi = dadi.PhiManip.reorder_pops(phi, [3,2,1])
    phi = dadi.Integration.three_pops(phi, xx, 0.2, 3, 2, 1, deme_ids=['CHB_final','CEU','YRI_final'])

    phi = dadi.PhiManip.filter_pops(phi, xx, [1])
    phi = dadi.Integration.one_pop(phi, xx, T=0.1, nu=2)

    return dadi.Spectrum.from_phi(phi, ns, [xx])

fs = one_test(None, [2], 4)
g = dadi.Demes.output()
demesdraw.tubes(g)

fs = two_test(None, [2,2], 4)
g = dadi.Demes.output(deme_mapping={'AMH':['d1_1'], 'YRI':['d1_2'], 'CEU':['d2_2']})
demesdraw.tubes(g)

fs = three_test(None, [2], 4)
g = dadi.Demes.output()
demesdraw.tubes(g)

fs = three_test(None, [2], 4)
g = dadi.Demes.output(deme_mapping={'YRI':['d1_1', 'd1_2', 'd1_3'], 'Bottle':['d2_2'],
                                            'CEU':['d2_3'], 'CHB':['d3_3']})
demesdraw.tubes(g)

fs = three_test_named(None, [2], 4)
g = dadi.Demes.output()
demesdraw.tubes(g)

demes_ex = dadi.Numerics.make_extrap_func(dadi.Demes.SFS)
def demes_test(fname, sampled_demes, sample_sizes, pts, Nref, generation_time, deme_mapping=None, draw=False):
    gin = demes.load(fname)
    print('{0}'.format(gin.description))
    if draw:
        demesdraw.tubes(gin)
    fsin = demes_ex(gin, sampled_demes, sample_sizes, pts)
    gout = dadi.Demes.output(Nref, deme_mapping, generation_time)
    if draw:
        demesdraw.tubes(gout)
    fsout = demes_ex(gout, sampled_demes, sample_sizes, pts)
    print(np.allclose(fsin,fsout, rtol=1e-3, atol=1e-4))
    return gin, gout, fsin, fsout

# XXX: TODO: demes_hist in CUDA Integration methods

demes_test('demes/bottleneck.yaml', ['our_population'], [5], 10, 1e4, 1)
demes_test('demes/browning_america.yaml', ['AFR', 'EAS', 'EUR', 'ADMIX'], [2,3,4,5], 5, 7310, 1) # cloning_example.yaml: dadi doesn't implement cloning
demes_test('demes/gutenkunst_ooa.yaml', ['YRI','CEU','CHB'], [3,5,8], [10,15,20], 7300, 25)
# jacobs_papuans.yaml: too many populations for dadi
demes_test('demes/linear_size_function_example.yaml', ['pop_1','pop_2'], [4,5], 20, 100, 1)
demes_test('demes/offshoots.yaml', ['ancestral','offshoot1','offshoot2'], [3,5,8], 20, 1000, 1)
# selfing_example.yaml: dadi doesn't implement selfing
demes_test('demes/two_epoch.yaml', ['deme0'], [10], 20, 1000, 1)
demes_test('demes/zigzag.yaml', ['generic'], [10], 20, 7156, 1)

import matplotlib.pyplot as plt
#plt.show()
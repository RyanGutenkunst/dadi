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

fs = one_test(None, [2], 4)
g = dadi.Demes.output()
ax = demesdraw.tubes(g)
ax.figure.savefig('temp_three_epoch.pdf')

fs = two_test(None, [2,2], 4)
g = dadi.Demes.output(deme_mapping={'YRI':['d1_1', 'd1_2'], 'CEU':['d2_2']})
ax = demesdraw.tubes(g)
ax.figure.savefig('temp_split_mig.pdf')

fs = three_test(None, [2], 4)
g = dadi.Demes.output()
ax = demesdraw.tubes(g)
ax.figure.savefig('temp_three.pdf')

fs = three_test(None, [2], 4)
g = dadi.Demes.output(deme_mapping={'YRI':['d1_1', 'd1_2', 'd1_3'], 'Bottle':['d2_2'],
                                            'CEU':['d2_3'], 'CHB':['d3_3']})
ax = demesdraw.tubes(g)
ax.figure.savefig('temp_three_mapped.pdf')

gin = demes.load('demes/bottleneck.yaml')
fs1 = dadi.Demes.SFS(gin, ['our_population'], [5], pts=10)
gout = dadi.Demes.output(deme_mapping={'our_population':['d1_1']}, Nref=1e4)
print('{0}'.format(gin.description))
print(gout.isclose(gin))

gin = demes.load('demes/gutenkunst_ooa.yaml')
fs1 = dadi.Demes.SFS(gin, ['YRI','CEU','CHB'], [5,5,5], pts=10)
gout = dadi.Demes.output(Nref=7300, generation_time=25, 
                         deme_mapping={'AMH':['d1_1'], 'YRI':['d1_2','d2_3'], 'OOA':['d2_2'], 
                                       'CHB':['d3_3'], 'CEU':['d1_3']})
fs2 = dadi.Demes.SFS(gout, ['YRI','CEU','CHB'], [5,5,5], pts=10)
print('{0}'.format(gin.description))
print(np.allclose(fs1,fs2, rtol=1e-3, atol=1e-4))

gin = demes.load('demes/offshoots.yaml')
# Fails to integrate, due to issue with Pulses
fs1 = dadi.Demes.SFS(gin, ['ancestral','offshoot1','offshoot2'], [10,10,10], pts=20)

gin = demes.load('demes/zigzag.yaml')
fs1 = dadi.Demes.SFS(gin, ['generic'], [10], pts=20)
gout = dadi.Demes.output(Nref=7156, deme_mapping={'generic':['d1_1']})
print('{0}'.format(gin.description))
print(gout.isclose(gin))
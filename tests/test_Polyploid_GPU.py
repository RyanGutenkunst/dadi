import pytest
import dadi
import dadi.Polyploidy.Integration as PolyInt
import numpy as np

# set of tests comparing polyploidy integration with CPU code to GPU/CUDA code

try:
    import dadi.Polyploidy.cuda
    skip = False
except:
    skip = True

@pytest.mark.skipif(skip, reason="Could not load dadi CUDA")
def test_2d_const_params_diploid():
    T = 0.1
    nu1, nu2 = 0.1, 2
    m12, m21 = 0.3, 0.5
    s1_dict = {'gamma': -0.2, 'h': .1}
    s2_dict = {'gamma': 2, 'h': 0.9}
    theta0 = 2

    pts = 20
    xx = np.linspace(0, 1, pts)
    phi = np.random.uniform(size=((pts, pts)))

    dadi.Polyploidy.cuda_enabled(False)
    phi_cpu = PolyInt.two_pops(phi.copy(), xx, T=T, nu1=nu1, nu2=nu2, m12=m12, m21=m21, 
                               sel_dict1=s1_dict, sel_dict2=s2_dict, theta0=theta0)

    dadi.Polyploidy.cuda_enabled(True)
    phi_gpu = PolyInt.two_pops(phi.copy(), xx, T=T, nu1=nu1, nu2=nu2, m12=m12, m21=m21, 
                               sel_dict1=s1_dict, sel_dict2=s2_dict, theta0=theta0,
                                enable_cuda_cached=True)

    assert(np.allclose(phi_cpu, phi_gpu))

@pytest.mark.skipif(skip, reason="Could not load dadi CUDA")
def test_2d_temporal_params_alloautohexaploid():
    T = 0.1
    # need nu1=nu2, m12=m21, and a single selection dictionary 
    # for the polyploidy integration
    nu = lambda t: 0.1+10*t
    e12 = e21 = 0.3
    s_dict = {'gamma': lambda t: -1+2*t}
    theta0 = 2

    HEX_tetra = PolyInt.PloidyType.HEX_tetra
    HEX_dip = PolyInt.PloidyType.HEX_dip

    pts = 20
    xx = np.linspace(0, 1, pts)
    phi = np.random.uniform(size=((pts, pts)))

    dadi.cuda_enabled(False)
    phi_cpu = PolyInt.two_pops(phi.copy(), xx, T, nu, nu, e12, e21, 
                               s_dict, s_dict, HEX_tetra, HEX_dip, theta0)

    dadi.cuda_enabled(True)
    phi_gpu = PolyInt.two_pops(phi.copy(), xx, T, nu, nu, e12, e21, 
                               s_dict, s_dict, HEX_tetra, HEX_dip, theta0)

    assert(np.allclose(phi_cpu, phi_gpu))

@pytest.mark.skipif(skip, reason="Could not load dadi CUDA")
def test_3d_const_params_alloallohexaploid():
    pts = 13

    # need nu1=nu2=nu3, a single selection dictionary, 
    # and m12=m21, m13=m31, m23=m32 for the polyploidy integration
    nu1 = nu2 = nu3 = .2
    e12 = e21 = 3
    e13 = e31 = 10
    e23 = e32 = 0.1
    s_dict = {'gamma': -1}
    theta0, initial_t, T = [10.2, 0.1, 0.1+0.1]
    frozen1, frozen2, frozen3 = False, False, False

    HEX_a = PolyInt.PloidyType.HEXa
    HEX_b = PolyInt.PloidyType.HEXb
    HEX_c = PolyInt.PloidyType.HEXc

    xx = np.linspace(0,1,pts)
    np.random.seed(213)
    phi = np.random.uniform(size=(pts,pts,pts))

    dadi.cuda_enabled(False)
    phi_cpu = PolyInt.three_pops(phi, xx, T, nu1, nu2, nu3,
                   e12, e13, e21, e23, e31, e32,
                   s_dict, s_dict, s_dict, HEX_a, HEX_b, HEX_c,
                   theta0, initial_t, frozen1, frozen2, frozen3)

    dadi.cuda_enabled(True)
    phi_gpu = PolyInt.three_pops(phi, xx, T, nu1, nu2, nu3,
                   e12, e13, e21, e23, e31, e32,
                   s_dict, s_dict, s_dict, HEX_a, HEX_b, HEX_c,
                   theta0, initial_t, frozen1, frozen2, frozen3,
                   enable_cuda_cached=True)
    
    assert(np.allclose(phi_cpu, phi_gpu))

    # Need to handle frozen populations carefully in the function,
    # so we test all cases here.
    e12, e13, e21, e23, e31, e32 = [0]*6
    for frozen1 in [True, False]:
        for frozen2 in [True, False]:
            for frozen3 in [True, False]:
                dadi.cuda_enabled(False)
                phi_cpu = PolyInt.three_pops(phi, xx, T, nu1, nu2, nu3,
                                             e12, e13, e21, e23, e31, e32,
                                             s_dict, s_dict, s_dict, HEX_a, HEX_b, HEX_c,
                                             theta0, initial_t, frozen1, frozen2, frozen3)

                dadi.cuda_enabled(True)
                phi_gpu = PolyInt.three_pops(phi, xx, T, nu1, nu2, nu3,
                                             e12, e13, e21, e23, e31, e32,
                                             s_dict, s_dict, s_dict, HEX_a, HEX_b, HEX_c,
                                             theta0, initial_t, frozen1, frozen2, frozen3,
                                             enable_cuda_cached=True)
    
                assert(np.allclose(phi_cpu, phi_gpu))

@pytest.mark.skipif(skip, reason="Could not load dadi CUDA")
def test_3d_temporal_params_autotetraploid():
    pts = 17

    nu1, nu2, nu3 = [2,1,0.1]
    nu1 = lambda t: 0.1+4*t
    m12, m13, m21, m23, m31, m32 = [0.1,3,10,0,0.3,0.1]
    m23 = lambda t: 9-10*t
    s_dict1 = {'gamma': -1}
    s_dict2 = {'gamma': 2.0}
    s_dict3 = {'gamma': lambda t: t, 'h1': 0.2, 'h2': lambda t: 2*t, 'h3': 0.9}
    theta0, initial_t, T = [10.2, 0.1, 0.1+0.1]
    frozen1, frozen2, frozen3 = False, False, False

    AUTO = PolyInt.PloidyType.AUTO

    xx = np.linspace(0,1,pts)
    np.random.seed(213)
    phi = np.random.uniform(size=(pts,pts,pts))

    dadi.cuda_enabled(False)
    phi_cpu = PolyInt.three_pops(phi, xx, T, nu1, nu2, nu3,
                   m12, m13, m21, m23, m31, m32,
                   s_dict1, s_dict2, s_dict3, AUTO, AUTO, AUTO,
                   theta0, initial_t, frozen1, frozen2, frozen3)

    dadi.cuda_enabled(True)
    phi_gpu = PolyInt.three_pops(phi, xx, T, nu1, nu2, nu3,
                   m12, m13, m21, m23, m31, m32,
                   s_dict1, s_dict2, s_dict3, AUTO, AUTO, AUTO,
                   theta0, initial_t, frozen1, frozen2, frozen3)
    
    assert(np.allclose(phi_cpu, phi_gpu))

    m12, m13, m21, m23, m31, m32 = [0]*6
    for frozen1 in [True, False]:
        for frozen2 in [True, False]:
            for frozen3 in [True, False]:
                dadi.cuda_enabled(False)
                phi_cpu = PolyInt.three_pops(phi, xx, T, nu1, nu2, nu3,
                                    m12, m13, m21, m23, m31, m32,
                                    s_dict1, s_dict2, s_dict3, AUTO, AUTO, AUTO,
                                    theta0, initial_t, frozen1, frozen2, frozen3)

                dadi.cuda_enabled(True)
                phi_gpu = PolyInt.three_pops(phi, xx, T, nu1, nu2, nu3,
                                    m12, m13, m21, m23, m31, m32,
                                    s_dict1, s_dict2, s_dict3, AUTO, AUTO, AUTO,
                                    theta0, initial_t, frozen1, frozen2, frozen3)
    
                assert(np.allclose(phi_cpu, phi_gpu))

@pytest.mark.skipif(skip, reason="Could not load dadi CUDA")
def test_4d_integration_allotetraploid():
    pts = 10

    # need nu1=nu2, nu3=nu4, m12=m21, m34=m43, and only two selection dictionaries
    nu1 = nu2 = lambda t: 0.5 + 5*t
    nu3 = nu4 = lambda t: 10-20*t

    e12 = e21 = 2.0
    e34 = e43 = lambda t: 0.5+3*t

    m13, m14 = 0.1, 3.2
    m23, m24 = lambda t: 0.5+3*t, 1.2
    m31, m32= 0.9, 1.7
    m41, m42 = 0.3, 1.9

    s_dict1 = s_dict2 = {'gamma': lambda t: -2*t}
    s_dict3 = s_dict4 = {'gamma': 3.0}

    theta0 = lambda t: 1 + 2*t
    f1,f2,f3,f4 = False, False, False, False
    T = 0.1

    ALLOa = PolyInt.PloidyType.ALLOa
    ALLOb = PolyInt.PloidyType.ALLOb

    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
    phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
    
    dadi.cuda_enabled(True)
    phi_gpu = PolyInt.four_pops(phi.copy(), xx, T=T, nu1=nu1, nu2=nu2, nu3=nu3, nu4=nu4,
                                m12=e12, m13=m13, m14=m14, m21=e21, m23=m23, m24=m24,
                                m31=m31, m32=m32, m34=e34, m41=m41, m42=m42, m43=e43,
                                sel_dict1=s_dict1, sel_dict2=s_dict2, sel_dict3=s_dict3, sel_dict4=s_dict4,
                                ploidyflag1=ALLOa, ploidyflag2=ALLOb, ploidyflag3=ALLOa, ploidyflag4=ALLOa,
                                theta0=theta0, frozen1=f1, frozen2=f2, frozen3=f3, frozen4=f4)
    dadi.cuda_enabled(False)
    phi_cpu = PolyInt.four_pops(phi.copy(), xx, T=T, nu1=nu1, nu2=nu2, nu3=nu3, nu4=nu4,
                                m12=e12, m13=m13, m14=m14, m21=e21, m23=m23, m24=m24,
                                m31=m31, m32=m32, m34=e34, m41=m41, m42=m42, m43=e43,
                                sel_dict1=s_dict1, sel_dict2=s_dict2, sel_dict3=s_dict3, sel_dict4=s_dict4,
                                ploidyflag1=ALLOa, ploidyflag2=ALLOb, ploidyflag3=ALLOa, ploidyflag4=ALLOa,
                                theta0=theta0, frozen1=f1, frozen2=f2, frozen3=f3, frozen4=f4)

    assert(np.allclose(phi_cpu, phi_gpu))

    e12, m13, m14, e21, m23, m24, m31, m32, e34, m41, m42, e43 = [0]*12
    for f1 in [True, False]:
        for f2 in [True, False]:
            for f3 in [True, False]:
                for f4 in [True, False]:
                    dadi.cuda_enabled(True)
                    phi_gpu = PolyInt.four_pops(phi.copy(), xx, T=T, nu1=nu1, nu2=nu2, nu3=nu3, nu4=nu4,
                                                m12=e12, m13=m13, m14=m14, m21=e21, m23=m23, m24=m24,
                                                m31=m31, m32=m32, m34=e34, m41=m41, m42=m42, m43=e43,
                                                sel_dict1=s_dict1, sel_dict2=s_dict2, sel_dict3=s_dict3, sel_dict4=s_dict4,
                                                ploidyflag1=ALLOa, ploidyflag2=ALLOb, ploidyflag3=ALLOa, ploidyflag4=ALLOa,
                                                theta0=theta0, frozen1=f1, frozen2=f2, frozen3=f3, frozen4=f4)
                    dadi.cuda_enabled(False)
                    phi_cpu = PolyInt.four_pops(phi.copy(), xx, T=T, nu1=nu1, nu2=nu2, nu3=nu3, nu4=nu4,
                                                m12=e12, m13=m13, m14=m14, m21=e21, m23=m23, m24=m24,
                                                m31=m31, m32=m32, m34=e34, m41=m41, m42=m42, m43=e43,
                                                sel_dict1=s_dict1, sel_dict2=s_dict2, sel_dict3=s_dict3, sel_dict4=s_dict4,
                                                ploidyflag1=ALLOa, ploidyflag2=ALLOb, ploidyflag3=ALLOa, ploidyflag4=ALLOa,
                                                theta0=theta0, frozen1=f1, frozen2=f2, frozen3=f3, frozen4=f4)
                    assert(np.allclose(phi_cpu, phi_gpu))

@pytest.mark.skipif(skip, reason="Could not load dadi CUDA")
def test_5d_integration_autohexaploid():
    AUTOHEX = PolyInt.PloidyType.AUTOHEX
    
    kwargs = {'T': 0.1,
              'nu1': 0.2, 'nu2': 1.3, 'nu3': 7.1, 'nu4': 27.1, 'nu5':lambda t: 0.9-8*t,
              'm12': 3, 'm13': 2.9, 'm14': 0.9, 'm15': 10,
              'm21': 3, 'm23': lambda t: 0.9+10*t, 'm24': 0.9, 'm25': lambda t: 2-15*t,
              'm31': 3.5, 'm32': 2.9, 'm34': 0.9, 'm35': 10,
              'm41': 3.3, 'm42': 2.2, 'm43': 0.1, 'm45': 9,
              'm51': 3.3, 'm52': 2.2, 'm53': 0.8, 'm54': 9.2,
              'sel_dict1': {'gamma': -1}, 
              'sel_dict2': {'gamma': 2},
              'sel_dict3': {'gamma': -1.9},
              'sel_dict4': {'gamma': lambda t: 10*t},
              'sel_dict5': {'gamma': -1},
              'ploidyflag1': AUTOHEX, 'ploidyflag2': AUTOHEX, 'ploidyflag3': AUTOHEX, 'ploidyflag4': AUTOHEX, 'ploidyflag5': AUTOHEX,
              'frozen1':False, 'frozen2':False, 'frozen3':False, 'frozen4':False, 'frozen5':False,
              'theta0': lambda t: 10*t}

    pts = 5
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx,xx,xx)
    phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx,xx,xx,xx)
    phi = dadi.PhiManip.phi_4D_to_5D(phi, 0,0,0, xx,xx,xx,xx,xx)
    dadi.cuda_enabled(True)
    phi_gpu = PolyInt.five_pops(phi.copy(), xx, **kwargs)
    dadi.cuda_enabled(False)
    phi_cpu = PolyInt.five_pops(phi.copy(), xx, **kwargs)

    assert(np.allclose(phi_cpu, phi_gpu))

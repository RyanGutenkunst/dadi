import numpy as np
import dadi
import dadi.Polyploidy.Integration as PolyInt

# set of tests comparing polyploidy integration to rescaled original dadi integrations

def test_2D_integration_diploids():
    """
    Integration test for diploids by comparison to original dadi code.
    """
    pts = 40
    T = 0.1
    nu1 = 1.5
    nu1_f = lambda t: np.exp(np.log(nu1)*t/T)
    nu2 = .5
    nu2_f = lambda t: np.exp(np.log(nu2)*t/T)
    m12 = lambda t: 2-t
    m21 = lambda t: 0.5+3*t
    gamma1 = lambda t: -2*t
    gamma2 = lambda t: 3*t
    h1 = lambda t: 0.2+t
    h2 = lambda t: 0.9-t
    # make dictionaries of selection coefficients for polyploidy integration
    sel1 = {'gamma': gamma1, 'h': h1}
    sel2 = {'gamma': gamma2, 'h': h2}

    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # integrate using the polyploidy module
    phi = PolyInt.two_pops(phi, xx, T=T, sel_dict1=sel1, sel_dict2=sel2, 
                                nu1=nu1_f, nu2=nu2_f, m12=m12, m21=m21, theta0=1)
    fs_poly = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # integrate using dadi
    phi = dadi.Integration.two_pops(phi, xx, T=T, gamma1=gamma1, gamma2=gamma2, h1=h1, h2=h2, 
                                    m12=m12, m21=m21, nu1=nu1_f, nu2=nu2_f, theta0=1)
    fs_dadi = dadi.Spectrum.from_phi(phi, [5,5], (xx,xx))
    
    assert(np.allclose(fs_poly, fs_dadi))

def test_3D_integration_autotetraploids():
    """
    Integration test for autotetraploids by comparison to original dadi code.
    """
    pts = 30
    T = 0.1
    nu1 = 1.5
    nu1_f = lambda t: np.exp(np.log(nu1)*t/T)
    nu2 = 1.25
    nu2_f = lambda t: np.exp(np.log(nu2)*t/T)
    nu3 = 2
    nu3_f = lambda t: np.exp(np.log(nu3)*t/T)
    m12 = 1
    m31 = 0.5
    # to test with selection, we need gammas to be constant
    gamma1 = -2
    gamma2 = 3
    gamma3 = 1

    # to properly test that the autotetraploid diffusion reduces to the diploid diffusion, 
    # we have to rescale time... that also means rescaling the nu and m functions
    # since the selection functions are already halved, we don't need to rescale them
    nu1_fa = lambda t: np.exp(np.log(nu1)*t/(2*T))
    nu2_fa = lambda t: np.exp(np.log(nu2)*t/(2*T))
    nu3_fa = lambda t: np.exp(np.log(nu3)*t/(2*T))
    # make dictionaries of selection coefficients for polyploidy integration
    # note: these will assume additive dominance
    sel1 = {'gamma': gamma1}
    sel2 = {'gamma': gamma2}
    sel3 = {'gamma': gamma3}

    auto_flag = PolyInt.PloidyType.AUTO

    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx, xx, xx)
    # integrate using the polyploidy module
    # note: theta0=0 here since mass accumulates differently for diploids and autotetraploids
    # also note, T = 2*T for autotetraploids and mij/2  
    phi = PolyInt.three_pops(phi, xx, T=2*T, ploidyflag1=auto_flag, ploidyflag2=auto_flag, ploidyflag3=auto_flag,
                             sel_dict1=sel1, sel_dict2=sel2, sel_dict3=sel3,
                             nu1=nu1_fa, nu2=nu2_fa, nu3=nu3_fa, m12=m12/2, m31=m31/2, theta0=0)
    fs_poly = dadi.Spectrum.from_phi(phi, [5,5,5], (xx,xx,xx))

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx, xx, xx)
    # integrate using dadi
    phi = dadi.Integration.three_pops(phi, xx, T=T, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, 
                                    m12=m12, m31=m31, nu1=nu1_f, nu2=nu2_f, nu3=nu3_f, theta0=0)
    fs_dadi = dadi.Spectrum.from_phi(phi, [5,5,5], (xx,xx,xx))

    assert(np.allclose(fs_poly, fs_dadi))
    
def test_3D_integration_alloallohexaploids():
    """
    Integration test alloallohexaploids by comparison to original dadi code.
    """
    pts = 20
    T = 0.1
    # here, nu = nu1=nu2=nu3 since this jointly specifies the population size for one alloallohexaploid population
    nu = 1.5
    nu_f = lambda t: np.exp(np.log(nu)*t/T)
    # here, mij must = mji since these two migration rates jointly specify a single exchange parameter
    m12 = m21 = lambda t: 1-t
    m31 = m13 = lambda t: 0.5+3*t
    m23 = m32 = lambda t: 0.5 - 0.1*t
    gamma1 = gamma2 = gamma3 = 0

    # make dictionaries of selection coefficients for polyploidy integration
    # note: these will assume additive dominance
    sel1 = {'gamma': gamma1}
    sel2 = {'gamma': gamma2}
    sel3 = {'gamma': gamma3}

    hex_a_flag = PolyInt.PloidyType.HEXa
    hex_b_flag = PolyInt.PloidyType.HEXb
    hex_c_flag = PolyInt.PloidyType.HEXc

    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx, xx, xx)
    # integrate using the polyploidy module
    phi = PolyInt.three_pops(phi, xx, T=T, ploidyflag1=hex_a_flag, ploidyflag2=hex_b_flag, ploidyflag3=hex_c_flag,
                             sel_dict1=sel1, sel_dict2=sel2, sel_dict3=sel3,
                             nu1=nu_f, nu2=nu_f, nu3=nu_f, 
                             m12=m12, m13=m13, m21=m21, m23=m23, m31=m31, m32=m32, theta0=1)
    fs_poly = dadi.Spectrum.from_phi(phi, [5,5,5], (xx,xx,xx))

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx, xx, xx)
    # integrate using dadi
    phi = dadi.Integration.three_pops(phi, xx, T=T, gamma1=gamma1, gamma2=gamma2, gamma3 = gamma3, 
                                      m12=m12, m13=m13, m21=m21, m23=m23, m31=m31, m32=m32,
                                      nu1=nu_f, nu2=nu_f, nu3=nu_f, theta0=1)
    fs_dadi = dadi.Spectrum.from_phi(phi, [5,5,5], (xx,xx,xx))
    
    assert(np.allclose(fs_poly, fs_dadi))

def test_4D_integration_allotetraploids():
    """
    Integration test for allotetraploids by comparison to original dadi code.
    """
    pts = 8
    T = 0.1
    # here, nu1=nu2 since these jointly specify the first allotetraploid population
    nu1 = 1.5
    nu1_f = nu2_f = lambda t: np.exp(np.log(nu1)*t/T)
    # similarly, nu3=nu4 since these jointly specify the second allotetraploid population
    nu3 = .5
    nu3_f = nu4_f = lambda t: np.exp(np.log(nu3)*t/T)
    # here, m12 = m21 and m34 = m43, as above
    m12 = m21 = lambda t: 2-t
    m34 = m43 = lambda t: 0.5+1.5*t
    m31 = lambda t: 0.5-.3*t
    m24 = lambda t: 0.5-.3*t
    gamma1 = gamma2 = gamma3 = gamma4 = 0

    # make dictionaries of selection coefficients for polyploidy integration
    # note: these will assume additive dominance
    sel1 = {'gamma': gamma1}
    sel2 = {'gamma': gamma2}
    sel3 = {'gamma': gamma3}
    sel4 = {'gamma': gamma4}

    alloa_flag = PolyInt.PloidyType.ALLOa
    allob_flag = PolyInt.PloidyType.ALLOb

    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx, xx, xx)
    phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx, xx, xx, xx)
    # integrate using the polyploidy module
    phi = PolyInt.four_pops(phi, xx, T=T, ploidyflag1=alloa_flag, ploidyflag2=allob_flag, ploidyflag3=alloa_flag, ploidyflag4=allob_flag,   
                             sel_dict1=sel1, sel_dict2=sel2, sel_dict3=sel3, sel_dict4=sel4,
                             nu1=nu1_f, nu2=nu2_f, nu3=nu3_f, nu4=nu4_f,
                             m12=m12, m21=m21, m24=m24, m31=m31, m34=m34, m43=m43, theta0=1)
    fs_poly = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx, xx, xx)
    phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx, xx, xx, xx)
    # integrate using dadi
    phi = dadi.Integration.four_pops(phi, xx, T=T, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, gamma4=gamma4, 
                                      nu1=nu1_f, nu2=nu2_f, nu3=nu3_f, nu4=nu4_f,
                                      m12=m12, m21=m21, m24=m24, m31=m31, m34=m34, m43=m43, theta0=1)
    fs_dadi = dadi.Spectrum.from_phi(phi, [5,5,5,5], (xx,xx,xx,xx))
    
    assert(np.allclose(fs_poly, fs_dadi))

def test_5D_integration_autohexaploids():
    """
    Integration test for autohexaploids by comparison to original dadi code.
    """
    pts = 5
    T = 0.05
    nu1 = 1.5
    nu1_f = lambda t: np.exp(np.log(nu1)*t/T)
    nu2 = 1.25
    nu2_f = lambda t: np.exp(np.log(nu2)*t/T)
    nu3 = 2
    nu3_f = lambda t: np.exp(np.log(nu3)*t/T)
    nu4 = .75
    nu4_f = lambda t: np.exp(np.log(nu4)*t/T)
    m12 = 1
    m31 = 0.5
    m54 = 1.5
    # to test with selection, we need gammas to be constant
    gamma1 = -2
    gamma2 = 3
    gamma3 = 1
    gamma4 = -4
    gamma5 = -1
    # to properly test that the autohexaploid diffusion reduces to the diploid diffusion, 
    # we have to rescale time... that also means rescaling the nu and m functions
    # since the selection functions are already weaker, we don't need to rescale them
    nu1_fa = lambda t: np.exp(np.log(nu1)*t/(3*T))
    nu2_fa = lambda t: np.exp(np.log(nu2)*t/(3*T))
    nu3_fa = lambda t: np.exp(np.log(nu3)*t/(3*T))
    nu4_fa = lambda t: np.exp(np.log(nu4)*t/(3*T))
    # make dictionaries of selection coefficients for polyploidy integration
    # note: these will assume additive dominance
    sel1 = {'gamma': gamma1}
    sel2 = {'gamma': gamma2}
    sel3 = {'gamma': gamma3}
    sel4 = {'gamma': gamma4}
    sel5 = {'gamma': gamma5}

    autohex_flag = PolyInt.PloidyType.AUTOHEX

    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx, xx, xx)
    phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx, xx, xx, xx)
    phi = dadi.PhiManip.phi_4D_to_5D(phi, 0, 0, 0, xx, xx, xx, xx, xx)
    # integrate using the polyploidy module
    # note: theta0=0 here since mass accumulates differently for diploids and autohexaploids
    # also note, T = 3*T for autohexaploids
    phi = PolyInt.five_pops(phi, xx, T=3*T, ploidyflag1=autohex_flag, ploidyflag2=autohex_flag, ploidyflag3=autohex_flag, ploidyflag4=autohex_flag, ploidyflag5=autohex_flag,
                             sel_dict1=sel1, sel_dict2=sel2, sel_dict3=sel3, sel_dict4=sel4, sel_dict5=sel5,
                             nu1=nu1_fa, nu2=nu2_fa, nu3=nu3_fa, nu4=nu4_fa, m12=m12/3, m31=m31/3, m54=m54/3, theta0=0)
    fs_poly = dadi.Spectrum.from_phi(phi, [5,5,5,5,5], (xx,xx,xx,xx,xx))

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 0, xx, xx, xx)
    phi = dadi.PhiManip.phi_3D_to_4D(phi, 0, 0, xx, xx, xx, xx)
    phi = dadi.PhiManip.phi_4D_to_5D(phi, 0, 0, 0, xx, xx, xx, xx, xx)
    # integrate using dadi
    phi = dadi.Integration.five_pops(phi, xx, T=T, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, gamma4=gamma4, gamma5=gamma5,
                                    m12=m12, m31=m31, m54=m54, nu1=nu1_f, nu2=nu2_f, nu3=nu3_f, nu4=nu4_f, theta0=0)
    fs_dadi = dadi.Spectrum.from_phi(phi, [5,5,5,5,5], (xx,xx,xx,xx,xx))

    assert(np.allclose(fs_poly, fs_dadi))
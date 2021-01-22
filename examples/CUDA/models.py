import math, time
import dadi

def timeit(func):
    def newfunc(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        return out, end-start
    return newfunc

@dadi.Numerics.make_extrap_func
def OutOfAfricaArchaicAdmixture_5R19(n, pts):
    """
    Integration of model from Ragsdale (2019) PLoS Genetics.
    """
    # Output
    # Pop 1: African
    # Pop 2: CEU
    # Pop 3: CHB

    # Parameter values from stdpopsim GitHub
    # From here ***
    # First we set out the maximum likelihood values of the various parameters
    # given in Table 1 (under archaic admixture).
    N_0 = 3600
    N_YRI = 13900
    N_B = 880
    N_CEU0 = 2300
    N_CHB0 = 650

    # Times are provided in years, so we convert into generations.
    # In the published model, the authors used a generation time of 29 years to
    # convert from genetic to physical units
    generation_time = 29

    T_AF = 300e3 / generation_time
    T_B = 60.7e3 / generation_time
    T_EU_AS = 36.0e3 / generation_time
    T_arch_afr_split = 499e3 / generation_time
    T_arch_afr_mig = 125e3 / generation_time
    T_nean_split = 559e3 / generation_time
    T_arch_adm_end = 18.7e3 / generation_time

    # We need to work out the starting (diploid) population sizes based on
    # the growth rates provided for these two populations
    r_CEU = 0.00125
    r_CHB = 0.00372
    N_CEU = N_CEU0 / math.exp(-r_CEU * T_EU_AS)
    N_CHB = N_CHB0 / math.exp(-r_CHB * T_EU_AS)

    # Migration rates during the various epochs.
    m_AF_B = 52.2e-5
    m_YRI_CEU = 2.48e-5
    m_YRI_CHB = 0e-5
    m_CEU_CHB = 11.3e-5
    m_AF_arch_af = 1.98e-5
    m_OOA_nean = 0.825e-5
    # To here ***

    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx)
    # Split off Neanderthal pop
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, (T_nean_split-T_arch_afr_split)/(2*N_0))
    # Split off archaic African pop
    phi = dadi.PhiManip.phi_2D_to_3D(phi, 1, xx,xx,xx)
    phi = dadi.Integration.three_pops(phi, xx, (T_arch_afr_split-T_AF)/(2*N_0))
    # African population growth
    phi = dadi.Integration.three_pops(phi, xx, (T_AF-T_arch_afr_mig)/(2*N_0), nu1=N_YRI/N_0)
    # Archaic African migration begins
    phi = dadi.Integration.three_pops(phi, xx, (T_arch_afr_mig-T_B)/(2*N_0), nu1=N_YRI/N_0,
                                      m13=m_AF_arch_af*2*N_0, m31=m_AF_arch_af*2*N_0)
    # Split of Eurasian ancestral pop
    phi = dadi.PhiManip.phi_3D_to_4D(phi, 1,0, xx,xx,xx,xx)
    phi = dadi.Integration.four_pops(phi, xx, (T_B-T_EU_AS)/(2*N_0), 
                                     nu1=N_YRI/N_0, nu4=N_B/N_0,
                                     m13=m_AF_arch_af*2*N_0, m31=m_AF_arch_af*2*N_0,
                                     m14=m_AF_B*2*N_0, m41=m_AF_B*2*N_0,
                                     m24=m_OOA_nean*2*N_0, m42=m_OOA_nean*2*N_0)
    # Split of European and Asian ancestral pops
    phi = dadi.PhiManip.phi_4D_to_5D(phi, 0,0,0, xx,xx,xx,xx,xx)
    nuEu_func = lambda t: N_CEU0/N_0*(N_CEU/N_CEU0)**(t * 2*N_0/(T_EU_AS))
    nuAs_func = lambda t: N_CHB0/N_0*(N_CHB/N_CHB0)**(t * 2*N_0/(T_EU_AS))
    phi = dadi.Integration.five_pops(phi, xx, (T_EU_AS-T_arch_adm_end)/(2*N_0), 
                                     nu1=N_YRI/N_0, nu4=nuEu_func, nu5=nuAs_func,
                                     m13=m_AF_arch_af*2*N_0, m31=m_AF_arch_af*2*N_0,
                                     m14=m_YRI_CEU*2*N_0, m41=m_YRI_CEU*2*N_0,
                                     m15=m_YRI_CHB*2*N_0, m51=m_YRI_CHB*2*N_0,
                                     m24=m_OOA_nean*2*N_0, m42=m_OOA_nean*2*N_0,
                                     m25=m_OOA_nean*2*N_0, m52=m_OOA_nean*2*N_0,
                                     m45=m_CEU_CHB*2*N_0, m54=m_CEU_CHB*2*N_0,
                                     )
    # End of archaic migration. Remove archaic pops for efficiency.
    phi = dadi.PhiManip.filter_pops(phi, xx, [1,4,5])
    # We use initial_t argument so we can reuse nuEu_func and nuAs_func
    phi = dadi.Integration.three_pops(phi, xx, T_arch_adm_end/(2*N_0), initial_t=(T_EU_AS-T_arch_adm_end)/(2*N_0), 
                                     nu1=N_YRI/N_0, nu2=nuEu_func, nu3=nuAs_func,
                                     m12=m_YRI_CEU*2*N_0, m21=m_YRI_CEU*2*N_0,
                                     m13=m_YRI_CHB*2*N_0, m31=m_YRI_CHB*2*N_0,
                                     m23=m_CEU_CHB*2*N_0, m32=m_CEU_CHB*2*N_0,
                                     )

    return dadi.Spectrum.from_phi(phi, (n,n,n), (xx,xx,xx))
OutOfAfricaArchaicAdmixture_5R19_timed = timeit(OutOfAfricaArchaicAdmixture_5R19)

@dadi.Numerics.make_extrap_func
def NewWorld_4G09(n, pts, variant='original'):
    """
    Integration of New World model from Gutenkunst (2009) PLoS Genetics.

    Note that in the original publication, the African population was removed late
    in the simulation, to keep the total number of populations to three. Now
    we can run with that fourth population.

    If variant=='no_admixture', then the final admixture event is ignored,
    for comparison with moments.
    """
    # From Table S3
    nuAf, nuB = 1.68, 0.287
    mAfB, mAfEu, mAfAs = 3.65, 0.44, 0.28
    TAf, TB = 0.607, 0.396
    # From Table S6
    nuEu0, nuEu = 0.208, 2.42 
    nuAs0, nuAs = 0.081, 4.186 
    mEuAs = 1.98
    TEuAs = 0.072
    nuMx0, nuMx = 0.103, 7.94
    TMx = 0.059
    fEuMx = 0.48

    xx = dadi.Numerics.default_grid(pts)

    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, TAf-TB-TEuAs-TMx, nu=nuAf)

    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, TB-TEuAs-TMx, nu1=nuAf, nu2=nuB, 
                               m12=mAfB, m21=mAfB)

    phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)
    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/(TEuAs+TMx))
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/(TEuAs+TMx))
    phi = dadi.Integration.three_pops(phi, xx, TEuAs, 
                                 nu1=nuAf,
                                 nu2=nuEu_func, nu3=nuAs_func, 
                                 m12=mAfEu, m21=mAfEu, m13=mAfAs, m31=mAfAs,
                                 m23=mEuAs, m32=mEuAs)

    # Split off New World population, with contribution entirely from
    # population 3 (East Asia)
    phi = dadi.PhiManip.phi_3D_to_4D(phi, 0,0, xx,xx,xx,xx)

    nuEu0 = nuEu_func(TEuAs)
    nuAs0 = nuAs_func(TEuAs)
    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TMx)
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TMx)
    nuMx_func = lambda t: nuMx0*(nuMx/nuMx0)**(t/TMx)
    phi = dadi.Integration.four_pops(phi, xx, TMx, 
                                nu1=nuAf, nu2=nuEu_func, 
                                nu3=nuAs_func, nu4=nuMx_func,
                                m12=mAfEu, m21=mAfEu, m13=mAfAs, m31=mAfAs,
                                m23=mEuAs, m32=mEuAs)
    if variant != 'no_admixture':
        phi = dadi.PhiManip.phi_4D_admix_into_4(phi, 0,fEuMx,0, xx,xx,xx,xx)

    return dadi.Spectrum.from_phi(phi, (n,n,n,n), (xx,xx,xx,xx))
NewWorld_4G09_timed = timeit(NewWorld_4G09)

@dadi.Numerics.make_extrap_func
def OutOfAfrica_3G09(n, pts):
    """
    Integration of Out-of-Africa model from Gutenkunst (2009) PLoS Genetics.
    """
    # Parameter values from https://github.com/popsim-consortium/stdpopsim/blob/master/stdpopsim/catalog/homo_sapiens.py
    generation_time = 25
    # First we set out the maximum likelihood values of the various parameters
    # given in Table 1.
    N_A = 7300
    N_B = 2100
    N_AF = 12300
    N_EU0 = 1000
    N_AS0 = 510
    # Times are provided in years, so we convert into generations.
    T_AF = 220e3 / generation_time
    T_B = 140e3 / generation_time
    T_EU_AS = 21.2e3 / generation_time
    # We need to work out the starting (diploid) population sizes based on
    # the growth rates provided for these two populations
    r_EU = 0.004
    r_AS = 0.0055
    # Migration rates during the various epochs.
    m_AF_B = 25e-5
    m_AF_EU = 3e-5
    m_AF_AS = 1.9e-5
    m_EU_AS = 9.6e-5

    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, (T_AF-T_B)/(2*N_A), nu=N_AF/N_A)
    
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, (T_B-T_EU_AS)/(2*N_A), nu1=N_AF/N_A, nu2=N_B/N_A, 
                                    m12=m_AF_B * 2*N_A, m21=m_AF_B * 2*N_A)
    
    phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)
    
    enable_cuda_cached = False
    def nuEu_func(t): return N_EU0/N_A * math.exp(r_EU * t*2*N_A)
    def nuAs_func(t): return N_AS0/N_A * math.exp(r_AS * t*2*N_A)

    phi = dadi.Integration.three_pops(phi, xx, T_EU_AS/(2*N_A), nu1=N_AF/N_A, nu2=nuEu_func, nu3=nuAs_func,
                                       m12=m_AF_EU *2*N_A, m13=m_AF_AS *2*N_A, m21=m_AF_EU *2*N_A,
                                       m23=m_EU_AS *2*N_A, m31=m_AF_AS *2*N_A, m32=m_EU_AS *2*N_A,
                                       enable_cuda_cached=enable_cuda_cached)

    return dadi.Spectrum.from_phi(phi, (n,n,n), (xx,xx,xx))
OutOfAfrica_3G09_timed = timeit(OutOfAfrica_3G09)

@dadi.Numerics.make_extrap_func
def OutOfAfrica_2L06(n, pts):
    """
    Integration of Out-of-Africa model from Li (2006) PLoS Genetics.
    """
    # Parameter values from https://github.com/popsim-consortium/stdpopsim/blob/master/stdpopsim/catalog/drosophila_melanogaster.py

    # African Parameter values from "Demographic History of the African
    # Population" section
    N_A0 = 8.603e06
    t_A0 = 600000  # assuming 10 generations / year
    N_A1 = N_A0 / 5.0 # This is the ancestral size

    # European Parameter values from "Demo History of Euro Population"
    N_E0 = 1.075e06
    N_E1 = 2200
    t_AE = 158000  # generations
    t_E1 = t_AE - 3400

    # Ancestral size is N_A1

    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, (t_A0 - t_AE)/(2*N_A1), nu=N_A0/N_A1)

    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)

    enable_cuda_cached = False

    phi = dadi.Integration.two_pops(phi, xx, (t_AE - t_E1)/(2*N_A1), nu1=N_A0/N_A1, nu2=N_E1/N_A1,
            enable_cuda_cached=enable_cuda_cached)
    phi = dadi.Integration.two_pops(phi, xx, t_E1/(2*N_A1), nu1=N_A0/N_A1, nu2=N_E0/N_A1,
            enable_cuda_cached=enable_cuda_cached)

    return dadi.Spectrum.from_phi(phi, (n,n), (xx,xx))
OutOfAfrica_2L06_timed = timeit(OutOfAfrica_2L06)

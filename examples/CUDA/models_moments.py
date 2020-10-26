import moments, numpy
from models import timeit

def OutOfAfrica_2L06_moments(n):
    """
    Integration of Out-of-Africa model from Li (2006) PLoS Genetics.

    Potential variants:
    'original': Model from the original publication
    'cached_int': Forcing CUDA code to use cached matrices
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

    n1,n2 = n,n

    sts = moments.LinearSystem_1D.steady_state_1D(n1+n2)
    fs = moments.Spectrum(sts)

    fs.integrate([N_A0/N_A1], (t_A0 - t_AE)/(2*N_A1), 0.05)

    fs = moments.Manips.split_1D_to_2D(fs, n1, n2)

    fs.integrate([N_A0/N_A1, N_E1/N_A1], (t_AE - t_E1)/(2*N_A1), 0.05)
    fs.integrate([N_A0/N_A1, N_E0/N_A1], t_E1/(2*N_A1), 0.05)

    return fs
OutOfAfrica_2L06_moments_timed = timeit(OutOfAfrica_2L06_moments)

def OutOfAfrica_3G09_moments(n):
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

    n1,n2,n3 = n,n,n

    sts = moments.LinearSystem_1D.steady_state_1D(n1+n2+n3)
    fs = moments.Spectrum(sts)

    nuAf = N_AF/N_A
    TAf = (T_AF-T_B)/(2*N_A)
    fs.integrate([nuAf], TAf, 0.05)

    fs = moments.Manips.split_1D_to_2D(fs, n1, n2+n3)

    mAfB = m_AF_B * 2*N_A
    TB = (T_B-T_EU_AS)/(2*N_A)
    nuB = N_B/N_A
    mig1 = numpy.array([[0, mAfB],[mAfB, 0]])
    fs.integrate([nuAf, nuB], TB, 0.05, m=mig1)

    fs = moments.Manips.split_2D_to_3D_2(fs, n2, n3)

    nuEu0, nuAs0 = N_EU0/N_A , N_AS0/N_A
    TEuAs = T_EU_AS/(2*N_A)
    nuEu = N_EU0/N_A * numpy.exp(r_EU * T_EU_AS)
    nuAs = N_AS0/N_A * numpy.exp(r_AS * T_EU_AS)
    mAfEu, mAfAs, mEuAs = m_AF_EU * 2*N_A, m_AF_AS * 2*N_A, m_EU_AS * 2*N_A

    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TEuAs) 
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TEuAs)
    nu2 = lambda t: [nuAf, nuEu_func(t), nuAs_func(t)]
    mig2 = numpy.array([[0, mAfEu, mAfAs], [mAfEu, 0, mEuAs], [mAfAs, mEuAs, 0]]) 
    fs.integrate(nu2, TEuAs, 0.05, m=mig2)

    return fs
OutOfAfrica_3G09_moments_timed = timeit(OutOfAfrica_3G09_moments)

def NewWorld_4G09_noadmix_moments(n):
    """
    Integration of New World model from Gutenkunst (2009) PLoS Genetics.

    Note that in the original publication, the African population was removed late
    in the simulation, to keep the total number of populations to three. Now
    we can run with that fourth population.
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

    n1,n2,n3,n4 = n,n,n,n

    sts = moments.LinearSystem_1D.steady_state_1D(n1+n2+n3+n4)
    fs = moments.Spectrum(sts)

    fs.integrate([nuAf], TAf-TB-TEuAs-TMx, 0.05)

    fs = moments.Manips.split_1D_to_2D(fs, n1, n2+n3+n4)
    mig1 = numpy.array([[0, mAfB],[mAfB, 0]])
    fs.integrate([nuAf, nuB], TB-TEuAs-TMx, 0.05, m=mig1)

    fs = moments.Manips.split_2D_to_3D_2(fs, n2, n3+n4)

    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/(TEuAs+TMx))
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/(TEuAs+TMx))
    nu2 = lambda t: [nuAf, nuEu_func(t), nuAs_func(t)]
    mig2 = numpy.array([[0, mAfEu, mAfAs], [mAfEu, 0, mEuAs], [mAfAs, mEuAs, 0]]) 
    fs.integrate(nu2, TEuAs, 0.05, m=mig2)

    # Split off New World population, with contribution entirely from
    # population 3 (East Asia)
    fs = moments.Manips.split_3D_to_4D_3(fs, n3, n4)

    nuEu0 = nuEu_func(TEuAs)
    nuAs0 = nuAs_func(TEuAs)
    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TMx)
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TMx)
    nuMx_func = lambda t: nuMx0*(nuMx/nuMx0)**(t/TMx)
    nu3 = lambda t: [nuAf, nuEu_func(t), nuAs_func(t), nuMx_func(t)]
    mig3 = numpy.array([[0, mAfEu, mAfAs,0], [mAfEu, 0, mEuAs,0], [mAfAs, mEuAs, 0, 0], [0, 0, 0, 0]]) 
    fs.integrate(nu3, TMx, 0.05, m=mig3)
    # Admixture skipped

    return fs
NewWorld_4G09_noadmix_moments_timed = timeit(NewWorld_4G09_noadmix_moments)

def OutOfAfricaArchaicAdmixture_5R19_moments(n):
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
    N_CEU = N_CEU0 / numpy.exp(-r_CEU * T_EU_AS)
    N_CHB = N_CHB0 / numpy.exp(-r_CHB * T_EU_AS)

    # Migration rates during the various epochs.
    m_AF_B = 52.2e-5
    m_YRI_CEU = 2.48e-5
    m_YRI_CHB = 0e-5
    m_CEU_CHB = 11.3e-5
    m_AF_arch_af = 1.98e-5
    m_OOA_nean = 0.825e-5
    # To here ***

    n1 = n2 = n3 = n4 = n5 = n

    sts = moments.LinearSystem_1D.steady_state_1D(n1+n2+n3+n4+n5)
    fs = moments.Spectrum(sts)

    # Split off Neanderthal pop
    fs = moments.Manips.split_1D_to_2D(fs, n1+n3+n4+n5, n2)
    fs.integrate([1., 1.],  (T_nean_split-T_arch_afr_split)/(2*N_0))
    # Split off archaic African pop
    fs = moments.Manips.split_2D_to_3D_1(fs, n1+n4+n5, n3)
    fs.integrate([1., 1., 1.],  (T_arch_afr_split-T_AF)/(2*N_0))
    # African population growth
    fs.integrate([N_YRI/N_0, 1., 1.],  (T_AF-T_arch_afr_mig)/(2*N_0))
    # Archaic African migration begins
    mig_mat = [[0, 0, m_AF_arch_af*2*N_0],
               [0, 0, 0],
               [m_AF_arch_af*2*N_0, 0, 0]]
    fs.integrate([N_YRI/N_0, 1., 1.], (T_arch_afr_mig-T_B)/(2*N_0), m=mig_mat)
    # Split of Eurasian ancestral pop
    fs = moments.Manips.split_3D_to_4D_1(fs, n1, n4+n5)
    nus = [N_YRI/N_0, 1.0, 1.0, N_B/N_0]
    mig_mat = [[0, 0, m_AF_arch_af*2*N_0, m_AF_B*2*N_0],
               [0, 0, 0, m_OOA_nean*2*N_0],
               [m_AF_arch_af*2*N_0, 0, 0, 0],
               [m_AF_B*2*N_0, m_OOA_nean*2*N_0, 0, 0]]
    fs.integrate(nus, (T_B-T_EU_AS)/(2*N_0), m=mig_mat)
    # Split of European and Asian ancestral pops
    fs = moments.Manips.split_4D_to_5D_4(fs, n4, n5) # order: [YRI, Neand, AA, CEU, CHB]
    nuEu_func = lambda t: N_CEU0/N_0*(N_CEU/N_CEU0)**(t * 2*N_0/(T_EU_AS))
    nuAs_func = lambda t: N_CHB0/N_0*(N_CHB/N_CHB0)**(t * 2*N_0/(T_EU_AS))
    nu_func = lambda t: [N_YRI/N_0, 1.0, 1.0, nuEu_func(t), nuAs_func(t)]
    mig_mat = [[0, 0, m_AF_arch_af*2*N_0, m_YRI_CEU*2*N_0, m_YRI_CHB*2*N_0],
               [0, 0, 0, m_OOA_nean*2*N_0, m_OOA_nean*2*N_0],
               [m_AF_arch_af*2*N_0, 0, 0, 0, 0],
               [m_YRI_CEU*2*N_0, m_OOA_nean*2*N_0, 0, 0, m_CEU_CHB*2*N_0],
               [m_YRI_CHB*2*N_0, m_OOA_nean*2*N_0, 0, m_CEU_CHB*2*N_0, 0]]
    fs.integrate(nu_func, (T_EU_AS-T_arch_adm_end)/(2*N_0), m=mig_mat)
    # End of archaic migration. Remove archaic pops for efficiency.
    fs = fs.marginalize([1, 2])

    _, _, _, nuEu_temp, nuAs_temp = nu_func((T_EU_AS-T_arch_adm_end)/(2*N_0))
    nu_func = lambda t: [N_YRI/N_0,
                         nuEu_temp * (N_CEU0/N_0/nuEu_temp)**(t * 2*N_0/(T_arch_adm_end)),
                         nuAs_temp * (N_CHB0/N_0/nuAs_temp)**(t * 2*N_0/(T_arch_adm_end))]

    # We use initial_t argument so we can reuse nuEu_func and nuAs_func
    mig_mat = [[0, m_YRI_CEU*2*N_0, m_YRI_CHB*2*N_0],
               [m_YRI_CEU*2*N_0, 0, m_CEU_CHB*2*N_0],
               [m_YRI_CHB*2*N_0, m_CEU_CHB*2*N_0, 0]]
    fs.integrate(nu_func, T_arch_adm_end/(2*N_0), m=mig_mat)

    return fs
OutOfAfricaArchaicAdmixture_5R19_moments_timed = timeit(OutOfAfricaArchaicAdmixture_5R19_moments)

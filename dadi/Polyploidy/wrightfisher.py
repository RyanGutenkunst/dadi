import numpy as np
import matplotlib.pyplot as plt
from dadi.Misc import ensure_1arg_func 

# Shared functions
def LB_dominance_model(k, delta, dosage):
    """
    Computes dominance coefficient using a parameterized CDF 
        of the LB system of equations of Tadikamalla and Johsnon (1982)

    k: central tendency parameter specifying dosage at which the returned dominance coefficient is 0.5 
        (e.g. k = .25 means a genotype with dosage of .25 will have a dominance coefficient of .5)
    delta: shape paramter specifying steepness of the distribution 
    dosage: gene dosage for the dominance coefficient; typically equals # mutants / ploidy

    Note: Look at model from Booker and Schrider 2024/Huber et al. 2018 too

    Returns:
        dom_coeff: a dominance coefficient
    """
    d = dosage

    dom_coeff = ((d/(1-d))**delta) / (((d/(1-d))**delta) + ((k/(1-k))**delta))

    return dom_coeff

def mask(arr_in):
    """
    Masks/removes allele frequencies that are lost (0) or fixed (1).

    arr_in: 1D array of allele frequencies

    Returns:
        arr_out: 1D array of polymorphic allele frequencies 
    """
    mask0 = arr_in != 0
    arr_out = arr_in[mask0]
    mask1 = arr_out != 1
    arr_out = arr_out[mask1]

    return arr_out

def summarize_phi(phi, xx, model_name="No name specified"):
    """
    Compute mean and variance of final allele frequency from dadi. 
    Also, plots the phi distribution.

    phi: integrated phi from dadi
    xx: grid used for integration
    model_name: string to label the plot
    """
    # normalize phi to be a probability density, since phi can lose mass in the theta0=0 regime
    # This also helps with comparing to the results of the WF sims when using density = True in plt.hist()
    phi /= np.trapezoid(phi, xx)
    # Then, we compute the mean
    mean = np.trapezoid(phi*xx, xx)
    print(f"Mean: {mean}")
    # and the variance
    var = np.trapezoid(phi*xx**2, xx) - mean**2
    print(f"Variance: {var}")

    # plot phi
    plt.plot(xx, phi)
    plt.xlabel('Allele Frequency')
    plt.ylabel('Density')
    plt.title(model_name)
    plt.ylim(0, 1.1*max(phi))
    plt.show()

# Diploid model

def dip_selection(q, s, h):
    """
    Evaluates new value of q after one generation of selection for diploids.

    q: allele frequency before selection
    s: selection coefficient
    h: dominance coefficient for heterozygotes

    Returns:
        q_post_sel: allele frequency post selection
    """
    #delta_q = s * 2*(h + (1-2*h)*q) * q*(1-q)
    #return delta_q
    #the RHS expression was computed symbolically in Matlab
    q_post_sel = (q**2*(2*s + 1) - 
                 q*(2*h*s + 1)*(q - 1))/((q - 1)**2 + 
                 q**2*(2*s + 1) - 2*q*(2*h*s + 1)*(q - 1))

    return q_post_sel

def dip_allelic_WF(N, T, gamma, init_q, nu=1, h=0.5, replicates = 1, plot = False, track_all = False):
    """
    Simple Wright-Fisher model of genetic drift in diploids.

    N: ancestral population size (number of individuals)
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    gamma: population-scaled selection coefficient (<0 is purifying, >0 is positive)
        gamma = 2Ns 
        if A is the selected allele, aa has fitness 1, 
        Aa has fitness 1+2sh, AA has fitness 1+2s
    h: dominance coefficient for heterozygote
    init_q: vector of initial allele frequencies for selected allele; must have size = replicates
    nu: population size relative to ancestral population size
        nu can be a constant or a function of diffusion-scaled time
    replicates: number of times to run the simulation
    plot: Boolean input to either show a plot of individual trajectories or not
    track_all: Boolean input to either track and return all trajectories or not
        This usage and plot are not recommended for analysis, but are reasonable usages for debuggging
        They are very memory intensive options and also slow down the code up to 50%
        
    Returns:
        allele_freqs: array of allele frequencies, either just final 
            or if track_all = True, then over time
    """

    if np.less(N, 0) or np.less(T, 0):
        raise(ValueError("The population size or time is less than zero." 
                        " Has the model been misspecified?"))
    
    if np.any(np.less(init_q, 0)) or np.any(np.greater(init_q, 1)):
        raise(ValueError("At least one initial q_value is less than zero"
                         " or greater than one."))
    
    if len(init_q) != replicates:
        raise ValueError("Length of init_q must equal number of replicates.")
    
    nu_f = ensure_1arg_func(nu)

    # if we want to plot, we need to track all of the trajectories
    if plot:
        track_all = True

    # calculate s from gamma because we will need it for our generation based simulation
    s = gamma/(2*N)

    # create matrix to store allele frequencies
    if track_all:
        allele_freqs = np.empty((replicates, int(2*N*T+1)))
        allele_freqs[:, 0] = init_q
    else:
        allele_freqs = init_q

    s_vec = np.full(replicates, s)
    h_vec = np.full(replicates, h)

    rng = np.random.default_rng()

    total_gens = int(2*N*T)  
    for t in range(total_gens):
        nu = nu_f(t/(2*N)) # this rescales from generations back to diffusion time
        samples = int(2*N*nu) 
        if track_all:
            q_post_sel = dip_selection(allele_freqs[:, t], s_vec, h_vec)
            allele_freqs[:, t+1] = rng.binomial(samples, q_post_sel)/(samples)
        else:
            q_post_sel = dip_selection(allele_freqs, s_vec, h_vec)
            allele_freqs = rng.binomial(samples, q_post_sel)/(samples)

    if plot:
        plt.plot(allele_freqs.T, color='gray', alpha=0.025)
        plt.plot(np.mean(allele_freqs, axis=0), color='black', lw=2)
        plt.xlabel('Generation')
        plt.ylabel('Mutant Allele Frequency')
        plt.title('Autotetraploid Allelic Drift Simulation')
        plt.ylim(-.1, 1.1)
        plt.show()

    return allele_freqs

# Autotetraploid model based on allele sampling
def auto_selection(q, s1, s2, s3, s4):
    """
    Evaluates new value of q after one generation of selection for autotetraploids.

    q: allele frequency before selection
    s1: selective effect for G1 individual
    s2: selective effect for G2 individual
    s3: selective effect for G3 individual
    s4: selective effect for G4 individual

    Returns:
        q_post_sel: allele frequency post selection
    """
    # the RHS expression was computed symbolically in Matlab
    q_post_sel = (q*(2*s1 - 6*q*s1 + 6*q*s2 + 6*q**2*s1 - 12*q**2*s2 - 2*q**3*s1
                     + 6*q**2*s3 + 6*q**3*s2 - 6*q**3*s3 + 2*q**3*s4 + 1)
                     )/(8*q*s1 - 24*q**2*s1 + 12*q**2*s2 + 24*q**3*s1 - 24*q**3*s2 
                        - 8*q**4*s1 + 8*q**3*s3 + 12*q**4*s2 - 8*q**4*s3 + 2*q**4*s4 + 1)

    return q_post_sel

def auto_allelic_WF(N, T, init_q, gamma1, gamma2, gamma3, gamma4, nu = 1, replicates = 1, plot = False, track_all = False):
    """
    Simple Wright-Fisher model of genetic drift in autotetraploids based on allele frequency sampling. 

    N: population size (number of individuals)
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    init_q: vector of initial allele frequencies for selected allele; must have size = replicates
    gamma1: population-scaled selection coefficient for G1 individuals
        (<0 is purifying, >0 is positive)
        gamma1 = 2*N*s1 where a G1 individual has fitness 1 + 2*s1
    gamma2: population-scaled selection coefficient for G2 individuals
    gamma3: population-scaled selection coefficient for G3 individuals
    gamma4: population-scaled selection coefficient for G4 individuals
        This is the closest to "gamma" for diploids
    nu: population size relative to ancestral population size
        nu can be a constant or a function of diffusion-scaled time
    replicates: number of times to run the simulation
    plot: Boolean input to either show a plot of individual trajectories or not
    track_all: Boolean input to either track and return all trajectories or not
        This usage and plot are not recommended for analysis, but are reasonable usages for debuggging
        They are very memory intensive options and also slow down the code up to 50%

    Returns: 
        allele_freqs: array of allele frequencies, either just final 
            or if track_all = True, then over time
    """

    if np.less(N, 0) or np.less(T, 0):
        raise(ValueError("The population size or time is less than zero." 
                         " Has the model been misspecified?"))
    
    if np.any(np.less(init_q, 0)) or np.any(np.greater(init_q, 1)):
        raise(ValueError("At least one initial q_value is less than zero"
                         " or greater than one."))
    
    if len(init_q) != replicates:
        raise ValueError("Length of init_q must equal number of replicates.")
    
    nu_f = ensure_1arg_func(nu)

    # if we want to plot, we need to track all of the trajectories
    if plot:
        track_all = True
    
    # calculate s values from gammas because we will need it for our generation based simulation
    s1, s2, s3, s4 = gamma1/(2*N), gamma2/(2*N), gamma3/(2*N), gamma4/(2*N)

    # create matrix to store allele frequencies
    if track_all:
        allele_freqs = np.empty((replicates, int(2*N*T+1)))
        allele_freqs[:, 0] = init_q
    else:
        allele_freqs = init_q

    s1_vec = np.full(replicates, s1)
    s2_vec = np.full(replicates, s2)
    s3_vec = np.full(replicates, s3)
    s4_vec = np.full(replicates, s4)

    rng = np.random.default_rng()
    total_gens = int(2*N*T)
    for t in range(total_gens):
        nu = nu_f(t/(2*N)) # this rescales from generations back to diffusion time
        samples = int(4*N*nu)
        if track_all:
            q_post_sel = auto_selection(allele_freqs[:, t], s1_vec, s2_vec, s3_vec, s4_vec)
            allele_freqs[:, t+1] = rng.binomial(samples, q_post_sel)/(samples)
        else:
            q_post_sel = auto_selection(allele_freqs, s1_vec, s2_vec, s3_vec, s4_vec)
            allele_freqs = rng.binomial(samples, q_post_sel)/(samples)

    if plot:
        plt.plot(allele_freqs.T, color='gray', alpha=0.025)
        plt.plot(np.mean(allele_freqs, axis=0), color='black', lw=2)
        plt.xlabel('Generation')
        plt.ylabel('Mutant Allele Frequency')
        plt.title('Autotetraploid Allelic Drift Simulation')
        plt.ylim(-.1, 1.1)
        plt.show()
    
    return allele_freqs

# Autotetraploid model based on gametic sampling
def auto_gamete_recursions(gamete_freqs, fitness):
    """
    gamete_freqs = (g0, g1, g2)

    Creates gamete pool for the next generation by deterministically 
    forming and selecting genotypes and a subsequent gamete pool.

    g0: frequency of g0 gamete
    g1: frequency of g1 gamete
    g2: frequency of g2 gamete
    fitness: numpy array of fitnesses for G0, G1, G2, G3, G4 individuals

    Returns:
        gamete_plus_1: gamete frequencies for the sampling function
    """

    # unpack vectors of gamete frequencies
    g0, g1, g2 = gamete_freqs
    # compute genotype frequencies
    genotype_freqs = np.array([g0**2, 2*g0*g1, 2*g0*g2 + g1**2, 2*g1*g2, g2**2])
    # compute average fitness
    average_fitness = np.dot(fitness, genotype_freqs)
    # create a compatible size matrix of fitness values
    fitness_vector = np.array([[fitness[0]], [fitness[1]], [fitness[2]], [fitness[3]], [fitness[4]]])
    fitness_matrix = np.repeat(fitness_vector, g0.size, axis = 1)
    # calculate normalized frequencies of genotypes post selection
    post_sel_genotypes = (fitness_matrix * genotype_freqs)/average_fitness
    # create frequencies of gametes post selection
    g0_plus_1 = post_sel_genotypes[0, :] + post_sel_genotypes[1, :]/2 + post_sel_genotypes[2, :]/6
    g1_plus_1 = post_sel_genotypes[1, :]/2 + 2*post_sel_genotypes[2, :]/3 + post_sel_genotypes[3, :]/2
    g2_plus_1 = post_sel_genotypes[2, :]/6 + post_sel_genotypes[3, :]/2 + post_sel_genotypes[4, :]

    gamete_plus_1 = np.array([g0_plus_1, g1_plus_1, g2_plus_1])
    
    return gamete_plus_1

def auto_gametic_WF(N, T, init_q, gamma1, gamma2, gamma3, gamma4, nu = 1, replicates = 1, plot = False, track_all = False):
    """
    Simple Wright-Fisher model of genetic drift in autotetraploids based on gamete frequency sampling.

    N: population size (number of individuals)
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    init_q: vector of initial allele frequencies for selected allele; must have size = replicates
    gamma1: population-scaled selection coefficient for G1 individuals
        (<0 is purifying, >0 is positive)
        gamma1 = 2*N*s1 where a G1 individual has fitness 1 + 2*s1
    gamma2: population-scaled selection coefficient for G2 individuals
    gamma3: population-scaled selection coefficient for G3 individuals
    gamma4: population-scaled selection coefficient for G4 individuals
        This is the closest to "gamma" for diploids
    nu: population size relative to ancestral population size
        nu can be a constant or a function of diffusion-scaled time
    init_q: vector of initial allele frequencies for selected allele; must have size = replicates
    replicates: number of times to run the simulation
    plot: Boolean input to either show a plot of individual trajectories or not
    track_all: Boolean input to either track and return all trajectories or not
        This usage and plot are not recommended for analysis, but are reasonable usages for debuggging
        They are very memory intensive options and also slow down the code up to 50%

    Returns:
        allele_freqs: array of allele frequencies, either just final 
            or if track_all = True, then over time 
    """
    if np.less(N, 0) or np.less(T, 0):
        raise(ValueError("The population size or time is less than zero." 
                         " Has the model been misspecified?"))
    
    if np.any(np.less(init_q, 0)) or np.any(np.greater(init_q, 1)):
        raise(ValueError("At least one initial q_value is less than zero"
                         " or greater than one."))
    
    if len(init_q) != replicates:
        raise ValueError("Length of init_q must equal number of replicates.")
    
    nu_f = ensure_1arg_func(nu)

    # if we want to plot, we need to track all of the trajectories
    if plot:
        track_all = True
    
    # calculate s from gamma because we will need it for our generation based simulation
    s1, s2, s3, s4 = gamma1/(2*N), gamma2/(2*N), gamma3/(2*N), gamma4/(2*N)

    fitness = np.array([1, 1+2*s1, 1+2*s2, 1+2*s3, 1+2*s4])

    if track_all:
        # create tensor to store gamete frequencies
        # first dimension delineates across gamete types (g0, g1, g2)
        # second = separate runs
        # third = time
        gamete_freqs = np.empty((3, replicates, int(2*N*T+1)))
        gamete_freqs[0, :, 0] = (1-init_q)**2
        gamete_freqs[1, :, 0] = 2*init_q*(1-init_q)
        gamete_freqs[2, :, 0] = init_q**2
    else:
        # create matrix to store gamete frequencies
        # first dimension delineates across gamete types (g0, g1, g2)
        # second = separate runs
        gamete_freqs = np.empty((3, replicates))
        gamete_freqs[0, :] = (1-init_q)**2
        gamete_freqs[1, :] = 2*init_q*(1-init_q)
        gamete_freqs[2, :] = init_q**2

    rng = np.random.default_rng()
    total_gens = int(2*N*T)
    for t in range(0, total_gens):
        nu = nu_f(t/(2*N)) # this rescales from generations back to diffusion time
        samples = int(2*N*nu)
        if track_all:
            gametes_post_sel = auto_gamete_recursions(gamete_freqs[:, :, t], fitness)
            # Note the transposes below. This could be addressed by fixing the shape of the gamete_freqs, 
            # but is a relatively trivial calculation especially for vectors
            gamete_freqs[:, :, t+1] = rng.multinomial(samples, gametes_post_sel.T).T/(samples)
        else:
            gametes_post_sel = auto_gamete_recursions(gamete_freqs, fitness)
            # Note the transposes below. This could be addressed by fixing the shape of the gamete_freqs, 
            # but is a relatively trivial calculation especially for vectors
            gamete_freqs = rng.multinomial(samples, gametes_post_sel.T).T/(samples)

    if track_all:
        allele_freqs = .5*gamete_freqs[1, :, :] + gamete_freqs[2, :, :]
    else:
        allele_freqs = .5*gamete_freqs[1, :] + gamete_freqs[2, :]

    if plot:
        plt.plot(allele_freqs.T, color='gray', alpha=0.025)
        plt.plot(np.mean(allele_freqs, axis=0), color='black', lw=2)
        plt.xlabel('Generation')
        plt.ylabel('Mutant Allele Frequency')
        plt.title('Autotetraploid Allelic Drift Simulation')
        plt.ylim(-.1, 1.1)
        plt.show()
           
    return allele_freqs

# Allotetraploid model based on allelic sampling

# Note: only the allelic model for allos has the exchange parameter from Blischak et al. incorporated
# The implementation for the gamete based model is unclear to me

def allo_selection(qa, qb, s01, s02, s10, s11, s12, s20, s21, s22):
    """
    Evaluates new values of qa and qb after one generation of selection

    qa: allele frequency in subgenome a before selection
    qb: allele frequency in subgenome b before selection
    sij: selection coefficient for an individual of type G_ij
        that is, with i mutant copies in the first subgenome and j in the second

    Returns:
        q_post_sel: matrix of vectors for qa and qb, the allele frequencies post selection
    """
    # the RHS expressions were computed symbolically in Matlab
    qa_post_sel = (qa**2*qb**2*(2*s22 + 1) + qa**2*(2*s20 + 1)*(qb - 1)**2 
                   - qa*(2*s10 + 1)*(qa - 1)*(qb - 1)**2 - qa*qb**2*(2*s12 + 1)*(qa - 1)
                   - 2*qa**2*qb*(2*s21 + 1)*(qb - 1) + 2*qa*qb*(2*s11 + 1)*(qa - 1)*(qb - 1)
                   )/((qa - 1)**2*(qb - 1)**2 + qa**2*qb**2*(2*s22 + 1) + qb**2*(2*s02 + 1)*(qa - 1)**2 
                      + qa**2*(2*s20 + 1)*(qb - 1)**2 - 2*qa*(2*s10 + 1)*(qa - 1)*(qb - 1)**2 
                      - 2*qb*(2*s01 + 1)*(qa - 1)**2*(qb - 1) - 2*qa*qb**2*(2*s12 + 1)*(qa - 1) 
                      - 2*qa**2*qb*(2*s21 + 1)*(qb - 1) + 4*qa*qb*(2*s11 + 1)*(qa - 1)*(qb - 1))

 
    qb_post_sel = (qa**2*qb**2*(2*s22 + 1) + qb**2*(2*s02 + 1)*(qa - 1)**2
                    - qb*(2*s01 + 1)*(qa - 1)**2*(qb - 1) - 2*qa*qb**2*(2*s12 + 1)*(qa - 1)
                    - qa**2*qb*(2*s21 + 1)*(qb - 1) + 2*qa*qb*(2*s11 + 1)*(qa - 1)*(qb - 1)
                    )/((qa - 1)**2*(qb - 1)**2 + qa**2*qb**2*(2*s22 + 1) + qb**2*(2*s02 + 1)*(qa - 1)**2
                       + qa**2*(2*s20 + 1)*(qb - 1)**2 - 2*qa*(2*s10 + 1)*(qa - 1)*(qb - 1)**2
                       - 2*qb*(2*s01 + 1)*(qa - 1)**2*(qb - 1) - 2*qa*qb**2*(2*s12 + 1)*(qa - 1)
                       - 2*qa**2*qb*(2*s21 + 1)*(qb - 1) + 4*qa*qb*(2*s11 + 1)*(qa - 1)*(qb - 1))
 
 
    # manual override to set fixation/loss in one subgenome 
    # because there seems to be some numerical error in the float calculations above
    # I checked these expressions manually in Matlab at the boundaries for qa = 0 or 1 
    # and qb in [0, 1] and they seem correct 
    # i.e. qa = 0 yields qa = 0 after the calculation for any value of qb
    for i in range(qa_post_sel.size):
        if qa[i] == 1:
            qa_post_sel[i] = 1
        if qa[i] == 0:
            qa_post_sel[i] = 0
        if qb[i] == 1:
            qb_post_sel[i] = 1
        if qb[i] == 0:
            qb_post_sel[i] = 0

    q_post_sel = np.array([qa_post_sel, qb_post_sel])

    return q_post_sel

def allo_allelic_WF(N, T, E, gamma01, gamma02, gamma10, gamma11, gamma12, gamma20, gamma21, gamma22, 
                    init_qa, init_qb, nu=1, replicates = 1, plot = False, track_all=False):
    """
    Simple Wright-Fisher model of genetic drift in allotetraploids based on allelic sampling.

    N: population size (number of individuals)
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    gammaij: population-scaled selection coefficient (<0 is purifying, >0 is positive)
        gamma = 2Nsij
        gammaij corresponds to the scaled s for an individual of type G_ij
    E: population scaled probability that meiosis results in the exchange 
        of genetic material between subgenomes from Blischak et al. (2023) Genetics
        E = 2Ne
    init_qa: vector of initial allele frequencies for selected allele in a subgenome; must have size = replicates
    init_qb: vector of initial allele frequencies for selected allele in b subgenome; must have size = replicates
    nu: population size relative to ancestral population size
        nu can be a constant or a function of diffusion-scaled time
    replicates: number of times to run the simulation
    plot: Boolean input to either show a plot of individual trajectories or not
    track_all: Boolean input to either track and return all trajectories or not
        This usage and plot are not recommended for analysis, but are reasonable usages for debuggging
        They are very memory intensive options and also slow down the code up to 50%

    Returns: 
        allele_freqs: matrix of allele frequencies over generations
            each row corresponds to a single simulation
            each column corresponds to a single point in time
    """

    if np.less(N, 0) or np.less(T, 0):
        raise(ValueError("The population size or time is less than zero." 
                         " Has the model been misspecified?"))
    
    if np.any(np.less(init_qa, 0)) or np.any(np.greater(init_qa, 1)):
        raise(ValueError("At least one initial q_value for subgenome a"
                         " is less than zero or greater than one."))
    
    if np.any(np.less(init_qb, 0)) or np.any(np.greater(init_qb, 1)):
        raise(ValueError("At least one initial q_value for subgenome b"
                         " is less than zero or greater than one."))
    
    if len(init_qa) != replicates:
        raise ValueError("Length of init_qa must equal number of replicates.")
    
    if len(init_qb) != replicates:
        raise ValueError("Length of init_qb must equal number of replicates.")

    nu_f = ensure_1arg_func(nu)

    # if we want to plot, we need to track all of the trajectories
    if plot:
        track_all = True

    # calculate sij from gammaij because we will need it for our generation based simulation
    s01, s02, s10, s11 = gamma01/(2*N), gamma02/(2*N), gamma10/(2*N), gamma11/(2*N)
    s12, s20, s21, s22 = gamma12/(2*N), gamma20/(2*N), gamma21/(2*N), gamma22/(2*N)

    # same for e
    e = E/(2*N)

    if track_all:
        # create tensor to store allele frequencies
        # first dimension delineates across subgenomes
        # second = separate runs
        # third = time
        allele_freqs = np.empty((2, replicates, int(2*N*T+1)))
    
        allele_freqs[0, :, 0] = init_qa
        allele_freqs[1, :, 0] = init_qb
    else:
        # create matrix to store allele frequencies
        # first dimension delineates across subgenomes
        # second = separate runs
        allele_freqs = np.empty((2, replicates))
    
        allele_freqs[0, :] = init_qa
        allele_freqs[1, :] = init_qb

    s01_vec = np.full(replicates, s01)
    s02_vec = np.full(replicates, s02)
    s10_vec = np.full(replicates, s10)
    s11_vec = np.full(replicates, s11)
    s12_vec = np.full(replicates, s12)
    s20_vec = np.full(replicates, s20)
    s21_vec = np.full(replicates, s21)
    s22_vec = np.full(replicates, s22)

    rng = np.random.default_rng()

    total_gens = int(2*N*T)
    for t in range(total_gens):
        nu = nu_f(t/(2*N)) # this rescales from generations back to diffusion time
        samples = int(2*N*nu)
        ### Note 
        # I am not sure if doing the allelic exchange before or after selection matters, but I will test both
        # It sees to have no effect, but ask Ryan and Justin about this still
        ###
        if track_all:
            q_next = allo_selection(allele_freqs[0, :, t], allele_freqs[1, :, t], s01_vec, s02_vec, 
                                        s10_vec, s11_vec, s12_vec, s20_vec, s21_vec, s22_vec)
            # add the changes due to migration/HEs
            # here, we use the old allele freqs to avoid issues with applying selection
            # and migration simultaneously
            q_next[0, :] += e*(allele_freqs[1, :, t] - allele_freqs[0, :, t])
            q_next[1, :] += e*(allele_freqs[0, :, t] - allele_freqs[1, :, t])
            
            allele_freqs[0, :, t+1] = rng.binomial(samples, q_next[0, :])/(samples)
            allele_freqs[1, :, t+1] = rng.binomial(samples, q_next[1, :])/(samples)
        else:
            q_next = allo_selection(allele_freqs[0, :], allele_freqs[1, :], s01_vec, s02_vec, 
                                        s10_vec, s11_vec, s12_vec, s20_vec, s21_vec, s22_vec)
            
            q_next[0, :] += e*(allele_freqs[1, :] - allele_freqs[0, :])
            q_next[1, :] += e*(allele_freqs[0, :] - allele_freqs[1, :])

            allele_freqs[0, :] = rng.binomial(samples, q_next[0, :])/(samples)
            allele_freqs[1, :] = rng.binomial(samples, q_next[1, :])/(samples)

    if plot:
        fig, axs = plt.subplots(3, 1, sharex = 'col', sharey = 'row')
        # axs[0] corresponds to subgenome a, axs[1] to subgenome b, axs[2] to overall allele frequency
        for i in range(replicates):
            axs[0].plot(allele_freqs[0, i, :], color = 'gray', alpha=0.025)
            axs[1].plot(allele_freqs[1, i, :], color = 'gray', alpha=0.025)
            axs[2].plot((allele_freqs[0, i, :] + allele_freqs[1, i, :])/2, color = 'gray', alpha=0.025)
        axs[0].set_ylabel('Subgenome A')
        axs[0].set_ylim(-.1, 1.1)
        axs[1].set_ylabel('Subgenome B')
        axs[1].set_ylim(-.1, 1.1)
        axs[2].set_ylabel('Overall')
        axs[2].set_ylim(-.1, 1.1)
        fig.supxlabel('Time (in generations)')
        fig.suptitle('Allele Frequencies Over Time')
        plt.show()
        
    return allele_freqs


# Allotetraploid model based on gametic sampling
def allo_gamete_recursions(gamete_freqs, fitness):
    """
    gamete_freqs = (g00, g01, g10, g11)

    Creates gamete pool for the next generation by deterministically 
    forming genotypes and a subsequent gamete pool.

    g00: frequency of g00 gamete
    g01: frequency of g01 gamete
    g10: frequency of g10 gamete
    g11: frequency of g11 gamete
    fitness: numpy array of fitnesses for G00, G01, G02, G10, G11, G12, G20, G21, G22 individuals

    Returns: 
        gamete_plus_1: gamete frequencies post selection and meiosis
    """

    # unpack vectors of gamete frequencies
    g00, g01, g10, g11 = gamete_freqs
    # compute genotype frequencies
    genotype_freqs = np.array([g00**2, 2*g00*g01, g01**2, 2*g00*g10, 2*(g00*g11 + g01*g10), 2*g01*g11, g10**2, 2*g10*g11, g11**2])
    # compute average fitness
    average_fitness = np.dot(fitness, genotype_freqs)
    # create a compatible size matrix of fitness values for element wise operations below
    fitness_matrix = np.repeat(fitness[:, np.newaxis], g00.size, axis = 1)
    # calculate normalized frequencies of genotypes post selection
    post_sel_genotypes = (fitness_matrix * genotype_freqs) / average_fitness
    psGs = post_sel_genotypes
    # create frequencies of gametes post selection
    g00_plus_1 = psGs[0] + psGs[1]/2 + psGs[3]/2 + psGs[4]/4
    g01_plus_1 = psGs[1]/2 + psGs[2] + psGs[4]/4 + psGs[5]/2
    g10_plus_1 = psGs[3]/2 + psGs[4]/4 + psGs[6] + psGs[7]/2
    g11_plus_1 = psGs[4]/4 + psGs[5]/2 + psGs[7]/2 + psGs[8]

    gamete_plus_1 = np.array([g00_plus_1, g01_plus_1, g10_plus_1, g11_plus_1])

    return gamete_plus_1

def allo_gametic_WF(N, T, gamma01, gamma02, gamma10, gamma11, gamma12, gamma20, gamma21, gamma22, 
                    init_qa, init_qb, nu = 1, replicates = 1, plot = False, track_all = False):
    """
    Simple Wright-Fisher model of genetic drift in allotetraploids based on gametic sampling. 

    N: population size (number of individuals)
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    gammaij: population-scaled selection coefficient (<0 is purifying, >0 is positive)
        gamma = 2Nsij
        gammaij corresponds to the scaled s for an individual of type G_ij
    init_qa: vector of initial allele frequencies for selected allele in a subgenome; must have size = replicates
    init_qb: vector of initial allele frequencies for selected allele in b subgenome; must have size = replicates
    nu: population size relative to ancestral population size
        nu can be a constant or a function of diffusion-scaled time
    replicates: number of times to run the simulation
    plot: Boolean input to either show a plot of individual trajectories or not
    track_all: Boolean input to either track and return all trajectories or not
        This usage and plot are not recommended for analysis, but are reasonable usages for debuggging
        They are very memory intensive options and also slow down the code up to 50%

    Returns:
        allele_frequencies: array of allele frequencies over time
    """

    if np.less(N, 0) or np.less(T, 0):
        raise(ValueError("The population size or time is less than zero." 
                         " Has the model been misspecified?"))
    
    if np.any(np.less(init_qa, 0)) or np.any(np.greater(init_qa, 1)):
        raise(ValueError("At least one initial q_value for subgenome a"
                         " is less than zero or greater than one."))
    
    if np.any(np.less(init_qb, 0)) or np.any(np.greater(init_qb, 1)):
        raise(ValueError("At least one initial q_value for subgenome b"
                         " is less than zero or greater than one."))
    
    if len(init_qa) != replicates:
        raise ValueError("Length of init_qa must equal number of replicates.")
    
    if len(init_qb) != replicates:
        raise ValueError("Length of init_qb must equal number of replicates.")
    
    nu_f = ensure_1arg_func(nu)

    # if we want to plot, we need to track all of the trajectories
    if plot:
        track_all = True

    # calculate s from gamma because we will need it for our generation based simulation
    s01, s02, s10, s11 = gamma01/(2*N), gamma02/(2*N), gamma10/(2*N), gamma11/(2*N)
    s12, s20, s21, s22 = gamma12/(2*N), gamma20/(2*N), gamma21/(2*N), gamma22/(2*N)

    fitness = np.array([1, 1+2*s01, 1+2*s02, 1+2*s10, 1+2*s11, 1+2*s12, 1+2*s20, 1+2*s21, 1+2*s22])

    if track_all:
        # create tensor to store gamete frequencies
        # first dimension delineates across gamete types (g00, g01, g10, g11)
        # second = separate runs
        # third = time
        gamete_freqs = np.empty((4, replicates, int(2*N*T+1)))
        gamete_freqs[0, :, 0] = (1-init_qa)*(1-init_qb)
        gamete_freqs[1, :, 0] = (1-init_qa)*init_qb
        gamete_freqs[2, :, 0] = init_qa*(1-init_qb)
        gamete_freqs[3, :, 0] = init_qa*init_qb
    else:
        # create matrix to store gamete frequencies
        # first dimension delineates across gamete types (g00, g01, g10, g11)
        # second = separate runs
        gamete_freqs = np.empty((4, replicates))
        gamete_freqs[0, :] = (1-init_qa)*(1-init_qb)
        gamete_freqs[1, :] = (1-init_qa)*init_qb
        gamete_freqs[2, :] = init_qa*(1-init_qb)
        gamete_freqs[3, :] = init_qa*init_qb

    rng = np.random.default_rng()
    total_gens = int(2*N*T)
    for t in range(total_gens):
        nu = nu_f(t/(2*N)) # this rescales from generations back to diffusion time
        samples = int(2*N*nu)

        if track_all:
            gametes_post_sel = allo_gamete_recursions(gamete_freqs[:, :, t], fitness)
            # Note the transposes below. This could be addressed by changing the shape of the gamete_freqs, 
            # but is a relatively trivial calculation especially for vectors
            gamete_freqs[:, :, t+1] = rng.multinomial(samples, gametes_post_sel.T).T/(samples)
        else:
            gametes_post_sel = allo_gamete_recursions(gamete_freqs[:, :], fitness)
            gamete_freqs[:, :] = rng.multinomial(samples, gametes_post_sel.T).T/(samples)

    if track_all:    
        q_freq_a = gamete_freqs[2, :, :] + gamete_freqs[3, :, :]
        q_freq_b = gamete_freqs[1, :, :] + gamete_freqs[3, :, :]
    else:
        q_freq_a = gamete_freqs[2, :] + gamete_freqs[3, :]
        q_freq_b = gamete_freqs[1, :] + gamete_freqs[3, :]
    
    overall_q_freq = (q_freq_a + q_freq_b) / 2
    allele_freqs = np.array([q_freq_a, q_freq_b])

    if plot: 
        fig, axs = plt.subplots(3, 1, sharex = 'col', sharey = 'row')
        # axs[0] = Subgenome A, axs[1] = Subgenome B, axs[2] = Overall
        for i in range(replicates):
            axs[0].plot(q_freq_a[i, :], color="gray", alpha=0.025)
            axs[1].plot(q_freq_b[i, :], color="gray", alpha=0.025)
            axs[2].plot(overall_q_freq[i, :], color="gray", alpha=0.025)
        axs[0].set_ylabel("Subgenome A")
        axs[0].set_ylim(-.1, 1.1)
        axs[1].set_ylabel("Subgenome B")
        axs[1].set_ylim(-.1, 1.1)
        axs[2].set_ylabel("Overall")
        axs[2].set_ylim(-.1, 1.1)
        fig.supxlabel('Time (in generations)')
        fig.suptitle('Allele Frequencies Over Time')
        plt.show()

    return allele_freqs

### 2D models

def auto_dip_migration_WF(N, T, init_q1, init_q2, sel1, sel2, M_12 = 0, M_21 = 0, nu1 = 1, nu2 = 1, replicates = 1, plot = False, track_all = False):
    """
    Wright-Fisher model of two populations. 
        pop1: diploids
        pop2: autotetraploids
    
    N: population size (number of individuals) for each population 
        I think this must be the same for the model to be defined appropriately
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    M_12: population scaled migration rate from autotetraploid to diploid population (2Nm_12)
    M_21: population scaled migration rate from diploid to autotetraploid population (2Nm_21)
    s1: vector of selection coefficients for diploids ([gamma, h, ...])
    s2: vector of selection coefficients for autotetraploids ([gamma1, gamma2, gamma3, gamma4, ...])
    init_q1: vector of initial allele frequencies for selected allele in diploids
        must have size = replicates
    init_q2: vector of initial allele frequencies for selected allele in autotetraploids
        must have size = replicates
    replicates: number of times to run the simulation
    plot: Boolean input to either show a plot of individual trajectories or not
    track_all: Boolean input to either track and return all trajectories or not

    Returns: 
        allele_freqs: tensor of allele frequencies over generations
            first dimension separates each population (autos = 0, dips = 1)
            second dimension corresponds to replicates
            third dimension corresponds to time
            e.g. allele_freqs[0, :, :] is the auto data
                 allele_freqs[1, :, -1] are the final freqs for the diploids
    """

    if np.any(np.less([N, T, M_12, M_21], 0)):
        raise(ValueError("A population size, time, or migration rate is less than zero." 
                         " Has the model been misspecified?"))
    
    if np.any(np.less(init_q1, 0)) or np.any(np.greater(init_q1, 1)) or np.any(np.less(init_q2, 0)) or np.any(np.greater(init_q2, 1)):
        raise(ValueError("At least one initial q_value is less than zero"
                         " or greater than one."))
    
    if len(init_q1) != replicates:
        raise ValueError("Length of init_q1 must equal number of replicates.")
    
    if len(init_q2) != replicates:
        raise ValueError("Length of init_q2 must equal number of replicates.")
    
    nu1_f, nu2_f = ensure_1arg_func(nu1), ensure_1arg_func(nu2)

    # if we want to plot, we need to track all of the trajectories
    if plot:
        track_all = True
    
    # calculate s from gamma because we will need it for our generation based simulation
    s_dip, h = sel1[0]/(2*N), sel1[1]
    s1, s2, s3, s4 = sel2[0]/(2*N), sel2[1]/(2*N), sel2[2]/(2*N), sel2[3]/(2*N)

    # calculate m from M similarly
    m_12 = M_12/(2*N)
    m_21 = M_21/(2*N)

    if track_all:
        # create matrices to store allele frequencies
        dip_freqs = np.empty((replicates, int(2*N*T+1)))
        dip_freqs[:, 0] = init_q1

        auto_freqs = np.empty((replicates, int(2*N*T+1)))
        auto_freqs[:, 0] = init_q2
    else:
        dip_freqs = init_q1
        auto_freqs = init_q2

    # create vectors of parameters for parallel evaluation of selection
    s1_vec = np.full(replicates, s1)
    s2_vec = np.full(replicates, s2)
    s3_vec = np.full(replicates, s3)
    s4_vec = np.full(replicates, s4)

    s_dip_vec = np.full(replicates, s_dip)
    h_vec = np.full(replicates, h)

    rng_auto = np.random.default_rng()
    rng_dip = np.random.default_rng()
    total_gens = int(2*N*T)
    for t in range(total_gens):
        nu1, nu2 = nu1_f(t/(2*N)), nu2_f(t/(2*N))

        # N_e_1 = dip_calc_N_e(s_dip, h, N)

        samples1 = int(2*N*nu1)
        samples2 = int(4*N*nu2) # autos, so 4N

        if track_all:
            dip_next = dip_selection(dip_freqs[:, t], s_dip_vec, h_vec)
            auto_next = auto_selection(auto_freqs[:, t], s1_vec, s2_vec, s3_vec, s4_vec)

            dip_next += m_12*(auto_freqs[:, t] - dip_freqs[:, t])
            auto_next += m_21*(dip_freqs[:, t] - auto_freqs[:, t])
            
            dip_freqs[:, t+1] = rng_dip.binomial(samples1, dip_next)/(samples1)
            auto_freqs[:, t+1] = rng_auto.binomial(samples2, auto_next)/(samples2)
        else:
            dip_next = dip_selection(dip_freqs, s_dip_vec, h_vec)
            auto_next = auto_selection(auto_freqs, s1_vec, s2_vec, s3_vec, s4_vec)

            dip_next += m_12*(auto_freqs - dip_freqs)
            auto_next += m_21*(dip_freqs - auto_freqs)

            dip_freqs = rng_dip.binomial(samples1, dip_next)/(samples1)
            auto_freqs = rng_auto.binomial(samples2, auto_next)/(samples2)


    if plot:
        plt.plot(auto_freqs.T, color='gray', alpha=0.025)
        plt.plot(np.mean(auto_freqs, axis=0), color='black', lw=2)
        plt.xlabel('Generation')
        plt.ylabel('Mutant Allele Frequency')
        plt.title('Autotetraploid Allelic Drift Simulation')
        plt.ylim(-.1, 1.1)
        plt.show()

        plt.plot(dip_freqs.T, color='gray', alpha=0.025)
        plt.plot(np.mean(dip_freqs, axis=0), color='black', lw=2)
        plt.xlabel('Generation')
        plt.ylabel('Mutant Allele Frequency')
        plt.title('Diploid Allelic Drift Simulation')
        plt.ylim(-.1, 1.1)
        plt.show()

    allele_freqs = np.array([auto_freqs, dip_freqs])
    
    return allele_freqs

def dip_dip_migration_WF(N, T, init_q1, init_q2, sel1, sel2, M_12 = 0, M_21 = 0, nu1 = 1, nu2 = 1, replicates = 1, plot = False, track_all = False):
    """
    Wright-Fisher model of two populations. 
        pop1: diploids
        pop2: diploids
    
    N: population size (number of individuals) for each population 
        I think this must be the same for the model to be defined appropriately
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    M_12: population scaled migration rate from pop2 to pop1 (2Nm_12)
    M_21: population scaled migration rate from pop1 to pop2 (2Nm_21)
    s1: vector of selection coefficients for pop1 ([gamma, h, ...])
    s2: vector of selection coefficients for pop2 ([gamma, h, ...])
    init_q1: vector of initial allele frequencies for selected allele in pop1
        must have size = replicates
    init_q2: vector of initial allele frequencies for selected allele in pop2
        must have size = replicates
    replicates: number of times to run the simulation
    plot: Boolean input to either show a plot of individual trajectories or not
    track_all: Boolean input to either track and return all trajectories or not

    Returns: 
        allele_freqs: tensor/matrix of allele frequencies over generations
            first dimension separates each population (pop1 = 0, pop2 = 1)
    """

    if np.any(np.less([N, T, M_12, M_21], 0)):
        raise(ValueError("A population size, time, or migration rate is less than zero." 
                         " Has the model been misspecified?"))
    
    if np.any(np.less(init_q1, 0)) or np.any(np.greater(init_q1, 1)) or np.any(np.less(init_q2, 0)) or np.any(np.greater(init_q2, 1)):
        raise(ValueError("At least one initial q_value is less than zero"
                         " or greater than one."))
    
    if len(init_q1) != replicates:
        raise ValueError("Length of init_q1 must equal number of replicates.")
    
    if len(init_q2) != replicates:
        raise ValueError("Length of init_q2 must equal number of replicates.")
    
    nu1_f, nu2_f = ensure_1arg_func(nu1), ensure_1arg_func(nu2)

    # if we want to plot, we need to track all of the trajectories
    if plot:
        track_all = True
    
    # calculate s from gamma because we will need it for our generation based simulation
    s1, h1 = sel1[0]/(2*N), sel1[1]
    s2, h2 = sel2[0]/(2*N), sel2[1]

    # calculate m from M similarly
    m_12 = M_12/(2*N)
    m_21 = M_21/(2*N)

    if track_all:
        # create matrices to store allele frequencies
        pop1_freqs = np.empty((replicates, int(2*N*T+1)))
        pop1_freqs[:, 0] = init_q1

        pop2_freqs = np.empty((replicates, int(2*N*T+1)))
        pop2_freqs[:, 0] = init_q2
    else:
        pop1_freqs = init_q1
        pop2_freqs = init_q2

    # create vectors of parameters for parallel evaluation of selection
    s1_vec = np.full(replicates, s1)
    h1_vec = np.full(replicates, h1)

    s2_vec = np.full(replicates, s2)
    h2_vec = np.full(replicates, h2)

    rng = np.random.default_rng()
    total_gens = int(2*N*T)
    for t in range(total_gens):
        nu1, nu2 = nu1_f(t/(2*N)), nu2_f(t/(2*N))
        samples1 = int(2*N*nu1)
        samples2 = int(2*N*nu2) 

        if track_all:
            q1_next = dip_selection(pop1_freqs[:, t], s1_vec, h1_vec)
            q2_next = dip_selection(pop2_freqs[:, t], s2_vec, h2_vec)

            q1_next += m_12*(pop2_freqs[:, t] - pop1_freqs[:, t])
            q2_next += m_21*(pop1_freqs[:, t] - pop2_freqs[:, t])
            
            pop1_freqs[:, t+1] = rng.binomial(samples1, q1_next)/(samples1)
            pop2_freqs[:, t+1] = rng.binomial(samples2, q2_next)/(samples2)

        else:
            q1_next = dip_selection(pop1_freqs, s1_vec, h1_vec)
            q2_next = dip_selection(pop2_freqs, s2_vec, h2_vec)

            q1_next += m_12*(pop2_freqs - pop1_freqs)
            q2_next += m_21*(pop1_freqs - pop2_freqs)

            pop1_freqs = rng.binomial(samples1, q1_next)/(samples1)
            pop2_freqs = rng.binomial(samples2, q2_next)/(samples2)


    if plot:
        plt.plot(pop1_freqs.T, color='gray', alpha=0.025)
        plt.plot(np.mean(pop1_freqs, axis=0), color='black', lw=2)
        plt.xlabel('Generation')
        plt.ylabel('Mutant Allele Frequency')
        plt.title('Diploid (pop1) Allelic Drift Simulation')
        plt.ylim(-.1, 1.1)
        plt.show()

        plt.plot(pop2_freqs.T, color='gray', alpha=0.025)
        plt.plot(np.mean(pop2_freqs, axis=0), color='black', lw=2)
        plt.xlabel('Generation')
        plt.ylabel('Mutant Allele Frequency')
        plt.title('Diploid (pop2) Allelic Drift Simulation')
        plt.ylim(-.1, 1.1)
        plt.show()

    allele_freqs = np.array([pop1_freqs, pop2_freqs])
    
    return allele_freqs

def auto_auto_migration_WF(N, T, init_q1, init_q2, sel1, sel2, M_12 = 0, M_21 = 0, nu1 = 1, nu2 = 1, replicates = 1, plot = False, track_all = False):
    """
    Wright-Fisher model of two populations. 
        pop1: autotetraploids
        pop2: autotetraploids
    
    N: population size (number of individuals) for each population 
        I think this must be the same for the model to be defined appropriately
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    M_12: population scaled migration rate from pop2 to pop1 (2Nm_12)
    M_21: population scaled migration rate from pop1 to pop2 (2Nm_21)
    s1: vector of selection coefficients for pop1 ([gamma1, gamma2, gamma3, gamma4, ...])
    s2: vector of selection coefficients for pop2 ([gamma1, gamma2, gamma3, gamma4, ...])
    init_q1: vector of initial allele frequencies for selected allele in pop1
        must have size = replicates
    init_q2: vector of initial allele frequencies for selected allele in pop2
        must have size = replicates
    replicates: number of times to run the simulation
    plot: Boolean input to either show a plot of individual trajectories or not
    track_all: Boolean input to either track and return all trajectories or not

    Returns: 
        allele_freqs: tensor/matrix of allele frequencies over generations
            first dimension separates each population (pop1 = 0, pop2 = 1)
    """

    if np.any(np.less([N, T, M_12, M_21], 0)):
        raise(ValueError("A population size, time, or migration rate is less than zero." 
                         " Has the model been misspecified?"))
    
    if np.any(np.less(init_q1, 0)) or np.any(np.greater(init_q1, 1)) or np.any(np.less(init_q2, 0)) or np.any(np.greater(init_q2, 1)):
        raise(ValueError("At least one initial q_value is less than zero"
                         " or greater than one."))
    
    if len(init_q1) != replicates:
        raise ValueError("Length of init_q1 must equal number of replicates.")
    
    if len(init_q2) != replicates:
        raise ValueError("Length of init_q2 must equal number of replicates.")
    
    nu1_f, nu2_f = ensure_1arg_func(nu1), ensure_1arg_func(nu2)

    # if we want to plot, we need to track all of the trajectories
    if plot:
        track_all = True
    
    # calculate s from gamma because we will need it for our generation based simulation
    s1_1, s1_2, s1_3, s1_4 = sel1[0]/(2*N), sel1[1]/(2*N), sel1[2]/(2*N), sel1[3]/(2*N)
    s2_1, s2_2, s2_3, s2_4 = sel2[0]/(2*N), sel2[1]/(2*N), sel2[2]/(2*N), sel2[3]/(2*N)

    # calculate m from M similarly
    m_12 = M_12/(2*N)
    m_21 = M_21/(2*N)

    if track_all:
        # create matrices to store allele frequencies
        pop1_freqs = np.empty((replicates, int(2*N*T+1)))
        pop1_freqs[:, 0] = init_q1

        pop2_freqs = np.empty((replicates, int(2*N*T+1)))
        pop2_freqs[:, 0] = init_q2
    else:
        pop1_freqs = init_q1
        pop2_freqs = init_q2

    # create vectors of parameters for parallel evaluation of selection
    s1_1_vec = np.full(replicates, s1_1)
    s1_2_vec = np.full(replicates, s1_2)
    s1_3_vec = np.full(replicates, s1_3)
    s1_4_vec = np.full(replicates, s1_4)

    s2_1_vec = np.full(replicates, s2_1)
    s2_2_vec = np.full(replicates, s2_2)
    s2_3_vec = np.full(replicates, s2_3)
    s2_4_vec = np.full(replicates, s2_4)

    rng = np.random.default_rng()
    total_gens = int(2*N*T)
    for t in range(total_gens):
        nu1, nu2 = nu1_f(t/(2*N)), nu2_f(t/(2*N))
        samples1 = int(4*N*nu1)
        samples2 = int(4*N*nu2) 

        if track_all:
            q1_next = auto_selection(pop1_freqs[:, t], s1_1_vec, s1_2_vec, s1_3_vec, s1_4_vec)
            q2_next = auto_selection(pop2_freqs[:, t], s2_1_vec, s2_2_vec, s2_3_vec, s2_4_vec)

            q1_next += m_12*(pop2_freqs[:, t] - pop1_freqs[:, t])
            q2_next += m_21*(pop1_freqs[:, t] - pop2_freqs[:, t])
            
            pop1_freqs[:, t+1] = rng.binomial(samples1, q1_next)/(samples1)
            pop2_freqs[:, t+1] = rng.binomial(samples2, q2_next)/(samples2)

        else:
            q1_next = auto_selection(pop1_freqs, s1_1_vec, s1_2_vec, s1_3_vec, s1_4_vec)
            q2_next = auto_selection(pop2_freqs, s2_1_vec, s2_2_vec, s2_3_vec, s2_4_vec)

            q1_next += m_12*(pop2_freqs - pop1_freqs)
            q2_next += m_21*(pop1_freqs - pop2_freqs)

            pop1_freqs = rng.binomial(samples1, q1_next)/(samples1)
            pop2_freqs = rng.binomial(samples2, q2_next)/(samples2)


    if plot:
        plt.plot(pop1_freqs.T, color='gray', alpha=0.025)
        plt.plot(np.mean(pop1_freqs, axis=0), color='black', lw=2)
        plt.xlabel('Generation')
        plt.ylabel('Mutant Allele Frequency')
        plt.title('Diploid (pop1) Allelic Drift Simulation')
        plt.ylim(-.1, 1.1)
        plt.show()

        plt.plot(pop2_freqs.T, color='gray', alpha=0.025)
        plt.plot(np.mean(pop2_freqs, axis=0), color='black', lw=2)
        plt.xlabel('Generation')
        plt.ylabel('Mutant Allele Frequency')
        plt.title('Diploid (pop2) Allelic Drift Simulation')
        plt.ylim(-.1, 1.1)
        plt.show()

    allele_freqs = np.array([pop1_freqs, pop2_freqs])
    
    return allele_freqs

### 3D model
def dip_allo_WF(N, T, M12, M21, M13, M31, M23, M32,
                    sel1, sel2, sel3, 
                    init_q1, init_q2, init_q3,
                    nu1=1, nu2=1, nu3=1,
                    replicates = 1):
    """
    Simple Wright-Fisher model of genetic drift in allotetraploids based on allelic sampling.
        pop1: diploids
        pop2: allo subgenome a
        pop3: allo subgenome b
    
    N: population size (number of individuals)
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    Mij: 2*N*mij where mij is the migration rate from pop. j to pop. i
        Note: M32 = M23 jointly specify an exchange parameter, E (see Blischak et al. 2023)
    sel1,2,3: vectors of selection parameters for diploids, subgenome a, and subgenome b
        necessarily, sel2=sel3 for allotetraploids
    init_q1: vector of initial allele frequencies for selected allele in diploids; must have size = replicates
    init_q2: vector of initial allele frequencies for selected allele in a subgenome; must have size = replicates
    init_q3: vector of initial allele frequencies for selected allele in b subgenome; must have size = replicates
    nu1,2,3: population size relative to ancestral population size
        nu can be a constant or a function of diffusion-scaled time
        nu2=nu3 necessarily for allotetraploids
    replicates: number of times to run the simulation
    
    Returns: 
        allele_freqs: matrix of allele frequencies over generations
            each row corresponds to a single simulation
            each column corresponds to a single point in time
    """

    if np.less(N, 0) or np.less(T, 0):
        raise(ValueError("The population size or time is less than zero." 
                         " Has the model been misspecified?"))
    
    if np.any(np.less(np.concatenate([init_q1, init_q2, init_q3]), 0)) or np.any(np.greater(np.concatenate([init_q1, init_q2, init_q3]), 1)):
        raise(ValueError("At least one initial q_value is less than zero or greater than one."))
    
    if (len(init_q1) != replicates) or (len(init_q2) != replicates) or (len(init_q3) != replicates):
        raise ValueError("Length of init_q's must equal number of replicates.")
    
    if sel2 != sel3:
        raise ValueError("Selection parameters for subgenome a and b must be equal.")
    
    if M23 != M32:
        raise ValueError("Migration rates for subgenomes a and b must be equal.")
    
    if np.isscalar(nu2) and np.isscalar(nu3):
        if nu2 != nu3:
            raise ValueError("Population 2 and 3 are allotetraploids, but populations 2 and 3 do not have the same population size."
                             "Has the model been misspecified?")
    elif nu2(0) != nu3(0):
        raise ValueError("Population 2 and 3 are allotetraploids, but populations 2 and 3 do not have the same population size."
                         "Has the model been misspecified?")
    
    nu1_f, nu2_f, nu3_f = ensure_1arg_func(nu1), ensure_1arg_func(nu2), ensure_1arg_func(nu3)

    # calculate sij from gammaij because we will need it for our generation based simulation
    s01, s02, s10, s11 = sel2[0]/(2*N), sel2[1]/(2*N), sel2[2]/(2*N), sel2[3]/(2*N)
    s12, s20, s21, s22 = sel2[4]/(2*N), sel2[5]/(2*N), sel2[6]/(2*N), sel2[7]/(2*N)

    s_dip, h = sel1[0]/(2*N), sel1[1]

    # same for mij's
    m12, m21, m13, m31, m23, m32 = M12/(2*N), M21/(2*N), M13/(2*N), M31/(2*N), M23/(2*N), M32/(2*N)

    # create matrix to store allele frequencies
    # first dimension delineates across subgenomes
    # second = separate runs
    q_freqs = np.empty((3, replicates))
    
    q_freqs[0, :] = init_q1 # diploids
    q_freqs[1, :] = init_q2 # allotet subgenome a
    q_freqs[2, :] = init_q3 # allotet subgenome b

    s01_vec = np.full(replicates, s01)
    s02_vec = np.full(replicates, s02)
    s10_vec = np.full(replicates, s10)
    s11_vec = np.full(replicates, s11)
    s12_vec = np.full(replicates, s12)
    s20_vec = np.full(replicates, s20)
    s21_vec = np.full(replicates, s21)
    s22_vec = np.full(replicates, s22)

    sdip_vec = np.full(replicates, s_dip)   
    h_vec = np.full(replicates, h)

    rng = np.random.default_rng()

    total_gens = int(2*N*T)
    for t in range(total_gens):
        nu1 = nu1_f(t/(2*N)) # this rescales from generations back to diffusion time
        samples1 = int(2*N*nu1)

        nu2 = nu2_f(t/(2*N)) 
        samples2 = int(2*N*nu2)

        q_dip_next = dip_selection(q_freqs[0, :], sdip_vec, h_vec)
        q_allos_next = allo_selection(q_freqs[1, :], q_freqs[2, :], s01_vec, s02_vec,
                                        s10_vec, s11_vec, s12_vec, s20_vec, s21_vec, s22_vec)
        # q_allos_next has shape (2, replicates) with [0, :] indexing the a subgenome 
        # and [1, :] indexing the b subgenome   
        
        q_dip_next += m12*(q_freqs[1, :] - q_freqs[0, :]) + m13*(q_freqs[2, :] - q_freqs[0, :])
        q_allos_next[0, :] += m21*(q_freqs[0, :] - q_freqs[1, :]) + m23*(q_freqs[2, :] - q_freqs[1, :])
        q_allos_next[1, :] += m31*(q_freqs[0, :] - q_freqs[2, :]) + m32*(q_freqs[1, :] - q_freqs[2, :])

        q_freqs[0, :] = rng.binomial(samples1, q_dip_next)/(samples1)
        q_freqs[1, :] = rng.binomial(samples2, q_allos_next[0, :])/(samples2)
        q_freqs[2, :] = rng.binomial(samples2, q_allos_next[1, :])/(samples2)

    return q_freqs


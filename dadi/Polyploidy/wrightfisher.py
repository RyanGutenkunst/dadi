import numpy as np
import matplotlib.pyplot as plt

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
    
    # the RHS expression was computed symbolically in Matlab
    q_post_sel = (q**2*(2*s + 1) - 
                  q*(2*h*s + 1)*(q - 1))/((q - 1)**2 + 
                  q**2*(2*s + 1) - 2*q*(2*h*s + 1)*(q - 1))
 
    return q_post_sel

def dip_allelic_WF(N, T, gamma, init_q, h=0.5, replicates = 1, plot = False, track_all = False):
    """
    Simple Wright-Fisher model of genetic drift in diploids.

    N: population size (number of individuals)
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    gamma: population-scaled selection coefficient (<0 is purifying, >0 is positive)
        gamma = 2Ns 
        if A is the selected allele, aa has fitness 1, 
        Aa has fitness 1+2sh, AA has fitness 1+2s
    h: dominance coefficient for heterozygote
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
        if track_all:
            q_post_sel = dip_selection(allele_freqs[:, t], s_vec, h_vec)
            allele_freqs[:, t+1] = rng.binomial(2*N, q_post_sel)/(2*N)
        else:
            q_post_sel = dip_selection(allele_freqs, s_vec, h_vec)
            allele_freqs = rng.binomial(2*N, q_post_sel)/(2*N)

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

def auto_allelic_WF(N, T, init_q, gamma1, gamma2, gamma3, gamma4, replicates = 1, plot = False, track_all = False):
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
        if track_all:
            q_post_sel = auto_selection(allele_freqs[:, t], s1_vec, s2_vec, s3_vec, s4_vec)
            allele_freqs[:, t+1] = rng.binomial(4*N, q_post_sel)/(4*N)
        else:
            q_post_sel = auto_selection(allele_freqs, s1_vec, s2_vec, s3_vec, s4_vec)
            allele_freqs = rng.binomial(4*N, q_post_sel)/(4*N)

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

def auto_gametic_WF(N, T, init_q, gamma1, gamma2, gamma3, gamma4, replicates = 1, plot = False, track_all = False):
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
        if track_all:
            gametes_post_sel = auto_gamete_recursions(gamete_freqs[:, :, t], fitness)
            # Note the transposes below. This could be addressed by fixing the shape of the gamete_freqs, 
            # but is a relatively trivial calculation especially for vectors
            gamete_freqs[:, :, t+1] = rng.multinomial(2*N, gametes_post_sel.T).T/(2*N)
        else:
            gametes_post_sel = auto_gamete_recursions(gamete_freqs, fitness)
            # Note the transposes below. This could be addressed by fixing the shape of the gamete_freqs, 
            # but is a relatively trivial calculation especially for vectors
            gamete_freqs = rng.multinomial(2*N, gametes_post_sel.T).T/(2*N)

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

def allelic_exchange(qa, qb, e):
    """
    Evaluates new values of qa and qb due to allelic exchange

    qa: allele frequency in subgenome a before allelic exchange
    qb: allele frequency in subgenome b before allelic exchange
    e: rate of allelic exchange between subgenomes

    Returns:
        q_post_exchange: matrix of vectors for qa and qb, the allele frequencies post exchange
    """

    qa_post = qa+e*(qb - qa)/2 # the factor of two is from the Blischak paper, but not sure why?
    qb_post = qb+e*(qa - qb)/2

    q_post_exchange = np.array([qa_post, qb_post])
    return(q_post_exchange)

def allo_allelic_WF(N, T, E, gamma01, gamma02, gamma10, gamma11, gamma12, gamma20, gamma21, gamma22, 
                    init_qa, init_qb, replicates = 1, plot = False, track_all=False):
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
        ### Note 
        # I am not sure if doing the allelic exchange before or after selection matters, but I will test both
        # It sees to have no effect, but ask Ryan and Justin about this still
        ###
        if track_all:
            q_post_exchange = allelic_exchange(allele_freqs[0, :, t], allele_freqs[1, :, t], e)
            q_post_sel = allo_selection(q_post_exchange[0, :], q_post_exchange[1, :], s01_vec, s02_vec, 
                                        s10_vec, s11_vec, s12_vec, s20_vec, s21_vec, s22_vec)
            allele_freqs[0, :, t+1] = rng.binomial(2*N, q_post_sel[0, :])/(2*N)
            allele_freqs[1, :, t+1] = rng.binomial(2*N, q_post_sel[1, :])/(2*N)
        else:
            q_post_exchange = allelic_exchange(allele_freqs[0, :], allele_freqs[1, :], e)
            q_post_sel = allo_selection(q_post_exchange[0, :], q_post_exchange[1, :], s01_vec, s02_vec, 
                                        s10_vec, s11_vec, s12_vec, s20_vec, s21_vec, s22_vec)
            allele_freqs[0, :] = rng.binomial(2*N, q_post_sel[0, :])/(2*N)
            allele_freqs[1, :] = rng.binomial(2*N, q_post_sel[1, :])/(2*N)

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
                    init_qa, init_qb, replicates = 1, plot = False, track_all = False):
    """
    Simple Wright-Fisher model of genetic drift in allotetraploids based on gametic sampling. 

    N: population size (number of individuals)
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    gammaij: population-scaled selection coefficient (<0 is purifying, >0 is positive)
        gamma = 2Nsij
        gammaij corresponds to the scaled s for an individual of type G_ij
    init_qa: vector of initial allele frequencies for selected allele in a subgenome; must have size = replicates
    init_qb: vector of initial allele frequencies for selected allele in b subgenome; must have size = replicates
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
        if track_all:
            gametes_post_sel = allo_gamete_recursions(gamete_freqs[:, :, t], fitness)
            # Note the transposes below. This could be addressed by changing the shape of the gamete_freqs, 
            # but is a relatively trivial calculation especially for vectors
            gamete_freqs[:, :, t+1] = rng.multinomial(2*N, gametes_post_sel.T).T/(2*N)
        else:
            gametes_post_sel = allo_gamete_recursions(gamete_freqs[:, :], fitness)
            gamete_freqs[:, :] = rng.multinomial(2*N, gametes_post_sel.T).T/(2*N)

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

### two population model for autos and diploids to test migration scaling.
### this primarily is to validate the scaling of migration rates in the diffusion solution in dadi
### such a biological scenario could exist in mixed cytotype populations with triploids

def auto_dip_migration_WF(N, T, init_q_auto, init_q_dip, M_da = 0, M_ad = 0, gamma1=0, gamma2=0, 
                          gamma3=0, gamma4=0, gamma_dip=0, h = .5, replicates = 1, plot = False, track_all = False):
    """
    Simple Wright-Fisher model of genetic drift in autotetraploids based on allele frequency sampling. 

    N: population size (number of individuals) for each population 
        I think this must be the same for the model to be defined appropriately
    T: "diffusion" time to run the forward sampling process (in terms of 2*N generations)
    M_da: population scaled migration rate from autotetraploid to diploid population (2Nm_da)
    M_ad: population scaled migration rate from diploid to autotetraploid population (2Nm_ad)
    gamma1: population-scaled selection coefficient for G1 individuals in auto population
        (<0 is purifying, >0 is positive)
        gamma1 = 2*N*s1 where a G1 individual has fitness 1 + 2*s1
    gamma2: population-scaled selection coefficient for G2 individuals
    gamma3: population-scaled selection coefficient for G3 individuals
    gamma4: population-scaled selection coefficient for G4 individualsgamma_dip: population-scaled selection coefficient for autotetraploids (2Ns_dip)
    h: dominance coefficient for heterozygote in diploid population
    init_q_auto: vector of initial allele frequencies for selected allele in autotetraploids
        must have size = replicates
    init_q_dip: vector of initial allele frequencies for selected allele in diploids
        must have size = replicates
    replicates: number of times to run the simulation
    plot: Boolean input to either show a plot of individual trajectories or not
    track_all: Boolean input to either track and return all trajectories or not

    Returns: 
        allele_freqs: tensor of allele frequencies over generations
            first dimension separates each population (autos = 0, dips = 0)
            second dimension corresponds to replicates
            third dimension corresponds to time
            e.g. allele_freqs[0, :, :] is the auto data
                 allele_freqs[1, :, -1] are the final freqs for the diploids
    """

    if np.any(np.less([N, T, M_da, M_ad], 0)):
        raise(ValueError("A population size, time, or migration rate is less than zero." 
                         " Has the model been misspecified?"))
    
    if np.any(np.less(init_q_auto, 0)) or np.any(np.greater(init_q_auto, 1)):
        raise(ValueError("At least one initial q_value is less than zero"
                         " or greater than one."))
    
    if np.any(np.less(init_q_dip, 0)) or np.any(np.greater(init_q_dip, 1)):
        raise(ValueError("At least one initial q_value is less than zero"
                         " or greater than one."))
    
    if len(init_q_auto) != replicates:
        raise ValueError("Length of init_q_auto must equal number of replicates.")
    
    if len(init_q_dip) != replicates:
        raise ValueError("Length of init_q_dip must equal number of replicates.")
    
    # if we want to plot, we need to track all of the trajectories
    if plot:
        track_all = True
    
    # calculate s from gamma because we will need it for our generation based simulation
    s_dip = gamma_dip/(2*N)
    s1, s2, s3, s4 = gamma1/(2*N), gamma2/(2*N), gamma3/(2*N), gamma4/(2*N)

    # calculate m from M similarly
    m_da = M_da/(2*N)
    m_ad = M_ad/(2*N)

    if track_all:
        # create matrices to store allele frequencies
        auto_freqs = np.empty((replicates, int(2*N*T+1)))
        auto_freqs[:, 0] = init_q_auto

        dip_freqs = np.empty((replicates, int(2*N*T+1)))
        dip_freqs[:, 0] = init_q_dip
    else:
        auto_freqs = init_q_auto
        dip_freqs = init_q_dip

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
        # manually coded migration... because two lines is not worth a function :)
        #auto_mig_freqs = auto_freqs[:, t] + m_ad*(dip_freqs[:, t] - auto_freqs[:, t])
        #dip_mig_freqs = dip_freqs[:, t] + m_da*(auto_freqs[:, t] - dip_freqs[:, t])

        #auto_post_sel = auto_selection(auto_mig_freqs, s1_vec, s2_vec, s3_vec, s4_vec)
        #dip_post_sel = dip_selection(dip_mig_freqs, s_dip_vec, h_vec)

        # Here let's try reversing the order of migration and selection to see if that matters
        # Note: that was not contributing to the error and the order shouldn't matter!
        if track_all:
            auto = auto_selection(auto_freqs[:, t], s1_vec, s2_vec, s3_vec, s4_vec)
            dip = dip_selection(dip_freqs[:, t], s_dip_vec, h_vec)

            auto_final = auto + m_ad*(dip - auto)
            dip_final = dip + m_da*(auto - dip)

            auto_freqs[:, t+1] = rng_auto.binomial(4*N, auto_final)/(4*N)
            dip_freqs[:, t+1] = rng_dip.binomial(2*N, dip_final)/(2*N)
        else:
            auto = auto_selection(auto_freqs, s1_vec, s2_vec, s3_vec, s4_vec)
            dip = dip_selection(dip_freqs, s_dip_vec, h_vec)

            auto_final = auto + m_ad*(dip - auto)
            dip_final = dip + m_da*(auto - dip)

            auto_freqs = rng_auto.binomial(4*N, auto_final)/(4*N)
            dip_freqs = rng_dip.binomial(2*N, dip_final)/(2*N)


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
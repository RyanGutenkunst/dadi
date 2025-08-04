import numpy as np
from scipy.special import hyp2f1, gamma
from scipy.optimize import fsolve

# Set of functions for testing the accuracy of the dadi results


# These first two are for single populations (auto or dip) under neutrality
def kimura_neutral_auto(x, t, p0, N, max_terms=1000):
    """
    Calculate the probability density of allele frequency x at time t
    using Kimura's 1964 equation 4.10, modified for autotetraploids by 
    replacing N in Kimura's soln. with 2N here.
    
    Parameters:
    x : float or array, allele frequency (0 < x < 1)
    t : float, time in generations
    p0 : float, initial allele frequency
    N : int, effective population size 
    max_terms : int, maximum number of terms in the series
    
    Returns:
    float or array, probability density
    """
    
    # Ensure x is an array for vectorized computation
    x = np.asarray(x)
    
    # Initialize the sum
    density = np.zeros_like(x)
    
    # Series summation
    for n in range(1, max_terms + 1):
        # Eigenvalue for nth term; this is 8*N because of the smaller variance for autos
        lambda_n = n * (n + 1) / (8 * N)
        
        coeff = n*(n+1)*(2*n+1)
        p_hypgeom = hyp2f1(1-n, n+2, 2, p0)
        x_hypgeom = hyp2f1(1-n, n+2, 2, x)
    
        # Term contribution
        term = p0*(1-p0) * coeff * p_hypgeom * x_hypgeom * np.exp(-lambda_n * t)

        # Add to sum
        density += term
    
    return density

def kimura_neutral_dip(x, t, p0, N, max_terms=1000):
    """
    Calculate the probability density of allele frequency x at time t
    using Kimura's 1964 equation 4.10.
    
    Parameters:
    x : float or array, allele frequency (0 < x < 1)
    t : float, time in generations
    p0 : float, initial allele frequency
    N : int, effective population size 
    max_terms : int, maximum number of terms in the series
    
    Returns:
    float or array, probability density
    """
    
    # Ensure x is an array for vectorized computation
    x = np.asarray(x)
    
    # Initialize the sum
    density = np.zeros_like(x)
    
    # Series summation
    for n in range(1, max_terms + 1):
        # Eigenvalue for nth term
        lambda_n = n * (n + 1) / (4 * N)
        
        coeff = n*(n+1)*(2*n+1)
        p_hypgeom = hyp2f1(1-n, n+2, 2, p0)
        x_hypgeom = hyp2f1(1-n, n+2, 2, x)

        # Term contribution
        term = p0*(1-p0) * coeff * p_hypgeom * x_hypgeom * np.exp(-lambda_n * t)

        # Add to sum
        density += term
    
    return density
    

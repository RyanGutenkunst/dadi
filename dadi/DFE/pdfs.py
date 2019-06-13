import numpy as np
import scipy.stats.distributions as ssd

def exponential(xx, params):
    """
    params: [scale]
    """
    return ssd.expon.pdf(xx, scale=params[0])

def lognormal(xx, params):
    """
    params: [mu, sigma] (in log scale)
    """
    mu, sigma = params
    return ssd.lognorm.pdf(xx, sigma, scale=np.exp(mu))

def gamma(xx, params):
    """
    params: [alpha, beta] = [shape, scale]
    """
    alpha, beta = params
    return ssd.gamma.pdf(xx, alpha, scale=beta)


def beta(xx, params):
    """
    params: [alpha, beta]
    """
    alpha, beta = params
    return ssd.beta.pdf(xx, alpha, beta)

def normal(mgamma, mu, sigma):
    """
    params: [mu, sigma]
    """
    return ssd.norm.pdf(xx, loc=mu, scale=sigma)
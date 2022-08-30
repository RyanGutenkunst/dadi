"""
Probability density functions for defining DFEs.
"""
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

def normal(xx, mu, sigma):
    """
    params: [mu, sigma]
    """
    return ssd.norm.pdf(xx, loc=mu, scale=sigma)

from dadi.DFE.PDFs_c import biv_lognormal, biv_ind_gamma
#import dadi.DFE.PDFs_cython as PDFs_cython
## The Cython functions need correct dimension and type array inputs,
## so we need to be careful here.
#def biv_lognormal(xx, yy, params):
#    xx = np.asarray(np.atleast_1d(xx), dtype=np.double)
#    yy = np.asarray(np.atleast_1d(yy), dtype=np.double)
#    params = np.asarray(np.atleast_1d(params), dtype=np.double)
#    return np.squeeze(PDFs_cython.biv_lognormal(xx, yy, params))
#def biv_indgamma(xx, yy, params):
#    xx = np.asarray(np.atleast_1d(xx), dtype=np.double)
#    yy = np.asarray(np.atleast_1d(yy), dtype=np.double)
#    params = np.asarray(np.atleast_1d(params), dtype=np.double)
#    return np.squeeze(PDFs_cython.biv_indgamma(xx, yy, params))

# Note: This method has been deprecated in favor of the much faster C version
# defined in PDFS_c
def biv_lognormal_py(xx, yy, params):
    """
    Bivariate lognormal pdf

    xx: x-coordinates at which to evaluate.
    yy: y-coordinates at which to evaluate.
    params: Input parameters. If len(params) == 3, then params = (mu,sigma,rho)
            and mu and sigma are assumed to be equal in the two dimensions. If
            len(params) == 5, then params = (mu1,mu2,sigma1,sigma2,rho)
    """
    # The atleast_1d calls here and the squeeze at the end enable this to work
    # for scalar and array arguments.
    xx = np.atleast_1d(xx)
    yy = np.atleast_1d(yy)
    if len(params) == 3:
        mu, sigma, rho = params
        mu1 = mu2 = mu
        sigma1 = sigma2 = sigma
    elif len(params) == 5:
        mu1, mu2, sigma1, sigma2, rho = params
    else:
        raise ValueError('Parameter array for bivariate lognormal must have '
                         'length 3 or 5.')
    delx = (np.log(xx[:,np.newaxis]) - mu1)/sigma1
    dely = (np.log(yy[np.newaxis,:]) - mu2)/sigma2
    norm = 2*np.pi * sigma1*sigma2 * np.sqrt(1.-rho**2) * np.outer(xx,yy)
    q = (delx**2 - 2.*rho*delx*dely + dely**2)/(1.-rho**2)

    return np.squeeze(np.exp(-q/2.)/norm)

# Note: This method has been deprecated in favor of the much faster C version
# defined in PDFS_c
def biv_ind_gamma_py(xx, yy, params):
    """
    Bivariate independent gamma pdf

    xx: x-coordinates at which to evaluate.
    yy: y-coordinates at which to evaluate.
    params: Input parameters. If len(params) == 2, then params = (alpha,beta)
            and alpha and beta are assumed to be equal in the two dimensions. If
            len(params) == 4, then params = (alpha1,alpha2,beta1,beta2).
            If len(params) in [3,5], then the last parameter is ignored (for
            compatibility with mixture model functions).

    For extensions to correlated gamma distributions, see
    Kibble (1941) and Smith and Adelfang (1981).
    """
    xx = np.atleast_1d(xx)
    yy = np.atleast_1d(yy)
    if len(params) in [2,3]:
        alpha1 = alpha2 = params[0]
        beta1 = beta2 = params[1]
    elif len(params) in [4,5]:
        alpha1, alpha2, beta1, beta2 = params[:4]
    else:
        raise ValueError('Parameter array for bivariate independent gamma must have '
                         'length 2,3,4, or 5.')

    xmarg = ssd.gamma.pdf(xx, alpha1, scale=beta1)
    ymarg = ssd.gamma.pdf(yy, alpha2, scale=beta2)
    return np.squeeze(np.outer(xmarg, ymarg))
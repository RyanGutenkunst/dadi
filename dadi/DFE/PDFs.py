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

def biv_lognormal(xx, yy, params):
    """
    Bivariate lognormal pdf

    xx: x-coordinates at which to evaluate.
    yy: y-coordinates at which to evaluate.
    params: Input parameters. If len(params) == 3, then params = (mu,sigma,rho)
            and mu and sigma are assumed to be equal in the two dimensions. If
            len(params) == 5, then params = (mu1,mu2,sigma1,sigma2,rho)
    """
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

    return np.exp(-q/2.)/norm

def biv_ind_gamma(xx, yy, params):
    """
    Bivariate independent gamma pdf

    xx: x-coordinates at which to evaluate.
    yy: y-coordinates at which to evaluate.
    params: Input parameters. If len(params) == 2, then params = (alpha,beta)
            and alpha and beta are assumed to be equal in the two dimensions. If
            len(params) == 4, then params = (alpha1,alpha2,beta1,beta2)

    For extensions to correlated gamma distributions, see
    Kibble (1941) and Smith and Adelfang (1981).
    """
    if len(params) == 2:
        alpha1 = alpha2 = params[0]
        beta1 = beta2 = params[1]
    elif len(params) == 4:
        alpha1, alpha2, beta1, beta2 = params
    else:
        raise ValueError('Parameter array for bivariate independent gamma must have '
                         'length 2 or 4.')

    xmarg = scipy.stats.distributions.gamma.pdf(xx, alpha1, scale=beta1)
    ymarg = scipy.stats.distributions.gamma.pdf(yy, alpha2, scale=beta2)
    return np.outer(xmarg, ymarg)

import logging
logger = logging.getLogger('Hessian')

import numpy

def hessian_elem(func, f0, params, i, j, hi, hj, args=()):
    """
    Second partial derivative of func w.r.t. parameters i and j

    f0: Value of the function at params
    eps: Stepsize to use
    """
    origPi, origPj = params[i], params[j]

    if i == j:
        params[i] = origPi + hi
        fp = func(params, *args)

        params[i] = origPi - hi
        fm = func(params, *args)

        element = (fp - 2*f0 + fm)/hi**2
    else:
        # f(xi + hi, xj + h)
        params[i] = origPi + hi
        params[j] = origPj + hj
        fpp = func(params, *args)

        # f(xi + hi, xj - hj)
        params[i] = origPi + hi
        params[j] = origPj - hj
        fpm = func(params, *args)

        # f(xi - hi, xj + hj)
        params[i] = origPi - hi
        params[j] = origPj + hj
        fmp = func(params, *args)

        # f(xi - hi, xj - hj)
        params[i] = origPi - hi
        params[j] = origPj - hj
        fmm = func(params, *args)

        element = (fpp - fpm - fmp + fmm)/(4 * hi * hj)

    params[i], params[j] = origPi, origPj

    return element

def hessian(func, params, eps, args=()):
    """
    Matrix of second partial derivatives of func. Hij = dfunc/(dp_i dp_j).

    func: Function to work with. This function should take params as its first
          argument, and then any number of *args. It will often be convenient
          to use lambda to define the appropriate function.
    params: Parameter values to take derivatives about.
    eps: Stepsize to use. This can be a vector, giving the size for each param.
    args: Optional additional arguments to pass to func.
    """
    params = numpy.asarray(params)
    # Convert eps from (possibly) a constant to a vector.
    eps = eps + numpy.zeros(len(params))
    # compute cost at f(x)
    f0 = func(params, *args)

    hess = numpy.zeros((len(params), len(params)))
    # compute all (numParams*(numParams + 1))/2 unique hessian elements
    for i in range(len(params)):
        for j in range(i, len(params)):
            hess[i][j] = hessian_elem(func, f0, params, i, j,
                                      eps[i], eps[j], args)
            hess[j][i] = hess[i][j]

    return hess

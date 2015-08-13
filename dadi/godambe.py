import numpy

from dadi import Inference
from dadi.Spectrum_mod import Spectrum

def hessian_elem(func, f0, p0, ii, jj, eps):
    """
    Calculate element [ii][jj] of the Hessian matrix, a matrix
    of partial second derivatives w.r.t. to parameters ii and jj
        
    func: Model function
    f0: Evaluation of func at p0
    p0: Parameters for func
    eps: List of absolute step sizes to use for each parameter when taking
         finite differences.
    """
    # Note that we need to specify dtype=float, to avoid this being an integer
    # array which will silently fail when adding fractional eps.
    pwork = numpy.array(p0, copy=True, dtype=float)
    if ii == jj:
        if pwork[ii] != 0:
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork)
            
            pwork[ii] = p0[ii] - eps[ii]
            fm = func(pwork)
            
            element = (fp - 2*f0 + fm)/eps[ii]**2
        if pwork[ii] == 0:
            pwork[ii] = p0[ii] + 2*eps[ii]
            fp = func(pwork)
            
            pwork[ii] = p0[ii] + eps[ii]
            fm = func(pwork)

            element = (fp - 2*fm + f0)/eps[ii]**2
    else:
        if pwork[ii] != 0 and pwork[jj] != 0:
            # f(xi + hi, xj + h)
            pwork[ii] = p0[ii] + eps[ii]
            pwork[jj] = p0[jj] + eps[jj]
            fpp = func(pwork)
            
            # f(xi + hi, xj - hj)
            pwork[ii] = p0[ii] + eps[ii]
            pwork[jj] = p0[jj] - eps[jj]
            fpm = func(pwork)
            
            # f(xi - hi, xj + hj)
            pwork[ii] = p0[ii] - eps[ii]
            pwork[jj] = p0[jj] + eps[jj]
            fmp = func(pwork)
            
            # f(xi - hi, xj - hj)
            pwork[ii] = p0[ii] - eps[ii]
            pwork[jj] = p0[jj] - eps[jj]
            fmm = func(pwork)

            element = (fpp - fpm - fmp + fmm)/(4 * eps[ii]*eps[jj])
        else:
            # f(xi + hi, xj + h)
            pwork[ii] = p0[ii] + eps[ii]
            pwork[jj] = p0[jj] + eps[jj]
            fpp = func(pwork)
            
            # f(xi + hi, xj)
            pwork[ii] = p0[ii] + eps[ii]
            pwork[jj] = p0[jj]
            fpm = func(pwork)
            
            # f(xi, xj + hj)
            pwork[ii] = p0[ii]
            pwork[jj] = p0[jj] + eps[jj]
            fmp = func(pwork)
            
            element = (fpp - fpm - fmp + f0)/(eps[ii]*eps[jj])
    return element

def get_hess(func, p0, eps):
    """
    Calculate Hessian matrix, a matrix of partial second derivatives. Hij = dfunc/(dp_i dp_j)
    
    func: Model function
    p0: Parameter values to take derivative around
    eps: Fractional stepsize to use when taking finite-difference derivatives
    """
    # Calculate step sizes for finite-differences.
    eps_in = eps
    eps = numpy.empty([len(p0)])
    for i, pval in enumerate(p0):
        if pval != 0:
            eps[i] = eps_in*pval
        else:
            # Account for zero parameters
            eps[i] = eps_in

    f0 = func(p0)
    hess = numpy.empty((len(p0), len(p0)))
    for ii in range(len(p0)):
        for jj in range(ii, len(p0)):
            hess[ii][jj] = hessian_elem(func, f0, p0, ii, jj, eps)
            hess[jj][ii] = hess[ii][jj]
    return hess

def get_grad(func, p0, eps):
    """
    Calculate gradient vector
    
    func: Model function
    p0: Parameters for func
    eps: Fractional stepsize to use when taking finite-difference derivatives
    """
    f0 = func(p0)
    # Calculate step sizes for finite-differences.
    eps_in = eps
    eps = numpy.empty([len(p0)])
    for i, pval in enumerate(p0):
        if pval != 0:
            eps[i] = eps_in*pval
        else:
            # Account for parameters equal to zero
            eps[i] = eps_in

    grad = numpy.empty([len(p0), 1])
    for ii in range(len(p0)):
        pwork = numpy.array(p0, copy=True, dtype=True)
        if pwork[ii] != 0:
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork)
            pwork[ii] = p0[ii] - eps[ii]
            fm = func(pwork)
            grad[ii] = (fp - fm)/(2*eps[ii])
        if pwork[ii] == 0:
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork)
            pwork[ii] = p0[ii]
            fm = func(pwork)
            grad[ii] = (fp - fm)/(eps[ii])
    return grad

def get_godambe(func_ex, ns, grid_pts, all_boot, p0, data, eps, log=True):
    """
    Godambe information and Hessian matrices

    NOTE: Assumes that last parameter in p0 is theta.

    func_ex: Model function
    ns: Number of samples in each population
    grid_pts: Number of grid points to evaluate the model function
    all_boot: List of bootstrap frequency spectra
    p0: Best-fit parameters for func_ex. 
        parameter in p0 is theta.
    data: Original data frequency spectrum
    eps: Fractional stepsize to use when taking finite-difference derivatives
    log: If True, calculate derivatives in terms of log-parameters
    """
    J = numpy.zeros((len(p0), len(p0)))
    ns = data.sample_sizes
    grid_pts = [ns[0]+20, ns[0]+30, ns[0]+40]
    func = lambda params: Inference.ll(params[-1]*func_ex(params[:-1], ns, grid_pts), data)
    hess = -get_hess(func, p0, eps)
    if log:
        func = lambda params: Inference.ll(numpy.exp(params[-1])*func_ex(numpy.exp(params[:-1]), ns, grid_pts), data)
        hess = -get_hess(func, numpy.log(p0), eps)
    for ii, boot in enumerate(all_boot):
        boot = Spectrum(boot)
        if not log:
            func = lambda params: Inference.ll(params[-1]*func_ex(params[:-1], ns, grid_pts), boot)
            grad_temp = get_grad(func, p0, eps)
        if log:
            func = lambda params: Inference.ll(numpy.exp(params[-1])*func_ex(numpy.exp(params[:-1]), ns, grid_pts), boot)
            grad_temp = get_grad(func, numpy.log(p0), eps)
        J_temp = numpy.outer(grad_temp, grad_temp)
        J = J + J_temp
    J = J/len(all_boot)
    J_inv = numpy.linalg.inv(J)
    # G = H*J^-1*H
    godambe = numpy.dot(numpy.dot(hess, J_inv), hess)
    return godambe, hess

def uncert(func_ex, all_boot, p0, data, eps, log=True):
    """
    Parameter uncertainties from Godambe Information Matrix

    Returns standard deviations of parameter values.

    NOTE: Assumes that last parameter in p0 is theta.

    func_ex: Model function
    all_boot: List of bootstrap frequency spectra
    p0: Best-fit parameters for func_ex
    data: Original data frequency spectrum
    eps: Fractional stepsize to use when taking finite-difference derivatives
    log: If True, assume log-normal distribution of parameters. Returned values 
         are then the standard deviations of the *logs* of the parameter values,
         which can be interpreted as relative parameter uncertainties.
    """
    godambe, hess = get_godambe(func_ex, ns, grid_pts, all_boot, p0, data, eps, log)
    return numpy.sqrt(numpy.diag(numpy.linalg.inv((godambe))))

def LRT(func_ex, all_boot, p0, data, eps, diff):
    """
    First-order moment matching adjustment factor for likelihood ratio test

    func_ex: Model function for complex model
    all_boot: List of bootstrap frequency spectra
    p0: Best-fit parameters for the simple model, with nested parameter explicity defined
    Although equal to values for simple model, should be in a list form that can be 
    taken in by the complex model you'd like to evaluate
    data: Original data frequency spectrum
    eps: Fractional stepsize to use when taking finite-difference derivatives
    diff: List of positions of nested parameters in complex model parameter list
    """
    ns = data.sample_sizes
    grid_pts = [ns[0]+20, ns[0]+30, ns[0]+40]
    func = lambda param: Inference.ll_multinom(func_ex([param[diff.index(i)] if i in diff else p0[i] for i in range(len(p0))], ns, grid_pts), data)
    H = -get_hess(func, [p0[i] for i in diff], eps)
    J_boot = numpy.zeros([len(diff), len(diff)])
    J_array = []
    for i in range(0, len(all_boot)):
        boot = Spectrum(all_boot[i])
        func = lambda param: Inference.ll_multinom(func_ex([param[diff.index(i)] if i in diff else p0[i] for i in range(len(p0))], ns, grid_pts), boot)
        cU_theta = get_grad(func, [p0[i] for i in diff], eps)
        J_theta = numpy.outer(cU_theta, cU_theta)
        J_boot = J_boot + J_theta
        J_array.append(J_theta)
    J = J_boot/len(all_boot)
    adjust = len(diff)/numpy.trace(numpy.dot(J, numpy.linalg.inv(H)))
    return adjust

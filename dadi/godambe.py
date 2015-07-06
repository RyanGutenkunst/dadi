import numpy

from dadi import Inference
from dadi.Spectrum_mod import Spectrum

def hessian_elem(func, f0, p0, ii, jj, eps):
    pwork = numpy.array(p0, copy=True)
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
        if pwork[ii] == 0 or pwork[jj] == 0:
            # f(xi + hi, xj + h)
            pwork[ii] = p0[ii] + eps[ii]
            pwork[jj] = p0[jj] + eps[jj]
            fpp = func(pwork)
            
            # f(xi + hi, xj)
            pwork[ii] = p0[ii] + eps[ii]
            fpm = func(pwork)
            
            # f(xi, xj + hj)
            pwork[jj] = p0[jj] + eps[jj]
            fmp = func(pwork)
            
            element = (fpp - fpm - fmp + f0)/(4 * eps[ii]*eps[jj])
    return element

def get_hess(func, p0, eps):
    f0 = func(p0)
    epp = eps
    eps = numpy.empty([len(p0), 1])
    for i in range(len(p0)):
        if p0[i] != 0:
            eps[i] = epp*p0[i]
        else:
            eps[i] = epp
    hess = numpy.empty((len(p0), len(p0)))
    for ii in range(len(p0)):
        for jj in range(ii, len(p0)):
            hess[ii][jj] = hessian_elem(func, f0, p0, ii, jj, eps)
            hess[jj][ii] = hess[ii][jj]
    return hess

def get_grad(func, p0, eps):
    f0 = func(p0)
    epp = eps
    eps = numpy.empty([len(p0)])
    for i in range(len(p0)):
        if p0[i] != 0:
            eps[i] = epp*p0[i]
        if p0[i] == 0:
            eps[i] = epp
    grad = numpy.empty([len(p0), 1])
    for ii in range(len(p0)):
        pwork = numpy.array(p0, copy=True)
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

def get_godambe(func_ex, all_boot, p0, data, eps, log=True):
    #assume that last element of p0 is theta, and the remaining elements are the demographic model parameters
    #log dictates whether parameters are regular are logarithmic
    #func_ex is dadi extrapolated function, all_boot is bootstrapped data, p0 is best_fit params for data/func_ex combination
    J = numpy.zeros((len(p0), len(p0)))
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
    godambe, hess = get_godambe(func_ex, all_boot, p0, data, eps, log)
    return numpy.sqrt(numpy.diag(numpy.linalg.inv((godambe))))

def LRT(func_ex, all_boot, p0, data, eps, diff=1)
    #p0 is the best fit parameters in the simple model with the complex model parameter(s) as the first diff number of parameters in p0
    adjust = 0.0
    func = lambda param: Inference.ll_multinom(func_ex([param[:diff]+p0[diff:]], ns, grid_pts), data)
    H = -get_hess_log(func, p0[:diff], eps)
    J_boot = numpy.zeros([diff, diff])
    J_array = []
    for i in range(0, len(all_boot)):
        boot = Spectrum(all_boot[i])
        func = lambda param: Inference.ll_multinom(func_ex([param[:diff]+p0[diff:]], ns, grid_pts), boot)
        cU_theta = get_grad_log(func, p0[:diff], eps)
        J_theta = numpy.outer(cU_theta, cU_theta)
        J_boot = J_boot + J_theta
        J_array.append(J_theta)
    J = J_boot/len(all_boot)
    adjust = diff/numpy.trace(numpy.dot(J, numpy.linalg.inv(H)))
    return adjust

import numpy as np, dadi
import numpy
import scipy, math
from scipy.special import gammaln
import scipy.optimize
from numpy import logical_and, logical_not
from . import numerics
import os,sys


###
### This is almost entirely adapted from the dadi.Inference functions
###
#: Counts calls to object_func
_counter = 0
#: Returned when object_func is passed out-of-bounds params or gets a NaN ll.
_out_of_bounds_val = -1e8

def ll(model, data):
    """
    The log-likelihood of the data given the model linkage frequency spectrum
    """
    ll_arr = ll_per_bin(model,data)
    return ll_arr.sum()

def ll_per_bin(model, data):
    """
    Poisson log-likelihood of each entry in the data given the model sfs
    """
    result = -model.data + data.data*np.log(model) - gammaln(data + 1.)
    return result

def ll_multinom(model,data):
    """
    LL of the data given the model, with optimal rescaling
    """
    ll_arr = ll_multinom_per_bin(model,data)
    return ll_arr.sum()

def ll_multinom_per_bin(model,data):
    """
    Multinomial log-likelihood of each entry in the data given the model
    """
    theta_opt = optimal_sfs_scaling(model,data)
    return ll_per_bin(theta_opt*model,data)

def optimal_sfs_scaling(model,data):
    """
    Optimal multiplicative scaling factor between model and data
    """
    model, data = dadi.Numerics.intersect_masks(model,data)
    return data.sum()/model.sum()

def optimally_scaled_sfs(model,data):
    """
    Optimally scaled model to data
    """
    return optimal_sfs_scaling(model,data) * model


def ll_over_rho_bins(model_list,data_list):
    """
    The log-likelihood of the binned data given the model spectra for the same bins
    Input list of models for rho bins, and list of data for rho bins
    """
    if len(model_list) != len(data_list):
        print('model list and data list must be of same length')
        return 0
    LL = 0
    for ii in range(len(model_list)):
        LL += ll(model_list[ii],data_list[ii])
    return LL


def ll_over_rho_bins_multinom(model_list,data_list):
    """
    The log-likelihood of the binned data given the model spectra for the same bins
    Input list of models for rho bins, and list of data for rho bins
    """
    if len(model_list) != len(data_list):
        print('model list and data list must be of same length')
        return 0
    LL = 0
    for ii in range(len(model_list)):
        LL += ll_multinom(model_list[ii],data_list[ii])
    return LL


### what if we want multinom log likelihoods, but the relative theta for each bin is known?


def _project_params_up(pin, fixed_params):
    """
    Fold fixed parameters into pin.
    """
    if fixed_params is None:
        return pin

    if numpy.isscalar(pin):
        pin = [pin]

    pout = numpy.zeros(len(fixed_params))
    orig_ii = 0
    for out_ii, val in enumerate(fixed_params):
        if val is None:
            pout[out_ii] = pin[orig_ii]
            orig_ii += 1
        else:
            pout[out_ii] = fixed_params[out_ii]
    return pout

def _project_params_down(pin, fixed_params):
    """
    Eliminate fixed parameters from pin.
    """
    if fixed_params is None:
        return pin

    if len(pin) != len(fixed_params):
        raise ValueError('fixed_params list must have same length as input '
                         'parameter array.')

    pout = []
    for ii, (curr_val,fixed_val) in enumerate(zip(pin, fixed_params)):
        if fixed_val is None:
            pout.append(curr_val)

    return numpy.array(pout)

def _object_func(params, data_list, model_func, pts, dts, rhos=[0],
                 lower_bound=None, upper_bound=None, 
                 verbose=0, multinom=True, flush_delay=0,
                 func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                 output_stream=sys.stdout, store_thetas=False):
    """
    Objective function for optimization.
    """
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    # Check our parameter bounds
    if lower_bound is not None:
        for pval,bound in zip(params_up, lower_bound):
            if bound is not None and pval < bound:
                return -_out_of_bounds_val/ll_scale
    if upper_bound is not None:
        for pval,bound in zip(params_up, upper_bound):
            if bound is not None and pval > bound:
                return -_out_of_bounds_val/ll_scale

#    ns = data.sample_sizes 
    ns = (len(data_list[0])-1,) ###XXX changed by AR
    if ns == (43499,):
        ns = (20,)
    
    all_args = [params_up, ns, pts, dts]
    # Pass the pts argument via keyword, but don't alter the passed-in 
    # func_kwargs
    func_kwargs = func_kwargs.copy()
    func_kwargs['rhos'] = rhos
    
    model_list = model_func(*all_args, **func_kwargs) ###XXX: need a model that constructs the model_list
                                                      #       which stores the expected two-locus fs for each
                                                      #       bin center
                                                      
    if multinom:
        result = ll_over_rho_bins_multinom(model_list, data_list)
    else:
        result = ll_over_rho_bins(model_list, data_list)

    if store_thetas:
        global _theta_store
        _theta_store[tuple(params)] = optimal_sfs_scaling(sfs, data)

    # Bad result
    if numpy.isnan(result):
        result = _out_of_bounds_val

    if (verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params_up]))
        output_stream.write('%-8i, %-12g, %s%s' % (_counter, result, param_str,
                                                   os.linesep))
        dadi.Misc.delayed_flush(delay=flush_delay)

    return -result/ll_scale

def _object_func_log(log_params, *args, **kwargs):
    """
    Objective function for optimization in log(params).
    """
    return _object_func(numpy.exp(log_params), *args, **kwargs)




def optimize_log_lbfgsb(p0, data_list, model_func, pts, dts, rhos=[0],
                        lower_bound=None, upper_bound=None,
                        verbose=0, flush_delay=0.5, epsilon=1e-3, 
                        pgtol=1e-5, multinom=True, maxiter=1e5, 
                        full_output=False,
                        func_args=[], func_kwargs={}, fixed_params=None, 
                        ll_scale=1, output_file=None):
    """
    Optimize log(params) to fit model to data using the L-BFGS-B method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum. This method is
    better than optimize_log if the optimum lies at one or more of the
    parameter bounds. However, if your optimum is not on the bounds, this
    method may be much slower.

    Because this works in log(params), it cannot explore values of params < 0.
    It should also perform better when parameters range over scales.

    p0: Initial parameters.
    data: Spectrum with data.
    model_function: Function to evaluate model spectrum. Should take arguments
                    (params, (n1,n2...), pts)
    lower_bound: Lower bound on parameter values. If not None, must be of same
                 length as p0. A parameter can be declared unbound by assigning
                 a bound of None.
    upper_bound: Upper bound on parameter values. If not None, must be of same
                 length as p0. A parameter can be declared unbound by assigning
                 a bound of None.
    verbose: If > 0, print optimization status every <verbose> steps.
    output_file: Stream verbose output into this filename. If None, stream to
                 standard out.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    epsilon: Step-size to use for finite-difference derivatives.
    pgtol: Convergence criterion for optimization. For more info, 
          see help(scipy.optimize.fmin_l_bfgs_b)
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    maxiter: Maximum algorithm iterations to run.
    full_output: If True, return full outputs as in described in 
                 help(scipy.optimize.fmin_bfgs)
    func_args: Additional arguments to model_func. It is assumed that 
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    func_kwargs: Additional keyword arguments to model_func.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change 
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    (See help(dadi.Inference.optimize_log for examples of func_args and 
     fixed_params usage.)
    ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
              too large. (This appears to be a flaw in the scipy
              implementation.) To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.

    The L-BFGS-B method was developed by Ciyou Zhu, Richard Byrd, and Jorge
    Nocedal. The algorithm is described in:
      * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
        Constrained Optimization, (1995), SIAM Journal on Scientific and
        Statistical Computing , 16, 5, pp. 1190-1208.
      * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
        FORTRAN routines for large scale bound constrained optimization (1997),
        ACM Transactions on Mathematical Software, Vol 23, Num. 4, pp. 550-560.
    
    """
    if output_file:
        output_stream = open(output_file, 'w')
    else:
        output_stream = sys.stdout

    args = (data_list, model_func, pts, dts, rhos,
            None, None, verbose,
            multinom, flush_delay, func_args, func_kwargs, fixed_params, 
            ll_scale, output_stream)

    # Make bounds list. For this method it needs to be in terms of log params.
    if lower_bound is None:
        lower_bound = [None] * len(p0)
    else:
        lower_bound = numpy.log(lower_bound)
        lower_bound[numpy.isnan(lower_bound)] = None
    lower_bound = _project_params_down(lower_bound, fixed_params)
    if upper_bound is None:
        upper_bound = [None] * len(p0)
    else:
        upper_bound = numpy.log(upper_bound)
        upper_bound[numpy.isnan(upper_bound)] = None
    upper_bound = _project_params_down(upper_bound, fixed_params)
    bounds = list(zip(lower_bound,upper_bound))

    p0 = _project_params_down(p0, fixed_params)

    outputs = scipy.optimize.fmin_l_bfgs_b(_object_func_log, 
                                           numpy.log(p0), bounds = bounds,
                                           epsilon=epsilon, args = args,
                                           iprint = -1, pgtol=pgtol,
                                           maxfun=maxiter, approx_grad=True)
    xopt, fopt, info_dict = outputs

    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, info_dict




def optimize_log_fmin(p0, data_list, model_func, pts, dts, rhos=[0],
                        lower_bound=None, upper_bound=None,
                        verbose=0, flush_delay=0.5, epsilon=1e-3, 
                        pgtol=1e-5, multinom=True, maxiter=1e5, 
                        full_output=False,
                        func_args=[], func_kwargs={}, fixed_params=None, 
                        ll_scale=1, output_file=None):
    if output_file:
        output_stream = open(output_file, 'w')
    else:
        output_stream = sys.stdout

    args = (data_list, model_func, pts, dts, rhos, lower_bound, upper_bound, verbose,
            multinom, flush_delay, func_args, func_kwargs, fixed_params, 1.0,
            output_stream)

    p0 = _project_params_down(p0, fixed_params)

    outputs = scipy.optimize.fmin(_object_func_log, numpy.log(p0), args = args,
                                  disp=False, maxiter=maxiter, full_output=True)
    
    xopt, fopt, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, iter, funcalls, warnflag 


# interpolation

def optimize_log_fmin_interp(p0, data_list, model_func, pts, dts, model_rhos=[0], data_rhos=[0],
                        lower_bound=None, upper_bound=None,
                        verbose=0, flush_delay=0.5, epsilon=1e-3, 
                        pgtol=1e-5, multinom=True, maxiter=1e5, 
                        full_output=False,
                        func_args=[], func_kwargs={}, fixed_params=None, 
                        ll_scale=1, output_file=None,
                        sorted_keys=None):
    """
    sorted_keys: 
    """
    if output_file:
        output_stream = open(output_file, 'w')
    else:
        output_stream = sys.stdout

    args = (data_list, model_func, pts, dts, model_rhos, data_rhos, lower_bound, upper_bound, verbose,
            multinom, flush_delay, func_args, func_kwargs, fixed_params, 1.0,
            output_stream)

    p0 = _project_params_down(p0, fixed_params)

    outputs = scipy.optimize.fmin(_object_func_log_interp, numpy.log(p0), args = args,
                                  disp=False, maxiter=maxiter, full_output=True)
    
    xopt, fopt, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, iter, funcalls, warnflag 

def _object_func_interp(params, data_list, model_func, pts, dts, model_rhos=[0], data_rhos = [0],
                 lower_bound=None, upper_bound=None, 
                 verbose=0, multinom=True, flush_delay=0,
                 func_args=[], func_kwargs={}, fixed_params=None, ll_scale=1,
                 output_stream=sys.stdout, store_thetas=False):
    """
    Objective function for optimization.
    """
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    # Check our parameter bounds
    if lower_bound is not None:
        for pval,bound in zip(params_up, lower_bound):
            if bound is not None and pval < bound:
                return -_out_of_bounds_val/ll_scale
    if upper_bound is not None:
        for pval,bound in zip(params_up, upper_bound):
            if bound is not None and pval > bound:
                return -_out_of_bounds_val/ll_scale

#    ns = data.sample_sizes 
    ns = (len(data_list[0])-1,) ###XXX changed by AR
    if ns == (43499,):
        ns = (20,)
    
    all_args = [params_up, ns, pts, dts]
    # Pass the pts argument via keyword, but don't alter the passed-in 
    # func_kwargs
    func_kwargs = func_kwargs.copy()
    func_kwargs['model_rhos'] = model_rhos
    func_kwargs['data_rhos'] = data_rhos
    
    model_list = model_func(*all_args, **func_kwargs) ###XXX: need a model that constructs the model_list
                                                      #       which stores the expected two-locus fs for each
                                                      #       bin center
                                                      
    if multinom:
        result = ll_over_rho_bins_multinom(model_list, data_list)
    else:
        result = ll_over_rho_bins(model_list, data_list)

    if store_thetas:
        global _theta_store
        _theta_store[tuple(params)] = optimal_sfs_scaling(sfs, data)

    # Bad result
    if numpy.isnan(result):
        result = _out_of_bounds_val

    if (verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params_up]))
        output_stream.write('%-8i, %-12g, %s%s' % (_counter, result, param_str,
                                                   os.linesep))
        dadi.Misc.delayed_flush(delay=flush_delay)

    return -result/ll_scale

def _object_func_log_interp(log_params, *args, **kwargs):
    """
    Objective function for optimization in log(params).
    """
    return _object_func_interp(numpy.exp(log_params), *args, **kwargs)


import numpy as np
from dadi.Inference import _project_params_down, _project_params_up, _object_func
import nlopt

def opt(p0, data, model_func, pts, multinom=True,
        lower_bound=None, upper_bound=None, fixed_params=None,
        algorithm=nlopt.LN_BOBYQA, stopval=0,
        ftol_abs=1e-6, xtol_abs=1e-6,
        maxeval=int(1e9), maxtime=np.inf,
        local_optimizer=nlopt.LN_BOBYQA,
        verbose=1,
        func_args=[], func_kwargs={},
        ):
    """
    p0: Initial parameters.
    data: Spectrum with data.
    model_func: Function to evaluate model spectrum. Should take arguments
                (params, (n1,n2...), pts)
    pts: Grid points list for evaluating likelihoods
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    lower_bound: Lower bound on parameter values. If not None, must be of same
                 length as p0.
    upper_bound: Upper bound on parameter values. If not None, must be of same
                 length as p0.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change 
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    algorithm: Optimization algorithm to employ. See
               https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
               for possibilities.
    stopval: Algorithm with stop when a log-likelihood of at least stopval
             is found.
    ftol_abs: Absolute tolerance on log-likelihood
    xtol_abs: Absolute tolerance in parameter values
    maxeval: Maximum number of function evaluations
    maxtime: Maximum optimization time, in seconds
    verbose: If > 0, print optimization status every <verbose> model evaluations.
    func_args: Additional arguments to model_func. It is assumed that 
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    func_kwargs: Additional keyword arguments to model_func.
    (See help(dadi.Inference.optimize_log for examples of func_args and 
     fixed_params usage.)
    """
    if lower_bound is None:
        lower_bound = [-np.inf] * len(p0)
    lower_bound = _project_params_down(lower_bound, fixed_params)
    if upper_bound is None:
        upper_bound = [np.inf] * len(p0)
    upper_bound = _project_params_down(upper_bound, fixed_params)

    p0 = _project_params_down(p0, fixed_params)

    opt = nlopt.opt(algorithm, len(p0))

    opt.set_lower_bounds(lower_bound)
    opt.set_upper_bounds(upper_bound)

    opt.set_stopval(stopval)
    opt.set_ftol_abs(ftol_abs)
    opt.set_xtol_abs(xtol_abs)
    opt.set_maxeval(maxeval)
    opt.set_maxtime(maxtime)

    # For some global optimizers, need to set local optimizer parameters.
    local_opt = nlopt.opt(local_optimizer, len(p0))
    local_opt.set_stopval(stopval)
    local_opt.set_ftol_abs(ftol_abs)
    local_opt.set_xtol_abs(xtol_abs)
    local_opt.set_maxeval(maxeval)
    local_opt.set_maxtime(maxtime)
    opt.set_local_optimizer(local_opt)

    def f(x, grad):
        if grad.size:
            raise ValueError("Cannot use optimization algorithms that require a derivative function.")
        return -_object_func(x, data, model_func, pts, 
                             verbose=verbose, multinom=multinom,
                             func_args=func_args, func_kwargs=func_kwargs, fixed_params=fixed_params)

    opt.set_max_objective(f)

    xopt = opt.optimize(p0)
    opt_val = opt.last_optimum_value()
    result = opt.last_optimize_result()

    xopt = _project_params_up(xopt, fixed_params)

    return xopt, opt_val, result

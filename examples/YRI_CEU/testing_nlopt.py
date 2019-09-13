from numpy import array
import dadi
import demographic_models

data = dadi.Spectrum.from_file('YRI_CEU.fs')
ns = data.sample_sizes
pts_l = [40,50,60]

func = demographic_models.prior_onegrow_mig

upper_bound = [100, 100, 100, 10, 3, 3]
lower_bound = [1e-2, 1e-2, 1e-2, 0, 0, 0]

# This is our initial guess for the parameters, which is somewhat arbitrary.
p0 = [2,0.1,2,1,0.2,0.2]
# Make the extrapolating version of our demographic model function.
func_ex = dadi.Numerics.make_extrap_log_func(func)

p0 = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound,
                              lower_bound=lower_bound)

real_llopt, real_popt = -1066.26, array([ 1.88012    ,  0.0725518  ,  1.75888    ,  0.931003   ,  0.362885   ,  0.112026   ])

# Testing with extremely large distance away from optimum

import dadi.NLopt_mod
import nlopt
import time
start = time.time()
popt,LLopt,result = dadi.NLopt_mod.opt(p0, data, func_ex, pts_l, 
        lower_bound=lower_bound, upper_bound=upper_bound, 
        verbose=0,
        algorithm=nlopt.LN_BOBYQA, stopval=real_llopt-10)
print(popt, LLopt)
print("Time:", time.time() - start)

try:
    start = time.time()
    popt_old = dadi.Inference.optimize_log(p0, data, func_ex, pts_l,
            lower_bound=lower_bound, upper_bound=upper_bound,
            verbose=0, stopval=real_llopt-10)
    print(popt_old)
except StopIteration as X:
    LLopt_old, popt_old = X.value
    print(popt_old, LLopt_old)
    print("Time:", time.time() - start)

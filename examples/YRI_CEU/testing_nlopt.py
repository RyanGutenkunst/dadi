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
#popt = dadi.Inference.optimize_log(p0, data, func_ex, pts_l, 
#                                   lower_bound=lower_bound,
#                                   upper_bound=upper_bound,
#                                   verbose=len(p0), maxiter=3)
import dadi.NLopt_mod
import nlopt
popt,LLopt,result = dadi.NLopt_mod.opt(p0, data, func_ex, pts_l, 
        lower_bound=lower_bound, upper_bound=upper_bound, 
        algorithm=nlopt.LN_BOBYQA)
print(popt, LLopt)
# These are the actual best-fit model parameters, which we found through
# longer optimizations and confirmed by running multiple optimizations.
# We'll work with them through the rest of this script.
# popt = [1.880, 0.0724, 1.764, 0.930, 0.363, 0.112]
# LLopt = -1066.26
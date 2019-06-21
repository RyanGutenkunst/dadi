"""
Modified from fitdadi's example file
"""
#! /usr/bin/env python

import pickle
import numpy as np
import dadi
import dadi.DFE as DFE

# Set demographic parameters and theta. This is usually inferred from
# synonymous sites. In this case, we'll be using a two-epoch model.
demog_params = [2, 0.05]
theta_ns = 4000.
ns = [250]

## Integrate over a range of gammas
#pts_l = [600, 800, 1000]
#spectra = DFE.Cache1D(demog_params, ns, DFE.DemogSelModels.two_epoch, pts_l=pts_l, 
#                      gamma_bounds=(1e-5, 500), gamma_pts=300, verbose=True,
#                      mp=True)
## The spectra can be pickled for usage later. This is especially convenient
## if the process of generating the spectra takes a long time.
#pickle.dump(spectra, open('example_spectra.bpkl','wb'))
# To load them, use this code
spectra = pickle.load(open('example_spectra.bpkl','rb'))


#load sample data
data = dadi.Spectrum.from_file('example.fs')

## Fit a DFE to the data
## Initial guess and bounds
#sel_params = [0.2, 1000.]
#lower_bound, upper_bound = [1e-3, 1e-2], [1, 50000.]
#p0 = dadi.Misc.perturb_params(sel_params, lower_bound=lower_bound,
#                              upper_bound=upper_bound)
#popt = dadi.Inference.optimize_log(p0, data, spectra.integrate, pts=None,
#                                   func_args=[DFE.PDFs.gamma, theta_ns],
#                                   lower_bound=lower_bound, upper_bound=upper_bound, 
#                                   verbose=len(sel_params), maxiter=30, multinom=False)
#
## Get expected SFS for MLE
#model_sfs = spectra.integrate(popt, None, DFE.PDFs.gamma, theta_ns, None)
#
## One possible characterization of the neutral+gamma DFE
## Written using numpy tricks to work with both scalar and array arguments
#def neugamma(xx, params):
#    pneu, alpha, beta = params
#    # Convert xx to an array
#    xx = np.atleast_1d(xx)
#    out = (1-pneu)*DFE.PDFs.gamma(xx, (alpha, beta))
#    # Assume gamma < 1e-4 is essentially neutral
#    out[np.logical_and(0 <= xx, xx < 1e-4)] += pneu/1e-4
#    # Reduce xx back to scalar if it's possible
#    return np.squeeze(out)
#
#sel_params = [0.2, 0.2, 1000.]
#lower_bound, upper_bound = [1e-3, 1e-3, 1e-2], [1, 1, 50000.]
#p0 = dadi.Misc.perturb_params(sel_params, lower_bound=lower_bound,
#                              upper_bound=upper_bound)
#popt = dadi.Inference.optimize_log(p0, data, spectra.integrate, pts=None,
#                                   func_args=[neugamma, theta_ns],
#                                   lower_bound=lower_bound, upper_bound=upper_bound, 
#                                   verbose=len(sel_params),
#                                   maxiter=30, multinom=False)
#
##
## Modeling ancestral state misidentification, using dadi's built-in function to 
## wrap fitdadi's integrate method.
##
#p_misid = 0.05
#data = dadi.Numerics.apply_anc_state_misid(data, p_misid)
#misid_func = dadi.Numerics.make_anc_state_misid_func(spectra.integrate)
#sel_params = [0.2, 1000., 0.2]
#lower_bound, upper_bound = [1e-3, 1e-2, 0], [1, 50000., 1]
#p0 = dadi.Misc.perturb_params(sel_params, lower_bound=lower_bound,
#                              upper_bound=upper_bound)
#popt = dadi.Inference.optimize_log(p0, data, misid_func, pts=None,
#                                   func_args=[DFE.PDFs.gamma, theta_ns],
#                                   lower_bound=lower_bound, upper_bound=upper_bound,
#                                   verbose=len(sel_params), maxiter=30,
#                                   multinom=False)
##
## Including a point mass of positive selection
##
#data = dadi.Spectrum.from_file('example.fs')
#ppos = 0.1
#sel_data = theta_ns*DFE.DemogSelModels.two_epoch(tuple(demog_params) + (5,), ns, pts_l)
#data_pos = (1-ppos)*data + ppos*sel_data
#
#sel_params = [0.2, 1000., 0.2, 2]
#lower_bound, upper_bound = [1e-3, 1e-2, 0, 0], [1, 50000., 1, 50]
#p0 = dadi.Misc.perturb_params(sel_params, lower_bound=lower_bound,
#                              upper_bound=upper_bound)
#popt = dadi.Inference.optimize_log(p0, data_pos, spectra.integrate_point_pos, pts=None,
#                                   func_args=[DFE.PDFs.gamma, theta_ns, DFE.DemogSelModels.two_epoch], 
#                                   lower_bound=lower_bound, upper_bound=upper_bound, 
#                                   verbose=len(sel_params), maxiter=30, multinom=False)

#
# Multiple point masses of positive selection
#
# Parameters are mu, sigma, ppops1, gammapos1, ppos2, gammapos2
sel_params = [3,2,0.1,2,0.3,6]
input_fs = spectra.integrate_point_pos(sel_params,None,DFE.PDFs.lognormal,theta_ns,
                                       DFE.DemogSelModels.two_epoch, 2)
data = input_fs.sample()
lower_bound, upper_bound = [-1,0.1,0,0,0,0], [5,5,1,10,1,20]
p0 = dadi.Misc.perturb_params(sel_params, lower_bound=lower_bound,
                              upper_bound=upper_bound)

def ieq_constraint(p,*args):
    # Our constraint is that ppop1+ppos2 must be less than 1.
    return [1-(p[2]-p[4])]

popt = dadi.Inference.optimize_cons(p0, data, spectra.integrate_point_pos, pts=None,
                                    func_args=[DFE.PDFs.lognormal, theta_ns,
                                               DFE.DemogSelModels.two_epoch, 2],
                                    lower_bound=lower_bound, upper_bound=upper_bound,
                                    ieq_constraint=ieq_constraint,
                                    # Fix gammapos1
                                    fixed_params=[None,None,None,2,None,None],
                                    verbose=len(sel_params),
                                    maxiter=30, multinom=False)

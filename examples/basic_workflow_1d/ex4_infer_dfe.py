#!/usr/bin/env python
#SBATCH --account=rgutenk
#SBATCH --qos=user_qos_rgutenk
#SBATCH --partition=high_priority
#SBATCH --job-name="dadi_ex4_results"
#SBATCH --output=%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --array 1-20

# Dadi workflow example 4 - Infer the DFE
import dadi
import dadi.DFE as DFE
import pickle
import nlopt

# Make a variable to store the name of the dataset you are working with
# so that you can esaily change it to work on different datasets
dataset = '1KG.YRI.20'

# Load synonymous frequency spectrum
data_fs = dadi.Spectrum.from_file('data/fs/'+dataset+'.nonsynonymous.snps.unfold.fs')

# Retrive the sample sizes from the data
ns = data_fs.sample_sizes

# Define the synonymous theta you got from the demography fit
theta0 = 6124

# Calculate the theta for the nonsynonymous data based
# on the ratio of nonsynonymous mutations to synonymous mutations.
# You will probably be get this ratio from the paper or calculate it,
# but typically it is larger grater than 1.
theta_ns = theta0 * 2.31

# Load the cache of spectra
cache1d = pickle.load(open('results/'+dataset+'_1d_cache.bpkl','rb'))

# Define the DFE function used
# cache.integrate or DFE.mixture
# Here we will use the cache.integration function
dfe_func = cache1d.integrate

# If the data is unfolded (the ancestral allele was known), as the example data is
# Wrap the dfe function in a function that adds a parameter to estimate the 
# rate of misidentification.
dfe_func = dadi.Numerics.make_anc_state_misid_func(dfe_func)

# Define the selection distribution you want to use.
# If using the mixture model you will want to define
# the selection distribution for both 1D and 2D distributions.
sele_dist1d = DFE.PDFs.gamma

# Optimization for the DFE requires extra arguments that
# are not included in the optimizer function, so we need
# to define them ourselves.
func_args = [sele_dist1d, theta_ns]
# For the DFE.mixture function the argument would be along the lines of:
# func_args = [cache1d, cache2d, sele_dist1d, sele_dist2d, theta_ns]

# Choose starting parameters for inference
params = [0.1, 5000, 0.01]

# Define boundaries of optimization
# It is a good idea to have boundaries for the DFE as
# the optimizer can take parameters to values that
# cause errors with calulating the spectrum.
# Ex. rho = 1
# If optimization runs up against boundaries, you can increase them
lower_bounds = [1e-2, 1e-2, 1e-3]
upper_bounds = [10, 10000, 1]

# Optional: create a file to store fits in
# For running on the HPC, it is a good idea to
# check if file is made before making it so
# that you don't overwrite other results
try:
	fid = open('results/'+dataset+'_dfe_fits.txt','a')
except:
	fid = open('results/'+dataset+'_dfe_fits.txt','w')

# Perturb parameters
# Optimizers dadi uses are mostly deterministic
# so you will want to randomize parameters for each optimization.
# It is recommended to optimize at least 20 time if you only have
# your local machin, 100 times if you have access to an HPC.
# If you want a single script to do multiple runs, you will want to
# start a for loop here
for i in range(1):
    p0 = dadi.Misc.perturb_params(params, fold=1, upper_bound=upper_bounds,
                                  lower_bound=lower_bounds)

    # Run optimization
    # At the end of the optimization you will get the
    # optimal parameters and log-likelihood.
    # You can modify verbose to watch how the optimizer behaves,
    # what number you pass it how many evaluations are done
    # before the evaluation is printed.
    # For the DFE, because we calculated the theta_ns, we want to
    # set multinom=False.
    popt, ll_model = dadi.Inference.opt(p0, data_fs, dfe_func, pts=None, 
                                func_args=func_args, 
                                lower_bound=lower_bounds, 
                                upper_bound=upper_bounds,
                                maxeval=600, multinom=False, verbose=0)
    # If were were using the DFE.mixture function, we might want to
    # fix the rho to 0, to have w be the weight of uncorrelated model
    # compared to the perfectly correlated model:
    # fixed_params=[None, None, 0, None, None]

    # Generate DFE spectrum
    model_fs = dfe_func(popt, None, sele_dist1d, theta_ns, None)
    # If we were using the DFE.mixture function

    # Optional save method 1:
    # Write results to fid
    res = [ll_model] + list(popt) + [theta_ns]
    fid.write('\t'.join([str(ele) for ele in res])+'\n')

# Optional save method 1: 
# Close the file
fid.close()

# You can use a BASH command like sort on the files to more easily tell what the
# best fit is and how the log-likelihoods compare.
# ex:
# sort results/1KG.YRI.20_dfe_fits.txt | head -10

# You can visualize the comparison between the data and model.
# This example is assuming you do not do this in the same script.
import dadi
import dadi.DFE as DFE
import pickle

# Loading data and setting up what we need for the model
dataset = '1KG.YRI.20'
data_fs = dadi.Spectrum.from_file('data/fs/'+dataset+'.nonsynonymous.snps.unfold.fs')
ns = data_fs.sample_sizes
theta0 = 6124
theta_ns = theta0 * 2.31
cache1d = pickle.load(open('results/'+dataset+'_1d_cache.bpkl','rb'))
dfe_func = cache1d.integrate
dfe_func = dadi.Numerics.make_anc_state_misid_func(dfe_func)
sele_dist1d = DFE.PDFs.gamma
func_args = [sele_dist1d, theta_ns]

# Using inference information to generate the model
popt = [0.14, 2734, 0.02]
fid = open('results/1KG.YRI.20_dfe_fits.txt','r').readlines()
ll_best = -100000
for line in fid:
    if float(line.split('\t')[0]) > ll_best:
        ll_best = float(line.split('\t')[0])
        popt = [float(ele) for ele in line.split('\t')[1:-1]]
model_fs = dfe_func(popt, None, sele_dist1d, theta_ns, None)

# Plot
import matplotlib.pyplot as plt
fig = plt.figure(219033)
fig.clear()
dadi.Plotting.plot_1d_comp_multinom(model_fs, data_fs)
fig.savefig('results/'+dataset+'_dfe_plot.png')




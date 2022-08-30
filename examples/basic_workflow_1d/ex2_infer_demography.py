#!/usr/bin/env python
#SBATCH --account=rgutenk
#SBATCH --qos=user_qos_rgutenk
#SBATCH --partition=high_priority
#SBATCH --job-name="dadi_ex2_results"
#SBATCH --output=%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --array 1-20

# Dadi workflow example 2 - Infer the demographic model
import dadi
import nlopt
import os
if not os.path.exists("results"):
    os.makedirs("results")

# Make a variable to store the name of the dataset you are working with
# so that you can easily change it to work on different datasets
dataset = '1KG.YRI.20'

# Load synonymous frequency spectrum
data_fs = dadi.Spectrum.from_file('data/fs/'+dataset+'.synonymous.snps.unfold.fs')

# Retrive the sample sizes from the data
ns = data_fs.sample_sizes

# Define the grid points based on the sample size
# For smaller data (largest sample size ~ <100) [ns+20, ns+30, ns+40] is a good starting point
# For larger data (largest sample size ~ >=100) [ns+100, ns+110, ns+120] is a good starting point
pts_l = [max(ns)+120, max(ns)+130, max(ns)+140]

# Define the demographic model
# Single population demographic models are in dadi.Demographics1D
# Two population demographic models are in dadi.Demographics2D
demo_model = dadi.Demographics1D.two_epoch

# If the data is unfolded (the ancestral allele was known), as the example data is,
# wrap the demographic model in a function that adds a parameter to estimate the 
# rate of misidentification.
demo_model = dadi.Numerics.make_anc_state_misid_func(demo_model)

# Wrap the demographic model in a function that utilizes grid points
# which increases dadi's ability to more accurately generate a model
# frequency spectrum
demo_model_ex = dadi.Numerics.make_extrap_func(demo_model)

# Choose starting parameters for inference
params = [1, 0.01, 0.01]

# Define boundaries of optimization
# It is a good idea to have boundaries to avoid optimization
# from trying parameter sets that are time consuming without
# nessicarily being correct.
# If optimization runs up against boundaries, you can increase them
lower_bounds = [1e-2, 1e-3, 1e-3]
upper_bounds = [5, 3, 1]

# Optionalsave method 1:
# create a file to store fits in
# For running on the HPC, it is a good idea to
# check if file is made before making it so
# that you don't overwrite other results
try:
	fid = open('results/'+dataset+'_demo_fits.txt','a')
except:
	fid = open('results/'+dataset+'_demo_fits.txt','w')

# Perturb parameters
# Optimizers dadi uses are mostly deterministic
# so you will want to randomize parameters for each optimization.
# It is recommended to optimize at least 20 time if you only have
# your local machin, 100 times if you have access to an HPC.
# If you want a single script to do multiple runs, you will want to
# start a for loop here
for i in range(10):
    p0 = dadi.Misc.perturb_params(params, fold=1, upper_bound=upper_bounds,
                                  lower_bound=lower_bounds)

    # Run optimization
    # At the end of the optimization you will get the
    # optimal parameters and log-likelihood.
    # You can modify verbose to watch how the optimizer behaves,
    # what number you pass it how many evaluations are done
    # before the evaluation is printed.
    popt, ll_model = dadi.Inference.opt(p0, data_fs, demo_model_ex, pts_l,
                                 lower_bound=lower_bounds,
                                 upper_bound=upper_bounds,
                                 algorithm=nlopt.LN_BOBYQA,
                                 maxeval=600, verbose=0)

    # Find the synonymous theta
    model_fs = demo_model_ex(popt, ns, pts_l)
    theta0 = dadi.Inference.optimal_sfs_scaling(model_fs, data_fs)

    # Optional save method 1:
    # Write results to fid
    res = [ll_model] + list(popt) + [theta0]
    fid.write('\t'.join([str(ele) for ele in res])+'\n')
fid.close()

# You can use a BASH command like sort on the files to more easily tell what the
# best fit is and how the log-likelihoods compare.
# ex:
# sort results/1KG.YRI.20_demo_fits.txt

# You can visualize the comparison between the data and model
# This example is assuming you do not do this in the same script
import dadi

# Loading data and setting up what we need for the model
dataset = '1KG.YRI.20'
data_fs = dadi.Spectrum.from_file('data/fs/'+dataset+'.synonymous.snps.unfold.fs')
ns = data_fs.sample_sizes
pts_l = [max(ns)+20, max(ns)+30, max(ns)+40]
demo_model = dadi.Demographics1D.two_epoch
demo_model = dadi.Numerics.make_anc_state_misid_func(demo_model)
demo_model_ex = dadi.Numerics.make_extrap_func(demo_model)

# Using inference information to generate the model
popt = [2.279, 0.608, 0.0249]
model_fs = demo_model_ex(popt, ns, pts_l)

import matplotlib.pyplot as plt
fig = plt.figure(219033)
fig.clear()
dadi.Plotting.plot_1d_comp_multinom(model_fs, data_fs)
fig.savefig('results/'+dataset+'_demo_plot.png')






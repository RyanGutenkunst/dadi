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

# Make a variable to store the name of the dataset you are working with
# so that you can esaily change it to work on different datasets
dataset = '1KG.YRI.CEU.20'

# Load synonymous frequency spectrum
data_fs = dadi.Spectrum.from_file('data/fs/'+dataset+'.synonymous.snps.unfold.fs')

# Retrive the sample sizes from the data
ns = data_fs.sample_sizes

# Define the grid points based on the sample size
# For smaller data (largest sample size ~ <100) [ns+20, ns+30, ns+40] is a good starting point
# For larger data (largest sample size ~ >=100) [ns+100, ns+110, ns+120] is a good starting point
pts_l = [max(ns)+20, max(ns)+30, max(ns)+40]

# Define the demographic model
# Single population demographic models are in dadi.Demographics1D
# Two population demographic models are in dadi.Demographics2D
demo_model = dadi.Demographics2D.split_mig

# If the data is unfolded (the ancestral allele was known), as the example data is
# Wrap the demographic model in a function that adds a parameter to estimate the 
# rate of misidentification.
demo_model = dadi.Numerics.make_anc_state_misid_func(demo_model)

# Wrap the demographic model in a function that utilizes grid points
# which increases dadi's ability to more accurately generate a model
# frequency spectrum
demo_model_ex = dadi.Numerics.make_extrap_func(demo_model)

# Choose starting parameters for inference
params = [1, 1, 0.01, 0.1, 0.01]

# Define boundaries of optimization
# It is a good idea to have boundaries to avoid optimization
# from trying parameter sets that are time consuming without
# nessicarily being correct.
# If optimization runs up against boundaries, you can increase them
lower_bounds = [1e-2, 1e-2, 1e-3, 1e-3, 1e-3]
upper_bounds = [5, 5, 3, 3, 1]

# # Optionalsave method 1:
# # create a file to store fits in
# # For running on the HPC, it is a good idea to
# # check if file is made before making it so
# # that you don't overwrite other results
# try:
# 	fid = open('results/'+dataset+'_demo_fits.txt','a')
# except:
# 	fid = open('results/'+dataset+'_demo_fits.txt','w')

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
    popt, ll_model = dadi.Inference.opt(p0, data_fs, demo_model_ex, pts_l,
                                 lower_bound=lower_bounds,
                                 upper_bound=upper_bounds,
                                 algorithm=nlopt.LN_BOBYQA,
                                 maxeval=600, verbose=0)

    # Find the synonymous theta
    model_fs = demo_model_ex(popt, ns, pts_l)
    theta0 = dadi.Inference.optimal_sfs_scaling(model_fs, data_fs)

    # # Optional save method 1:
    # # Write results to fid
    # res = [ll_model] + list(popt) + [theta0]
    # fid.write('\t'.join([str(ele) for ele in res])+'\n')

    # # Optional save method 2:
    # # You can make files with the items of interest.
    # # This is potentially safer when running on the HPC,
    # # but it takes up more space. 
    # # This is a bit nicer than parsing the HPC output files,
    # # as you can use ls to look at the fits in order of log-likelihood
    # # You should clear the demo_temp out once you have noted the optimal parameters.
    # fid_name = 'results/demo_temp/'+dataset+'_ll_%.5f_theta_%.5f_params' %tuple([ll_model, theta0])
    # fid_name += '_%.5f'*len(popt) %tuple(popt)
    # # So that file names don't overwrite eachother,
    # # you can add a random number into the file name
    # import random
    # fid_name += '_tag_' + str(random.randint(1,1000000))
    # fid = open(fid_name+'.txt','w')
    # fid.close()

    # # Optional save method 3: 
    # # You can save the optimal parameters as a plot
    # # instead of just a blank text file.
    # # For plotting two population models, we want to set a vmin
    # # and resid_range for visualization purposes.
    # import matplotlib.pyplot as plt
    # fig = plt.figure(219033)
    # fig.clear()
    # fid_name = 'results/demo_temp/'+dataset+'_ll_%.5f_theta_%.5f_params' %tuple([ll_model, theta0])
    # fid_name += '_%.5f'*len(popt) %tuple(popt)
    # # So that file names don't overwrite eachother,
    # # you can add a random number into the file name
    # import random
    # fid_name += '_tag_' + str(random.randint(1,1000000))
    # dadi.Plotting.plot_2d_comp_multinom(model_fs, data_fs, resid_range=3, vmin=1e-3, show=False)
    # fig.savefig(fid_name + '.png')

# # Optional save method 1: 
# # Close the file
# fid.close()

# You can use a BASH command like sort on the files to more easily tell what the
# best fit is and how the log-likelihoods compare.
# ex:
# sort results/1KG.YRI.20_demo_fits.txt



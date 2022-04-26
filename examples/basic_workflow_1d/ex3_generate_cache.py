#!/usr/bin/env python
#SBATCH --account=rgutenk
#SBATCH --qos=user_qos_rgutenk
#SBATCH --partition=high_priority
#SBATCH --job-name="dadi_ex3_results"
#SBATCH --output=%x-%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --time=24:00:00

# Dadi workflow example 3 - Creating a cache of demographic model spectra with selection
import dadi
import dadi.DFE as DFE
import pickle

# Make a variable to store the name of the dataset you are working with
# so that you can esaily change it to work on different datasets
dataset = '1KG.YRI.20'

# Define the sample sizes of the data
ns = [20]
# You can also load the data if you want
# to not worry about being specific.
# Just remember the DFE will be used to fit the
# nonsynonymous data, so you need to use that
# dataset's sample size
fs = dadi.Spectrum.from_file('data/fs/'+dataset+'.nonsynonymous.snps.unfold.fs')
ns = fs.sample_sizes

# Define the grid points based on the sample size
# Because you are adding selection, you might want to increase
# the sizes of the grid points, as spectra with higher levels
# of selection are harder for dadi to calculate accurately
pts_l = [max(ns)+140, max(ns)+150, max(ns)+160]

# Get the selection version of the demographic model
# Depending if you want the version with selection
# being independent or shared between populations
# the model name is slightly different.
# Extrapolation happens in the process of making the cache,
# so you do not need to to wrap the extrapolation function.
demo_sel_model = DFE.DemogSelModels.two_epoch_sel

# Define the optimial parameters from the demography fits
# If you used misidentification model, you can remove the
# misidentification 
popt = [2.273, 0.6057]
# You could also loop through your results file and extract the parameters that way.

# Generate cache
# The gamma_bounds argument defines the range of the gamma distribution.
# The gamma_pts argument can be used to specify the number of
# selection coefficients that will be selected in that range to generate your cache.
# It is recommended to use gamma_bounds=[1e-4, 2000], gamma_pts=50 for either 1D or 2D cache generation
# on the HPC.
# If you want to generate the 2D cache (independent selection coefficients), use:
# DFE.Cache1D
# NOTE: When testing locally, having mp = True might cause a bug, so set it to mp=False until you start working on an HPC
cache1d = DFE.Cache1D(popt, ns, demo_sel_model, pts=pts_l, gamma_bounds=[1e-2, 2000], gamma_pts=50, mp=False)

# Check if the cached spectra have any large negative values
if (cache1d.spectra<0).sum() > 0:
    print(
        '!!!WARNING!!!\nPotentially large negative values!\nMost negative value is: '+str(cache1d.spectra.min())+
        '\nIf negative values are very negative (<-0.001), rerun with larger values for pts_l'
        )

# Save the cache with pickle
fid = open('results/'+dataset+'_1d_cache.bpkl', 'wb')
pickle.dump(cache1d, fid, protocol=2)
fid.close()


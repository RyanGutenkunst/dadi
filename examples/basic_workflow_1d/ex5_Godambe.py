import dadi
import dadi.DFE as DFE
import pickle
import random
import glob

# Example code for how you can generate bootstraps for synonymous and nonsynonymous fs.
# We don't have the data dictionaries in the dadi repository due to size.
# But the bootstraps are provided.
# # Godambe for demographic fits
# # Make a variable to store the name of the dataset you are working with
# # so that you can easily change it to work on different datasets
# dataset = '1KG.YRI.20'

# # Load synonymous data dictionary
# # We need the data dictionary because it contains SNP positions
# # which we need in order to break up and randomize new genomes
# # for bootstraping.
# dd_syn = pickle.load(open('data/data_dictionaries/'+dataset+'.synonymous.snps.unfold.bpkl','rb'))

# # Nboot is the number of bootstraped datasets we want
# # chunk_size is the genome chunks we use to make the
# # randomized genomes.
# Nboot, chunk_size = 100, 1e7

# # Break up the data dictionari into a list of
# # dictionary entries for each chunk of the genome.
# # Ex, the fist 10^7 base pairs will be element 0,
# # the next 10^7 base pairs will be element 1, etc.
# # to the last 10^7 base pairs will be the last element/element -1.
# chunks_syn = dadi.Misc.fragment_data_dict(dd_syn, chunk_size)

# # Define the sample sizes for the data
# ns = [20]

# # Define the population ID(s).
# pop_ids = ['YRI']

# # If we want to do bootstrapping on both the
# # synonymous and nonsynonymous variants,
# # we want to set a seed so that the same random
# # chunks of genome are picked for both.
# random.seed(12345)

# # Get a list containin the SFS of the bootstraped genomes.
# boots_syn = dadi.Misc.bootstraps_from_dd_chunks(chunks_syn, Nboot, pop_ids, ns)
# for i in range(len(boots_syn)):
#     boots_syn[i].to_file('data/fs/bootstraps_syn/'+dataset+'.synonymous.snps.unfold.boot_{0}.fs'.format(str(i)))

# # Load nonsynonymous data dictionary
# dd_non = pickle.load(open('data/data_dictionaries/'+dataset+'.nonsynonymous.snps.unfold.bpkl','rb'))

# # Break up the data dictionari into a list of
# # dictionary entries for each chunk of the genome.
# # Ex, the fist 10^7 base pairs will be element 0,
# # the next 10^7 base pairs will be element 1, etc.
# # to the last 10^7 base pairs will be the last element/element -1.
# chunks_non = dadi.Misc.fragment_data_dict(dd_non, chunk_size)

# # If we want to do bootstrapping on both the
# # synonymous and nonsynonymous variants,
# # we want to set a seed so that the same random
# # chunks of genome are picked for both.
# random.seed(12345)

# # Get a list containin the SFS of the bootstraped genomes.
# boots_non = dadi.Misc.bootstraps_from_dd_chunks(chunks_non, Nboot, pop_ids, ns)
# for i in range(len(boots_non)):
#     boots_non[i].to_file('data/fs/bootstraps_non/'+dataset+'.nonsynonymous.snps.unfold.boot_{0}.fs'.format(str(i)))
# # Load cache
# cache1d = pickle.load(open('results/'+dataset+'_1d_cache.bpkl', 'rb'))

#
#
# Demography Uncertainties
#
#

# Make a variable to store the name of the dataset you are working with
# so that you can easily change it to work on different datasets
dataset = '1KG.YRI.20'

# Define the sample sizes for the data
ns = [20]

# Define the population ID(s).
pop_ids = ['YRI']

# Define the demographic model
# Single population demographic models are in dadi.Demographics1D
# Two population demographic models are in dadi.Demographics2D
demo_model = dadi.Demographics1D.two_epoch

# If the data is unfolded (the ancestral allele was known), as the example data is
# Wrap the demographic model in a function that adds a parameter to estimate the 
# rate of misidentification.
demo_model = dadi.Numerics.make_anc_state_misid_func(demo_model)

# Wrap the demographic model in a function that utilizes grid points
# which increases dadi's ability to more accurately generate a model
# frequency spectrum.
demo_model_ex = dadi.Numerics.make_extrap_func(demo_model)

# Define the grid points based on the sample size
# For smaller data (largest sample size ~ <100) [ns+20, ns+30, ns+40] is a good starting point
# For larger data (largest sample size ~ >=100) [ns+100, ns+110, ns+120] is a good starting point
pts_l = [max(ns)+120, max(ns)+130, max(ns)+140]

# Define the bestfit parameters
popt_demo = [2.28, 0.61, 0.025]

# Load the SFS you fit the demography to
fs_syn = dadi.Spectrum.from_file('data/fs/'+dataset+'.synonymous.snps.unfold.fs')

# Load synonymous bootstraps
boots_syn_fids = glob.glob('data/fs/bootstraps_syn/'+dataset+'.synonymous.snps.unfold.boot_*.fs')
boots_syn = [dadi.Spectrum.from_file(fid) for fid in boots_syn_fids]

# Godambe uncertainties
# Will contain uncertainties for the
# estimated demographic parameters and theta.

# Start a file to contain the confidence intervals
fi = open('results/'+dataset+'_demographic_confidence_intervals.txt','w')
fi.write('Optimized parameters: {0}\n\n'.format(popt_demo))

# we want to try a few different step sizes (eps) to see if
# uncertainties very wildly with changes to step size.
for eps in [0.01, 0.001, 0.0001]:
    uncerts_adj_demo= dadi.Godambe.GIM_uncert(demo_model_ex, pts_l, boots_syn, popt_demo, fs_syn, eps=eps)
    fi.write('Estimated 95% uncerts (with step size '+str(eps)+'): {0}\n'.format(1.96*uncerts_adj_demo[:-1]))
    fi.write('Lower bounds of 95% confidence interval : {0}\n'.format(popt_demo-1.96*uncerts_adj_demo[:-1]))
    fi.write('Upper bounds of 95% confidence interval : {0}\n\n'.format(popt_demo+1.96*uncerts_adj_demo[:-1]))
fi.close()



#
#
# DFE Uncertainties
#
#

# Make a variable to store the name of the dataset you are working with
# so that you can easily change it to work on different datasets
dataset = '1KG.YRI.20'

# Define the sample sizes for the data
ns = [20]

# Define the population ID(s).
pop_ids = ['YRI']

# Load cache
cache1d = pickle.load(open('results/'+dataset+'_1d_cache.bpkl', 'rb'))

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

# Load the SFS you fit the DFE to
fs_non = dadi.Spectrum.from_file('data/fs/'+dataset+'.nonsynonymous.snps.unfold.fs')

# Load nonsynonymous bootstraps
boots_non_fids = glob.glob('data/fs/bootstraps_non/'+dataset+'.nonsynonymous.snps.unfold.boot_*.fs')
boots_non = [dadi.Spectrum.from_file(fid) for fid in boots_non_fids]

# Load synonymous bootstraps
boots_syn_fids = glob.glob('data/fs/bootstraps_syn/'+dataset+'.synonymous.snps.unfold.boot_*.fs')
boots_syn = [dadi.Spectrum.from_file(fid) for fid in boots_syn_fids]

# The DFE analysis requires the theta estimation from the demographic analysis.
# Because bootstraped genomes can change the theta estimate from the demographic analysis,
# we use the synonymous bootstraps the approximate the difference in theta by comparing the
# amount of data in the bootstraps compared to the real data.
# You will want to make the bootstraps using and setting a shared seed with
# the sysnonymous bootstrap.
boot_theta_adjusts = [boot_fs.sum()/fs_syn.sum() for boot_fs in boots_syn]

# Define the optimal DFE parameters
popt_dfe = [0.139, 2734.679, 0.0198]

# Define nonsynonymous theta
theta_ns = 6124 * 2.31

# Create a dadi model function that extra arguments can be passed in through
def dfe_func_new(params, ns, pts):
    return dfe_func(params, None, sele_dist1d, theta_ns, pts=None)

# Godambe uncertainties
# Will contain uncertainties for the
# estimated DFE parameters.
# Since we pass in a specific theta and adjustments,
# we don't worry about an uncertainty for theta.

# Start a file to contain the confidence intervals
fi = open('results/'+dataset+'_DFE_confidence_intervals.txt','w')
fi.write('Optimized parameters: {0}\n\n'.format(popt_dfe))

# we want to try a few different step sizes (eps) to see if
# uncertainties very wildly with changes to step size.
for eps in [0.01, 0.001, 0.0001]:
    uncerts_adj_dfe = dadi.Godambe.GIM_uncert(dfe_func_new, [], boots_non, popt_dfe, fs_non, eps=eps, multinom=False, boot_theta_adjusts=boot_theta_adjusts)
    fi.write('Estimated 95% uncerts (with step size '+str(eps)+'): {0}\n'.format(1.96*uncerts_adj_dfe))
    fi.write('Lower bounds of 95% confidence interval : {0}\n'.format(popt_dfe-1.96*uncerts_adj_dfe))
    fi.write('Upper bounds of 95% confidence interval : {0}\n\n'.format(popt_dfe+1.96*uncerts_adj_dfe))
fi.close()












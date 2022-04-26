# Basic workflow for single population demographic inference
The example code can be ran from `dadi/examples/basic_workflow/` directory. You may need to be on the developmental branch of dadi.

## Creating a frequency spectrum (FS)
Much of this section currently repeates the "Frequency spectrum from data" example, please view that example for more details on FS.
First we load the modules we will use for our workflow:
```python
# Pickle is used to save variables as files for future use
import pickle
# NLopt is the optimization libary dadi uses
import nlopt
# MatPlotLib is a libary dadi uses for plotting frequency spectrum
import matplotlib.pyplot as plt
import dadi
```

Use dadi functions to parse the VCF file to generate a data dictionary.
```python
datafile = '../fs_from_data/1KG.YRI.CEU.biallelic.synonymous.snps.withanc.strict.subset.vcf.gz'
dd = dadi.Misc.make_data_dict_vcf(datafile, '../fs_from_data/1KG.YRI.CEU.popfile.txt')
```

Extract the SFS for the Yoruba (YRI) and Central European (CEU) population from that dictionary, with both projected down to 20. 
We project down for this example to make it doable on a laptop. 
For a real analysis we would probably not project so severely (ex. the 1000 Genomes Yoruba population's sample size can be as high as 216).
```python
pop_ids, ns = ['YRI', 'CEU'], [20, 20]
data_fs = dadi.Spectrum.from_data_dict(dd, pop_ids, ns)

# Save our extracted spectrum to disk.
data_fs.to_file('data/fs/1KG.YRI.CEU.biallelic.synonymous.snps.withanc.strict.subset.fs')
```

If we didn't have outgroup information, we could fold the fs.
```python
data_fs_folded = dadi.Spectrum.from_data_dict(dd, pop_ids, ns, polarized=False)
```

We can see how much data is in the SFS. 
As a rule of thumb we want to maximize this number. 
We can do this by trying different sample sizes for our population(s).
```python
print(data_fs.S())
```

We can save the data dictionary for later, incase we want a new population from the VCF, generate FS with a bigger sample size, or to make bootstrapped genomes to get confidence intervals.
```python
# Make directory for saving data dictionaries
import os
if not os.path.exists('data/data_dictionaries'):
  os.makedirs('data/data_dictionaries')

pick = open('data/data_dictionaries/1KG.YRI.CEU.biallelic.synonymous.snps.withanc.strict.subset.bpkl','wb')
pickle.dump(dd, pick, 2)

# We can open the pickled data dictionary with:
dd = pickle.load(open('data/data_dictionaries/1KG.YRI.CEU.biallelic.synonymous.snps.withanc.strict.subset.bpkl','rb'))
```

We can generate and save the FS for bootstraped genomes:
```python
# Nboot is the number of bootstraped datasets we want
# chunk_size is the genome chunks we use to make the
# randomized genomes.
Nboot, chunk_size = 100, 1e7

# Break up the data dictionari into a list of
# dictionary entries for each chunk of the genome.
# Ex, the fist 10^7 base pairs will be element 0,
# the next 10^7 base pairs will be element 1, etc.
# to the last 10^7 base pairs will be the last element/element -1.
chunks = dadi.Misc.fragment_data_dict(dd, chunk_size)

# If we want to do bootstrapping on both the
# synonymous and nonsynonymous variants,
# we want to set a seed so that the same random
# chunks of genome are picked for both.
# The seed needs to be reset to the same value between
# runs of dadi.Misc.bootstraps_from_dd_chunks()
import random
random.seed(12345)

# Get a list containin the SFS of the bootstraped genomes.
boots = dadi.Misc.bootstraps_from_dd_chunks(chunks, Nboot, pop_ids, ns)

# Save the bootstraps
for i in range(len(boots)):
  boots[i].to_file('data/fs/bootstraps_syn/1KG.YRI.CEU.biallelic.synonymous.snps.withanc.strict.subset.boot_{0}.fs'.format(str(i)))
```

We can plot the data spectra.
```python
fig = plt.figure(219033)
fig.clear()
dadi.Plotting.plot_single_2d_sfs(data_fs)
```

## Infer the demographic model
We start by generating the FS (in these examples, the data is different from the previous, as this data is more complete but would have been time-consuming to use with the previous examples) and some properties of it that are needed for inference.
```python
# If we have multiple datasets to work with, we can make a variable to store the name of the dataset that we can easily change and redo the inference with a different dataset.
dataset = '1KG.YRI.CEU.20'

# Load the synonymous frequency spectrum
data_fs = dadi.Spectrum.from_file('data/fs/'+dataset+'.synonymous.snps.unfold.fs')

# Retrive the sample sizes from the data
ns = data_fs.sample_sizes

# Define the grid points based on the sample size.
# For smaller data (largest sample size is about <100) [ns+20, ns+30, ns+40] is a good starting point.
# For larger data (largest sample size is about >=100) or for heavily down projected data [ns+100, ns+110, ns+120] is a good starting point.
pts_l = [max(ns)+20, max(ns)+30, max(ns)+40]
```

Define the demographic model.

Single population demographic models are in ```dadi.Demographics1D```.

Two population demographic models are in ```dadi.Demographics2D```.
```python
demo_model = dadi.Demographics2D.split_mig

# If the data is unfolded (the ancestral allele was known), as the example data is,
# wrap the demographic model in a function that adds a parameter to estimate the rate of misidentification.
demo_model = dadi.Numerics.make_anc_state_misid_func(demo_model)

# Wrap the demographic model in a function that utilizes grid points which increases dadi's ability to more accurately generate a model frequency spectrum.
demo_model_ex = dadi.Numerics.make_extrap_func(demo_model)
```

Choose starting parameters and boundaries for inference
```python
# Define starting parameters
params = [1, 1, 0.01, 0.01, 0.01]

# Define boundaries of optimization.
# It is a good idea to have boundaries to avoid optimization
# from trying parameter sets that are time consuming without
# nessicarily being correct.
# If optimization infers parameters very close to the boundaries, we should increase them.
lower_bounds = [1e-2, 1e-2, 1e-3, 1e-3, 1e-3]
upper_bounds = [3, 3, 1, 1, 1]
```
Create a file to store fits in.

For running on an HPC job array, it is a good idea to
check if file is made before making it so
that we don't overwrite other results.
```python
# Create or append to an file to store optimization results
try:
  fid = open('results/'+dataset+'_demo_fits.txt','a')
except:
  fid = open('results/'+dataset+'_demo_fits.txt','w')
```
Optimize parameters
```python
# Perturb parameters
# Optimizers dadi uses are mostly deterministic
# so we will want to randomize parameters for each optimization.
# It is recommended to optimize at least 20 time if we only have
# our local machin, 100 times if we have access to an HPC.
# If we want a single script to do multiple runs, we will want to
# start a for loop here
p0 = dadi.Misc.perturb_params(params, fold=1, upper_bound=upper_bounds,
                              lower_bound=lower_bounds)

# Run optimization
# At the end of the optimization we will get the
# optimal parameters and log-likelihood.
# We can modify verbose to watch how the optimizer behaves,
# what number we pass it how many evaluations are done
# before the evaluation is printed.

popt, ll_model = dadi.Inference.opt(p0, data_fs, demo_model_ex, pts_l,
                                    lower_bound=lower_bounds,
                                    upper_bound=upper_bounds,
                                    algorithm=nlopt.LN_BOBYQA,
                                    maxeval=400, verbose=100)

# Calculate the synonymous theta
model_fs = demo_model_ex(popt, ns, pts_l)
theta0 = dadi.Inference.optimal_sfs_scaling(model_fs, data_fs)

# Write results to fid
res = [ll_model] + list(popt) + [theta0]
fid.write('\t'.join([str(ele) for ele in res])+'\n')
fid.close()
```

If we use a loop we will want to close the file (```fid.close()```) after the loop, or open for appending (```fid = open('results/'+dataset+'_demo_fits.txt','a')```) it for each step of the loop.
We can use a BASH command like sort on the files to more easily tell what the
best fit is and how the log-likelihoods compare. The best log-likelihood is the lest negative (the closest to 0).

ex:
```bash
sort results/1KG.YRI.CEU.20_demo_fits.txt
```
## Plotting demographic inference
```python
# Using inference information to generate the model the best fit parameters with the two_epoch model is [2.279, 0.608, 0.0249]
popt = [2.8, 0.520, 0.144, 0.023, 0.017]
model_fs = demo_model_ex(popt, ns, pts_l)

import matplotlib.pyplot as plt
fig = plt.figure(219033)
fig.clear()
dadi.Plotting.plot_2d_comp_multinom(model_fs, data_fs)
fig.savefig('results/'+dataset+'_demo_plot.png')
```

## Godambe for demographic fits
```python
# Load bootstraped frequency spectrum (if they haven't been made, there is an example in the "Creating a frequency spectrum" section)
import glob
boots_fids = glob.glob('data/fs/bootstraps_syn/'+dataset+'.synonymous.snps.unfold.boot_*.fs')
boots_syn = [dadi.Spectrum.from_file(fid) for fid in boots_fids]

# Godambe uncertainties
# Will contain uncertainties for the
# estimated demographic parameters and theta.

# Start a file to contain the confidence intervals
fi = open('results/'+dataset+'_demographic_confidence_intervals.txt','w')
fi.write('Optimized parameters: {0}\n\n'.format(popt))

# we want to try a few different step sizes (eps) to see if
# uncertainties very wildly with changes to step size.
for eps in [0.01, 0.001, 0.0001]:
    uncerts_adj = dadi.Godambe.GIM_uncert(demo_model_ex, pts_l, boots_syn, popt, data_fs, eps=eps)
    fi.write('Estimated 95% uncerts (with step size '+str(eps)+'): {0}\n'.format(1.96*uncerts_adj[:-1]))
    fi.write('Lower bounds of 95% confidence interval : {0}\n'.format(popt-1.96*uncerts_adj[:-1]))
    fi.write('Upper bounds of 95% confidence interval : {0}\n\n'.format(popt+1.96*uncerts_adj[:-1]))
fi.close()
```
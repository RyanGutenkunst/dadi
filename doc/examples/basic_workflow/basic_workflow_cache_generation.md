# Basic workflow for cache generation
The example code can be ran from `dadi/examples/basic_workflow/` directory. You may need to be on the developmental branch of dadi.

The cache we generate is multiple demographic model frequency spectrum with different levels of selection. We can gnerate a cache with shared or independant selection generating, depending on the number of populations we are interested in.
Generating a cache can take a long time and it is recommended that you use a script dedicated to generating it on an HPC.

```python
import dadi
import dadi.DFE as DFE
import pickle
```
## Single Population Cache Generation
This example will demonstraite generating a cache for a single population from the dadi/examples/basic_workflow_1d/ directory.
```python
# Make a variable to store the name of the dataset you are working with
# so that you can esaily change it to work on different datasets
dataset = '1KG.YRI.20'

# Define the sample sizes of the data
ns = [20]
# You can also load the data if you want
# to not worry about being specific.
# Just remember the DFE will be used to fit the
# nonsynonymous data, so you'll need to use that
# dataset's sample size
# or
# you'll want to project the nonsynonymous data
# to the sample size of the synonymous data
fs = dadi.Spectrum.from_file('data/fs/'+dataset+'.nonsynonymous.snps.unfold.fs')
ns = fs.sample_sizes

# Define the grid points based on the sample size
# Because you are adding selection, you might want to increase
# the sizes of the grid points, as spectra with higher levels
# of selection are harder for dadi to calculate accurately
pts_l = [max(ns)+140, max(ns)+150, max(ns)+160]

# Get the selection version of the demographic model
# Extrapolation happens in the process of making the cache,
# so you do not need to to wrap it in the extrapolation function.
demo_sel_model = DFE.DemogSelModels.two_epoch_sel

# Define the optimial parameters from the demography fits
# If you used misidentification model, you can remove the
# misidentification parameter.
popt = [2.27, 0.61]
# You could also loop through your results file and extract the parameters that way.

# Generate cache
# The gamma_bounds argument defines the range of the gamma distribution.
# The gamma_pts argument can be used to specify the number of
# selection coefficients that will be selected in that range to generate your cache.
# It is recommended to use gamma_bounds=[1e-4, 2000], gamma_pts=50 for either 1D or 2D cache generation
# on the HPC.
cache1d = DFE.Cache1D(popt, ns, demo_sel_model, pts=pts_l, gamma_bounds=[1e-2, 20], gamma_pts=5, cpus=1)
```
We can check if the cached spectra have any large negative values:
```python
if (cache1d.spectra<0).sum() > 0:
    print(
        '!!!WARNING!!!\nPotentially large negative values!\nMost negative value is: '+str(cache1d.spectra.min())+
        '\nIf negative values are very negative (<-0.001), rerun with larger values for pts_l'
        )
```
We can save the cache with pickle:
```python
fid = open('results/'+dataset+'_1d_cache.bpkl', 'wb')
pickle.dump(cache1d, fid, protocol=2)
fid.close()
```
## Two Population Cache Generation
These examples will demonstraite generating a two population cache. Two population caches can have shared selection (denoted as 1d or sel_single_gamma in the code) and independant selection (denoted as 2d or just sel in the code).
```python
# Make a variable to store the name of the dataset you are working with
# so that you can esaily change it to work on different datasets
dataset = '1KG.YRI.CEU.20'

# Define the sample sizes of the data
ns = [20, 20]
# You can also load the data if you want
# to not worry about being specific.
# Just remember the DFE will be used to fit the
# nonsynonymous data, so you'll need to use that
# dataset's sample size
# or
# you'll want to project the nonsynonymous data
# to the sample size of the synonymous data
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
# split_mig_sel is the version of the split_mig demographic model with independant selection
demo_sel_model = DFE.DemogSelModels.split_mig_sel
# split_mig_sel_single_gamma is the version of the split_mig demographic model with shared selection
demo_sel_single_gamma_model = DFE.DemogSelModels.split_mig_sel_single_gamma

# Define the optimial parameters from the demography fits
# If you used misidentification model, you can remove the
# misidentification parameter.
popt = [2.8, 0.520, 0.144, 0.023]
# You could also loop through your results file and extract the parameters that way.

# Generate cache
# The gamma_bounds argument defines the range of the gamma distribution.
# The gamma_pts argument can be used to specify the number of
# selection coefficients that will be selected in that range to generate your cache.
# It is recommended to use gamma_bounds=[1e-4, 2000], gamma_pts=50 for either 1D or 2D cache generation
# on the HPC.
# If you want to generate the 2D cache (independent selection coefficients), use:
# DFE.Cache2D
# NOTE: When testing locally, having mp = True might cause a bug, so set it to mp=False until you start working on an HPC
cache1d = DFE.Cache1D(popt, ns, demo_sel_single_gamma_model, pts=pts_l, gamma_bounds=[1e-2, 20], gamma_pts=5, mp=False)

cache2d = DFE.Cache2D(popt, ns, demo_sel_model, pts=pts_l, gamma_bounds=[1e-2, 20], gamma_pts=5, mp=False)
```
We can check if the cached spectra have any large negative values:
```python
if (cache1d.spectra<0).sum() > 0:
    print(
        '!!!WARNING!!!\nPotentially large negative values!\nMost negative value is: '+str(cache1d.spectra.min())+
        '\nIf negative values are very negative (<-0.001), rerun with larger values for pts_l'
        )
if (cache2d.spectra<0).sum() > 0:
    print(
        '!!!WARNING!!!\nPotentially large negative values!\nMost negative value is: '+str(cache2d.spectra.min())+
        '\nIf negative values are very negative (<-0.001), rerun with larger values for pts_l'
        )
```
We can save the cache with pickle:
```python
fid = open('results/'+dataset+'_1d_cache.bpkl', 'wb')
pickle.dump(cache1d, fid, protocol=2)
fid.close()

fid = open('results/'+dataset+'_2d_cache.bpkl', 'wb')
pickle.dump(cache2d, fid, protocol=2)
fid.close()
```
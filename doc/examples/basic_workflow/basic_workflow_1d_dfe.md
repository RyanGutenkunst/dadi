# Basic workflow for single population distribution of fitness effects (DFE) inference
The example code can be ran from `dadi/examples/basic_workflow_1d/` directory. You may need to be on the developmental branch of dadi.
Load modules used for DFE inference:
```python
import dadi
import dadi.DFE as DFE
import pickle
import nlopt
```
## Infer DFE
```python
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

# Define the DFE function used.
# For single populations only cache.integration is used
dfe_func = cache1d.integrate

# If the data is unfolded (the ancestral allele was known), as the example data is
# Wrap the dfe function in a function that adds a parameter to estimate the 
# rate of misidentification.
dfe_func = dadi.Numerics.make_anc_state_misid_func(dfe_func)

# Define the selection distribution you want to use.
# This example will be the gamma distribution.
sele_dist1d = DFE.PDFs.gamma

# Optimization for the DFE requires extra arguments that
# are not included in the optimizer function, so we need
# to define them ourselves.
func_args = [sele_dist1d, theta_ns]

# Choose starting parameters for inference
# This is an example for the gamma distribution.
# Most importantly are the first two parameters
# shape and scale (also called alpha and beta).
params = [0.1, 15000, 0.01]

# Define boundaries of optimization
# It is a good idea to have boundaries for the DFE as
# the optimizer can take parameters to values that
# cause errors with calulating the spectrum.
# If optimization runs up against boundaries, you can increase them
lower_bounds = [1e-2, 1e-2, 1e-3]
upper_bounds = [10, 10000, 1]

# If you want to use a lognormal distribution,
# more relistic mu and sigma parameters and
# boundaries (for humans) would be:
# params = [2, 2, 0.01]
# lower_bounds = [1e-2, 1e-2, 1e-3]
# upper_bounds = [10, 10, 1]

# For running on the HPC, it is a good idea to
# check if file is made before making it so
# that you don't overwrite other results
try:
    fid = open('results/'+dataset+'_1d_dfe_fits.txt','a')
except:
    fid = open('results/'+dataset+'_1d_dfe_fits.txt','w')

# Perturb parameters
# Optimizers dadi uses are mostly deterministic
# so you will want to randomize parameters for each optimization.
# It is recommended to optimize at least 20 time if you only have
# your local machin, 100 times if you have access to an HPC.
# If you want a single script to do multiple runs, you will want to
# start a for loop here.
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
                                    maxeval=400, multinom=False, verbose=100)

# Generate DFE spectrum
model_fs = dfe_func(popt, None, sele_dist1d, theta_ns, None)
# If we were using the DFE.mixture function

# Optional save method 1:
# Write results to fid
res = [ll_model] + list(popt) + [theta_ns]
fid.write('\t'.join([str(ele) for ele in res])+'\n')

# Close the file
fid.close()
```
## Plotting the DFE
The main difference plotting the DFE is we use `dadi.Plotting.plot_1d_comp_Poisson` instead of `dadi.Plotting.plot_1d_comp_multinom`. The `_multinom` function adjusts the model to help match the data, because the DFE is adjusted to better match the data using theta, we use the `_Poisson` plotting function, which does not adjust the DFE.
```python
# Using inference information to generate the model
popt = [0.133, 4125, 0.02]

# Plot
import matplotlib.pyplot as plt
fig = plt.figure(219033)
fig.clear()
dadi.Plotting.plot_1d_comp_Poisson(model_fs, data_fs)
fig.savefig('results/'+dataset+'_dfe_plot.png')
```
## Godambe analysis with the DFE
```python
# Load bootstraped frequency spectrum
# (if they haven't been made, there is an example in the "Creating a frequency spectrum"
# section from the demographics example)
import glob
boots_non_fids = glob.glob('data/fs/bootstraps_non/'+dataset+'.nonsynonymous.snps.unfold.boot_*.fs')
boots_non = [dadi.Spectrum.from_file(fid) for fid in boots_non_fids]
# The DFE analysis requires the theta estimation from the demographic analysis.
# Because bootstraped genomes can change the theta estimate from the demographic analysis,
# we use the synonymous bootstraps the approximate the difference in theta by comparing the
# amount of data in the bootstraps compared to the real data.
# You will want to make the bootstraps using and setting a shared seed with
# the sysnonymous bootstrap.
boots_syn_fids = glob.glob('data/fs/bootstraps_syn/'+dataset+'.synonymous.snps.unfold.boot_*.fs')
boots_syn = [dadi.Spectrum.from_file(fid) for fid in boots_syn_fids]

# The DFE analysis requires the theta estimation from the demographic analysis.
# Because bootstraped genomes can change the theta estimate from the demographic analysis,
# we use the synonymous bootstraps the approximate the difference in theta by comparing the
# amount of data in the bootstraps compared to the real data.
# You will want to make the bootstraps using and setting a shared seed with
# the sysnonymous bootstrap.
fs_syn = dadi.Spectrum.from_file('data/fs/'+dataset+'.synonymous.snps.unfold.fs')
boot_theta_adjusts = [boot_fs.sum()/fs_syn.sum() for boot_fs in boots_syn]

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
fi.write('Optimized parameters: {0}\n\n'.format(popt))

# we want to try a few different step sizes (eps) to see if
# uncertainties very wildly with changes to step size.
for eps in [0.01, 0.001, 0.0001]:
    uncerts_adj_dfe = dadi.Godambe.GIM_uncert(dfe_func_new, [], boots_non, popt, data_fs, eps=eps, multinom=False, boot_theta_adjusts=boot_theta_adjusts)
    fi.write('Estimated 95% uncerts (with step size '+str(eps)+'): {0}\n'.format(1.96*uncerts_adj_dfe))
    fi.write('Lower bounds of 95% confidence interval : {0}\n'.format(popt-1.96*uncerts_adj_dfe))
    fi.write('Upper bounds of 95% confidence interval : {0}\n\n'.format(popt+1.96*uncerts_adj_dfe))
fi.close()
```
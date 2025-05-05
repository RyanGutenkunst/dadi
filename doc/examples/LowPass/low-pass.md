# Low-Pass Sequencing Pipeline Example

### Overview

A pipeline example for low-pass sequencing at 3x coverage depth, detailing each step from loading necessary modules and importing data to performing demographic inference and visualizing results.

```python
# Load Required Modules
import dadi  # Demographic Analysis with dadi
from dadi.LowPass import LowPass  # Low-pass sequencing tools in dadi
import nlopt  # Non-linear optimization library
import matplotlib.pyplot as plt  # Plotting library for visualization

# Define Input Files
datafile = './low-pass-example-3x.vcf.gz'
popfile = './low-pass-popfile-3x.txt'

# Create Data Dictionary
data_dict = dadi.Misc.make_data_dict_vcf(datafile, popfile, subsample={'pop1': 16}, calc_coverage=True)

# Set Outgroup Allele and Context
for chrom_pos in data_dict:
    data_dict[chrom_pos]['outgroup_allele'] = data_dict[chrom_pos]['segregating'][0]
    data_dict[chrom_pos]['outgroup_context'] = data_dict[chrom_pos]['segregating'][0]

# Generate Site Frequency Spectrum (SFS)
data_fs = dadi.Spectrum.from_data_dict(data_dict, ['pop1'], [32])

# Calculate Coverage Distribution
cov_dist = LowPass.compute_cov_dist(data_dict, data_fs.pop_ids)

# Define the Demographic Model and Integrate it with the Low-Pass Function
demo_model_ex = dadi.Numerics.make_extrap_func(dadi.Demographics1D.growth)
demo_model_ex = LowPass.make_low_pass_func_GATK_multisample(demo_model_ex, cov_dist, data_fs.pop_ids, [40], [32], 1e-2)

# Set Grid Points for Analysis
pts_l = [max(data_fs.sample_sizes)+20, max(data_fs.sample_sizes)+30, max(data_fs.sample_sizes)+40]

# Initialize Parameters and Optimization Boundaries
params = [1, 0.01]
lower_bounds = [1e-2, 1e-3]
upper_bounds = [100, 3]

# Perturb Parameters
p0 = dadi.Misc.perturb_params(params, fold=1, upper_bound=upper_bounds, lower_bound=lower_bounds)

# Run Optimization
# It is recommended to optimize the model at least 100 times
popt, ll_model = dadi.Inference.opt(p0, data_fs,demo_model_ex, pts_l, lower_bound=lower_bounds, upper_bound=upper_bounds, maxeval=600, verbose=0)

# Calculate Synonymous Theta
model_fs = demo_model_ex(popt, data_fs.sample_sizes, pts_l)
theta0 = dadi.Inference.optimal_sfs_scaling(model_fs, data_fs)

# Plot Demographic Model Results
dadi.Plotting.plot_1d_comp_multinom(model_fs, data_fs)
```

**Notes**:
- When working with low-pass sequencing data, avoid projecting down the data, as this can distort AFS estimates due to an excess of homozygosity. Instead, use subsampling methods to maintain data integrity and reflect the original AFS accurately.
- When creating the data dictionary, ensure that `calc_coverage=True` is set to include coverage information.
# Low-Pass Sequencing

To address the impact of low-pass sequencing on the Allele Frequency Spectrum (AFS), we developed a probabilistic model of low-pass biases from the Genome Analysis Toolkit (GATK) multi-sample calling pipeline that corrects for biases, particularly the loss of low-frequency variants and the misidentification of heterozygotes as homozygotes. These biases lead to fewer observed variant sites and skewed AFS ([Figure 1](LowPassSFS.png)). Our model calculates the probabilities of correctly calling variable sites, subsampling individuals, and estimating allele frequencies with higher accuracy.

The model considers a site as variant if at least two alternate allele reads are detected. Individuals are classified as missing if no reads are observed, homozygous if all reads match one allele, and heterozygous if reads for both alleles are present. The model then calculates the probability of misidentifying heterozygotes and adjusts allele frequencies accordingly. Subsampling is used to generate an AFS when not all individuals have been called.

In inbred populations, where homozygosity is elevated, the model accounts for these effects by incorporating a user-defined inbreeding coefficient \(F\), which represents the reduction in heterozygosity across the population. This coefficient is provided as a single value that reflects the overall level of inbreeding. When applying low-pass bias correction for inbred populations, users must initially specify an inbreeding coefficient. However, during optimization, they may iteratively adjust this value to refine the model's accuracy. Our simulations with moderate inbreeding (F = 0.5) demonstrated that if the initial inbreeding coefficient is underestimated, the optimization process tends to infer larger inbreeding values. Conversely, overestimating the coefficient results in smaller inferred values. A substantial discrepancy between the inbreeding coefficient used in the correction and the value inferred during optimization indicates that the initial assumption was suboptimal. Therefore, users should iteratively update the inbreeding coefficient in the low-pass model, refining the estimate with each iteration until the best possible inference is achieved.

![Low-Pass example SFS](LowPassSFS.png)

<p align="center"><b>Figure 1:</b> SFS from various depths of coverage: high (30x), medium (10x), and low (5x and 3x).</p>

### Usage

```python
make_low_pass_func_GATK_multisample(func, cov_dist, nseq, nsub, sim_threshold=1e-2, Fx=[0])
```

- **demo_model**: Specified demographic model in *dadi*.
- **cov_dist**: coverage distribution information calculated using the LowPass.compute_cov_dist function.
- **nseq**: Sequenced sample size (in haplotypes).
- **nsub**: Final sample size (in haplotypes).
- **sim_threshold**: This method switches between analytic and simulation-based methods. 
  - Setting this threshold to `0` will always use simulations, 
  - Setting it to `1` will always use analytics. 
  - Values in between indicate that simulations will be employed for thresholds below that value.
- **Fx**: Inbreeding coefficient. A single value or a sequence of values representing the inbreeding levels for different parts of the genome.

#### Define the demographic model (exponential growth)
```python
demo_model = dadi.Numerics.make_extrap_func(dadi.Demographics1D.growth)
```

#### Wrap the demographic model with low-pass model
```python
demo_model_lp = LowPass.make_low_cov_func(demo_model, cov_dist, nseq, nsub, sim_threshold=1e-2, Fx=[0])
```


#### Applying the Low-Pass Model to Any AFS
The `LowPass.make_low_pass_func_GATK_multisample` method enables users to seamlessly apply the low-pass model to any AFS. By leveraging a straightforward passthrough function `(_passthrough_sfs)`, the method processes the AFS directly through the low-pass model.

```python
def _passthrough_sfs(sfs, *args, **kwargs):
    return sfs
apply_low_pass_GATK_multisample = LowPass.make_low_pass_func_GATK_multisample(_passthrough_sfs, cov_dist, [32], [32], sim_threshold=1e-2, Fx=[0])
sfs_lowpass = apply_low_pass_GATK_multisample(sfs)
```

**Notes**:
- When working with low-pass sequencing data, avoid projecting down the data, as this can distort AFS estimates due to an excess of homozygosity. Instead, use subsampling methods to maintain data integrity and reflect the original AFS accurately.
- When creating the data dictionary, ensure that `calc_coverage=True` is set to include coverage information.
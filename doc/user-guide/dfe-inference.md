# DFE inference

Note: This section is adaptive from the fit∂a∂i manual written by Bernard Kim and Kirk Lohmueller. If you use this code, please be sure to cite their paper<sup>[8](./references.md)</sup>.

The code examples shown here are meant to work with the example dataset. For simplicity's sake, I have generated an example dataset with PReFerSIM<sup>[9](./references.md)</sup>. Furthermore, we will work with a relatively small sample size and simple demographic model so that the examples can be worked through quickly on a laptop. Lastly, all the example code is provided in the `example.py` script as well as in this document.

Another important thing to note: dadi characterizes genotype fitnesses as: 1, 1 + 2*sh*, and 1 + 2*s*, where 1 + 2*sh* is the fitness of the heterozygote. Furthermore, the DFEs inferred and scaled in terms of the ancestral population size: γ = 2*N*<sub><i>A</i></sub>*s*. This means that the selection coefficients must sometimes be rescaled, for instance when using the program SLiM<sup>[10](./references.md)</sup>. 

### Example dataset

The example dataset used in the example script was generated with forward simulations under the PRF model, with the simulation program PReFerSIM. Additionally, we will assume we know the true underlying demographic model rather than trying to fit one.

This dataset is summarized with a site frequency spectrum, has sample size 2*n* = 250 (125 diploids), and is saved in the file `sample.sfs` file. It was generated with a single size change demography and an underlying gamma DFE. Specifically, a population of size *N* = 10,000 diploids expands to 20,000 diploids 1000 generations ago and the gamma DFE has shape parameter 0.186 and scale parameter 686.7. This is the same gamma DFE that we inferred from the 1000 Genomes EUR dataset, but the scale parameter has been rescaled to the ancestral population size of 10,000 diploids. Finally, the amount of diversity in the sample dataset matches *θ*<sub><i>NS</i></sub> = 4000 = 4*N*<sub><i>A</i></sub>*μL*<sub><i>NS</i></sub>.

### Demographic inference

Because the usage of dadi for demographic inference is extensively documented, it will not be discussed in detail here. In practice, we find that, as long as the demographic model that fits the synonymous sites reasonably well also works well for inference of the DFE.

Briefly, we fit a demographic model to synonymous sites, which are assumed to be evolving in a neutral or nearly neutral manner. We believe this accurately represents the effects of linked selection and population structure, and condition upon this demographic model to fit the DFE. However, note the assumption of neutral synonymous variants may not hold for species with large population sizes, since this will increase the efficacy of selection on mutations with small fitness effects.

Our sample dataset was generated with a two epoch (single size change) demography. We will assume we infer a 2-fold population expansion 0.05 * 2*N*<sub><i>A</i></sub> generations ago, where *N*<sub><i>A</i></sub> is the ancestral population size. Therefore, the parameter vector is: `[nu, T]`.

Again, we assume that the population scaled nonsynonymous mutation rate, *θ*<sub><i>NS</i></sub> = 4000. In practice, we compute the synonymous mutation rate, *θ*<sub><i>S</i></sub>, by using the multinomial likelihood to fit the demographic model. Because this method only fits the proportional SFS, *θ*<sub><i>S</i></sub> is estimated with the `dadi.Inference.optimal_sfs_scaling` method. Then, we multiply *θ*<sub><i>S</i></sub> by 2.31 to get *θ*<sub><i>NS</i></sub>, *θ*<sub><i>S</i></sub> * 2.31 = *θ*<sub><i>NS</i></sub>. Remember that our sample size is 125 diploids (250 chromosomes).

### Pre-computing of the SFS for many γ

Next, we must generate frequency spectra for a range of gammas. The demographic function is modified to allow for a single selection coefficient. Here, each selection coefficient is scaled with the ancestral population size, γ = 2*N*<sub><i>A</i></sub>*s*. In other words, if `gamma = 0`, this function is the same as the original demographic function. This function is defined as `two_epoch` in `dadi.DFE.DemogSelModels.py`. Note the use of a Python decorator to easily define this as an extrapolating function.

	def two_epoch(params, ns, pts):
		nu, T, gamma = params
		xx = Numerics.default_grid(pts)

		phi = PhiManip.phi_1D(xx, gamma = gamma)
		phi = Integration.one_pop(phi, xx, T, nu, gamma = gamma)
		fs = Spectrum. from_phi(phi, ns, (xx,))

		return fs

### Fitting a DFE

#### Fitting simple DFEs

#### Fitting complex DFEs

### Fitting joint DFEs

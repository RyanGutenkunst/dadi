# DFE inference

Note: This section is adaptive from the fit∂a∂i manual written by Bernard Kim and Kirk Lohmueller. If you use this code, please be sure to cite their paper<sup>[8](./references.md)</sup>.

The code examples shown here are meant to work with the example dataset. For simplicity's sake, I have generated an example dataset with PReFerSIM<sup>[9](./references.md)</sup>. Furthermore, we will work with a relatively small sample size and simple demographic model so that the examples can be worked through quickly on a laptop. Lastly, all the example code is provided in the `example.py` script as well as in this document.

Another important thing to note: dadi characterizes genotype fitnesses as: \\(1\\), \\(1 + 2sh\\), and \\(1 + 2s\\), where \\(1 + 2sh\\) is the fitness of the heterozygote. Furthermore, the DFEs inferred and scaled in terms of the ancestral population size: \\(\gamma = 2N_As\\). This means that the selection coefficients must sometimes be rescaled, for instance when using the program SLiM<sup>[10](./references.md)</sup>. 

### Example dataset

The example dataset used in the example script was generated with forward simulations under the PRF model, with the simulation program PReFerSIM. Additionally, we will assume we know the true underlying demographic model rather than trying to fit one.

This dataset is summarized with a site frequency spectrum, has sample size \\(2n = 250\\) (125 diploids), and is saved in the file `sample.sfs` file. It was generated with a single size change demography and an underlying gamma DFE. Specifically, a population of size \\(N = 10,000\\) diploids expands to 20,000 diploids 1000 generations ago and the gamma DFE has shape parameter 0.186 and scale parameter 686.7. This is the same gamma DFE that we inferred from the 1000 Genomes EUR dataset, but the scale parameter has been rescaled to the ancestral population size of 10,000 diploids. Finally, the amount of diversity in the sample dataset matches \\(\theta_{NS} = 4000 = 4N_A\mu L_{NS}\\).

### Demographic inference

Because the usage of dadi for demographic inference is extensively documented, it will not be discussed in detail here. In practice, we find that, as long as the demographic model that fits the synonymous sites reasonably well also works well for inference of the DFE.

Briefly, we fit a demographic model to synonymous sites, which are assumed to be evolving in a neutral or nearly neutral manner. We believe this accurately represents the effects of linked selection and population structure, and condition upon this demographic model to fit the DFE. However, note the assumption of neutral synonymous variants may not hold for species with large population sizes, since this will increase the efficacy of selection on mutations with small fitness effects.

Our sample dataset was generated with a two epoch (single size change) demography. We will assume we infer a 2-fold population expansion \\(0.05 * 2N_A\\)generations ago, where \\(N_A\\)is the ancestral population size. Therefore, the parameter vector is: `[nu, T]`.

Again, we assume that the population scaled nonsynonymous mutation rate, \\(\theta_{NS} = 4000\\). In practice, we compute the synonymous mutation rate, \\(\theta_S\\), by using the multinomial likelihood to fit the demographic model. Because this method only fits the proportional SFS, \\(\theta_S\\) is estimated with the `dadi.Inference.optimal_sfs_scaling` method. Then, we multiply \\(\theta_S\\) by 2.31 to get \\(\theta_{NS}\\), i.e. \\(\theta_S * 2.31 = \\theta_{NS}\\). Remember that our sample size is 125 diploids (250 chromosomes).

### Pre-computing of the SFS for many \\(\gamma\\)

Next, we must generate frequency spectra for a range of gammas. The demographic function is modified to allow for a single selection coefficient. Here, each selection coefficient is scaled with the ancestral population size, \\(\gamma = 2N_As\\). In other words, if `gamma = 0`, this function is the same as the original demographic function. This function is defined as `two_epoch` in `dadi.DFE.DemogSelModels.py`.

	def two_epoch(params, ns, pts):
		nu, T, gamma = params
		xx = Numerics.default_grid(pts)

		phi = PhiManip.phi_1D(xx, gamma = gamma)
		phi = Integration.one_pop(phi, xx, T, nu, gamma = gamma)
		fs = Spectrum. from_phi(phi, ns, (xx,))

		return fs

Then, we generate the frequency spectra for a range of gammas. The following code generates expected frequency spectra, conditional on the demographic model fit to synonymous sites, over `gamma_pts` log-spaced points over the range of `gamma_bounds`. Additionally, the `mp=True` argument specifies usage of multiple cores/threads, which is convenient since this step takes the longest. If the argument `cpus` is passed, we will utilize that many cores, but if `mp=True` and no `cpus` argument is passed, we will use `n-1` threads, where `n` is the number of threads available. If `mp=False`, each SFS will be computed in serial. This step should take 1-10 minutes depending on your CPU.

    spectra = DFE.Cache1D(demog_params, ns, DFE.DemogSelModels.two_epoch, pts_l=pts_l, 
                          gamma_bounds=(1e-5, 500), gamma_pts=100, verbose=True,
                          mp=True)

Note, one error message that will come up often with very negative selection coefficients is:
`WARNING:Numerics:Extrapolation may have failed. Check resulting frequency spectrum for unexpected results.`

One way to fix this is by increasing the `pts_l` grid sizes -- this will need to increase as the sample size increases and/or if the integration is done over a range which includes stronger selection coefficients. `dadi.Numerics.make_extrap_func` is used to extrapolate the frequency spectra to infinitely many gridpoints, but will sometimes return tiny negative values (often \\(|X_i|<1\text{e}^{-50}\\)) due to floating point rounding errors. In practice, it seems that the tiny negative values do not affect the integration because they are insignificant, but if the error message appears it is good to manually double-check each SFS. Alternately, the small negative values can be manually approximated with 0.

In the example, the pre-computed SFSs are saved in the list `spectra.spectra`. For convenience's sake, the `spectra` object can be pickled.

    pickle.dump(spectra, open('example_spectra.bpkl','wb'))
    spectra = pickle.load(open('example_spectra.bpkl','rb'))

### Fitting a DFE

#### Fitting simple DFEs

Fitting a DFE is the quickest part of this procedure, especially for simple distributions such as the gamma distribution. If you wish to get an SFS for a specific DFE, you can use the `integrate` method that is built into the spectra object: `spectra.integrate(sel_params, None, sel_dist, theta, None)`. `sel_params` is a list containing the DFE parameters, `sel_dist` is the distribution used for the DFE, and `theta` is \\(\theta_{NS}\\). To compute the expected SFS for our simulations with the true parameter values, we would use `spectra.integrate([0.186, 686.7], None, Selection.gamma_dist, 4000., None)`. (The `None` arguments are for `ns` and `pts`, which are ignored. These are useful to ensure compatibility with dadi's optimization functions.)

First, load the sample data:

    data = dadi.Spectrum.from_file('example.fs')

Similar to the way in which vanilla dadi is used, you should have a starting guess at the parameters. Set an upper and lower bound. Perturb the parameters to select a random starting point, then fit the DFE. This should be done multiple times from different starting points. We use the `spectra.integrate` methods to generate the expected SFSs during each step of the optimization. The following lines of code fit a gamma DFE to the example data:

    sel_params = [0.2, 1000.]
    lower_bound, upper_bound = [1e-3, 1e-2], [1, 50000.]
    p0 = dadi.Misc.perturb_params(sel_params, lower_bound=lower_bound,
                                  upper_bound=upper_bound)
    popt, ll = dadi.Inference.opt(p0, data, spectra.integrate, pts=None,
                                       func_args=[DFE.PDFs.gamma, theta_ns],
                                       lower_bound=lower_bound, upper_bound=upper_bound, 
                                       verbose=len(sel_params), maxiter=10, multinom=False)

If this runs correctly, you should infer something close to, but not exactly, the true DFE. The final results will be stored in `popt`.
The expected SFS at the MLE can be computed with:

    model_sfs = spectra.integrate(popt, None, DFE.PDFs.gamma, theta_ns, None)

#### Fitting complex DFEs

Fitting complex distributions is similar to fitting simple DFEs, but requires a few additional steps. The following code can be used to fit a neutral+gamma mixture DFE to the data. Note that the gamma DFE should fit better if assessing model fit using AIC. Additionally, we assume that every selection coefficient \\(\gamma < 1\text{e}^{-4}\\) is effectively neutral. Since this is a mixture of two distributions, we infer the proportion of neutral mutations, \\(p_\text{neu}\\), and assume the complement of that (i.e. \\(1-p_\text{neu}\\)) is the proportion of new mutations drawn from a gamma distribution. Therefore, the parameter vector should be: [\\(p_\text{neu}\\),shape, scale]. The gamma DFE is still the true DFE.

    def neugamma(xx, params):
        pneu, alpha, beta = params
        xx = np.atleast_1d(xx)
        out = (1-pneu)*DFE.PDFs.gamma(xx, (alpha, beta))
        # Assume gamma < 1e-4 is essentially neutral
        out[np.logical_and(0 <= xx, xx < 1e-4)] += pneu/1e-4
        # Reduce xx back to scalar if it's possible
        return np.squeeze(out)

Fit the DFE as before, accounting for the extra parameter to describe the proportion of neutral new mutations. Note that \\(p_\text{neu}\\) is explicitly bounded to be \\(0 < p_\text{neu} \leq 1\\).

    sel_params = [0.2, 0.2, 1000.]
    lower_bound, upper_bound = [1e-3, 1e-3, 1e-2], [1, 1, 50000.]
    p0 = dadi.Misc.perturb_params(sel_params, lower_bound=lower_bound,
                                  upper_bound=upper_bound)
    popt = dadi.Inference.opt(p0, data, spectra.integrate, pts=None,
                              func_args=[neugamma, theta_ns],
                              lower_bound=lower_bound, upper_bound=upper_bound, 
                              verbose=len(sel_params),
                              maxiter=10, multinom=False)

For fitting with ancestral state misidentification or including a point mass of positive selection, see `example1D.py`.

### Fitting joint DFEs

dadi can also fit joint DFEs between populations, in a similar fashion to one-dimensional DFEs.

Caching is similar to the one-dimensional case, although note that it is generally much more computationally expensive.

        s2 = Cache2D(demo_params, ns, DemogSelModels.IM, pts=pts_l, gamma_pts=100,
                     gamma_bounds=(1e-2, 10), verbose=True, mp=True,
                     additional_gammas=[1.2, 4.3])

dadi currently includes a few simple models for joint DFEs in `dadi.DFE.PDFs`.
Note that the semi-analytic integration of the distribution over the regime not covered by the cache is expensive.
Therefore, C implementations of the PDFs can make a big difference in computational time, and we provide
C implementations for the default PDFs.

Calculating individual spectra is very similar to the 1D case.

    input_params, theta = [0.5,0.5,-0.8], 1e5
    sel_dist = PDFs.biv_lognormal
    target = s2.integrate(input_params, None, sel_dist, theta, None)

As is optimization.

    p0 = [0,1.,0.8]
    popt = dadi.Inference.opt(p0, data, s2.integrate, pts=None,
                              func_args=[sel_dist, theta],
                              lower_bound=[None,0,-1],
                              upper_bound=[None,None,1],
                              verbose=30, multinom=False)

Note that when a point mass of positive selection is included in a 2D DFE, the assumed value for the positive \\(\gamma\\) must be cached, otherwise evaluation would be too expensive.

dadi also implements mixture models, in which the total DFE is a sum of a 2D distribution plus a 1D distribution representing perfect correlation.
These are implemented by `dadi.DFE.mixture`.

    input_params, theta = [0.5,0.3,0,0.2,1.2,0.2], 1e5
    target = mixture_symmetric_point_pos(input_params,None,s1,s2,PDFs.lognormal,
                                         PDFs.biv_lognormal, theta)

Using a mixture model requires both a 1D and a 2D cache.

For additional examples, see `examples/DFE/example2D.py` in the source distribution.

# Triallelic spectra

The triallelic frequency spectrum is the distribution of frequencies of triallelic, instead of biallelic, SNPs. The triallelic spectrum stores the counts of observed alleles with given major and minor derived allele frequencies, where the major and minor derived alleles are those appearing at higher or lower frequency, respectively. We use a `dadi.Spectrum` object for the triallelic spectrum as well, with entries for infeasible triallelic frequencies masked. The `dadi.Triallele` methods can handle selection at one or both derived alleles, and can produce expected frequency spectra under arbitrary, single-population demography. By folding a triallelic frequency spectrum, we assume that we do not know which derived allel arose first.

### Build in models

In `dadi.Triallele.demographics.py`, you will find three pre-built demographic models: `equilibrium`, `two_epoch`, and `three_epoch`. The methods take demographic, selection, and integration parameters as inputs, as well as number of sampled individuals (`ns`), and number of grid points to use for integration (`pts`). For example, the parameters for the equilibrium model takes parameters `[sig1, sig2, theta1, theta2, misid, dt]`. The `sig` parameters are the selection coefficients for each derived allele, `theta` are the scaled mutation rates for each derived allele, `misid` is the probability of ancestral misidentification, and `dt` is the integration time step. For non-equilibrium demography, those models also take population sizes `nu` relative to the ancestral population size, times `T` for which the population is at size `nu`. An example can be found in the `examples/` directory from the [dadi source distribution](https://bitbucket.org/gutenkunstlab/dadi/src/master/).

### Faster triallele with Cython

The Triallele methods are written in Python; however, considerable speed-up can be achieved by generating some of the code in C. Some methods are written using Cython, and these can be installed by compiling the Cythonized code when dadi is built. To build the Cython extensions, use the flag `--cython` when installing dadi, by running `python setup.py install --cython`.

If for some reason installation fails to build the Cython modules, dadi can be installed without the Cythonized Triallele methods, and the Triallele methods will still be functional.

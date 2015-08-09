Diffusion Approximation for Demographic Inference

∂a∂i implements methods for demographic history and selection inference from genetic data, based on diffusion approximations to the allele frequency spectrum. One of ∂a∂i's main benefits is speed: fitting a two-population model typically takes around 10 minutes, and run time is independent of the number of SNPs in your data set. ∂a∂i is also flexible, handling up to three simultaneous populations, with arbitrary timecourses for population size and migration, plus the possibility of admixture and population-specific selection.

Originally ∂a∂i was initially developed by  Ryan Gutenkunst in the labs of Scott Williamson and Carlos Bustamante [http://www.bustamantelab.org] in Cornell's Department of Biological Statistics and Computational Biology. Ryan is now faculty in Molecular and Cellular Biology at the University of Arizona, and his group continues to work on ∂a∂i [http://gutengroup.mcb.arizona.edu].

**Getting started**

Please see the wiki pages on Getting Started [https://bitbucket.org/RyanGutenkunst/dadi/wiki/GettingStarted], Installation [https://bitbucket.org/RyanGutenkunst/dadi/wiki/Installation], and our Data Format [ [https://bitbucket.org/RyanGutenkunst/dadi/wiki/DataFormats].

Also, please sign up for our mailing list, where we help the community with ∂a∂i. Please do search the archives before posting. Many questions come up repeatedly. [http://groups.google.com/group/dadi-user]

**∂a∂i version 1.6.3 released**; Jul 12, 2012

This release improves `optimize_grid`, in response to a request by Xueqiu, and also adds the option to push optimization output to a file. It also includes a fix contributed by Simon Gravel for errors in extrapolation for very large spectra. Finally, spectra calculation for SNPs ascertained by sequencing a single individual has been added, in response to a request by Joe Pickrell.

**∂a∂i version 1.6.2 released**; Dec 4, 2011

This release fixes a long-standing bug in `make_fux_table`. (Testing suggests that in almost all cases the bug had little effect on results.) Also fixed is a bug that prevented optimizations from succeeding.

**∂a∂i version 1.6.1 released**; Nov 8, 2011

This release improves support for modeling admixture. It turns out that when the admixture methods in `PhiManip` are used, this can result in oscillations in the calculated frequency spectrum as the grid size is changes, which invalidates extrapolation. One solution is to run with `pts_l` of length one, so there is no extrapolation. Another solution, if the admixture is the last event in your model, is to implement admixture using the new `admix_props` argument to `Spectrum.from_phi`, rather than using the `PhiManip` methods.

[https://bitbucket.org/RyanGutenkunst/dadi/wiki/OldNews]
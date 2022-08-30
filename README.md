**Diffusion Approximation for Demographic Inference**

dadi implements methods for demographic history and selection inference from genetic data, based on diffusion approximations to the allele frequency spectrum. One of dadi's main benefits is speed: fitting a two-population model typically takes around 10 minutes, and run time is independent of the number of SNPs in your data set. dadi is also flexible, handling up to three simultaneous populations, with arbitrary timecourses for population size and migration, plus the possibility of admixture and population-specific selection.

Originally dadi was initially developed by  Ryan Gutenkunst in the labs of Scott Williamson and Carlos Bustamante (https://bustamantelab.stanford.edu/) in Cornell's Department of Biological Statistics and Computational Biology. Ryan is now faculty in Molecular and Cellular Biology at the University of Arizona, and his group continues to work on dadi and other topics (http://gutengroup.mcb.arizona.edu).

If you use dadi in your research, please cite RN Gutenkunst, RD Hernandez, SH Williamson, CD Bustamante "Inferring the joint demographic history of multiple populations from multidimensional SNP data" PLoS Genetics 5:e1000695 (2009).

**Getting started**

See the manual (https://dadi.readthedocs.io) and the example files in the source distribution. Full API documenation is available (https://dadi.readthedocs.io/en/latest/api/dadi/) and is in the source distribution, under doc/api/dadi/index.html .

Also, please sign up for our mailing list, where we help the community with ∂a∂i. Please do search the archives before posting. Many questions come up repeatedly. (http://groups.google.com/group/dadi-user)

The easiest way to install dadi is via conda, `conda install -c conda-forge dadi`.

**Notebook examples**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fbitbucket.org%2Fgutenkunstlab%2Fdadi%2Fsrc%2Fmaster/HEAD)

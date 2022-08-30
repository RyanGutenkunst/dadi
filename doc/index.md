# Welcome to dadi

### Introduction

dadi is a powerful software tool for simulating the joint frequency spectrum (FS) of genetic variation among multiple populations and employing the FS for population-genetic inference. An important aspect of dadi is its flexibility, particularly in model specification, but with that flexibility comes some complexity. dadi is not a GUI program, nor can dadi be run usefully with a single command at the command-line; using dadi requires at least rudimentary Python scripting. Luckily for us, Python is a beautiful and simple language. Together with a few examples, this manual will quickly get you productive with dadi even if you have no prior Python experience.

#### Getting help

Please join the `dadi-user` Google group: [https://groups.google.com/group/dadi-user](https://groups.google.com/group/dadi-user). `dadi-user` is the preferred forum for asking questions and getting help. Before posting a question, take a moment to look through the `dadi-user` archives to see if your question has already been addressed. There are example scripts included in the source distribution: [https://bitbucket.org/gutenkunstlab/dadi/src/master/examples](https://bitbucket.org/gutenkunstlab/dadi/src/master/examples).

#### Helping us

As we do our own research, dadi is constantly improving. Our philosophy is to include in dadi any code we develop for our own projects that may useful to others. Similarly, if you develop dadi-related code that you think might be useful to others, please let us know so we can include it with the main distribution. If you have particular needs that modification to dadi may fulfill, please contact the developers and we may be able to help.

### Citations

If you find dadi useful in your research, please cite:
[RN Gutenkunst, RD Hernandez, SH Williamson, CD Bustamante "Inferring the joint demographic history of multiple populations from multidimensional SNP data" PLoS Genetics 5:e1000695 (2009)](http://doi.org/10.1371/journal.pgen.1000695).

If you find the Godambe Information Matrix methods useful, please cite:
[AJ Coffman, P Hsieh, S Gravel, RN Gutenkunst "Computationally efficient composite likelihood statistics for demographic inference" Molecular Biology and Evolution 33:591 (2016)](http://doi.org/10.1093/molbev/msv255).

If you find the DFE inference methods useful, please cite:
[BY Kim, CD Huber, KE Lohmueller "Inference of the Distribution of Selection Coefficients for New Nonsynonymous Mutations Using Large Samples" Genetics 206:345 (2017)](https://doi.org/10.1534/genetics.116.197145).

If you find the triallelic methods useful, please cite:
[AP Ragsdale, AJ Coffman, P Hsieh, TJ Struck, RN Gutenkunst "Triallelic population genomics for inferring correlated fitness effects of same site nonsynonymous mutations" Genetics 203:513 (2016)](http://doi.org/10.1534/genetics.115.184812).

If you find the two-locus methods useful, please cite:
[AP Ragsdale, RN Gutenkunst "Inferring demographic history using two-locus statistics" Genetics 206:1037 (2017)](http://doi.org/10.1534/genetics.117.201251).

If you find the joint DFE inference methods useful, please cite:
[X Huang, AL Fortier, AJ Coffman, TJ Struck, MN Irby, JE James, JE León-Burguete, AP Ragsdale, RN Gutenkunst "Inferring Genome-Wide Correlations of Mutation Fitness Effects between Populations" Molecular Biology and Evolution 38:4588–4602 (2021)](https://doi.org/10.1093/molbev/msab162).


### Suggested workflow

One of Python’s major strengths is its interactive nature. This is very useful in the ex-ploratory stages of a project: for examining data and testing models. If you intend to use dadi’s plotting commands, which rely on `matplotlib`, they you’ll almost certainly want to install IPython, an enhanced Python shell that fixes several difficulties with interactive plotting using `matplotlib`.

My preferred workflow involves one window editing a Python script (e.g. `script.py`) and another running an IPython session (started as `ipython --pylab`). In the IPython session I can interactively use dadi, while I record my work in `script.py`. IPython’s `%run script.py` magic command lets me apply changes I’ve made to script.py to my interactive session. (Note that you will need to reload other Python modules used by your script if you change them.) Once I’m sure I’ve defined my model correctly and have a useful script, I run that from the command line (python `script.py`) for extended optimizations and other long computations.

Note that to access dadi’s functions, you will need to `import dadi` at the start of your script or interactive session.

If you are comfortable with Matlab, this workflow should seem very familiar. Moreover the `numpy`, `scipy`, and `matplotlib` packages replicate much of Matlab’s functionality.

### Interactive examples with Jupyter notebooks

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fbitbucket.org%2Fgutenkunstlab%2Fdadi%2Fsrc%2Fmaster/HEAD)

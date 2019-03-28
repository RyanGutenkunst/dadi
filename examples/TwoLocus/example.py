import dadi
import dadi.TwoLocus
import numpy as np

"""
Examples for
1. Loading and manipulating two-locus spectrum
2. Calculating expected spectrum using demographic history functions
    Integrating the frequency spectrum with population size changes begins
    from equlibrium, which we attempt to load from cache for a given grid
    points, dt, rho, etc.
    First computing the equilibrium spectrum can take a while, since we must
    integrate for ~20 time units.
    The default cache for two locus spectra is ~/.dadi/TwoLocus_cache/
    The cached spectra for these examples are included in the directory
        TwoLocus.
    Move these cached npz files to ~/.dadi/TwoLocus_cache/ to avoid having to
    cache these yourself!
    Note that changing rho, grid points, dts, gammas, etc will require new
    cached spectra
3. Inference demographic parameters from example spectrum
"""

## 1. We'll first import a saved two-locus spectrum
# this was simulated/sampled with rho=1, nu=2.0, T=0.1
fs = dadi.TwoLocus.TLSpectrum.from_file('example_observed_fs.fs')
# get the sample size of the spectrum
ns = fs.sample_size
# fold the spectrum (ancestral state unknown)
fs_folded = fs.fold()
# project to a smaller sample size
fs_project = dadi.TwoLocus.numerics.project(fs, ns//2)
# compute mean r squared for observed variable sites
r2 = fs.mean_r2()


## 2. Create a two locus spectrum under a demographic history model
# two epoch model, with a recent size change of 2.0 for 0.1 (2*N_e) time units
nu,T = 2.0, 0.1
# sample size
ns = 20
# population size-scaled recombination rate between loci
rho = 1.0
# set grid points and time steps to integrate over
# we want the number of grid points to be larger than the sample size (here 20)
gridpts = [40,50,60]
# we also extrapolate over dt values, which improves accuracy
dts = [0.005,0.0025,0.001]

# note that the first time computing the equilibrium spectrum for a given 
# dt and pts can take a while, which is why we cache those spectra

# Here we copy the precached spectra we have provided to the central cache
# location (by default, ~/.dadi/TwoLocus_cache). In general, a user won't need
# to do this. But if you are moving between machines, you may want to copy your
# cache over as well, to save substantial calculation time.
import glob, shutil
for fname in glob.glob('TwoLocus_cache/*.npz'):
    shutil.copy(fname, dadi.TwoLocus.demographics.cache_path)

spectra = {}
for dt in dts:
    spectra.setdefault(dt,{})
    for pts in gridpts:
        print('computing for dt, pts = {0}, {1}'.format(dt,pts))
        spectra[dt][pts] = dadi.TwoLocus.demographics.two_epoch((nu,T), pts, ns, rho=rho, dt=dt)

# extrap_dt_pts takes a dictionary in the form spectra[dts][pts] and
# extrapolates the calculated spectra over dt and pts to output the expected
# frequency spectrum
F = dadi.TwoLocus.numerics.extrap_dt_pts(spectra)

# save this spectrum to working directory
F.to_file('./two_epoch_example.fs')

## 3. Fit a two-epoch model to the spectrum that we loaded in 1.
fs = dadi.TwoLocus.TLSpectrum.from_file('example_observed_fs.fs')
# The inference/optimize functions takes a list of two-locus frequency spectra,
# and a list of the rho values for each of the spectra in the list, in the same
# order as the spectra in the list. Here, we only are optimizing over a single
# frequency spectrum for rho = 1.
data_list = [fs]
rho_list = [1.0]
# we know that the sample spectrum was generated under a demography with 
# nu = 2.0, T = 0.1, so for illustration purposes we pick initial values close
# to the true values
initial_guess = dadi.Misc.perturb_params([2.0,0.1], fold=0.2)

gridpts = [40,50,60]
dts = [0.005,0.0025,0.001]

# do_calculation computes the expected frequency spectrum for a given sample
# size and rho
def do_calculation(arguments):
    params, pts, ns, rho, dt = arguments
    # create spectrum for given param set, rho, etc
    temps = {}
    for dt in dts:
        temps.setdefault(dt,{})
        for numpts in pts:
            temps[dt][numpts] = dadi.TwoLocus.demographics.two_epoch(params,numpts,ns,rho=rho,dt=dt)
    return dadi.TwoLocus.TLSpectrum(dadi.TwoLocus.numerics.extrap_dt_pts(temps))

# optimize functions needs a function to output a list of spectra over the rho-values in the rhos list
def model_func(params, ns, pts, dts, rhos=[0]):
    inputs = [(params,pts,ns,rho,dts) for rho in rhos]
    output = [do_calculation(input) for input in inputs]
    return output

opt_params = dadi.TwoLocus.inference.optimize_log_fmin(initial_guess, data_list, model_func, gridpts, dts, rhos=rho_list, verbose=1)

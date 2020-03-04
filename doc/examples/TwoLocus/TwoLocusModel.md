# Two Locus Model


```python
import dadi
import dadi.TwoLocus
import numpy as np
```

### Loading and manipulating two-locus spectrum


```python
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
```

### Calculating expected spectrum using demographic history functions


```python
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
```

    computing for dt, pts = 0.005, 40
    computing for dt, pts = 0.005, 50
    computing for dt, pts = 0.005, 60
    computing for dt, pts = 0.0025, 40
    computing for dt, pts = 0.0025, 50
    computing for dt, pts = 0.0025, 60
    computing for dt, pts = 0.001, 40
    computing for dt, pts = 0.001, 50
    computing for dt, pts = 0.001, 60


### Inference demographic parameters from example spectrum


```python
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
```

    1       , -3413.32    , array([ 1.84088    ,  0.093625   ])
    2       , -3405.48    , array([ 1.89791    ,  0.093625   ])
    3       , -3430.38    , array([ 1.84088    ,  0.083169   ])
    4       , -3396.61    , array([ 1.89791    ,  0.105396   ])
    5       , -3396.23    , array([ 1.92709    ,  0.118646   ])
    6       , -3399.13    , array([ 1.98679    ,  0.118646   ])
    7       , -3446.44    , array([ 2.01734    ,  0.150353   ])
    8       , -3395.18    , array([ 1.92709    ,  0.105396   ])
    9       , -3398.65    , array([ 1.86918    ,  0.105396   ])
    10      , -3395.77    , array([ 1.89791    ,  0.108562   ])
    11      , -3401.33    , array([ 1.89791    ,  0.0964382  ])
    12      , -3394.74    , array([ 1.91975    ,  0.112655   ])
    13      , -3394.55    , array([ 1.94926    ,  0.109369   ])
    14      , -3394.83    , array([ 1.97546    ,  0.109774   ])
    15      , -3396.39    , array([ 1.94184    ,  0.116902   ])
    16      , -3394.67    , array([ 1.93077    ,  0.108161   ])
    17      , -3394.27    , array([ 1.96045    ,  0.105006   ])
    18      , -3394.26    , array([ 1.98112    ,  0.101379   ])
    19      , -3394.09    , array([ 2.0001     ,  0.10251    ])
    20      , -3393.92    , array([ 2.03569    ,  0.0997966  ])
    21      , -3395.43    , array([ 2.06896    ,  0.0925057  ])
    22      , -3394.09    , array([ 1.97852    ,  0.104885   ])
    23      , -3394.22    , array([ 2.03303    ,  0.103248   ])
    24      , -3393.95    , array([ 2.01992    ,  0.102778   ])
    25      , -3394.06    , array([ 2.07829    ,  0.0977916  ])
    26      , -3394.05    , array([ 2.05289    ,  0.0995186  ])
    27      , -3393.97    , array([ 2.00301    ,  0.103065   ])
    28      , -3394.22    , array([ 2.01536    ,  0.102166   ])
    29      , -3394.08    , array([ 2.02779    ,  0.101276   ])
    30      , -3393.97    , array([ 2.04427    ,  0.0996575  ])
    31      , -3393.96    , array([ 2.05224    ,  0.0982017  ])
    32      , -3393.97    , array([ 2.04362    ,  0.0983388  ])


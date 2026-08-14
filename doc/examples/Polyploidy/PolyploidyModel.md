# Polyploidy model examples

This notebook demonstrates how to use the `dadi.Polyploidy.Integration` module to simulate and define more complex polyploidy models.

### Model of allotetraploids with diploid progenitors

Here, we outline a demographic model in which we jointly model the SFS of the two subgenomes of an allotetraploid with both diploid progenitors. In this model, one of the diploid progenitors will grow exponentially, the other will have a single size change, and the allotetraploid will have a bottlegrowth size history. We will also include symmetric migration between the diploid progenitors and homoeologous exchange between the subgenomes. 

We start by loading the relevant modules from `dadi` and `numpy`.

```python
# load required modules
from dadi import Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum
from dadi.Polyploidy import Integration
from dadi.Polyploidy.Integration import PloidyType
import numpy
```

Then, we can define the demographic model

```python
# define the demographic model function
def allotetraploid_with_diploid_progenitors(params, ns, pts):
    """
    Model of allotetraploid formation with diploid progenitors.
    
    Parameters:
        params (tuple): (T_dip_div, nu_f_dip1, nu_f_dip2, M, T_WGD, nu_b_allo, nu_f_allo, H)
            - T_dip_div: Total time of the diploid progenitor divergence
               (in units of 2*Na generations).
            - nu_f_dip1: Final/contemporary effective population size of the first diploid progenitor 
                after a period of exponential growth begining at the.
            - nu_f_dip2: Final/contemporary effective population size of the second diploid progenitor
                after a single size change at the time of WGD.
            - M: Migration rate between the two diploid progenitors for all epochs (2*Na*m).
            - T_WGD: Time in the past at which the WGD occurred, creating the  
               allotetraploid population (in units of 2*Na generations).
            - nu_b_allo: Bottlenecked effective population size of the allotetraploid population.
            - nu_f_allo: Final/contemporary effective population size of the allotetraploid population.
            - H: homoeologous exchange rate between the two subgenomes, = 2*Na*eta
        ns (tuple): Sample sizes (n_dip1, n_dip2, n_allo, n_allo).
        pts (int): Number of grid points to use in integration.
    
    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    # unpack the parameters
    T_dip_div, nu_f_dip1, nu_f_dip2, M, T_WGD, nu_b_allo, nu_f_allo, H = params
    
    # initialize the numerical grid and phi
    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)
    
    # model the split of the diploid progenitors
    phi = PhiManip.phi_1D_to_2D(xx, phi)

    # function for the size of the first diploid progenitor
    T_total = T_dip_div + T_WGD
    nu_func_dip1 = lambda t: numpy.exp(numpy.log(nu_f_dip1) * t/T_total)

    # integrate to model the diploid progentitor divergence
    phi = Integration.two_pops(phi, xx, T_dip_div, 
                               nu1=nu_func_dip1, nu2=nu_f_dip2, 
                               m12=M, m21=M)
    
    # model the allotetraploid formation by splitting phi twice more
    # the third dimension is subgenome A derived entirely from the first diploid progenitor
    phi = PhiManip.phi_2D_to_3D(phi, 1, xx, xx, xx)
    # the fourth dimension is subgenome B derived entirely from the second diploid progenitor
    phi = PhiManip.phi_3D_to_4D(phi, 0, 1, xx, xx, xx, xx)

    # for the second integration, time will be reset to run from 0 to T_WGD, so we need to
    # redefine the function for the size of the first diploid progenitor
    nu_func_dip1 = lambda t: numpy.exp(numpy.log(nu_f_dip1) * (t+T_dip_div)/T_total)
    # and define the function for the bottlegrowth size of the allotetraploid
    nu_allo = lambda t: nu_b_allo * numpy.exp(numpy.log(nu_f_allo/nu_b_allo) * t/T_WGD)

    # then, integrate for a period of T_WGD to reach the present
    phi = Integration.four_pops(phi, xx, T_WGD, 
                                nu1=nu_func_dip1, nu2=nu_f_dip2, nu3=nu_allo, nu4=nu_allo,
                                m12=M, m21=M, m34=H, m43=H,
                                ploidyflag3=PloidyType.ALLOa, ploidyflag4=PloidyType.ALLOb)
    
    # and calculate and return the SFS
    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx, xx))
    return fs
```

and use it to simulate a sample SFS
```python
# then, we can generate an example SFS
# params = (T_dip_div, nu_f_dip1, nu_f_dip2, M, T_WGD, nu_b_allo, nu_f_allo, H)
params = [1.0, 2, 0.5, 0.5, 0.25, 0.6, 1.5, 0.1]
ns = (24, 36, 30, 30)
sfs = 5000 * allotetraploid_with_diploid_progenitors(params, ns, max(ns)+11)
# optionally, we can save the SFS to a file by uncommenting the following line
sfs.to_file('allotetraploid_with_diploid_progenitors.sfs')
```

Then we can use many of the existing methods in `dadi` to visualize the model and the SFS. For example, we can plot visualize the demographic model using the `demes` and `demesdraw` packages:

```python
import demes, demesdraw
# define the demes graph with relabeled demes/populations
g = dadi.Demes.output(Nref=10000, generation_time=2, 
                      deme_mapping={'Ancestral':['d1_1'],
                                    'Progenitor_A':['d2_1', 'd3_1'],
                                    'Progenitor_B':['d2_2', 'd3_2'],
                                    'Subgenome_A':['d3_3'],
                                    'Subgenome_B':['d4_4']})

# optionally, we can save the demes graph to a file
g.description = 'Example model of allotetraploid formation with diploid progenitors'
demes.dump(g, 'polyploidy_example.yaml')

# and, we can also visualize the demographic model with the demesdraw package
ax = demesdraw.tubes(g)
ax.figure.savefig('allotetraploid_w_dips_demesdraw.png')
```

![png](allotetraploid_w_dips_demesdraw.png)

Or, using the `Plotting` module in `dadi` to visualize the SFS:

```python
import matplotlib.pyplot as plt
# if saved to a file, we can load the SFS
sfs = dadi.Spectrum.from_file('allotetraploid_with_diploid_progenitors.sfs')
# then, we combine the two subgenomes for plotting
sfs_collapsed = sfs.combine_pops([3,4])

dadi.Plotting.plot_3d_pairwise(sfs_collapsed, pop_ids=['Progenitor A', 'Progenitor B', 'Allotetraploid'], show=False)
fig = plt.gcf()
fig.set_size_inches(6.5,2)
fig.savefig('allotetraploid_w_dips_sfs.png', bbox_inches='tight')
```

![png](allotetraploid_w_dips_sfs.png)



This model is relatively complex and is primarily meant to demonstrate how to handle a more realistic model of allotetraploid populations. In general, we recommend starting with a simpler model than this one and adding complexity as needed to better fit the data.

### Model of autotetraploids with non-additive selection

### Model of autohexaploid formation


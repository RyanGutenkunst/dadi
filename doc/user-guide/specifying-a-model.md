# Specifying a model

A demographic model specifies population sizes and migration rates as function of time, and it also includes discrete events such as population splittings and admixture. Unlike many coalescent-based simulators, demographic models in dadi are specified forward in time. Also note that all population sizes within a demographic model are specified relative to some reference population size \\(N_{\text{ref}}\\).

One important subtlety is that within the demographic model function, by default the mutation parameter \\(\theta = 4N_{\text{ref}}\mu\\) is set to 1. This is because the optimal \\(\theta\\) for a given model and set of data is trivial to calculate, so dadi by default does this automatically in optimization (so-called "multinomial" optimization). See the Fixed \\(\theta\\) section for how to fix theta to a particular value in a demographic model.

### Implementation

Demographic models are specified by defining a Python function. This function employs various methods defined by dadi to specify the demography.

When defining a demographic function the arguments must be specified in a particular order. The *first* argument must be a list of free parameters that will be optimized. The *second* argument (usually called `ns`) must be a list of sample sizes. The `last` argument (usually called `pts`) must be the number of grid points used in the calculation. Any additional arguments (between the second and last) can be used to pass additional non-optimized parameters, using the `func_args` argument of the optimization methods. (See Listing 8 for an example.) The demographic model function tracks the evolution of \\(\phi\\) the density of mutations within the populations at given frequencies. This continuous density \\(\phi\\) is approximated by its values on a grid of points, represented by the `numpy` array phi. Thus the first step in a demographic model is to specify that grid:

	xx = dadi.Numerics.default_grid(pts)

Here `pts` is the number of grid points in each dimension for representing \\(\phi\\).

All demographic models employed in dadi must begin with an equilibrium population of non-zero size. \\(\phi\\) for such a population can be generated using the method `PhiManip.phi_1D`. The most important parameter to this method is `nu`, which specifies the relative size of this ancestral population to the reference population. Most often, the reference population is the ancestral, so `nu` defaults to 1.

Once we've created an initial \\(\phi\\), we can begin to manipulate it. First, we can split \\(\phi\\) to simulate population splits. This can be done using the methods `PhiManip.phi_1D_to_2D`, `PhiManip.phi_2D_to_3D_split_1`, and `PhiManip.phi_2D_to_3D_split_2`. These methods take in an input \\(\phi\\) of either one or two dimensions, and output a \\(\phi\\) of one greater dimension, corresponding to addition of a population. The added population is the last dimension of \\(\phi\\). For example, if `PhiManip.phi_2D_to_3D_split_1` is used, population 1 will split into populations 1 and 3. `phi_2D_to_3D_admix` is a more advanced version of the `2D_to_3D` methods that incorporates admixture. In this method, the proportions of pop 3 that are derived from pop 1 and pop 2 may be specified.

Direct admixture events can be specified using the methods `phi_2D_admix_1_into_2`, `phi_2D_admix_2_into_1`, `phi_3D_admix_1_and_2_into_3`, `phi_3D_admix_1_and_3_into_2`, and `phi_3D_admix_2_and_3_into_1`. These methods do not change the dimensionality of \\(\phi\\), but rather simulate discrete admixture events. For example, `phi_2D_admix_1_into_2` can be used to simulate a large discrete influx of individuals from pop 1 into pop 2. For example, this might model European (pop 1) admixture into indigenous Americans (pop 2). Note that the `PhiManip` methods for admixture can compromise the effectiveness of extrapolation for evaluating entries in the frequency spectrum corresponding to SNPs private to the recipient population. If your model involves admixture, you may obtain better accuracy by avoiding extrapolation and instead setting `pts_l` to be a list of length 1. Alternatively, if the admixture is the final event in your model, you can model admixture using the `admix_props` arguments for `Spectrum.from_phi`.

Along with these discrete manipulations of \\(\phi\\), we have the continuous transformations as time passes, due to genetic drift at different population sizes or migration. This is handled by `Integration` methods, `Integration.one_pop`, `Integration.two_pops`, and `Integration.three_pops`. Each of these methods must be used with a `phi` of the appropriate dimensionality. `Integration.one_pop` takes two crucial parameters, `T` and `nu`. `T` specifies the time of this integration and `nu` specifies the size of this population relative to the reference during this time period. `Integration.two_pop` takes an integration time `T`, relative sizes for populations 1 and 2 `nu1` and `nu2`, and migration parameters `m12` and `m21`. The migration parameter `m12` specifies the rate of migration *from pop2 into pop1*. It is equal to the fraction of individuals each generation in pop 1 that are new migrants from pop 2, times the \\(2N_{\text{ref}}\\). `Integration.three_pops` is a straightforward extension of `two_pops` but now there are three population sizes and six migration parameters.

Note that for all these methods, the integration time `T` must be positive. To ensure this, it is best to define your time parameters as the *interval between* events rather than the absolute time of those events. For example, a size change happened a time `Tsize` before a population split `Tsplit` in the past.

Importantly, population sizes and migration rates (and selection coefficents) may be functions of time. This allows one to simulate exponential growth and other more complex scenarios. To do so, simply pass a function that takes a single argument (the time) and returns the given variable. The Python `lambda` expression is a convenient way to do this. For example, to simulate a single population growing exponentially from size `nu0` to size `nuF` over a time `T`, one can do:

	nu_func = lambda t: nu0 * (nuF/nu0) ** (t/T)
	phi = Integration.one_pop(nu = nu_func, T = T)

Numerous examples are provided in Listings 2 through 8.

### Units

The units dadi uses are slightly different than those used by some other programs, *ms* in particular.

In dadi, \\(\theta = 4N_{\text{ref}}\mu\\), as is typical.

Times are given in units of \\(2N_{\text{ref}}\\) generations. This differs from *ms*, where time is in units of \\(4N_{\text{ref}}\\) generations. So to convert from a time in dadi to a time in *ms*, *divide* by 2.

Migration rates are given in units of \\(M_{ij} = 2N_{\text{ref}}m_{ij}\\). Again, this differs from *ms*, where the scaling factor is \\(4N_{\text{ref}}\\) generations. So to get equivalent migration (\\(m_{ij}\\) in *ms* for a given rate in dadi, *multiply* by 2.

### Fixed \\(\theta\\)

If you wish to set a fixed value of \\(\theta = 4N_0\mu\\) in your analysis, that information must be provided to the initial \\(\phi\\) creation function and the `Integration` functions. For an example, see Listing 7, which defines a demographic model in which \\(\theta\\) is fixed to be 137 for derived population 1. Derived pop 1 is thus the reference population for specifying all population sizes, so its size is set to 1 in the call to `Integration.two_pops`. When fixing \\(\theta\\), every `Integration` function must be told what the reference \\(\theta\\) is, using the option `theta0`. In addition, the methods for creating an initial \\(\phi\\) distribution must be passed the appropriate value of \\(\theta\\) using the `theta0` option.

### Ancient sequences

If you have DNA samples from multiple timepoints, you can construct a frequency spectrum in which different axes correspond to samples from different timepoints. To support this in dadi, the `Integration.one_pop`, `two_pops`, and `three_pops` support `freeze` arguments. When `True`, these arguments will "freeze" a particular population so that it no longer changes (although the relationship between SNPs in the frozen and unfrozen populations will change). Note that because time in dadi models is in genetic units, you need to be careful in how you specify the time of collection of your frozen sample. In this case, you likely want to run a model that explicitly includes \\(\theta\\) as parameter (See the Fixed \\(\theta\\) section), so that you can convert from physical to genetic units within the model function.

### Code examples

	def bottleneck(params, ns, pts):
		nuB, nuF, TB, TF = params
		xx = Numerics.default_grid(pts)

		phi = PhiManip.phi_1D(xx)
		phi = Integration.one_pop(phi, xx, TB, nuB)
		phi = Integration.one_pop(phi, xx, TF, nuF)

		fs = Spectrum.from_phi(phi, ns, (xx,))
		return fs

<p><strong>Listing 2 Bottleneck</strong>: At time <code>TF + TB</code> in the past, an equilibrium population goes through a bottleneck of depth <code>nuB</code>, recovering to relative size <code>nuF</code></p>

	def growth(params, ns, pts):
		nu, T = params
		xx = Numerics.default_grid(pts)
		phi = PhiManip.phi_1D(xx)

		nu_func = lambda t : numpy.exp(numpy.log(nu) * t/T)
		phi = Integration.one_pop(phi, xx, T, nu_func)

		fs = Spectrum.from_phi(phi, ns, (xx,))
		return fs

<p><strong>Listing 3 Exponential growth</strong>: At time <code>T</code> in the past, an equilibrium population begins growing exponentially, reaching size <code>nu</code> at present</p>

	def split_mig(params, ns, pts):
		nu1, nu2, T, m = params
		xx = Numerics.default_grid(pts)

		phi = PhiManip.phi_1D(xx)
		phi = PhiManip.phi_1D_to_2D(xx, phi)
		phi = Integration.two_pops(phi, xx, T, nu1, nu2, m12 = m, m21 = m)

		fs = Spectrum.from_phi(phi, ns, (xx, xx))
		return fs

<p><strong>Listing 4 Split with migration</strong>: At time <code>T</code> in the past, two populations diverge from an equilibrium population, with relative sizes <code>nu1</code> and <code>nu2</code> and with symmetric migration at rate <code>m</code></p>

	def IM(params, ns, pts):
		s, nu1, nu2, T, m12, m21 = params
		xx = Numerics.default_grid(pts)

		phi = PhiManip.phi_1D(xx)
		phi = PhiManip.phi_1D_to_2D(xx, phi)

		nu1_func = lambda t : s * (nu1/s) ** (t/T)
		nu2_func = lambda t : (1-s) * (nu2/(1-s)) ** (t/T)

		phi = Integration.two_pops(phi, xx, T, nu1_func, nu2_func, m12 = m12, m21 = m21)

		fs = Spectrum.from_phi(phi, ns, (xx, xx))
		return fs

<p><strong>Listing 5 Two-population isolation-with-migration</strong>: The ancestral population splits into two, with a fraction <code>s</code> going into pop 1 and fraction <code>1-s</code> into pop 2. The populations then grow exponentially, with asymmetric migration allowed between them</p>

	from dadi import Numerics, PhiManip, Integration, Spectrum
	
	def OutOfAfrica(params, ns, pts):
		nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs = params
		xx = Numerics.default_grid(pts)

		phi = PhiManip.phi_1D(xx)
		phi = Integration.one_pop(phi, xx, TAf, nu = nuAf)

		phi = PhiManip.phi_1D_to_2D(xx, phi)
		phi = Integration.two_pops(phi, xx, TB, nu1 = nuAf, nu2 = nuB, m12 = mAfB, m21 = mAfB)

		phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
		
		nuEu_func = lambda t : nuEu0 * (nuEu/nuEu0) ** (t/TEuAs)
		nuAs_func = lambda t : nuAs0 * (nuAs/nuAs0) ** (t/TEuAs)
		phi = Integration.three_pops(phi, xx, TEuAs, nu1 = nuAf, nu2 = nuEu_func, nu3 = nuAs_func, m12 = mAfEu, m13 = mAfAs, m21 = mAfEu, m23 = mEuAs, m31 = mAfAs, m32 = mEuAs)

		fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
		return fs

<p><strong>Listing 6 Out-of-Africa model from Gutenkust (2009)</strong>: This model involves a size change in the ancestral population, a split, another split, and then exponential growth of populations 1 and 2. (The <code>from dadi import</code> line imports those modules from the <code>dadi</code> namespace into the local namespace, so we don't have to type <code>dadi.</code> to access them.)</p>

	def fixed_theta(params, ns, pts):
		nuA, nu2, T = params
		theta1 = 137

		xx = dadi.Numerics.default_grid(pts)

		phi = dadi.PhiManip.phi_1D(xx, nu = nuA, theta0 = theta1)
		phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
		phi = dadi.Integration.two_pops(phi, xx, T, nu1 = 1, nu2 = nu2, theta0 = theta1)

		fs = dadi.Spectrum.from_phi(phi, ns, (xx, xx))
		return fs

<p><strong>Listing 7 Fixed <i>θ</i></strong>: A split demographic model function with a fixed value of <i>θ</i> = 137 for derived population 1. The free parameters are the sizes of the ancestral pop, <code>nuA</code>, and derived pop 2, <code>nu2</code>, (relative to derived pop 1), along with the divergence time <code>T</code> between the two derived pops</p>

	from dadi import Numerics, PhiManip, Integration, Spectrum

	def NewWorld(params, ns, fixed_params, pts):
		nuEu0, nuEu, nuAs0, nuAs, nuMx0, nuMx, mEuAs, TEuAs, TMx, fEuMx = params
		theta0, nuAf, nuB, mAfB, mAfEu, mAfAs, TAf, TB = fixed_params
		xx = Numerics.default_grid(pts)

		phi = PhiManip.phi_1D(xx)
		phi = Integration.one_pop(phi, xx, TAf, nu = nuAf)

		phi = PhiManip.phi_1D_to_2D(xx, phi)
		phi = Integration.two_pops(phi, xx, TB, nu1 = nuAf, nu2 = nuB, m12 = mAfB, m21 = mAfB)

		# Integrate out the YRI population
		phi = Numerics.trapz(phi, xx, axis = 0)

		phi = PhiManip.phi_1D_to_2D(xx, phi)
		nuEu_func = lambda t : nuEu0 * (nuEu/nuEu0) ** (t/(TEuAs+TMx))
		nuAs_func = lambda t : nuAs0 * (nuAs/nuAs0) ** (t/(TEuAs+TMx))
		phi = Integration.two_pops(phi, xx, TEuAs, nu1 = nuEu_func, nu2 = nuAs_func, m12 = mEuAs, m21 = mEuAs)

		phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)

		# Initial population sizes for this stretch of integration
		nuEu0 = nuEu_func(TEuAs)
		nuAs0 = nuAs_func(TEuAs)
		nuEu_func = lambda t : nuEu0 * (nuEu/nuEu0) ** (t/TMx)
		nuAs_func = lambda t : nuAs0 * (nuAs/nuAs0) ** (t/TMx)
		nuMx_func = lambda t : nuMx0 * (nuMx/nuMx0) ** (t/TMx)
		phi = Integration.three_pops(phi, xx, TMx, nu1 = nuEu_func, nu2 = nuAs_func, nu3 = nuMx_func, m12 = mEuAs, m21 = mEuAs, m23 = mAsMx, m32 = mAsMx)
		phi = PhiManip.phi_3D_admix_1_and_2_into_3(phi, fEuMx, 0, xx, xx, xx)

		fs = Sepctrum.from_phi(phi, ns, (xx, xx, xx))
		# Apply our theta0. (All previous methods default to theta0 = 1.)
		return theta0 * fs

<p><strong>Listing 8 Settlement-of-New-World model from Gutenkunst (2009)</strong>: Because dadi is limited to 3 simultaneous populations, we need to integrate out the African population, using <code>Numerics.trapz.</code> This model also employs a fixed <i>θ</i>, and ancillary parameters passed in using the third argument</p>

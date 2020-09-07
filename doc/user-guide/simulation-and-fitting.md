# Simulation and fitting

### Grid sizes and extrapolation

To simulate the frequency spectrum, dadi solves a partial differential equation, approximating the solution using a grid of points in population frequency space (the `phi` array). Importantly, a single evaluation of the frequency spectrum with a fixed grid size is apt to be inaccurate, because computational limits mean the grid must be relatively coarse. To overcome this, dadi solves the problem at a series (typically 3) of grid sizes and extrapolates to an infinitely fine grid. To transform the demographic model function you have created (call it `my_demo_func`) into a function that does this extrapolation. wrap it using a call to `Numerics.make_extrap_func`, e.g.:

	my_extrap_func = Numerics.make_extrap_func(my_demo_func)

Having done this, the final argument to `my_extrap_func` is now a *sequence* of grid sizes, which will be used for extrapolation. In our experience, good results are obtained by setting the smallest grid size slightly larger than the largest population sample size. For example, if you have sample sizes of 16, 24, and 12 samples in the three populations you're working with, a good choice of grid sizes is probably `pts_l = [40, 50, 60]`. This can be altered depending on your usage. For example, if you are fitting a complex slow model, it may speed up the analysis considerably to first run an optimization at small grid sizes (even less than the maximum number of samples). This should get your parameter values approximately correct. They can be refined by running another optimization with a finer grid.

A simulated frequency spectrum is thus obtained by calling

	model = my_extrap_func(params, ns, pts_l)

Here `ns` is the sequenc of sample sizes for the populations in the model, `params` is the model parameters, and `pts_l` is the grid sizes.

### Likelihoods

dadi offers two complimentary ways of calculating the likelihood of the data FS given a model FS. The first is the Poisson approach, and the second is the multinomial approach.

In the Poisson approach, the likelihood is the product of Poisson likelihoods for each entry in the data FS, given an expected value from the model FS. This approach is relevant if \\(\theta_0\\) is an explicit parameter in your demographic function. Then the likelihood `ll` is 

	ll = dadi.Inference.ll(model, data)

In the multinomial approach, before calculating the likelihood, dadi will calculate the optimal \\(\theta_0\\) for comparing model and data. (It turns out that this is just \\(\theta_0 = \sum\text{data}/\sum\text{model}\\).) Because \\(\theta_0\\) is so trivial to estimate given the other parameters in the model, it is most efficient for it *not* to be an explicit parameter in the demographic function. Then the likelihood `ll` is

	ll = dadi.Inference.ll_multinomial(model, data)

The optimal \\(\theta_0\\) can be requested via

	theta0 = dadi.Inference.optimal_sfs_scaling(model, data)

### Fitting

To find the maximum-likelihood model parameters for a given data set, dadi employs non-linear optimization. Several optimization methods are provided, as detailed in the "Which optimizer should I use" section.

#### Parameter bounds

In their exploration, the optimization methods typically try a wide range of parameter values. For the methods that work in terms of log parameters, that range can be very wide indeed. As a consequence, the algorithms may sometimes try parameter values that are very far outside the feasible range and that cause *very* slow evaluation of the model FS. Thus, it is important to place upper and lower bounds on the values they may try. For divergence times and migration rates, large values cause slow evaluation, so it is okay to put the lower bound to 0 as long as the upper bound is kept reasonable. In our analyses, we often set the upper bound on times to be 10 and the upper bound on migration rates to be 20. For population sizes, very small sizes lead to very fast drift and consequently slow solution of the model equations; thus a non-zero lower bound is important, with the upper bound less so. In our analyses, we often set the lower bound on population sizes to be 10<sup>-2</sup> or 10<sup>-3</sup> (i.e. `1e-2` or `1e-3`).

If your fits often push the bounds of your parameter space (i.e., results are often at the bounds of one or more parameters), this indicates a problem. It may be that your bounds are too conservative, so try widening them. It may also be that your model is misspecified or that there are unaccounted biases in your data.

### Fixing parameters

It is often useful to optimize only a subset of model parameters. A common example is doing likelihood-ratio tests on nested models. The optional argument `fixed_params` to the optimization methods facilitates this. As an example, if `fixed_params = [None, 1.0, None, 2.0]`, the first and third model parameters will be optimized, with the second and fourth parameters fixed to 1 and 2 respectively. Note that when using this option, a full length initial parameter set `p0` should be passed in.

### Which optimizer should I use?

dadi provides a multitude of optimization algorithms through the [nlopt library](https://nlopt.readthedocs.io/), each of which perform best in particular circumstances. For a summary of all the algorithms available, see [the nlop documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/).

The most general purpose and default routine is `algorithm=nlopt.LN_BOBYQA`. This performs a local search from a specified set of parameters, using an algorithm which attempts to estimate the curvature of the likelihood surface. Also available is `algorithm=nlopt.LN_COBYLA`, which supports constrained optimization that restricts combinations of parameter values.

You may also want to set `log_opt=True`, which will do the optimization in terms of logs of the parameter values. This is useful
if the parameters differ dramatically in scale, with some very small and some very large parameter values.

We generally employ local optimizers. These are efficient, but are not guaranteed to find the global optimum. Thus, it is important to run several optimizations for each data set, starting from different initial parameters. If all goes well, multiple such runs will converge to the same set of parameters and likelihood, and this likelihood will be the highest found. This is strong evidence that you have indeed found the global optimum. To facilitate this, dadi provides a method `dadi.Misc.perturb_params` that randomly perturbs the parameters passed in to generate a new initial point for the optimization. 

If you have the computational resources and are struggling to optimize your model to your data, you may consider exploring the [global optimization algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#global-optimization) available in nlopt.
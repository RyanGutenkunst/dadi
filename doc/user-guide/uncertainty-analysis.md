# Uncertainty analysis

dadi can also perform uncertainty analysis using the Godambe Information Matrix (GIM), which is equivalent to the Fisher Information Matrix, but for composite likelihoods. The function call is

	uncert = dadi.Godambe.GIM_uncert(func_ex, grid_pts, all_boot, p0, data, log, multinom, eps, return_GIM)

Here `func_ex` is the model function, `grid_pts` is the set of grid points used in extrapolation, `all_boot` is a list containing bootstrapped data sets, `p0` is the best-fit parameters, and `data` is the original data. If `log = True`, then uncertainties will be calculated for the logs of the parameters; these can be interpreted as relative uncertainties for the parameters themselves. If `multinom = True`, it is assumed that \\(\theta\\) is not an explicit parameter of the model (this is the most common case). `eps` is the relative step size to use when taking numerical derivatives; the default value is often sufficient. The returned `uncert` is an array equal in length to `p0`, where each entry in `uncert` is the estimated standard deviation of the parameter it corresponds to in `p0`. If `multinom = True`, there will be one extra entry in `uncert`, corresponding to \\(\theta\\). If `return_GIM = True`, then the return value will be `(uncert, GIM)`, where `GIM` is the full Godambe Information Matrix, for use in propagating uncertainties.

Using the GIM is often preferable to directly fitting the bootstrapped datasets, because such fitting is computaionally time consuming. However, the GIM approach approximates parameter uncertainties as normal, which may not be a good approximation if they are large. To check this, one can evaluate the GIM uncertainties and compare them with the parameter values themselves. If the GIM uncertainties are large compared to the parameter values (for example, if a standard deviation is half the parameter value itself), then fitting the bootstrap data sets may be necessary to get accurate uncertainty estimates.

Parameter uncertainties for correlated parameters can also be determined using the GIM with uncertainty propagation techniques. An example of this would be if one wanted to know the uncertainty in the total time of a demographic model, \\(T_{total} = T_1 + T_2\\) that contains two events occurring at times \\(T_1\\) and \\(T_2\\). If the variance for \\(T_1\\) and \\(T_2\\) are given by \\(\sigma_{T_1}^2\\) and \\(\sigma_{T_2}^2\\), with a covariance term between the two \\(\sigma_{T_1T_2}\\), then the uncertainty in \\(T_{total}\\) is

$$\sigma_{T_{total}} = \sqrt{\sigma_{T_1}^2+\sigma_{T_2}^2+2\sigma_{T_1T_2}}$$

Another example where error propagation is necessary is when determining theta for an individual population, \\(\theta_A = \theta\times\nu_A\\), from the overall theta, \\(\theta\\), and the relative population size \\(\nu_A\\). For variance in \\(\nu_A\\) and \\(\theta\\) given by \\(\sigma_{\nu_A}^2\\) and \\(\sigma_{\theta}^2\\), respectively, and covariance between the two \\(\sigma_{\nu_A\theta}\\), the equation for uncertainty in \\(\theta_A\\) is

$$\sigma_{\theta_A} = \sqrt{\theta^2\sigma_{\nu_A}^2+\nu_A^2\sigma_{\theta}^2+2\nu_A\theta\sigma_{\nu_A\theta}}$$

The full GIM can be obtained from the `dadi.Godambe.GIM_uncert` function by setting `return_GIM = True`. Variance and covriances can be taken directly from the inverse of the GIM (obtained by `numpy.linalg.inv(GIM)`), in which diagonal terms present variance terms and off-diagonal terms represent covariance terms. For more complex scenarios, see [5](./references.md).

The `dadi.Godambe.FIM_uncert` function calculates uncertainties using the Fisher Information Matrix, which is sufficient if your data are unlinked.

# Frequently asked questions

### 1. What does the message `WARNING: Inference: Model is < 0 where data is not masked.` mean?

This waring comes from the likelihood calculation function. It indicates that the model frequency spectrum has negative values that are trying to be compared with data. Negative values in the frequency spectrum are nonsense, so this most likely indicates numerical difficulties. If you're running an optimization, occasional warnings like this likely not a problem. The optimization explores a wide range of parameter values, most of which are bad fits. If these errors crop up for parameter values that will be a bad fit anyways, the errors won't change the final result. On the other hand, if you are getting these warnings near good-fitting sets of parameters, you'll need to fix them. There are two possible causes, and thus two possible solutions.

- The negative values might be arising from the extrapolation process (over different grid sizes). In this case, replace any calls to `Numerics.make_extrap_func` to `Numeric.make_extrap_log_func`. This will do the extrapolation based on the logarithms of the value in the frequency spectrum, guaranteeing positive results.

- The negative values might be arising from calculating an individual spectrum (for a fixed grid size). This typically only happens in cases of very rapid exponential growth. In this case, you can try a finer gird size (increase the elements of the `pts_l` list) or smaller time steps. The time step is set by the call to `dadi.Integration.set_timescale_factor(pts_l[-1], factor=10)`. To shorten the time setp, increase `factor`. First try shortening the time step, as this will typically increase computation time less than increasing the grid size.

### 2. I'm projecting my data down to a smaller frequency spectrum. What sample sizes should I project down to?

At this time, we have not done any formal power testing to judge the optimal level of projection, but we do have a rule-of-thumb. As you project down to smaller sample sizes, more SNPs can be used in constructing the FS (because they have enough successful calls). However, as you project downward, some SNPs will "disappear" from the FS because they get partially projected down to 0 observations in all the populations. Our rule of thumb is to use the projection that maximizes the number or segregating SNPs. The number of segregating SNPs can be calculated as `fs.S()`.

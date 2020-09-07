# Frequently asked questions

### 1. I'm projecting my data down to a smaller frequency spectrum. What sample sizes should I project down to?

At this time, we have not done any formal power testing to judge the optimal level of projection, but we do have a rule-of-thumb. As you project down to smaller sample sizes, more SNPs can be used in constructing the FS (because they have enough successful calls). However, as you project downward, some SNPs will "disappear" from the FS because they get partially projected down to 0 observations in all the populations. Our rule of thumb is to use the projection that maximizes the number or segregating SNPs. The number of segregating SNPs can be calculated as `fs.S()`.

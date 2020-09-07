# Bootstrapping

Because dadi's likelihood function treats all variants as independent, and they are often not, standard likelihood theory should not be used to estimate parameter uncertainties and significance levels for hypothesis test. 

To do such tests, one can bootstrap. For estimating parameter uncertainties, one can use a nonparameteric bootstrap, i.e. sampling with replacement from independent units of your data (genes or chromosomes) to generate new data sets to fit. 

For hypothesis tests, the parametric bootstrap is preferred. This involves using a coalescent simulator (such as *ms*) to generate simulated data sets. Care must be taken to simulate the sequencing strategy as closely as possible.

Both the above options are computationally very expensive. Much more efficient analysis is enabled by the Godambe Information Matrix approach (see Uncertainty Anlaysis section). For
that analysis, you will need nonparameteric bootstrap data.
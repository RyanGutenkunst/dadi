# Manipulating spectra

Frequency spectra are stored in `dadi.Spectrum` objects. Computationally, these are a subclass of `numpy.masked_array`, so most of the standard array manipulation techniques can be used. (In the examples here, I will typically be considering two-dimensional spectra, although all these features apply to higher-dimensional spectra as well.)

You can do arithmetic with `Spectrum` objects:

	fs3 = fs1 + fs2
	fs2 = fs1 * 2

Note that most operations involving two `Spectrum` objects only make sense if they correspond to data with the same sample sizes.

Standard indexing and slicing operations work as well. For example, to access the counts corresponding to 3 observations in population 1 and 5 observations in population 2, simpliy

	counts = fs[3, 5]

More complicated slices are also possible. The slice notation `:` indicates taking all corresponding entries. For example, to access the slice of the `Spectrum` corresponding to entries with 2 derived allele observations in population 2, take

	slice = fs[:, 2]

### Summary statistics

The frequency spectrum encompasses many common summary statistics, and dadi provides methods to calculate them from `Spectrum` objects.

#### Single-population statistics

Watterson's theta can be calculated as

	thetaW = fs.Watterson_theta()

The expected heterozygosity π assuming random mating is 

	pi = fs.pi()

Tajima's D is

	D = fs.Tajima_D()

#### Multi-population statistics

The number of segregating sites *S* is simply the sum of all entries in the FS (except for the absent-in-all and derived-in-all entries). This can be calculated as

	S = fs.S()

Wright's *F<sub>ST</sub>* can be calculated as

	Fst = fs.Fst()

This estimate of Fst assumes random mating, because the FS does not store heterozygote. Calculation is by the method of Weir and Cockerham<sup>[2](./references.md)</sup>. For a single SNP, the relevant formula is at the top of page 1363. To combine results between SNPs, we use the weighted average indicated by equation 10.

### Folding

By default, dadi considers the data in the `Spectrum` to be polarized, i.e. that the ancestral state of each variant is known. In some cases, however, this may not be possible, and the FS must be *folded*, indicating that only the minor allele frequency is known. To fold a `Spectrum` object, simply

	folded = fs.fold()

The `Spectrum` object will record the fact that it has been folded, so that the likelihood and optimization machinery can automatically fold model spectra when the data are folded.

### Masking

Finally, `Spectrum` arrays are *masked*, i.e. certain entries can be set to be ignored. Most typically, the ignored entries are the two corners: `[0, 0]` and `[n1, n2]`, corresponding to variants observed in zero samples or in all samples. More sophisticated masking is possible, however. For example, if your calling algorithm is such that singletons in population 1 cannot be confidently called, you may want to ignore those entries of the FS in your analysis. To do so, simply

	fs.mask[1, :] = True

Note that care must be taken when doing arithmetic with `Spectrum` objects that are masked in different ways.

### Marginalizing

If one has a multidimensional `Spectrum` it may be useful to examine the marginalized `Spectrum` corresponding to a subset of populations. To do so, use the `marginalize` method. For example, consider a three-dimensional `Spectrum` consisting of data from populations A, B, and C. To consider the marginal two dimensional spectrum for populations A and C, we need marginalize over population B.

	fsAC = fsABC.marginalize([1])

And to consider the marginal one-dimensional FS for population B, we marginalize over populations A and C.

	fsB = fsABC.marginalize([0, 2])

Note that the argument to `marginalize` is a list of dimensions to marginalize over, *indexed from* \\(\theta\\).

### Projection

One can also project an FS down from a larger sample size to a smaller sample size. Implicitly, this involves averaging over all possible re-samplings of the larger sample size data. This is very often done in the case of missing data: if some sites could not be called in all individuals, one can set a lower bound on the number of successfull calls necessary to include a SNP in the analysis; SNPs with more successful calls can then be projected down to that number of calls.

In dadi, this is implemented with the `project` method. For example, to project a two-dimensional FS down to sample sizes of 14 and 26, use

	proj = fs.project([14, 26])

### Sampling

One can simulate Poisson sampling from an FS using the `sample` method.

	sample = fs.sample()

Each entry in the `sample` output FS will have a Poisson number of counts, with mean given by the corresponding entry in `fs`. If all sites are completely unlinked, this is a proper parametric bootstrap from your FS.

### Scrambling

Occasionally, one may wish to ask whether the FS really represents samples from two populations or rather subsamples from a single population. A rough check of this is to consider what the FS would look like if the population identifiers were scrambled amongst the individuals for whom you have data. The `scramble` method will do this.

	scrambled = fs.scramble()

As an example, one could consider whether the FS for JPT and CHB shows evidence of differentiation between the two populations. Note that this is an informal test, and we have not developed the theory to assign statistical significance to the results. It is, nevertheless, a useful guide.

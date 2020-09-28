# Importing data

dadi represents frequency spectra using `dadi.Spectrum` objects. As desccribed in the [Manipulating spectra section](./manipulating-spectra.md), `Spectrum` objects are subclassed from `numpy.masked_array` and thus can be constructed similarly. The most basic way to create a `Spectrum` is manually:

	fs = dadi.Spectrum([0, 100, 20, 10, 1, 0])

This creates a `Spectrum` object representing the FS from a single population, from which we have 5 samples (The first and last entries in `fs` correspond to mutations observed in zero or all samples. These are thus not polymorphisms, and by default dadi masks out those entries so they are ignored.)

For nontrivial data sets, entering the FS manually is infeasible, so we will focus on automatic methods of generating a `Spectrum` object. The most direct way is to load a pre-generated FS from a file, using

	fs = dadi.Spectrum.from_file(filename)

The appropriate file format is detailed in the next section. We have also added fuction to generate the FS from a VCF file.

### Frequency spectrum file format

dadi uses a simple fie format for storing the FS. Each file begins with any number of comment lines beginning with `#`. the first non-comment line contain *P* integers giving the dimensions of the FS array, where *P* is the number of populations represented. For a FS representing data from 4×4×2 samples, this would be `5 5 3`. (Each dimension is one larger than the number of samples, because the number of observations can range, for example, from 0 to 4 if there are 4 samples, for a total of 5 possibilities.) On the same line, the string `folded` or `unfolded` denoting whether or not the stored FS is folded.

The actual data is stored in a single line listing all the FS elements separated by spaces, in the order `fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1] ...` This is followed by a single line giving the elements of the mask in the same order as the data, with `1` indicating masked and `0` indicating unmasked.

The file corresponding to the `Spectrum fs` can be written using the ccommand `fs.to_file(filename)`

### Frequency spectra from VCF files

Data can be loaded from VCF files. The main function for accomplishing this is `make_data_dict_vcf` in the `dadi.Misc` submodule. The function has two required arguments: 

1. The name of the VCF (can be gzipped [*.vcf.ga])
2. The name of a file describing how individuals map to populations.

This second file is a plain-text, two-column file containing the individual names in column one and their respective populations in column two:

	i0	pop0
	i1	pop0
	i2	pop0
	...
	iN	pop2

Examples of these files can be found in the `examples/fs_from_data/` folder in the [dadi source distribution](https://bitbucket.org/gutenkunstlab/dadi/src/master/). Generating a frequency spectrum with these files can then be achieved through the creation of a data dictionary with the following two lines of code:

	dd = dadi.Misc.make_data_dict_vcf("example.vcf.gz", "popfile.txt")
	fs = dadi.Spectrum.from_data_dict(dd, ['pop0', 'pop1'], projections = [20, 30], polarized = False)

Projection is often used to deal with missing data. But projection cannot be used when modeling inbreeding. It also turns the dadi's likelihood function into a composite likelihood, even for unlinked data. We have thus included an option to take a smaller subsample of individuals from a population at each site, so that variants with less missing data than the specified subsampling size are not completely ignored. To specify how many individuals should be subsampled, the function takes an additional dictionary as an argument, where the dictionary simply maps the population names to the desired number of individuals to subsample.

	# create the subsample dictionary
	ss = {'pop0':5, 'pop1':10}
	# pass it as an additional argument
	dd = dadi.Misc.make_data_dict_vcf("example.vcf.gz", "popfile.txt", subsample = ss)
	fs = dadi.Spectrum.from_data_dict(dd, ['pop0', 'pop1'], projections = [10, 20], polarized = False)

Subsampling offers an alternative to down projecting your data that preserves individual genotypes. Projecting will consider sampled chromosomes/alleles as exchangeable across all individuals, which is usually OK for a randomly mating population. However, if you want to include inbreeding in your model (see the [Inbreeding section](./inbreeding.md)), then projecting will erase the signal of excess homozygosity that inbreeding creates by sampling chromosomes instead of individuals. When generating a frequency spectrum from a subsampled data dictionary, be sure to set the projections argument to 2 times the subsample size you specified for each population so that no down projecting is done.

### SNP data methods

From a data dictionary, the method `Spectrum.from_data_dict` can be used to create a `Spectrum`.

	fs = Spectrum.from_data_dict(dd, pop_id = ['YRI', 'CEU'], projections = [10, 12], polarized = True)

The `pop_ids` argument specifies which populations to use to create the FS, and their order. `projections` denotes the population sample sizes for resulting FS. (Recall that for a diploid organism, assuming random mating, we get two samples from each individual.) Note that the total number of calls to `Allele1` and `Allele2` in a given population need not be the same for each SNP. When constructing the Spectrum, each SNP will be projected down to the requested number of samples in each population. (Note that SNPs cannot be projected up, so SNPs without enough calls in any population will be ignored.) `polarized` specifies whether dadi should use outgroup information to polarize the SNPs. If `polarized = True`, SNPs withouth outgroup information, or with that information —— will be ingored. If `polarized = False`, outgroup information will be ignored and the resulting `Spectrum` will be folded.

If your data have missing calls for some individuals, projecting down to a smaller sample size will increase the number of SNPs you can use for analysis. On the other hand, some fraction of the SNPs will now project down to frequency 0, and thus be uninformative. As a rule of thumb, we often choose our projection to maximize the number of segregating sites in our final FS (assessed via `fs.S()`), although we have not formally tested whether this maximizes statistical power.

### SNP data format

As a convenience, dadi includes several methods for generating frequency spectra directly from SNP data. That relevant SNP file format is described here. An large example can be found in the `examples/fs_from_data/data.txt` file included with the [dadi source distribution](https://bitbucket.org/gutenkunstlab/dadi/src/master/), and a small example is shown in Listing 1.

	Human	Chimp	Allele1	YRI	CEU	Allele2	YRI	CEU	Gene	Position
	ACG		ATG		C		29	24	T		1	0	abcb1	289
	CCT		CCT		C		29	23	G		3	2	abcb1	345

<p align="center"><strong>Listing 1</strong>: Example of SNP file format</p>

The data file begins with any number of comment lines that being with `#`. The first parsed line is a column header line. Whitespace is used to separate entries within the table, so no spaces are allowed within any entry. Individual rows make be commented out using `#`.

The first column contains the in-group reference sequence at that SNP, including the flanking bases. If the flanking bases are unknown, they can be denoted by `-`. The header label is arbitrary.

The second column contains the aligned outgroup reference sequency at that SNP, including the flanking bases. Unknown entries can be denoted by `-. The header label is arbitrary.

The third column gives the first segregating allele. The column header must be exactly `Allele1`.

Then follows an arbitrary number of columns, one for each population, each giving the number of times `Allele1` was observed in that population. The header for each column should be the population identifier.

The next column gives the second segregating allele. The column header must be exactly `Allele2`.

Then follows one column for each population, each giving the number of times `Allele2` was observed in that population. The header form each column should be the population identifier, and the columns should be in the same order as for the `Allele1` entries.

Then follows an arbitrary number of columns which will be concatenated with `_` to assign a label for each SNP.

The `Allele1` and `Allele2` headers must be exactly those values because the number of columns between those two is used to infer the number of population in the file.

The method `Misc.make_data_dict` reads the above SNP file format to generate a Python data dictionary describing the data:

	dd = Misc.make_data_dict(filename)

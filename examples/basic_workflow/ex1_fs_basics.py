# Dadi workflow example 1 - Creating a SFS and functions
import dadi

# Parse the VCF file to generate a data dictionary
datafile = '../fs_from_data/1KG.YRI.CEU.biallelic.synonymous.snps.withanc.strict.subset.vcf.gz'
dd = dadi.Misc.make_data_dict_vcf(datafile, '../fs_from_data/1KG.YRI.CEU.popfile.txt')

# Extract the spectrum for ['YRI','CEU'] from that dictionary, with both
# YRI projected down to 20 and CEU projected down to 24.
# We project down like this just to make fitting faster. For a real analysis
# we would not project so severely.
pop_ids, ns = ['YRI','CEU'], [20,24]
fs = dadi.Spectrum.from_data_dict(dd, pop_ids, ns)
# We can save our extracted spectrum to disk
fs.to_file('1KG.YRI.CEU.biallelic.synonymous.snps.withanc.strict.subset.fs')

# If we didn't have outgroup information, we could fold the fs.
# ex:
# fs_folded = dadi.Spectrum.from_data_dict(dd, pop_ids, ns, polarized=False)

# See how much data is in the SFS
print(fs.S())
# As a rule of thumb we want to maximize this number. Which we can do by trying
# different sample sizes for our population(s)

# Plot comparing the multiple versions of our data spectra.
import matplotlib.pyplot as plt

fig = plt.figure(1, figsize=(10,6))
fig.clear()

# Note that projection creates fractional entries in the spectrum,
# so vmin < 1 is sensible.
ax = fig.add_subplot(2,3,1)
dadi.Plotting.plot_single_2d_sfs(fs, vmin=1e-2, ax=ax)
ax.set_title('1000 Genomes SFS')
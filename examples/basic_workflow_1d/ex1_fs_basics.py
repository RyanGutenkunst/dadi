# Dadi workflow example 1 - Creating a SFS and functions
import dadi

# Parse the VCF file to generate a data dictionary
datafile = 'data/vcf/1KG.YRI.CEU.biallelic.synonymous.snps.withanc.strict.subset.vcf.gz'
dd = dadi.Misc.make_data_dict_vcf(datafile, 'data/vcf/1KG.YRI.CEU.popfile.txt')

# Extract the spectrum for ['YRI'] from that dictionary, with both
# YRI projected down to 20.
# We project down like this just to make fitting faster. For a real analysis
# we would not project so severely.
pop_ids, ns = ['YRI'], [20]
fs = dadi.Spectrum.from_data_dict(dd, pop_ids, ns)
# We can save our extracted spectrum to disk
fs.to_file('data/fs/1KG.YRI.biallelic.synonymous.snps.withanc.strict.subset.fs')

# If we didn't have outgroup information, we could fold the fs.
# ex:
# fs_folded = dadi.Spectrum.from_data_dict(dd, pop_ids, ns, polarized=False)

# See how much data is in the SFS
print(fs.S())
# As a rule of thumb we want to maximize this number. Which we can do by trying
# different sample sizes for our population(s)

# Plot our data spectra.
import matplotlib.pyplot as plt

fig = plt.figure(1, figsize=(10,6))
fig.clear()
dadi.Plotting.plot_1d_fs(fs)
ax.set_title('1000 Genomes SFS')
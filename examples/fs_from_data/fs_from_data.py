"""
Simple example of extracting a frequency spectrum from a SNP data file.
"""
import dadi

def make_data_dict():
    """
    Parses data.txt and store SNP info in a properly formatted dictionary.

    This could be made more flexible at the cost of complexity.
    """
    f = file('data.txt')

    # Skip the header
    header = f.readline()
    pops = header.split()[3:9]

    # The empty data dictionary
    data_dict = {}

    # Now walk down the file
    for line in f:
        # Split the into fields by whitespace
        spl = line.split()

        data_this_snp = {}
        data_this_snp['context'] = spl[0]
        data_this_snp['outgroup_context'] = spl[1]
        data_this_snp['outgroup_allele'] = spl[1][1]
        data_this_snp['segregating'] = spl[2],spl[9]

        calls_dict = {}
        for ii,pop in enumerate(pops):
            calls_dict[pop] = int(spl[3+ii]), int(spl[10+ii])
        data_this_snp['calls'] = calls_dict    

        # We name our SNPs by the gene they're on and their position.
        # (Negative indices count from the end.)
        snp_id = '%s_%s' %(spl[-2], spl[-1])

        data_dict[snp_id] = data_this_snp

    return data_dict

# Parse the data file to generate the data dictionary
dd = make_data_dict()
# Extract the spectrum for ['YRI','CEU'] from that dictionary, with both
# projected down to 20 samples per population.
fs = dadi.Spectrum.from_data_dict(dd, ['YRI','CEU'], [20,20])

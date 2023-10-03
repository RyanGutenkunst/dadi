import itertools
from io import StringIO as SIO
import numpy as np, pandas as pd
import dadi
import warnings

import math
from scipy.special import comb
import scipy.stats as ss, scipy.stats.distributions as ssd

rng = np.random.default_rng()


def make_data_dict_vcf(vcf_filename, popinfo_filename, subsample=None, filter=True,
                       flanking_info=[None, None]):
    """
    Parse a VCF file containing genomic sequence information, along with a file
    identifying the population of each sample, and store the information in
    a properly formatted dictionary.
    
    Each file may be zipped (.zip) or gzipped (.gz). If a file is zipped,
    it must be the only file in the archive, and the two files cannot be zipped
    together. Both files must be present for the function to work.
    
    vcf_filename : Name of VCF file to work with. The function currently works
                   for biallelic SNPs only, so if REF or ALT is anything other
                   than a single base pair (A, C, T, or G), the allele will be
                   skipped. Additionally, genotype information must be present
                   in the FORMAT field GT, and genotype info must be known for
                   every sample, else the SNP will be skipped. If the ancestral
                   allele is known it should be specified in INFO field 'AA'.
                   Otherwise, it will be set to '-'.
    
    popinfo_filename : Name of file containing the population assignments for
                       each sample in the VCF. If a sample in the VCF file does
                       not have a corresponding entry in this file, it will be
                       skipped. See _get_popinfo for information on how this
                       file must be formatted.
    
    subsample : Dictionary with population names used in the popinfo_filename
                as keys and the desired sample size (in number of individuals)
                for subsampling as values. E.g., {"pop1": n1, "pop2": n2} for
                two populations.
    
    filter : If set to True, alleles will be skipped if they have not passed
             all filters (i.e. either 'PASS' or '.' must be present in FILTER
             column.
    
    flanking_info : Flanking information for the reference and/or ancestral
                    allele can be provided as field(s) in the INFO column. To
                    add this information to the dict, flanking_info should
                    specify the names of the fields that contain this info as a
                    list (e.g. ['RFL', 'AFL'].) If context info is given for
                    only one allele, set the other item in the list to None,
                    (e.g. ['RFL', None]). Information can be provided as a 3
                    base-pair sequence or 2 base-pair sequence, where the first
                    base-pair is the one immediately preceding the SNP, and the
                    last base-pair is the one immediately following the SNP.
    """
    do_subsampling = False
    if subsample is not None:
        do_subsampling = True
        warnings.warn('Note on subsampling: If you will be including inbreeding in your model, '
                      'do not project your data to smaller sample sizes in later steps of your analysis.')
    
    if os.path.splitext(popinfo_filename)[1] == '.gz':
        import gzip
        popinfo_file = gzip.open(popinfo_filename)
    elif os.path.splitext(popinfo_filename)[1] == '.zip':
        import zipfile
        archive = zipfile.ZipFile(popinfo_filename)
        namelist = archive.namelist()
        if len(namelist) != 1:
            raise ValueError("Must be only a single popinfo file in zip "
                             "archive: {}".format(popinfo_filename))
        popinfo_file = archive.open(namelist[0])
    else:
        popinfo_file = open(popinfo_filename)
    # pop_dict has key, value pairs of "SAMPLE_NAME" : "POP_NAME"
    try:
        popinfo_dict = _get_popinfo(popinfo_file)
    except:
        raise ValueError('Failed in parsing popinfo file.')
    popinfo_file.close()
    
    # Open VCF file
    if os.path.splitext(vcf_filename)[1] == '.gz':
        import gzip
        vcf_file = gzip.open(vcf_filename)
    elif os.path.splitext(vcf_filename)[1] == '.zip':
        import zipfile
        archive = zipfile.ZipFile(vcf_filename)
        namelist = archive.namelist()
        if len(namelist) != 1:
            raise ValueError("Must be only a single vcf file in zip "
                             "archive: {}".format(vcf_filename))
        vcf_file = archive.open(namelist[0])
    else:
        vcf_file = open(vcf_filename)
    
    data_dict = {}
    for line in vcf_file:
        # decoding lines for Python 3 - probably a better way to handle this
        try:
            line = line.decode()
        except AttributeError:
            pass
        # Skip metainformation
        if line.startswith('##'):
            continue
        # Read header
        if line.startswith('#'):
            header_cols = line.split()
            # Ensure there is at least one sample
            if len(header_cols) <= 9:
                raise ValueError("No samples in VCF file")
            # Use popinfo_dict to get the order of populations present in VCF
            poplist = [popinfo_dict[sample] if sample in popinfo_dict else None
                       for sample in header_cols[9:]]
            continue
        
        # Read SNP data
        # Data lines in VCF file are tab-delimited
        # See https://samtools.github.io/hts-specs/VCFv4.2.pdf
        cols = line.split("\t")
        snp_id = '_'.join(cols[:2]) # CHROM_POS
        snp_dict = {}
        
        # Skip SNP if filter is set to True and it fails a filter test
        if filter and cols[6] != 'PASS' and cols[6] != '.':
            continue
        
        # Add reference and alternate allele info to dict
        ref, alt = (allele.upper() for allele in cols[3:5])
        if ref not in ['A', 'C', 'G', 'T'] or alt not in ['A', 'C', 'G', 'T']:
            # Skip line if site is not an SNP
            continue
        snp_dict['segregating'] = (ref, alt)
        snp_dict['context'] = '-' + ref + '-'
        
        # Add ancestral allele information if available
        info = cols[7].split(';')
        for field in info:
            if field.startswith('AA=') or field.startswith('AA_ensembl=') or field.startswith('AA_chimp='):
                outgroup_allele = field.split('=')[1].upper()
                if outgroup_allele not in ['A','C','G','T']:
                    # Skip if ancestral not single base A, C, G, or T
                    outgroup_allele = '-'
                break
        else:
            outgroup_allele = '-'
        snp_dict['outgroup_allele'] = outgroup_allele
        snp_dict['outgroup_context'] = '-' + outgroup_allele + '-'
        
        # Add flanking info if it is present
        rflank, aflank = flanking_info
        for field in info:
            if rflank and field.startswith(rflank):
                flank = field[len(rflank)+1:].upper()
                if not (len(flank) == 2 or len(flank) == 3):
                    continue
                prevb, nextb = flank[0], flank[-1]
                if prevb not in ['A','C','T','G']:
                    prevb = '-'
                if nextb not in ['A','C','T','G']:
                    nextb = '-'
                snp_dict['context'] = prevb + ref + nextb
                continue
            if aflank and field.startswith(aflank):
                flank = field[len(aflank)+1:].upper()
                if not (len(flank) == 2 or len(flank) == 3):
                    continue
                prevb, nextb = flank[0], flank[-1]
                if prevb not in ['A','C','T','G']:
                    prevb = '-'
                if nextb not in ['A','C','T','G']:
                    nextb = '-'
                snp_dict['outgroup_context'] = prevb + outgroup_allele + nextb
        
        calls_dict = {}
        subsample_dict = {}
        gtindex = cols[8].split(':').index('GT')
        
        # === New Feature Addition ===
        coverage_dict = {}
        
        try:
            covindex = cols[8].split(':').index('AD')
        except:
            covindex = None
        
        if do_subsampling:
            # Collect data for all genotyped samples
            for pop, sample in zip(poplist, cols[9:]):
                if pop is None:
                    continue
                gt = sample.split(':')[gtindex]
                
                if pop not in subsample_dict:
                    subsample_dict[pop] = []
                    coverage_dict[pop] = ()
                if '.' not in gt:
                    subsample_dict[pop].append(gt)
            
                if covindex is not None:
                    coverages = coverage_dict[pop]
                    coverage = sample.split(':')[covindex].split(',')
                    coverage_count = sum(int(cov) for cov in coverage if cov.isdigit())
                    coverage_dict[pop] = coverages + (coverage_count, )
            
            # key-value pairs here are population names
            # and a list of genotypes to subsample from
            for pop, genotypes in subsample_dict.items():
                if pop not in calls_dict:
                    calls_dict[pop] = (0, 0)
                if len(genotypes) < subsample[pop]:
                    # Not enough calls for this SNP
                    break
                # Choose which individuals to use
                idx = numpy.random.choice([i for i in range(0,len(genotypes))], subsample[pop], replace=False)
                for ii in idx:
                    gt = subsample_dict[pop][ii]
                    refcalls, altcalls = calls_dict[pop]
                    refcalls += gt[::2].count('0')
                    altcalls += gt[::2].count('1')
                    calls_dict[pop] = (refcalls, altcalls)
            else:
                # Only runs if we didn't break out of this loop
                snp_dict['calls'] = calls_dict
                
                # === New Feature Addition ===
                if covindex is not None:
                    snp_dict['coverage'] = coverage_dict
                else:
                    snp_dict['coverage'] = '-'
                
                data_dict[snp_id] = snp_dict
        else:
            for pop, sample in zip(poplist, cols[9:]):
                if pop is None:
                    continue
                if pop not in calls_dict:
                    calls_dict[pop] = (0,0)
                    coverage_dict[pop] = ()
                # Genotype in VCF format 0|1|1|0:...
                gt = sample.split(':')[gtindex]
                #g1, g2 = gt[0], gt[2]
                #if g1 == '.' or g2 == '.':
                #    continue
                    #full_info = False
                    #break
                
                refcalls, altcalls = calls_dict[pop]
                #refcalls += int(g1 == '0') + int(g2 == '0')
                #altcalls += int(g1 == '1') + int(g2 == '1')
                
                # Assume biallelic variants
                refcalls += gt[::2].count('0')
                altcalls += gt[::2].count('1')
                calls_dict[pop] = (refcalls, altcalls)
                
                # === New Feature Addition ===
                coverages = coverage_dict[pop]
                
                coverage = sample.split(':')[covindex].split(',')
                coverage_count = sum(int(cov) for cov in coverage if cov.isdigit())
                coverage_dict[pop] = coverages + (coverage_count, )
            
            snp_dict['calls'] = calls_dict
            
            # === New Feature Addition ===
            if covindex is not None:
                snp_dict['coverage'] = coverage_dict
            else:
                snp_dict['coverage'] = '-'
            
            data_dict[snp_id] = snp_dict
    
    vcf_file.close()
    return data_dict


def _get_popinfo(popinfo_file):
    """
    Helper function for make_data_dict_vcf. Takes an open file that contains
    information on the population designations of each sample within a VCF file,
    and returns a dictionary containing {"SAMPLE_NAME" : "POP_NAME"} pairs.
    
    The file should be formatted as a table, with columns delimited by
    whitespace, and rows delimited by new lines. Lines beginning with '#' are
    considered comments and will be ignored. Each sample must appear on its own
    line. If no header information is provided, the first column will be assumed
    to be the SAMPLE_NAME column, while the second column will be assumed to be
    the POP_NAME column. If a header is present, it must be the first
    non-comment line of the file. The column positions of the words "SAMPLE" and
    "POP" (ignoring case) in this header will be used to determine proper
    positions of the SAMPLE_NAME and POP_NAME columns in the table.
    
    popinfo_file : An open text file of the format described above.
    """
    popinfo_dict = {}
    sample_col = 0
    pop_col = 1
    header = False
    
    # check for header info
    for line in popinfo_file:
        if line.startswith('#'):
            continue
        cols = [col.lower() for col in line.split()]
        if 'sample' in cols:
            header = True
            sample_col = cols.index('sample')
        if 'pop' in cols:
            header = True
            pop_col = cols.index('pop')
        break
    
    # read in population information for each sample
    popinfo_file.seek(0)
    for line in popinfo_file:
        if line.startswith('#') or not line.strip():
            continue
        cols = line.split()
        sample = cols[sample_col]
        pop = cols[pop_col]
        # avoid adding header to dict
        if (sample.lower() == 'sample' or pop.lower() == 'pop') and header:
            header = False
            continue
        popinfo_dict[sample] = pop
    
    
    return popinfo_dict


def compute_cov_dist(data_dict, pop_ids):
    try:
        # Dictionary comprehension to compute the coverage distribution
        coverage_distribution = {pop: numpy.array([
            *numpy.unique(numpy.concatenate([numpy.array(list(entry['coverage'][pop])) for entry in data_dict.values()]), return_counts=True)
        ]) for pop in pop_ids}
        
        # Normalize counts
        coverage_distribution = {pop: numpy.array([elements, counts / counts.sum()]) for pop, (elements, counts) in coverage_distribution.items()}
        
        return coverage_distribution
    except:
        raise ValueError("Information about allelic depths for the reference and alternative alleles not found in the data dictionary")


def partitions_and_probabilities(n_sequenced, partition_type, allele_frequency=None):
    """
    Generate allele count partitions and calculate their probabilities for a given number of alleles.
    
    Args:
        n_sequenced (int): The number of sequenced haplotypes.
        partition_type (str): Type of partition to generate ("allele_frequency" or "genotype").
        allele_frequency (int): The total number of alleles (required only for "allele_frequency" partition_type).
    
    Returns:
        tuple: A tuple containing two elements:
            - list: A list of all possible allele count partitions.
            - numpy.ndarray: An array containing the probability of each partition.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
         
        if partition_type == 'allele_frequency':
            # Check if n_sequenced is even, as partitions are calculated for even numbers
            if n_sequenced % 2 != 0:
                raise ValueError("Genotype partitions can only be calculated for an even number of haplotypes")
            
            # Generate partitions
            partitions = dadi.Numerics.cached_part(allele_frequency, n_sequenced/2)
            
            # Calculate partition probabilities
            partition_ways = np.array([np.exp(dadi.Numerics.multinomln([part.count(0), part.count(1), part.count(2)])) * 2 ** part.count(1) for part in partitions])
            partition_probabilities = partition_ways / np.sum(partition_ways)
        
        elif partition_type == 'genotype':
            # Generate an array of allele counts from 0 to n_sequenced
            allele_counts = np.arange(n_sequenced + 1)
            
            # Generate partitions
            partitions = [dadi.Numerics.cached_part(allele_count, n_sequenced / 2) for allele_count in allele_counts]
            
            # Calculate partition probabilities using multinomial likelihood
            partition_ways = [
                [
                    np.exp(dadi.Numerics.multinomln([part.count(0), part.count(1), part.count(2)])) * 2 ** part.count(1)
                    for part in parts
                ]
                for parts in partitions
            ]
            
            # Normalize partition probabilities
            partition_ways_sum = [[np.sum(part)] if len(part) > 1 else part for part in partition_ways]
            partition_probabilities = [np.array(pw) / np.array(pwb) for pw, pwb in zip(partition_ways, partition_ways_sum)]
        
        else:
            raise ValueError("Invalid partition_type. Use 'allele_frequency' or 'genotype'.")
        
        return partitions, partition_probabilities


def split_list_by_lengths(input_list, lengths_list):
    """
    Split a list into sublists of specified lengths.
    
    Args:
        input_list (list): The list to be split.
        lengths_list (list): A list of integers representing the lengths of the sublists.
    
    Returns:
        list: A list of sublists created based on the specified lengths.
    """
    split_list = []  # Initialize an empty list to store the sublists
    start = 0  # Initialize the starting index
    
    # Iterate through the lengths in the lengths_list
    for length in lengths_list:
        # Extract a sublist from input_list
        sublist = input_list[start:start + length]
        
        # Append the sublist to the result list
        split_list.append(sublist)
        
        # Update the starting index for the next iteration
        start += length
    
    return split_list


def flatten_nested_list(nested_list, operation):
    """
    Flatten a nested list by concatenating or multiplying elements.
    
    Args:
        nested_list (list of lists): The nested list to be flattened.
        operation (str): The operation to apply between elements ('+' for concatenation or '*' for multiplication).
    
    Returns:
        list: A flattened list resulting from applying the operation to elements.
    """
    if len(nested_list) == 1:
        # If there's only one sublist, return it (no need to flatten)
        return nested_list[0]
    else:
        # Generate combinations of elements from sublists
        # Apply the specified operation (evaluated as a string) to each combination
        flattened_list = [
            eval(f"{operation.join(map(str, sublist))}")
            for sublist in itertools.product(*nested_list)
        ]
        
        return flattened_list


def simulate_reads(coverage_distribution, flattened_partition, pop_n_sequenced, number_simulations):
    """
    Simulate reads across a given genotype partition.
    
    Args:
        coverage_distribution (list): Coverage distribution for each population.
        flattened_partition (list): Flattened genotype partition.
        pop_n_sequenced (list): Number of sequenced haplotypes for each population.
        number_simulations (int): Number of simulations to perform.
    
    Returns:
        tuple: A tuple containing two arrays:
            - numpy.ndarray: Arrays of reference allele counts for each simulated individual.
            - numpy.ndarray: Arrays of alternative allele counts for each simulated individual.
    """
    flattened_partition = np.array(flattened_partition)
    
    # Split partition for different populations
    partition = split_list_by_lengths(flattened_partition, pop_n_sequenced)
    
    # Empty array for storing the simulated data, initialized with zeros
    simulated_coverage = np.zeros((number_simulations, len(flattened_partition)), dtype=int)
    
    # Create population breaks
    splits = np.concatenate((np.array([0]), np.cumsum(pop_n_sequenced)), axis=0)
    
    # Simulate reads for each population
    for i, _ in enumerate(partition):
        cov_distribution = np.asarray(coverage_distribution[i])
        cov_sampling = ss.rv_discrete(values=[np.arange(len(cov_distribution)), cov_distribution])
        coverages = cov_sampling.rvs(size=(number_simulations, len(partition[i])))
        simulated_coverage[:, splits[i]:splits[i+1]] = coverages
    
    # Initialize arrays for reference and alternative allele counts
    n_ref, n_alt = np.zeros((number_simulations, len(flattened_partition)), dtype=int), np.zeros((number_simulations, len(flattened_partition)), dtype=int)
    
    # Calculate reference and alternative allele counts based on genotype partition
    n_ref[:, flattened_partition == 0] = simulated_coverage[:, flattened_partition == 0]
    n_alt[:, flattened_partition == 2] = simulated_coverage[:, flattened_partition == 2]
    n_alt[:, flattened_partition == 1] = ss.binom.rvs(simulated_coverage[:, flattened_partition == 1], 0.5)
    n_ref[:, flattened_partition == 1] = simulated_coverage[:, flattened_partition == 1] - n_alt[:, flattened_partition == 1]
    
    return n_ref, n_alt


def subsample_genotypes_1D(genotype_calls, n_subsampling):
    """
    Subsample genotypes to create a smaller dataset with a specified number of haplotypes.
    
    Args:
        genotype_calls (numpy.ndarray): Genotype array with 99 assumed to represent missing data.
        n_subsampling (int): Number of haplotypes in the final subsample.
    
    Returns:
        numpy.ndarray: Subsampled genotype data.
    """
    # Handle a special case where the input genotype_calls is empty
    if len(genotype_calls) == 0:
        return genotype_calls[:, :n_subsampling // 2]
    
    # Count the number of called individuals for each locus
    n_called = np.count_nonzero(genotype_calls != 99, axis=1)
    
    # Sort the genotypes within each locus, placing uncalled genotypes at the end
    sorted_genotype_calls = np.sort(genotype_calls, axis=1)
    
    subsampled_data = []
    # Iterate through unique counts of called individuals
    for calls in np.sort(np.unique(n_called)):
        if calls < n_subsampling // 2:
            continue  # Skip loci without enough calls
        
        # Extract loci with exactly 'calls' called individuals and keep only the called genotypes
        loci_with_calls = sorted_genotype_calls[n_called == calls][:, :calls]
        
        # Permute the order of genotypes within each locus
        permuted_loci = rng.permuted(loci_with_calls, axis=1)
        
        # Take the first 'n_subsampling // 2' genotypes from each (permuted) locus
        subsampled_data.append(permuted_loci[:, :n_subsampling // 2])
    
    # Concatenate the subsamples from each group of called individuals
    return np.concatenate(subsampled_data)


def simulate_genotype_calling(coverage_distribution, allele_frequency, n_sequenced, n_subsampling, number_simulations):
    """
    Simulate the calling algorithm for alleles of a given frequency.
    
    Args:
        coverage_distribution (list): Coverage distribution for each population.
        allele_frequency (list): True allele frequency in sequenced samples for each population.
        n_sequenced (list): Number of sequenced haplotypes for each population.
        n_subsampling (list): Number of haplotypes after subsampling to account for missing data for each population.
        number_simulations (int): Number of loci to simulate for.
    
    Returns:
        numpy.ndarray: Frequency spectrum for n_subsampling haplotypes with observed allele frequencies resulting from the calling process.
    """
    # Initialize an array to record the allele frequencies from each simulation
    output_freqs = np.zeros(([x + 1 for x in n_subsampling]))
    
    # Record the number of individuals in each population
    pop_n_sequenced = [x // 2 for x in n_sequenced]
    
    # Extract partitions and their probabilities for each allele frequency value and population
    population_partitions = [partitions_and_probabilities(n, 'allele_frequency', af) for (af, n) in zip(allele_frequency, n_sequenced)]
    
    # Flatten the partitions and their probabilities for all populations
    combined_partitions = flatten_nested_list([_[0] for _ in population_partitions], '+')
    combined_part_probabilities = flatten_nested_list([_[1] for _ in population_partitions], '*')
    
    # Iterate through partitions and their probabilities
    for partition, partition_probability in zip(combined_partitions, combined_part_probabilities):
        nsimulations = number_simulations * partition_probability
        
        # Generate reads for all loci for this aggregate partition
        n_ref, n_alt = simulate_reads(coverage_distribution, partition, pop_n_sequenced, int(nsimulations))
        
        # Keep only loci identified as polymorphic
        t_alt = np.sum(n_alt, axis=1)
        n_ref, n_alt = n_ref[t_alt >= 2], n_alt[t_alt >= 2]
        
        # Update the allele frequency spectrum
        output_freqs.flat[0] += np.sum(t_alt < 2)
        
        # Calculate genotype calls for remaining loci
        genotype_calls = np.empty(n_ref.shape, dtype=int)
        genotype_calls[(n_ref == 0) & (n_alt == 0)] = 99 
        genotype_calls[(n_ref > 0) & (n_alt == 0)] = 0
        genotype_calls[(n_ref > 0) & (n_alt > 0)] = 1
        genotype_calls[(n_ref == 0) & (n_alt > 0)] = 2
        
        # Split genotype calls by population
        splits = np.cumsum(pop_n_sequenced)[:-1]
        split_genotype_calls = np.split(genotype_calls, splits, axis=1)
        
        # Handle subsampling to account for missing data
        split_nind_called = [np.sum(genotype_calls != 99, axis=1) for genotype_calls in split_genotype_calls]
        split_enough_calls = [ind_called >= n_subsampling_ // 2 for (ind_called, n_subsampling_) in zip(split_nind_called, n_subsampling)]
        all_enough_calls = np.logical_and.reduce(split_enough_calls)
        split_genotype_calls = [genotype_calls[all_enough_calls] for genotype_calls in split_genotype_calls]
        
        # Record loci without enough calls
        output_freqs.flat[0] += np.sum(all_enough_calls == False)
        
        # Calculate called allele frequencies for each population
        called_freqs = np.empty((len(split_genotype_calls[0]), len(n_sequenced)), int)
        for pop_ii, (n_subsampling_ii, n_sequenced_ii, genotype_calls) in enumerate(zip(n_subsampling, n_sequenced, split_genotype_calls)):
            if n_subsampling_ii != n_sequenced_ii:
                genotype_calls = subsample_genotypes_1D(genotype_calls, n_subsampling_ii)
            called_freqs[:,pop_ii] = np.sum(genotype_calls, axis=1)
        
        # Use the histogramdd function to generate the frequency spectrum for these genotype calls
        binning = [np.arange(n_subsampling_ii + 2) - 0.5 for n_subsampling_ii in n_subsampling]
        called_fs, _ = np.histogramdd(called_freqs, bins=binning)
        
        # Update the output frequency spectrum
        output_freqs += called_fs
        
    return output_freqs / np.sum(output_freqs)


def probability_of_no_call_1D(coverage_distribution, n_sequenced):
    """
    Calculate the probability of no genotype call for all allele frequencies.
    
    Args:
        coverage_distribution (numpy.ndarray): Coverage distribution.
        n_sequenced (int): Number of sequenced haplotypes.
    
    Returns:
        numpy.ndarray: Array containing the probability of no genotype call for each allele frequency.
    """
    # Extract partitions and their probabilities for a given number of samples
    partitions, partitions_probabilities = partitions_and_probabilities(n_sequenced, 'genotype')
    
    # Array of depths corresponding to coverage_distribution
    depths = np.arange(len(coverage_distribution))
    
    # Create an empty array to store the final probabilities of no genotype calling
    all_prob_nocall = np.empty(n_sequenced + 1)
        
    for allele_freq, (partitions, part_probs) in enumerate(zip(partitions, partitions_probabilities)):
        prob_nocall = 0
        
        for part, part_prob in zip(partitions, part_probs):
            # Number of homozygous ref, heterozygous, and homozygous alt
            num_heterozygous, num_hom_alt = part.count(1), part.count(2)
            
            # Probability of getting no reads for the homozygous alt
            P_case0 = (
                coverage_distribution[0]**num_hom_alt *
                np.sum(coverage_distribution * 0.5**depths)**num_heterozygous
            )
            
            # Probability of getting one read for the homozygous alt
            # P_case1a: Probability of 1 read in homozygous alt and 0 in heterozygous
            P_case1a = (
                num_hom_alt * coverage_distribution[1] * coverage_distribution[0]**(num_hom_alt - 1) *
                np.sum(coverage_distribution * 0.5**depths)**num_heterozygous
            )
            
            # P_case1b: Probability of 1 read in heterozygous and 0 in homozygous alt
            P_case1b = (
                coverage_distribution[0]**num_hom_alt *
                np.sum(coverage_distribution * 0.5**depths)**(num_heterozygous - 1) *
                num_heterozygous * np.sum(depths * coverage_distribution * 0.5**depths)
            )
            
            # Calculate the probability of no call
            prob_nocall += part_prob * (P_case0 + P_case1a + P_case1b)
        
        all_prob_nocall[allele_freq] = prob_nocall
    
    return all_prob_nocall


def probability_enough_individuals_covered(coverage_distribution, n_sequenced, n_subsampling):
    """
    Calculate the probability of having enough individuals covered to obtain n_subsampling successful genotypes.
    
    Args:
        coverage_distribution (numpy.ndarray): Coverage distribution.
        n_sequenced (int): Number of sequenced haplotypes.
        n_subsampling (int): Number of haplotypes to subsample.
    
    Returns:
        float: Probability of having enough individuals covered.
      """
    # Initialize the probability of having enough individuals covered
    prob_enough_individuals_covered = 0
        
    # Use math.ceil to round up, as you need a minimum of n_subsampling//2 covered individuals
    for covered in range(int(math.ceil(n_subsampling/2)), n_sequenced//2+1):
        # Calculate the probability of having enough individuals covered
        prob_enough_individuals_covered += (
            coverage_distribution[0]**(n_sequenced//2 - covered) *
            np.sum(coverage_distribution[1:]) ** covered *
            comb(n_sequenced//2, covered)
        )
    
    return prob_enough_individuals_covered


def projection_matrix(n_sequenced, n_subsampling):
    """
    Create a projection matrix for down-sampling haplotypes.
    
    Args:
        n_sequenced (int): Number of haplotypes sequenced.
        n_subsampling (int): Number of haplotypes to project down to.
    
    Returns:
        numpy.ndarray: Projection matrix.
    
    """
    # Create an empty matrix to store the projection
    projection_matrix = np.empty((n_sequenced + 1, n_subsampling + 1))
    
    # Calculate the projection for each allele frequency
    for allele_freq in range(n_sequenced + 1):
        projection_matrix[allele_freq, :] = dadi.Numerics._cached_projection(n_subsampling, n_sequenced, allele_freq)
    
    return projection_matrix


def calling_error_matrix(coverage_distribution, n_subsampling):
    """
        Calculate the calling error matrix based on the coverage distribution and subsampling.
        
        Args:
            coverage_distribution (numpy.ndarray): Coverage distribution.
            n_subsampling (int): Number of haplotypes to subsample.
        
        Returns:
            numpy.ndarray: Calling error matrix.
        """
    # Extract partitions and their probabilities for a given number of samples
    partitions, partitions_probabilities = partitions_and_probabilities(n_subsampling, 'genotype')
    
    # Array of depths corresopnding to coverage distribution
    depths = np.arange(1, len(coverage_distribution[1:])+1, 1)
    
    # Probability that a heterozygote is called incorrectly
    # Note that zero reads would be no call, so it isn't included here
    coverage_distribution_ = [x/coverage_distribution[1:].sum() for x in coverage_distribution[1:]]
    prob_het_err = np.sum(coverage_distribution_ * 0.5**depths)
    
    # Transformation matrix
    trans_matrix = np.zeros((n_subsampling+1,n_subsampling+1))
    for allele_freq, (partitions_, part_probs) in enumerate(zip(partitions, partitions_probabilities)):
        for part, part_prob in zip(partitions_, part_probs):
            # Extract the number of heterozygous
            n_heterozygous = part.count(1)
            
            # For each possible number of heterozygous errors
            for n_error in range(n_heterozygous+1):
                # Probability that n_error heterozygous are called incorrectly out of heterozygous
                p_nerr = ssd.binom.pmf(n_error, n_heterozygous, prob_het_err)
                
                # Potential numbers of references erros, alternative erros, and the net change in allele frequency
                n_ref = np.arange(n_error+1)
                n_alt = n_error - n_ref
                net_change = n_alt - n_ref
                
                # Probabilities of each possible number of reference erros
                p_nref = ssd.binom.pmf(n_ref, n_error, 0.5)
                
                # Allele frequencies after errors
                afs_after_error = allele_freq + net_change
                
                # Record where error alleles end up
                trans_matrix[afs_after_error, allele_freq] += part_prob * p_nerr * p_nref
    
    return trans_matrix


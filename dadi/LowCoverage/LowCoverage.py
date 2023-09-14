import itertools
from io import StringIO as SIO
import numpy as np, pandas as pd
import dadi
import warnings

import math
from scipy.special import comb
import scipy.stats as ss, scipy.stats.distributions as ssd

rng = np.random.default_rng()


def read_vcf(vcf_file):
    """
    Reads data from a VCF file and returns it as a pandas DataFrame.
    
    Args:
        vcf_file (str): The path to the VCF file to be read.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the VCF information.
    
    """
    # Open the VCF file for reading
    with open(vcf_file, 'r') as input_file:
        # Filter out lines starting with '##' (header lines)
        data_lines = [line for line in input_file if not line.startswith('##')]
        
        # Read the VCF data into a DataFrame using pandas
        vcf_to_table = pd.read_csv(SIO(''.join(data_lines)),dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str, 'QUAL': str, 'FILTER': str, 'INFO': str}, sep='\t').rename(columns={'#CHROM': 'CHROM'})
    
    return vcf_to_table


def extract_coverage(vcf_element, target_format):
    """
    Extracts coverage information from a VCF element based on a specified target format.
    
    Args:
        vcf_element (str): The VCF element from which to extract coverage information.
        target_format (str): The format string specifying the structure of the VCF element.
    
    Returns:
        int: The extracted coverage value, or 0 if no coverage information is found (i.e., missing data).
    
    Raises:
        ValueError: If the read depth parameter is not found in the target format.
    """
    try:
        # Find the index of 'AD' in the target format
        i = target_format.index('AD')
        
        # Extract the raw coverage information from the VCF element
        raw_coverage = vcf_element.split(':')[i]
        
        # Check if raw_coverage contains multiple values, representing reference and alternative allele information
        if len(raw_coverage) > 1:
            # Convert the comma-separated values to integers and calculate the sum (i.e., total number of reads)
            return sum(map(int, raw_coverage.split(',')))
        else:
            # Return 0 if there's only one value in raw_coverage
            return 0
    except ValueError:
        # Raise an exception if alleletic depths is not found in the VCF file
        raise ValueError("Information about allelic depths for the reference and alternative alleles not found in the input file")


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
            partition_ways = np.array([
                [
                    np.exp(dadi.Numerics.multinomln([part.count(0), part.count(1), part.count(2)])) * 2 ** part.count(1)
                    for part in parts
                ]
                for parts in partitions
            ])
            
            # Normalize partition probabilities
            partition_ways_sum = np.array([np.sum(np.array(part)) if len(part) > 1 else np.array(part) for part in partition_ways])
            partition_probabilities = partition_ways / partition_ways_sum
        
        else:
            raise ValueError("Invalid partition_type. Use 'allele_frequency' or 'genotype'.")
        
        return partitions, partition_probabilities


def calculate_coverage_distribution(vcf_file, population_file, population):
    """
    Calculate coverage distribution for specified populations from a VCF file.
    
    Args:
        vcf_file (str): The path to the VCF file.
        population_file (str): The path to the population information file.
        population (list): A list of population names to calculate coverage for.
    
    Returns:
        list: A list of Pandas Series, each representing the coverage distribution for a population.
    """
    # Read VCF data
    vcf_data = read_vcf(vcf_file)  
    
    # Get the index of the 'FORMAT' column and extract format information
    format_index = list(vcf_data.columns).index('FORMAT')
    format_info = vcf_data.iloc[1, format_index].split(':')
    
    # Read the population information
    population_info = pd.read_csv(population_file, header=None, sep='\s+|\t', engine='python')
    
    # Calculate coverage for all individuals in the VCF data
    coverage = vcf_data.iloc[:, 9:].applymap(lambda x: extract_coverage(x, format_info))
    
    coverage_distribution = []
    for pop in population:
        # Filter the population info DataFrame for the specified population
        population_sub = population_info[population_info.iloc[:, 1] == pop]
        population_sub = population_sub.iloc[:, 0]
        
        # Calculate individual coverage distribution for the population
        individual_coverage = coverage[population_sub].stack().value_counts(normalize=True).sort_index()
        coverage_distribution.append(individual_coverage)
    
    return coverage_distribution


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
                trans_matrix[allele_freq, afs_after_error] += part_prob * p_nerr * p_nref
    
    return trans_matrix


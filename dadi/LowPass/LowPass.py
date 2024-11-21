import itertools
import numpy, numpy as np
import dadi
import warnings

import math
from scipy.special import comb
import scipy.stats as ss, scipy.stats.distributions as ssd
from itertools import combinations
from math import factorial

rng = numpy.random.default_rng()


def compute_cov_dist(data_dict, pop_ids):
    """
    Compute the depth of coverage distribution for each population.

    Args:
    - data_dict (dict): A dictionary containing data entries.
    - pop_ids (list): A list of population identifiers for which depth of coverage distribution is computed.

    Returns:
    - dict: A dictionary where keys are population identifiers, and values are arrays representing
            the depth of coverage distribution for each population.
    
    Raises:
    - ValueError: If information about allelic depths for the reference and alternative alleles
                  is not found in the data dictionary.
    """
    try:
        coverage_distribution = {}
        for pop in pop_ids:
            all_depths = numpy.concatenate([entry['coverage'][pop] for entry in data_dict.values()])
            max_depth = all_depths.max()
            binning = numpy.arange(max_depth+2)-0.5
            coverage_distribution[pop] = [np.arange(max_depth+1), numpy.histogram(all_depths, bins=binning)[0]]
        # Normalize counts
        coverage_distribution = {pop: numpy.array([elements, counts / counts.sum()]) for pop, (elements, counts) in coverage_distribution.items()}
        
        return coverage_distribution
    except:
        raise ValueError("Information about allelic depths for the reference and alternative alleles not found in the data dictionary")


def part_inbreeding_probability(parts, Fx):
    """
    Calculate genotype partition probabilities under inbreeding.

    Parameters:
    - parts (list of lists): List of genotype partitions.
    - Fx (float): Inbreeding coefficient.

    Returns:
    - numpy.array: Normalized genotype partition probabilities.
    """
    part_prob = numpy.array([])
    for part in parts:
        if sum(part) != 0 and sum(part) != 2*len(part):
            p = (2 * part.count(2) + part.count(1))/(2*len(part))
            alpha = p*((1.0-Fx)/Fx)
            beta  = (1.0-p)*((1.0-Fx)/Fx)
            
            p00, p01, p11 = numpy.exp([dadi.Numerics.BetaBinomln(_,2,alpha,beta) for _ in range(2+1)])
            n, n00, n01, n11 = len(part), part.count(0), part.count(1), part.count(2)
            
            part_prob = numpy.append(part_prob, (factorial(n) / (factorial(n00) * factorial(n01) * factorial(n11))) * (p00 ** n00) * (p01 ** n01) * (p11 ** n11))
        else:
            part_prob = numpy.append(part_prob, 1)
        
    return part_prob / sum(part_prob)


def partitions_and_probabilities(n_sequenced, partition_type, Fx=0, allele_frequency=None):
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
            
            if Fx == 0:
                # Calculate partition probabilities
                partition_ways = numpy.array([numpy.exp(dadi.Numerics.multinomln([part.count(0), part.count(1), part.count(2)])) * 2 ** part.count(1) for part in partitions])
                partition_probabilities = partition_ways / numpy.sum(partition_ways)

            else:
                partition_probabilities = part_inbreeding_probability(partitions, Fx)
        
        elif partition_type == 'genotype':
            # Generate an array of allele counts from 0 to n_sequenced
            allele_counts = numpy.arange(n_sequenced + 1)
            
            # Generate partitions
            partitions = [dadi.Numerics.cached_part(allele_count, n_sequenced / 2) for allele_count in allele_counts]
            
            if Fx == 0:
                # Calculate partition probabilities using multinomial likelihood
                partition_ways = [
                    [
                        numpy.exp(dadi.Numerics.multinomln([part.count(0), part.count(1), part.count(2)])) * 2 ** part.count(1)
                        for part in parts
                    ]
                    for parts in partitions
                ]
                
                # Normalize partition probabilities
                partition_ways_sum = [[numpy.sum(part)] if len(part) > 1 else part for part in partition_ways]
                partition_probabilities = [numpy.array(pw) / numpy.array(pwb) for pw, pwb in zip(partition_ways, partition_ways_sum)]

            else:
                    partition_probabilities = [part_inbreeding_probability(part, Fx) for part in partitions]
         
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
        coverage_distribution (list): Depth of coverage distribution for each population.
        flattened_partition (list): Flattened genotype partition.
        pop_n_sequenced (list): Number of sequenced haplotypes for each population.
        number_simulations (int): Number of simulations to perform.
    
    Returns:
        tuple: A tuple containing two arrays:
            - numpy.ndarray: Arrays of reference allele counts for each simulated individual.
            - numpy.ndarray: Arrays of alternative allele counts for each simulated individual.
    """
    flattened_partition = numpy.array(flattened_partition)
    
    # Split partition for different populations
    partition = split_list_by_lengths(flattened_partition, pop_n_sequenced)
    
    # Empty array for storing the simulated data, initialized with zeros
    simulated_coverage = numpy.zeros((number_simulations, len(flattened_partition)), dtype=int)
    
    # Create population breaks
    splits = numpy.concatenate((numpy.array([0]), numpy.cumsum(pop_n_sequenced)), axis=0)
    
    # Simulate reads for each population
    pops = list(coverage_distribution.keys())
    for i, _ in enumerate(partition):
        cov_distribution = coverage_distribution[pops[i]][1]
        cov_sampling = ss.rv_discrete(values=[numpy.arange(len(cov_distribution)), cov_distribution])
        coverages = cov_sampling.rvs(size=(number_simulations, len(partition[i])))
        simulated_coverage[:, splits[i]:splits[i+1]] = coverages
    
    # Initialize arrays for reference and alternative allele counts
    n_ref, n_alt = numpy.zeros((number_simulations, len(flattened_partition)), dtype=int), numpy.zeros((number_simulations, len(flattened_partition)), dtype=int)
    
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
    n_called = numpy.count_nonzero(genotype_calls != 99, axis=1)
    
    # Sort the genotypes within each locus, placing uncalled genotypes at the end
    sorted_genotype_calls = numpy.sort(genotype_calls, axis=1)
    
    subsampled_data = []
    # Iterate through unique counts of called individuals
    for calls in numpy.sort(numpy.unique(n_called)):
        if calls < n_subsampling // 2:
            continue  # Skip loci without enough calls
        
        # Extract loci with exactly 'calls' called individuals and keep only the called genotypes
        loci_with_calls = sorted_genotype_calls[n_called == calls][:, :calls]
        
        # Permute the order of genotypes within each locus
        permuted_loci = rng.permuted(loci_with_calls, axis=1)
        
        # Take the first 'n_subsampling // 2' genotypes from each (permuted) locus
        subsampled_data.append(permuted_loci[:, :n_subsampling // 2])
    
    # Concatenate the subsamples from each group of called individuals
    return numpy.concatenate(subsampled_data)


def simulate_GATK_multisample_calling(coverage_distribution, allele_frequency, n_sequenced, n_subsampling, number_simulations, Fx):
    """
    Simulate the GATK multi-sample calling algorithm for alleles of a given frequency.
    
    Args:
        coverage_distribution (list): Depth of coverage distribution for each population.
        allele_frequency (list): True allele frequency in sequenced samples for each population.
        n_sequenced (list): Number of sequenced haplotypes for each population.
        n_subsampling (list): Number of haplotypes after subsampling to account for missing data for each population.
        number_simulations (int): Number of loci to simulate for.
    
    Returns:
        numpy.ndarray: Frequency spectrum for n_subsampling haplotypes with observed allele frequencies resulting from the calling process.
    """
    # Initialize an array to record the allele frequencies from each simulation
    output_freqs = numpy.zeros(([x + 1 for x in n_subsampling]))
    
    # Record the number of individuals in each population
    pop_n_sequenced = [x // 2 for x in n_sequenced]
    
    # Extract partitions and their probabilities for each allele frequency value and population
    population_partitions = [partitions_and_probabilities(n, 'allele_frequency', Fx_, af) for (af, n, Fx_) in zip(allele_frequency, n_sequenced, Fx)]
    
    # Flatten the partitions and their probabilities for all populations
    combined_partitions = flatten_nested_list([_[0] for _ in population_partitions], '+')
    combined_part_probabilities = flatten_nested_list([_[1] for _ in population_partitions], '*')
    
    # Iterate through partitions and their probabilities
    for partition, partition_probability in zip(combined_partitions, combined_part_probabilities):
        nsimulations = number_simulations * partition_probability
        
        # Generate reads for all loci for this aggregate partition
        n_ref, n_alt = simulate_reads(coverage_distribution, partition, pop_n_sequenced, int(nsimulations))
        
        # Keep only loci identified as polymorphic
        t_alt = numpy.sum(n_alt, axis=1)
        n_ref, n_alt = n_ref[t_alt >= 2], n_alt[t_alt >= 2]
        
        # Update the allele frequency spectrum
        output_freqs.flat[0] += numpy.sum(t_alt < 2)
        
        # Calculate genotype calls for remaining loci
        genotype_calls = numpy.empty(n_ref.shape, dtype=int)
        genotype_calls[(n_ref == 0) & (n_alt == 0)] = 99 
        genotype_calls[(n_ref > 0) & (n_alt == 0)] = 0
        genotype_calls[(n_ref > 0) & (n_alt > 0)] = 1
        genotype_calls[(n_ref == 0) & (n_alt > 0)] = 2
        
        # Split genotype calls by population
        splits = numpy.cumsum(pop_n_sequenced)[:-1]
        split_genotype_calls = numpy.split(genotype_calls, splits, axis=1)
        
        # Handle subsampling to account for missing data
        split_nind_called = [numpy.sum(genotype_calls != 99, axis=1) for genotype_calls in split_genotype_calls]
        split_enough_calls = [ind_called >= n_subsampling_ // 2 for (ind_called, n_subsampling_) in zip(split_nind_called, n_subsampling)]
        all_enough_calls = numpy.logical_and.reduce(split_enough_calls)
        split_genotype_calls = [genotype_calls[all_enough_calls] for genotype_calls in split_genotype_calls]
        
        # Record loci without enough calls
        output_freqs.flat[0] += numpy.sum(all_enough_calls == False)
        
        # Calculate called allele frequencies for each population
        called_freqs = numpy.empty((len(split_genotype_calls[0]), len(n_sequenced)), int)
        for pop_ii, (n_subsampling_ii, n_sequenced_ii, genotype_calls) in enumerate(zip(n_subsampling, n_sequenced, split_genotype_calls)):
            if n_subsampling_ii != n_sequenced_ii:
                genotype_calls = subsample_genotypes_1D(genotype_calls, n_subsampling_ii)
            called_freqs[:,pop_ii] = numpy.sum(genotype_calls, axis=1)
        
        # Use the histogramdd function to generate the frequency spectrum for these genotype calls
        binning = [numpy.arange(n_subsampling_ii + 2) - 0.5 for n_subsampling_ii in n_subsampling]
        called_fs, _ = numpy.histogramdd(called_freqs, bins=binning)
        
        # Update the output frequency spectrum
        output_freqs += called_fs
        
    return output_freqs / numpy.sum(output_freqs)


def probability_of_no_call_1D_GATK_multisample(coverage_distribution, n_sequenced, Fx):
    """
    Calculate the GATK multi-sample probability of no genotype call for all allele frequencies.
    
    Args:
        coverage_distribution (numpy.ndarray): Depth of coverage distribution.
        n_sequenced (int): Number of sequenced haplotypes.
    
    Returns:
        numpy.ndarray: Array containing the probability of no genotype call for each allele frequency.
    """
    # Extract partitions and their probabilities for a given number of samples
    partitions, partitions_probabilities = partitions_and_probabilities(n_sequenced, 'genotype', Fx)
    
    # Array of depths corresponding to depth of coverage_distribution
    depths = numpy.arange(len(coverage_distribution[0]))
    
    # Create an empty array to store the final probabilities of no genotype calling
    all_prob_nocall = numpy.empty(n_sequenced + 1)
        
    for allele_freq, (partitions_, part_probs) in enumerate(zip(partitions, partitions_probabilities)):
        prob_nocall = 0
        
        for part, part_prob in zip(partitions_, part_probs):
            # Number of homozygous ref, heterozygous, and homozygous alt
            num_heterozygous, num_hom_alt = part.count(1), part.count(2)
            
            # Probability of getting no reads for the homozygous alt
            P_case0 = (
                coverage_distribution[1][0]**num_hom_alt *
                numpy.sum(coverage_distribution[1] * 0.5**depths)**num_heterozygous
            )
            
            # Probability of getting one read for the homozygous alt
            # P_case1a: Probability of 1 read in homozygous alt and 0 in heterozygous
            if num_hom_alt > 0:
                P_case1a = (
                    num_hom_alt * coverage_distribution[1][1] * coverage_distribution[1][0]**(num_hom_alt - 1) *
                    numpy.sum(coverage_distribution[1] * 0.5**depths)**num_heterozygous
                )
            else:
                # The above expression can return nan when it should be 0, if num_hom_alt==0 and coverage_distribution[1][0]==0.
                P_case1a = 0
            
            # P_case1b: Probability of 1 read in heterozygous and 0 in homozygous alt
            P_case1b = (
                coverage_distribution[1][0]**num_hom_alt *
                numpy.sum(coverage_distribution[1] * 0.5**depths)**(num_heterozygous - 1) *
                num_heterozygous * numpy.sum(depths * coverage_distribution[1] * 0.5**depths)
            )
            
            # Calculate the probability of no call
            prob_nocall += part_prob * (P_case0 + P_case1a + P_case1b)
        
        all_prob_nocall[allele_freq] = prob_nocall
    
    return all_prob_nocall


def probability_enough_individuals_covered(coverage_distribution, n_sequenced, n_subsampling):
    """
    Calculate the probability of having enough individuals covered to obtain n_subsampling successful genotypes.

    Note: Because we consider this only for sites already called as variant, it is already guaranteed that
          at least one individual has coverage.
    
    Args:
        coverage_distribution (numpy.ndarray): Depth of coverage distribution.
        n_sequenced (int): Number of sequenced haplotypes.
        n_subsampling (int): Number of haplotypes to subsample.
    
    Returns:
        float: Probability of having enough individuals covered.
      """
    # Initialize the probability of having enough individuals covered
    prob_enough_individuals_covered = 0
        
    # Use math.ceil to round up, as you need a minimum of n_subsampling//2-1 covered individuals
    # (We already know at least one individual is covered, if the site is already called as variant.)
    for covered in range(int(math.ceil(n_subsampling/2))-1, n_sequenced//2+1-1):
        # Calculate the probability of having enough individuals covered
        prob_enough_individuals_covered += (
            coverage_distribution[1][0]**(n_sequenced//2-1 - covered) *
            numpy.sum(coverage_distribution[1][1:]) ** covered *
            comb(n_sequenced//2-1, covered)
        )
    
    return prob_enough_individuals_covered


import numpy
from itertools import combinations

def projection_inbreeding(partition, k):
    """
    Calculate the distribution of inbreeding coefficients for a given partition.
    
    Args:
    - partition (iterable): A collection of indices representing the individuals in the partition.
    - k (int): The total number of individuals considered in each combination.
    
    Returns:
    - numpy.ndarray: An array representing the distribution of inbreeding coefficients. Each index in the array corresponds 
      to the total number of shared alleles in a combination, and the values represent the corresponding frequencies normalized 
      by the total number of combinations.
    ```
    """
    result = numpy.zeros_like(range(k+1))

    # Generate all possible combinations of partition indices
    partitions = list(combinations(partition, k//2))

    for p in partitions:
        # Calculate the sum for each partition and update the result array
        result[sum(p)] += 1

    return result/sum(result)


def projection_matrix(n_sequenced, n_subsampling, F):
    """
    Create a projection matrix for down-sampling haplotypes.
    
    Args:
        n_sequenced (int): Number of haplotypes sequenced.
        n_subsampling (int): Number of haplotypes to project down to.
    
    Returns:
        numpy.ndarray: Projection matrix.
    
    """
    # Create an empty matrix to store the projection
    projection_matrix = numpy.empty((n_sequenced + 1, n_subsampling + 1))
    
    # Calculate the projection for each allele frequency
    for allele_freq in range(n_sequenced + 1):
        if F != 0:
            partitions, partition_probabilities = partitions_and_probabilities(n_sequenced, 'allele_frequency', F, allele_freq)
            
            proj = numpy.zeros_like(range(n_subsampling+1), dtype=float)
            for partition, part_prob in zip(partitions, partition_probabilities):
                proj += projection_inbreeding(partition, n_subsampling) * part_prob
            projection_matrix[allele_freq, :] = proj
        else:
            projection_matrix[allele_freq, :] = dadi.Numerics._cached_projection(n_subsampling, n_sequenced, allele_freq)
    
    return projection_matrix


def calling_error_matrix(coverage_distribution, n_subsampling, Fx=0):
    """
        Calculate the calling error matrix based on the depth of coverage distribution and subsampling.
        
        Args:
            coverage_distribution (numpy.ndarray): Depth of coverage distribution.
            n_subsampling (int): Number of haplotypes to subsample.
        
        Returns:
            numpy.ndarray: Calling error matrix.
        """
    # Extract partitions and their probabilities for a given number of samples
    partitions, partitions_probabilities = partitions_and_probabilities(n_subsampling, 'genotype', Fx)
    
    # Array of depths corresopnding to depth of coverage distribution
    depths = coverage_distribution[0][1:]
    
    # Probability that a heterozygote is called incorrectly
    # Note that zero reads would be no call, so it isn't included here
    coverage_distribution_ = [x/coverage_distribution[1][1:].sum() for x in coverage_distribution[1][1:]]
    prob_het_err = 2*numpy.sum(coverage_distribution_ * 0.5**depths)
    
    # Transformation matrix
    trans_matrix = numpy.zeros((n_subsampling+1,n_subsampling+1))
    for allele_freq, (partitions_, part_probs) in enumerate(zip(partitions, partitions_probabilities)):
        for part, part_prob in zip(partitions_, part_probs):
            # Extract the number of heterozygous
            n_heterozygous = part.count(1)
            
            # For each possible number of heterozygous errors
            for n_error in range(n_heterozygous+1):
                # Probability that n_error heterozygous are called incorrectly out of heterozygous
                p_nerr = ssd.binom.pmf(n_error, n_heterozygous, prob_het_err)
                
                # Potential numbers of references erros, alternative erros, and the net change in allele frequency
                n_ref = numpy.arange(n_error+1)
                n_alt = n_error - n_ref
                net_change = n_alt - n_ref
                
                # Probabilities of each possible number of reference erros
                p_nref = ssd.binom.pmf(n_ref, n_error, 0.5)
                
                # Allele frequencies after errors
                afs_after_error = allele_freq + net_change
                
                # Record where error alleles end up
                trans_matrix[allele_freq, afs_after_error] += part_prob * p_nerr * p_nref
    
    return trans_matrix

def low_cov_precalc_GATK_multisample_GATK_multisample(nsub, nseq, cov_dist, sim_threshold=1e-2, Fx=0, nsim=1e4):
    """
    Calculate transformation matrices for the low-pass calling model based on the GATK multi-sample algorithm.

    Args:
        nsub: Final sample size (in haplotypes)
        nseq: Sequenced sample size (in haplotypes)
        cov_dist: Depth of coverage distribution (list of one array per population)
        sim_threshold: This method uses the probability an allele is not called
                    to switch between analytic and simulation-based methods.
                    Setting this threshold to 0 will always use simulations,
                    while setting it to 1 will always use analytics.
        nsim: For simulations, number of simulations per allele frequency combination
        """
    
    # As a lower bound on the probability that a allele with a given frequency is not called,
    # use the probability it is not called considering only the reads in each individual population.
    # We calculate them separately, then combine them into a single matrix
    prob_nocall_by_pop = [probability_of_no_call_1D_GATK_multisample(cov_dist_, nseq_, Fx_) for (cov_dist_, nseq_, Fx_) in zip(cov_dist.values(), nseq, Fx)]
    prob_nocall_ND = 1
    for apn_1D in prob_nocall_by_pop:
        prob_nocall_ND = numpy.multiply.outer(prob_nocall_ND, apn_1D)
    
    # Identify those entries that should be simulated, as opposed to analytically calculated
    use_sim_mat = prob_nocall_ND > sim_threshold
    
    ### For analytic calling model
    # Probability that enough individuals got at least one read to subsample
    # This doesn't depend on allele frequency
    prob_enough_covered = numpy.prod([probability_enough_individuals_covered(cov_dist_, nseq_, nsub_) for (cov_dist_, nseq_, nsub_) in zip(cov_dist.values(), nseq, nsub)])
    # Precalculate analytic projection and heterozygote error matrices
    proj_mats = [prob_enough_covered * projection_matrix(nseq_, nsub_, Fx_) for (nseq_, nsub_, Fx_) in zip(nseq, nsub, Fx)]
    heterr_mats = [calling_error_matrix(cov_dist_, nsub_, Fx_) for (cov_dist_, nsub_, Fx_) in zip(cov_dist.values(), nsub, Fx)]
    
    ### For simulation calling model
    # Indices of entries in afs where we should use simulation
    simulated_indices = numpy.argwhere(use_sim_mat)
    # Do the simulations
    sim_outputs = {tuple(af):simulate_GATK_multisample_calling(cov_dist, af, nseq, nsub, nsim, Fx) for af in simulated_indices}
    
    return prob_nocall_ND, use_sim_mat, proj_mats, heterr_mats, sim_outputs

def make_low_pass_func_GATK_multisample(func, cov_dist, pop_ids, nseq, nsub, sim_threshold=1e-2, Fx=None, nsim=1000):
    """
    Generate a version of func accounting for low-pass distortion based on the GATK multi-sample algorithm.

    Args:
        demo_model: Specified demographic model in dadi.
        data_dict: A data dictionary comprising information extracted from a VCF file.
        pop_ids: Population names to be analyzed.
        nseq: Sequenced sample size (in haplotypes).
        nsub: Final sample size (in haplotypes)
        sim_threshold: This method switches between analytic and simulation-based methods. 
            Setting this threshold to 0 will always use simulations, while setting it to 1 will always use analytics. 
            Values in between indicate that simulations will be employed for thresholds below that value.
        Fx: Inbreeding coefficient.
        nsim: Number of simulations to use per potential allele frequency combination
    """
    # # Compute depth of coverage distribution
    # cov_dist = compute_cov_dist(dd, pop_ids)
    
    # Used to cache matrices used for low-pass transformation
    precalc_cache = {}

    if Fx is None:
        Fx = [0] * len(nseq)
    elif numpy.any(numpy.asarray(Fx) == 1):
        raise ValueError("Cannot apply low-coverage correction assuming perfect inbreeding "
                         "Fx=1. If organism is truly perfectly inbreed, then it can be "
                         "treated as haploid, so low coverge does not introduce bias.")
    
    def lowpass_func(*args, **kwargs):
        nonlocal Fx
        new_args = [args[0]] + [nseq] + list(args[2:])
        model = func(*new_args, **kwargs)
        if model.folded:
            raise ValueError('Low-pass model not tested for folded model spectra yet.')
        
        if tuple(nsub) not in precalc_cache:
            precalc_cache[tuple(nsub)] = low_cov_precalc_GATK_multisample_GATK_multisample(nsub, nseq, cov_dist, sim_threshold, Fx, nsim=nsim)
        prob_nocall_ND, use_sim_mat, proj_mats, heterr_mats, sim_outputs = precalc_cache[tuple(nsub)]
        # First, transform entries we do analytically. We zero out the entries
        # we'll simulate, since we'll handle their contribution later.
        analytic = model * (1-use_sim_mat)
        # Account for sites that aren't called 
        analytic *= (1-prob_nocall_ND)
        # Apply projection and heterozygote error transformations
        for pop_ii, (proj_mat, heterr_mat) in enumerate(zip(proj_mats, heterr_mats)):
            analytic = analytic.swapaxes(pop_ii, -1) # Swap axes to use efficient matrix multiplication
            analytic = analytic.dot(proj_mat)
            analytic = analytic.dot(heterr_mat)
            analytic = analytic.swapaxes(pop_ii, -1)
        
        # Use the simulated outputs
        simulated = numpy.sum([model[af]*output for (af, output) in sim_outputs.items()], axis=0)
        
        output = analytic + simulated
        # Not sure why the folding status got undefined in the above manipulations...
        output.folded = model.folded
        output.extrap_x = model.extrap_x
        
        return output
    lowpass_func.__name__ = func.__name__ + '_lowcov'
    lowpass_func.__doc__ = func.__doc__
    return lowpass_func
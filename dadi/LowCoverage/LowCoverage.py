import dadi
import numpy as np
import pandas as pd
from io import StringIO as SIO
import warnings
import itertools

def read_vcf(path):
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
        
        df = pd.read_csv(
                SIO(''.join(lines)),
                dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
                    'QUAL': str, 'FILTER': str, 'INFO': str},
                sep='\t'
            ).rename(columns={'#CHROM': 'CHROM'})
    
    return df

def extract_coverage(vcf_table, zero_reads):
    ind_cov = vcf_table.split(':')[1]

    if zero_reads:
        z_reads = 0
    else:
        z_reads = pd.NA
    
    if len(ind_cov) > 1:
        ind_cov = map(int, ind_cov.split(','))
        coverage = sum(ind_cov)
    else:
        coverage = z_reads
    
    return coverage

def calc_error(vcf_file):
    vcf = read_vcf(vcf_file)    
    ind_coverage = vcf.iloc[:, 9:].applymap(lambda x: extract_coverage(x, True))
    coverage_distribution = ind_coverage.stack().value_counts(normalize=True).sort_index()
    
    return coverage_distribution

def create_partitions_single(n):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        allele_counts = np.arange(n+1)
        
        all_partitions = [dadi.Numerics.cached_part(ac, n/2) for ac in allele_counts]
        
        partition_probabilities = np.array([
            [np.exp(dadi.Numerics.multinomln([part.count(0), part.count(1), part.count(2)])) for part in parts]
            for parts in all_partitions
        ])
        
        partition_probabilitie_sum = np.array([np.sum(np.array(elem)) if len(elem) > 1 else np.array(elem) for elem in partition_probabilities])
        partition_probabilities /= partition_probabilitie_sum
    
    return all_partitions, partition_probabilities


def calculate_calling_probabilities(coverage_distribution):
    """
    Calculate probabilities of not calling genotypes and the probability of having two derived alleles based on the number of reads supporting the alternative allele.
    """

    error_not_call = []  # List to store the error probability of not calling genotypes
    probability_double_alt = []  # List to store the probability of having two derived alleles
    probability_not_call_probs = []  # List to store the probability of not calling genotypes

    for read in [1, 0]:  # Loop for read values of 1 and 0
        for gen in ([0, 1], [1]):  # Loop for genotypes [0, 1] and [1]

            probability_not_call = []  # Temporary list to store the probability of not calling values
            probability_double_alt_ = []  # Temporary list to store the probability of getting two derived alleles

            for i in coverage_distribution.index:  # Loop over coverage distribution
                if i > 20:
                    probability_not_call.append(0)  # If i > 20, set probability_not_call to 0 since it will take a long time to calculate all combinations and the probability will be very low.
                else:
                    genotypes = list(itertools.product(gen, repeat=i))  # Generate possible genotypes for i number of reads
                    genotypes_count = len(genotypes)  # Count the total number of genotypes
                    matches_count = sum(1 for genotype in genotypes if genotype.count(1) <= read)
                    # Count the number of genotypes with at most 'read' alternative alleles
                    probability_not_call.append(matches_count / genotypes_count)
                    # Calculate the probability of not calling the genotype based on the count

                    if gen == [0, 1]:  # For genotype [0, 1]
                        count_total = sum(1 for elem in genotypes if sum(elem) > read)
                        # Count the number of genotypes with more than 'read' alternative alleles
                        count_double_alt = sum(1 for elem in genotypes if sum(elem) == i)
                        # Count the number of genotypes with exactly 'i' alternative alleles

                        if i > 1 and read == 1:
                            probability_double_alt_.append(count_double_alt / count_total)
                            # Calculate the probability of having double alternative alleles for read = 1 and i > 1
                        elif i >= 1 and read == 0:
                            probability_double_alt_.append(count_double_alt / count_total)
                            # Calculate the probability of having double alternative alleles for read = 0 and i >= 1
                        else:
                            probability_double_alt_.append(0)  # Set probability to 0 otherwise

            probability_double_alt.append(probability_double_alt_)
            # Store the temporary probability_double_alt_ list for the current gen

            # 'error_not_call' will store the probabilities of not making a genotype call when there are either 1 or 0 reads supporting the alternative allele for the genotypes [0,1] and [1,1].
            error_not_call.append(
                sum(
                    probability_not_call[i] * coverage_distribution.to_list()[i]
                    for i in range(len(probability_not_call))
                )
            )
            # Calculate the error_not_call by summing probability_not_call multiplied by coverage_distribution

            probability_not_call_probs.append(probability_not_call)
            # Store the temporary probability_not_call list for the current gen and call combination

    return error_not_call, probability_double_alt, probability_not_call_probs

def sfs_redist(vcf_file, all_partitions, all_part_probs, n):
    all_weights = np.zeros((n,n))

    coverage_distribution = calc_error(vcf_file)
    error_not_call, probability_double_alt, probability_not_call_probs = calculate_calling_probabilities(coverage_distribution)

    for allele_count in range(1, n):
        for part, part_prob in zip(all_partitions[allele_count], all_part_probs[allele_count]):
            nhet = part.count(1)
            ndouble = part.count(2)

            # Taking into consideration different scenarios, one example would be singletons, where we have the conditions nhet > 1 and ndouble == 0. In such cases,
            # our task would involve calculating the probability of observing 1 or 0 reads for the alternative allele. In the case of doubletons, where we observe 
            # two heterozygous alleles, we must consider the probability of observing 1 or 0 reads for the alternative allele in one individual and zero reads for 
            # the alternative allele in the other individual. Additionally, we need to account for the number of combinations of such occurrences.
            if nhet > 0 and ndouble == 0:
                error_not_call_ = (error_not_call[0] * error_not_call[1] ** (nhet - 1) * nhet) * part_prob
            elif nhet == 0 and ndouble > 0:
                error_not_call_ = error_not_call[2] * error_not_call[3] ** (ndouble - 1) * ndouble * part_prob
            elif nhet > 0 and ndouble > 0:
                error_not_call_ = error_not_call[0] * error_not_call[1] ** (nhet - 1) * nhet * error_not_call[3] ** ndouble + error_not_call[2] * error_not_call[3] ** (ndouble - 1) * ndouble * error_not_call[2] ** nhet * part_prob
            
            if nhet > 0:
                # Calculate the error probability for upwards distortion for singletons 
                if allele_count == 1:
                    allele_prob_called_1_or_0 = [1 - x for x in probability_not_call_probs[0]] # The probability of calling a position is derived by subtracting the probability of not calling it from 1.
                    allele_prob_called_1_or_0 = [x * y for x, y in zip(allele_prob_called_1_or_0, coverage_distribution)] # This will be conditional to the coverage distribution
                    allele_prob_called_1_or_0 = sum(allele_prob_called_1_or_0)
                    allele_prob_called_1_or_0 = [x / allele_prob_called_1_or_0 for x in allele_prob_called_1_or_0]
                    het_error_prob_1_or_0 = [prob * allele_prob for prob, allele_prob in zip(probability_double_alt[0], allele_prob_called_1_or_0)]
                    het_error_prob_up = sum(het_error_prob_1_or_0)
                # Calculate the error probability for upwards for doubletons, tripletons, so on.
                else:
                    allele_prob_called_0 = [1 - x for x in probability_not_call_probs[2]]
                    allele_prob_called_0 = [x * y for x, y in zip(allele_prob_called_0, coverage_distribution)]
                    allele_prob_called_0 = sum(allele_prob_called_0)
                    allele_prob_called_0 = [x / allele_prob_called_0 for x in allele_prob_called_0]
                    het_error_prob_0 = [prob * allele_prob for prob, allele_prob in zip(probability_double_alt[2], allele_prob_called_0)]
                    het_error_prob_down = sum(het_error_prob_0)

                # The probability of downward distortion is equivalent to the probability of losing an individual, which means observing a genotype of [0, 0].
                het_error_prob_down = error_not_call[1]
                
                probability_call = 1 - error_not_call_ 
                all_weights[allele_count, 0] += error_not_call_

                for ndown in range(nhet+1):
                    for ncorrect in range(nhet-ndown + 1):
                        nup = nhet - ndown - ncorrect
                        pr = het_error_prob_down**ndown * het_error_prob_up**nup * (1-(het_error_prob_down + het_error_prob_up))**ncorrect
                        ncomb = np.exp(dadi.Numerics.multinomln([ndown, ncorrect, nup]))
                        net_change = nup - ndown
                        if (allele_count+net_change) > 0:
                            all_weights[allele_count, allele_count+net_change] += part_prob * ncomb * pr * probability_call
            else:
                all_weights[allele_count, allele_count] += probability_call * part_prob

    return all_weights

def precalc(n, vcf_file):
    all_partitions, all_part_probs = create_partitions_single(n)
    weights = sfs_redist(vcf_file, all_partitions, all_part_probs, n+1)
    
    for i in range(1, n):
        weights[i][1:] = weights[i][1:]/sum(weights[i][1:]) * (1 - weights[i][0])

    return weights

def coverage_distortion(sfs, sfs_weights):

    coverage_distortion_ = np.dot(sfs, sfs_weights)

    return coverage_distortion_



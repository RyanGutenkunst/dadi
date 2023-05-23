import dadi
import numpy as np
import pandas as pd
from io import StringIO as SIO

# Read a file vcf and convert it to a pandas dataframe
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

# Generate a data frame containing the number of reads for all individuals at each position
def extract_coverage(vcf_table):
    ind_cov = vcf_table.split(':')[1]
    
    if len(ind_cov) != 1:
        ind_cov = list(map(int, ind_cov.split(',')))
        coverage = sum(ind_cov) if 0 in ind_cov else pd.NA
    else:
        coverage = pd.NA
    
    return coverage

# Calculate the expected amount of error using the vcf file
def calc_error(vcf_file, nsamples):
    vcf = read_vcf(vcf_file)
    first_sample_index = len(vcf.columns) - nsamples
    ind_coverage = vcf.iloc[:, first_sample_index:].applymap(extract_coverage).stack().value_counts(normalize=True)
    ind_coverage_dict = ind_coverage.to_dict()
    error_prob = np.sum(0.5 ** np.asarray(list(ind_coverage_dict.keys())) * np.array(list(ind_coverage_dict.values())))

    return error_prob

def create_partitions_single(n):
    allele_counts = np.arange(n+1)
    
    all_partitions = [dadi.Numerics.cached_part(ac, n/2) for ac in allele_counts]
    
    partition_probabilities = np.array([
        [np.exp(dadi.Numerics.multinomln([part.count(0), part.count(1), part.count(2)])) for part in parts]
        for parts in all_partitions
    ])
    
    partition_probabilitie_sum = np.array([np.sum(np.array(elem)) if len(elem) > 1 else np.array(elem) for elem in partition_probabilities])
    partition_probabilities /= partition_probabilitie_sum
    
    return all_partitions, partition_probabilities

def sfs_redist(het_error_prob, all_partitions, all_part_probs, n):
    # Entry (i,j) represents the flow from allele count i to allele count j
    all_weights = np.zeros((n+1,n+1))

    for allele_count in range(n+1):
        for part, part_prob in zip(all_partitions[allele_count], all_part_probs[allele_count]):
            nhet = part.count(1)
            # Each heterozygote can be called to reduce, maintain, or increase the observed allele count
            for ndown in range(nhet+1):
                for ncorrect in range(nhet-ndown + 1):
                    # XXX: There might be a faster way to do this if we thought hard about how to
                    #      enumerate the potential changes within each partition
                    nup = nhet - ndown - ncorrect
                    # XXX: To match Emanuel's code, need to double het_error_prob, which seems wrong
                    #pr = (het_error_prob/2)**(nup + ndown) * (1-het_error_prob)**ncorrect
                    # Emanuel: No need to double het_error_prob anymore
                    pr = (het_error_prob)**(ndown) * (het_error_prob)**(nup) * (1-(2 * het_error_prob))**ncorrect
                    ncomb = np.exp(dadi.Numerics.multinomln([ndown, ncorrect, nup]))
                    # Once I have this working, I only need to calculate probabilities if there is a net change
                    net_change = nup - ndown
                    all_weights[allele_count, allele_count+net_change] += part_prob * ncomb * pr
    return all_weights

def precalc(n, vcf_file, nsamples):
    error_prob = calc_error(vcf_file, nsamples)
    all_partitions, all_part_probs = create_partitions_single(n)
    weights = sfs_redist(error_prob, all_partitions, all_part_probs, n)
    
    weights_ = weights.copy()
    weights_[1] *= 1 - (2 * error_prob) # There is an excess of singletons. Trying to correct it.

    return weights_

def coverage_distortion(sfs, sfs_weights):      
    coverage_distortion =  np.dot(sfs, sfs_weights) 

    return coverage_distortion

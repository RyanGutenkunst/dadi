import dadi
import numpy as np
import pandas as pd
from io import StringIO as SIO
import warnings
import random

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
def extract_coverage(vcf_table, remove_het=True):
    ind_cov = vcf_table.split(':')[1]
    
    if len(ind_cov) > 1:
        ind_cov = list(map(int, ind_cov.split(',')))
        coverage = sum(ind_cov) if remove_het or 0 in ind_cov else pd.NA
    else:
        coverage = pd.NA
    
    return coverage

def genotype_dict(genotypes, ssample, seed):
    genotypes_ = genotypes.dropna()
    
    count = 0
    genotype_sum = 0
    
    condition1 = len(genotypes_) >= int(ssample/2)
    condition2 = sum(genotypes_) != 0
    condition3 = sum(genotypes_) != len(genotypes_) * 2
    
    if condition1 and condition2 and condition3:
        while genotype_sum == 0 or genotype_sum == ssample:
            random.seed(seed + count)

            ind_to_keep = random.sample(list(genotypes_.index), int(ssample/2))

            genotypes_sub = genotypes_[ind_to_keep]
            genotype_sum = sum(genotypes_sub)
            count += 1
            
        genot_dict = sum(genotypes_sub)

    else:
        genot_dict = pd.NA

    return genot_dict

def extract_genotype(genotype_matrix):
        genotype = genotype_matrix.split(':')[0]

        if '.' not in genotype:
            genot = [int(gen) for gen in genotype if gen.isdigit()]
            genot = sum(genot)
        else:
            genot = pd.NA
        
        return genot

def calc_error(vcf_file, nsamples, ssample, seed):
    vcf = read_vcf(vcf_file)
    first_sample_index = len(vcf.columns) - nsamples
    ind_coverage = vcf.iloc[:, first_sample_index:].applymap(lambda x: extract_coverage(x, remove_het=True))
    
    digits = [int(''.join(filter(str.isdigit, l))) for l in ind_coverage.columns]
    sorted_columns = ind_coverage.columns[np.argsort(digits)]
    ind_coverage = ind_coverage[sorted_columns]
    
    genotype_matrix = vcf.iloc[:, first_sample_index:].applymap(extract_genotype)
    genotype_matrix = genotype_matrix[sorted_columns]

    genotype_matrix[ind_coverage.isna()] = pd.NA

    genotype_matrix_summed = genotype_matrix.apply(lambda x: genotype_dict(x, ssample=ssample, seed=seed), axis=1)
    
    error_prob_dict = {}
    for i in range(1, genotype_matrix_summed.max()+1):
        genotype_subset = genotype_matrix_summed[genotype_matrix_summed == i].index
        ind_coverage_ = ind_coverage.loc[genotype_subset].stack().reset_index(drop=True)
        ind_coverage_dict = ind_coverage_.value_counts(normalize=True).to_dict()
        error_prob = np.sum(0.5 ** np.asarray(list(ind_coverage_dict.keys())) * np.array(list(ind_coverage_dict.values())))
        error_prob_dict[i] = error_prob

    return error_prob_dict

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

def sfs_redist(het_error_prob, all_partitions, all_part_probs, n):
    # Entry (i,j) represents the flow from allele count i to allele count j
    all_weights = np.zeros((n+1,n+1))

    het_error_prob[0], het_error_prob[n+1] = 0, 0
    het_error_prob = dict(sorted(het_error_prob.items()))

    for allele_count in range(n+1):
        het_error = list(het_error_prob.values())[allele_count]
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
                    pr = (het_error)**(ndown) * (het_error)**(nup) * (1-(2 * het_error))**ncorrect
                    ncomb = np.exp(dadi.Numerics.multinomln([ndown, ncorrect, nup]))
                    # Once I have this working, I only need to calculate probabilities if there is a net change
                    net_change = nup - ndown
                    all_weights[allele_count, allele_count+net_change] += part_prob * ncomb * pr
    return all_weights

def precalc(n, vcf_file, nsamples, seed):
    error_prob = calc_error(vcf_file, nsamples, n, seed)
    all_partitions, all_part_probs = create_partitions_single(n)
    weights = sfs_redist(error_prob, all_partitions, all_part_probs, n)
    
    weights_ = weights.copy()
    weights_[1] *= 1 - (2 * error_prob[1]) # Trying to correct the excess of singletons.

    return weights_

def coverage_distortion(sfs, sfs_weights):
    coverage_distortion =  np.dot(sfs, sfs_weights) 

    return coverage_distortion

import dadi
import numpy as np
import itertools
import scipy.stats.distributions as ssd

def low_coverage_distortion(sfs, ns, b1, coverages):

    # Create a copy of the SFS and extract the number of segregating sites.
    sfs1 = sfs.copy()
    seg_sites = sfs.S()
    
    # Extract the number of rows and columns in the FS
    # Then create an empty Spectrum representing the FS
    # The new FS will be used to storage the corrected FS
    shape_sfs = sfs1.shape
    corr_sfs = np.zeros(np.prod(shape_sfs)).reshape(*shape_sfs)
    
    # Create an empty list to store the overall error probability
    # of each population. The overall error probability is calculated 
    # based on the mean coverage of each population
    error_pr = []

    for coverage in coverages:

        # Calculating the lower and upper boundaries of the coverage distribution
        # For example, for a mean coverage equal to 5, the lower and upper
        # boundaries are 1 and 8, respectively.
        calling_threshold = [int(coverage/3), int(coverage * 1.5)]
        if calling_threshold[0] < 2:
            calling_threshold[0] = 1
        
        # Generating a probability distribution using a negative binomial distribution with mean equal to coverage of each population
        coverages_ = range(calling_threshold[0], calling_threshold[1]+1)
        poisson_pmf = ssd.nbinom.pmf(coverages_, coverage, 0.5)
        pr = {k: poisson_pmf[i] for i, k in enumerate(coverages_)}
        pr = {x: i/sum(pr.values()) for x, i in pr.items()}
        
        # Store the probability of error of different reads. This probability is
        # rescaled based on the probability of getting this number of reads under 
        # the negative binomial distribution. For instance, For two reads, 
        # there are four sequencing possibilities [0,0], [0,1], [1,0], [1,1].
        # Importantly, 0 and 1 refer to reads from different chromossomes.  
        # So, the probability of error, in this case, is equal to 50% (probability 
        # of getting [0,0] or [1,1]) multiplied by the probability of getting two 
        # reads under the negative binomial (let's say 10%). Therefore, the probability
        # of error for two reads is equal to 5% (or 50% x 10%).
        error_pr_ = {}
        for k, v in pr.items():
            reads_permutations = set(itertools.combinations_with_replacement([0,1], k))
            permutation_list = []
            for reads_permutation in reads_permutations:  
                n = [reads_permutation.count(x) for x in [0,1]]
                permutation_list.append(np.exp(dadi.Numerics.multinomln(n)))
            permutation_prob = [i/sum(permutation_list) for i in permutation_list]
            
            error = []
            for i, x in enumerate(permutation_prob):
                if sum(list(reads_permutations)[i]) == 0 or sum(list(reads_permutations)[i]) == len(list(reads_permutations)[i]):
                    error.append(permutation_prob[i])
            error_pr_[k] = sum(error) * v
        
        error_pr_ = sum(error_pr_.values())
        error_pr.append(error_pr_)
    
    # Create dictionaries to store information about individual entry in the FS and
    # genotype partitions and their relative probabilities. These information is
    # stored individually for each population 
    sfs_bins = {}
    genotype_partitions = {}
    genotype_prob = {}
    
    for i, nsize in enumerate(ns): 
        sfs_bins_ = []
        genotype_partitions_ = []
        genotype_prob_ = []   
        for n_snps in range(0, nsize+1):    
            partitions = dadi.Numerics.cached_part(n_snps, nsize/2)
            probability_list = []
            for partition in partitions:  
                # Calculate the number of unique permutations for a given genotype
                n = [partition.count(x) for x in [0,1,2]]
                probability_list.append(np.exp(dadi.Numerics.multinomln(n)))
            
            probability_list = [i/sum(probability_list) for i in probability_list]
            
            sfs_bins_.append(n_snps)
            genotype_partitions_.append(partitions)
            genotype_prob_.append(probability_list)
        
        sfs_bins['pop' + str(i + 1)] = sfs_bins_
        genotype_partitions['pop' + str(i + 1)] = genotype_partitions_
        genotype_prob['pop' + str(i + 1)] = genotype_prob_
    
    # Extracting all possible combinations between each entry and dimension in the FS.
    # This information will be used to calculate the amount of distortion, evaluating
    # entry by entry.
    sfs_bins_comb = [list(i) for i in itertools.product(*sfs_bins.values())]
    genotype_partitions_comb = [list(i) for i in itertools.product(*genotype_partitions.values())]
    genotype_prob_comb = [list(i) for i in itertools.product(*genotype_prob.values())]
    
    # List to save masked entries
    masked = []

    # Loop over all entry combinations. For a 2D FS: [0,0], [0,1] ... [n1,n2].
    # In each entry, the algorithm will distort the FS based on the overall
    # expected error
    # corr_pr: probability of calling an SNP correctly
    # error_left: probability of underestimating the number of SNPs.
    # error_rigth: probability of overestimating the number of SNPs.
    # b1: free parameter, there is a asymmetrical distortion in singletons

    for bin_index, sfs_bin in enumerate(sfs_bins_comb):
        corr_pr = []
        error_left = []
        error_rigth = []
        for bi, sfs_bin_ in enumerate(sfs_bin):
            if sfs_bin_ == 1:
                error_left.append(error_pr[bi] * 2 * b1)
                error_rigth.append(error_pr[bi] * 2 * (1 - b1))
            else:
                error_left.append(error_pr[bi])
                error_rigth.append(error_pr[bi])
            
            corr_pr.append(1 - error_left[bi] - error_rigth[bi])
        
        # The Combination among all partitions
        genotype_partition = genotype_partitions_comb[bin_index]
        genotype_partition = [i for i in itertools.product(*genotype_partition)]
        
        # The Combination among all partition probabilities
        partition_pr = genotype_prob_comb[bin_index]
        partition_pr = [i for i in itertools.product(*partition_pr)]
        
        # If the entry is masked, then we will include it in the "masked" dictionary.
        # Later, we are going to use this dictionary to mask those entries
        sfs_value = sfs1[tuple(sfs_bin)]
        if str(sfs_value) == '--':
            sfs_value = 0
            masked.append(sfs_bin)
        
        # Loop over each genotype partition combination
        for partition_index, partitions in enumerate(genotype_partition):
                partitions = list(partitions)
                partition_probability = np.prod(partition_pr[partition_index])
                
                # Calculating the number of heterozygous
                number_01 = [i.count(1) for i in partitions]
                # Calculating the number of homorozygous
                number_11 = [i.count(2)*2 for i in partitions]

                
                # Calculating possible permutations among heterozygous sites. A site can be correctly or incorrectly called (i.e., 01 or 00 and 11, respectively)
                heterozygous_permutations = []
                heterozygous_probabilities = []
                for i, partition in enumerate(partitions):
                    heterozygous_permutations_ = set(itertools.combinations_with_replacement([0,1,2], number_01[i])) if number_01[i] > 0 else [-1]
                    if len(heterozygous_permutations_) > 1:
                        heterozygous_permutations_ = [list(i) for i in heterozygous_permutations_]
                        permutation_list = []
                        for genotypes_perm in heterozygous_permutations_:  
                            n = [genotypes_perm.count(x) for x in [0,1,2]]
                            permutation_list.append(np.exp(dadi.Numerics.multinomln(n)))
                    else:
                        permutation_list = [1.0]
                    
                    heterozygous_permutations.append(heterozygous_permutations_)
                    heterozygous_probabilities.append(permutation_list)
                
                # Calculating all possible permutations among heterozygous. For
                # example, for a 2D FS, let's assume that populations 1 and 2 have one
                # heterozygous. There are nine possible combinations: 1)[0,0]; 2)[0,1];
                # 3)[0,2]; 4)[1,0]; 5)[1,1]; 6)[1,2]; 7)[2,0]; 8)[2,1]; 9)[2,2]. 
                # Legend = 1 --> 00; 1 --> 01; 2 --> 11 
                
                heterozygous_permutations_comb = [i for i in itertools.product(*heterozygous_permutations)]
                heterozygous_probabilities_comb = [i for i in itertools.product(*heterozygous_probabilities)]
                
                # Last step is to calculate the joint probility. The Fonseca and
                # Gutenkunst model only cares about heterozygous sites since only
                # them can cause a distortion in the FS. The model uses the overall
                # probability of error of each population jointly with all possible
                # permutations among heterozygous (i.e., a heterozygous site can be
                # inferred as 0, 1, or 2) under a given partition to calculate the
                # probability of distortion. Importantly, if a combination causes a
                # distortion in the FS, its associated probability is redistributed to
                # the expected FS. Legend: 0 --> a site with two ancestral alleles;
                # 2 --> a site with two alternative alleles; 1 --> heterozygous site. 
                for ii, permutation in enumerate(heterozygous_permutations_comb):
                    permutation = list(permutation)
                    het_perm = []
                    homo_perm_00 = []
                    homo_perm_11 = []
                    position = []
                    perm_list = np.prod(list(heterozygous_probabilities_comb[ii]))
                    for iii, perm in enumerate(permutation):
                        if perm != -1:
                            het_perm.append(perm.count(1))
                            homo_perm_00.append(perm.count(0))
                            homo_perm_11.append(perm.count(2))
                            position.append(number_11[iii] + sum(perm))
                        else:
                            het_perm.append(0)
                            homo_perm_00.append(0)
                            homo_perm_11.append(0)
                            position.append(sfs_bin[iii])
                        
                    perm_list = np.prod(list(heterozygous_probabilities_comb[ii]))
                    
                    corr = np.prod([corr_pr[k] ** het_perm[k] for k, v in enumerate(homo_perm_00)])
                    error_00 = np.prod([error_left[k] ** homo_perm_00[k] for k, v in enumerate(homo_perm_00)])
                    error_11 = np.prod([error_rigth[k] ** homo_perm_11[k] for k, v in enumerate(homo_perm_11)])
                    
                    corr_sfs[tuple(position)] += corr * error_00 * error_11 * perm_list * sfs_value * partition_probability
    

    # Recreate a FS compatible with dadi, including original masked sites.
    if len(ns) == 1:
        for index, i in enumerate(corr_sfs):
            if [index] not in masked:
                sfs1[index] = i
    else:
        for i, entry in enumerate(corr_sfs):
            for ii, entry2 in enumerate(entry):
                if [i, ii] not in masked:
                    sfs1[i][ii] = entry2
    
    sfs1 = sfs1/sum(sfs1.compressed()) * seg_sites
    
    return(sfs1)



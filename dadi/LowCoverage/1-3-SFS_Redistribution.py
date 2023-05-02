import dadi
import itertools
import numpy as np

def sfs_redistribution(error_pr, all_partitions, partition_threshold):
    
    sfs_bins_comb = all_partitions[0]
    genotype_partitions_comb = all_partitions[1]
    genotype_prob_comb = all_partitions[2]

    all_calculations = {}
    all_calculations = {str(k):str(0) for k in sfs_bins_comb}

    for bin_index, sfs_bin in enumerate(sfs_bins_comb):
        corr_pr = []
        error_pr1 = []
        error_pr2 = []
        for bi, sfs_bin_ in enumerate(sfs_bin):            
            error_pr1.append(error_pr[bi])
            error_pr2.append(error_pr[bi])                    
            corr_pr.append(1 - error_pr1[bi] - error_pr2[bi])
        
            genotype_partition = genotype_partitions_comb[bin_index]
            genotype_partition = [i for i in itertools.product(*genotype_partition)]
            
            partition_pr = genotype_prob_comb[bin_index]
            partition_pr = [i for i in itertools.product(*partition_pr)]
            
            for partition_index, partitions in enumerate(genotype_partition):
                    partitions = list(partitions)
                    partition_probability = np.prod(partition_pr[partition_index])
                    
                    number_01 = [i.count(1) for i in partitions]
                    number_11 = [i.count(2)*2 for i in partitions]
                    
                    genotypes_permutations = []
                    genotypes_probabilities = []
                    for i, partition in enumerate(partitions):
                        genotypes_permutations_ = set(itertools.combinations_with_replacement([0,1,2], number_01[i])) if number_01[i] > 0 else [-1]
                        if len(genotypes_permutations_) > 1:
                            genotypes_permutations_ = [list(i) for i in genotypes_permutations_]
                            permutation_list = []
                            for genotypes_perm in genotypes_permutations_:  
                                n = [genotypes_perm.count(x) for x in [0,1,2]]
                                permutation_list.append(np.exp(dadi.Numerics.multinomln(n)))
                        else:
                            permutation_list = [1.0]
                        
                        genotypes_permutations.append(genotypes_permutations_)
                        genotypes_probabilities.append(permutation_list)
                    
                    genotypes_permutations_comb = [i for i in itertools.product(*genotypes_permutations)]
                    genotypes_probabilities_comb = [i for i in itertools.product(*genotypes_probabilities)]
                    
                    for ii, permutation in enumerate(genotypes_permutations_comb):
                        permutation = list(permutation)
                        het_perm = []
                        homo_perm_00 = []
                        homo_perm_11 = []
                        position = []
                        perm_list = np.prod(list(genotypes_probabilities_comb[ii]))
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
                            
                        perm_list = np.prod(list(genotypes_probabilities_comb[ii]))
                        
                        corr = np.prod([corr_pr[k] ** het_perm[k] for k, v in enumerate(homo_perm_00)])
                        
                        error_00 = np.prod([error_pr1[k] ** homo_perm_00[k] for k, v in enumerate(homo_perm_00)])
                        error_11 = np.prod([error_pr2[k] ** homo_perm_11[k] for k, v in enumerate(homo_perm_11)])
                        
                        rd = corr * error_00 * error_11 * perm_list * partition_probability
                        if rd > partition_threshold:
                            if sfs_bin_ == 1 and position[0] != 1:
                                if position[0] == 0:
                                    calculation = 'sfsc[tuple(' + str(sfs_bin) + ')]*2*b1*' + str(rd)
                                else:
                                    calculation = 'sfsc[tuple(' + str(sfs_bin) + ')]*2*(1-b1)*' + str(rd)
                            else:
                                calculation = 'sfsc[tuple(' + str(sfs_bin) + ')]*' + str(rd)
                            
                            all_calculations[str(position)] = '+'.join([all_calculations[str(position)],calculation])
                            
    return(all_calculations)




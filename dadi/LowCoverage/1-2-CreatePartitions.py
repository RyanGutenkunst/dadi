import numpy as np    
import dadi
import itertools

def create_partitions(ns):
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
            
            i_prob = [i for i, prob in enumerate(probability_list)]
            
            if len(i_prob) > 0:
                sfs_bins_.append(n_snps)
                
                partitions_ = [prob for i, prob in enumerate(partitions) if i in i_prob]
                genotype_partitions_.append(partitions_)
                
                probability_list = [partition for i, partition in enumerate(probability_list) if i in i_prob]
                probability_list = list(probability_list/sum(probability_list))
                genotype_prob_.append(probability_list)
                
        sfs_bins['pop' + str(i + 1)] = sfs_bins_
        genotype_partitions['pop' + str(i + 1)] = genotype_partitions_
        genotype_prob['pop' + str(i + 1)] = genotype_prob_
    
    sfs_bins_comb = [list(i) for i in itertools.product(*sfs_bins.values())]
    genotype_partitions_comb = [list(i) for i in itertools.product(*genotype_partitions.values())]
    genotype_prob_comb = [list(i) for i in itertools.product(*genotype_prob.values())]

    partitions = [sfs_bins_comb, genotype_partitions_comb, genotype_prob_comb]
    
    return partitions
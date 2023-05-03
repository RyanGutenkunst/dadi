import numpy as np    
import scipy.stats.distributions as ssd
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

def calc_error(coverages):
    error_pr = []
    for coverage in coverages:
        calling_threshold = [int(coverage/3), int(coverage + coverage * 0.5)]
        if calling_threshold[0] < 2:
            calling_threshold[0] = 1
        
        # Generate reads using a negative binomial distribution with mean equal to coverage
        coverages_ = range(calling_threshold[0], calling_threshold[1]+1)
        poisson_pmf = ssd.nbinom.pmf(coverages_, coverage, 0.5)
        pr = {k: poisson_pmf[i] for i, k in enumerate(coverages_)}
        pr = {x: i/sum(pr.values()) for x, i in pr.items()}
        
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

    return(error_pr)

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

def sfs_redistribution_dict(ns, coverages, partition_threshold=0):

    error_pr = calc_error(coverages)
    all_partitions = create_partitions(ns)
    sfs_redist = sfs_redistribution(error_pr, all_partitions, partition_threshold)
    
    return(sfs_redist)

def coverage_distortion(sfs, sfs_redistribution_dict, b1):
    sfsc = sfs.copy()
    
    scope = {'sfsc': sfsc, 'b1':b1}
    sfs_ = [eval(x, scope) for x in sfs_redistribution_dict.values()]    
    
    if len(sfs.sample_sizes) == 1:
        for i, entry in enumerate(sfs_):
            if entry != 'masked':
                sfsc[i] = entry/np.ma.sum(sfs_)
    else:
        for i, entry in enumerate(sfs_):
            for ii, entry2 in enumerate(entry):
                if [i, ii] not in 'masked':
                    sfsc[i][ii] = entry2/np.ma.sum(sfs_)
    
    sfsc = sfsc * sfs.S()
    
    return(sfsc)

if __name__ == "__main__":
    sfs = dadi.Spectrum(np.random.uniform(size=5))
    sfs_redis = sfs_redistribution_dict(ns=sfs.sample_sizes, coverages=[5], partition_threshold=0)
    model = coverage_distortion(sfs=sfs, sfs_redistribution_dict=sfs_redis, b1=0.9)

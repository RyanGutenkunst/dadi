import dadi
import scipy.stats.distributions as ssd
import itertools
import numpy as np

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


import numpy as np

def sfs_redistribution_dict(ns, coverages, partition_threshold=0):

    error_pr = calc_error(coverages)
    all_partitions = create_partitions(ns)
    sfs_redist = sfs_redistribution(error_pr, all_partitions, partition_threshold)
    
    return(sfs_redist)

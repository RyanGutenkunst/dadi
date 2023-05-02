import numpy as np

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

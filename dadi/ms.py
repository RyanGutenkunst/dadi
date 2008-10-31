import numpy

def ms_simple(theta, ns, core, iter, recomb=None, rsites=None):
    ms_command = "ms %(total_chrom)i %(iter)i -t %(theta)f -I %(numpops)i "\
            "%(sample_sizes)s %(core)s"
    if recomb:
        ms_command = ms_command + " -r %(recomb)f %(rsites)i"
        if not rsites:
            rsites = theta*10
    sub_dict = {'total_chrom': numpy.sum(ns), 'iter': iter, 'theta': theta,
                'numpops': len(ns), 'sample_sizes': ' '.join(map(str, ns)),
                'core': core, 'recomb': recomb, 'rsites': rsites}

    return ms_command % sub_dict

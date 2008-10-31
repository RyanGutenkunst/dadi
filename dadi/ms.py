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

def ms_simulate(params, ns, core, scanned_length_file, tbs_filename, 
                recomb_factor=0.5):
    theta = params[0]
    sl = file(scanned_length_file, 'r').readlines()
    iter = len(sl)

    ms_command = "ms %(total_chrom)i %(iter)i -t tbs %(core)s "\
            "-r tbs tbs < %(tbs_filename)s"
    sub_dict = {'total_chrom': numpy.sum(ns), 'iter': iter, 
                'core': core, 'tbs_filename': tbs_filename}

    sl = file(scanned_length_file, 'r').readlines()
    total_scanned = numpy.sum([int(s.split()[1]) for s in sl])
    tbs_out = []
    for scanned in sl:
        scanned = int(scanned.split()[1])
        # Note factor of 2 here that makes per site recombination rate = 1/2 of
        # the per-site mutation rate.
        tbs_out.append('%f %f %i' % (theta*scanned/total_scanned, 
                                     theta*scanned/total_scanned*recomb_factor,
                                     scanned))
    tbs_file = file(tbs_filename, 'w')
    tbs_file.write('\n'.join(tbs_out))
    tbs_file.close()

    return ms_command % sub_dict

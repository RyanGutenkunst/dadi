"""
Miscellaneous utility functions. Including ms simulation.
"""

import collections,os,sys,time

import numpy
import scipy.linalg

# Nucleotide order assumed in Q matrices.
code = 'CGTA'

#: Storage for times at which each stream was flushed.
__times_last_flushed = {}
def delayed_flush(stream=sys.stdout, delay=1):
    """
    Flush a stream, ensuring that it is only flushed every 'delay' *minutes*.
    Note that upon the first call to this method, the stream is not flushed.

    stream: The stream to flush. For this to work with simple 'print'
            statements, the stream should be sys.stdout.
    delay: Minimum time *in minutes* between flushes.

    This function is useful to prevent I/O overload on the cluster.
    """
    global __times_last_flushed

    curr_time = time.time()
    # If this is the first time this method has been called with this stream,
    # we need to fill in the times_last_flushed dict. setdefault will do this
    # without overwriting any entry that may be there already.
    if stream not in __times_last_flushed:
        __times_last_flushed[stream] = curr_time
    last_flushed = __times_last_flushed[stream]

    # Note that time.time() returns values in seconds, hence the factor of 60.
    if (curr_time - last_flushed) >= delay*60:
        stream.flush()
        __times_last_flushed[stream] = curr_time

def ensure_1arg_func(var):
    """
    Ensure that var is actually a one-argument function.

    This is primarily used to convert arguments that are constants into
    trivial functions of time for use in integrations where parameters are
    allowed to change over time.
    """
    if numpy.isscalar(var):
        # If a constant was passed in, use lambda to make it a nice
        #  simple function.
        var_f = lambda t: var
    else:
        var_f = var
    if not callable(var_f):
        raise ValueError('Argument is not a constant or a function.')
    try:
        var_f(0.0)
    except TypeError:
        raise ValueError('Argument is not a constant or a one-argument '
                         'function.')
    return var_f

def ms_command(theta, ns, core, iter, recomb=0, rsites=None, seeds=None):
    """
    Generate ms command for simulation from core.

    theta: Assumed theta
    ns: Sample sizes
    core: Core of ms command that specifies demography.
    iter: Iterations to run ms
    recomb: Assumed recombination rate
    rsites: Sites for recombination. If None, default is 10*theta.
    seeds: Seeds for random number generator. If None, ms default is used.
           Otherwise, three integers should be passed. Example: (132, 435, 123)
    """
    if len(ns) > 1:
        ms_command = "ms %(total_chrom)i %(iter)i -t %(theta)f -I %(numpops)i "\
                "%(sample_sizes)s %(core)s"
    else:
        ms_command = "ms %(total_chrom)i %(iter)i -t %(theta)f  %(core)s"

    if recomb:
        ms_command = ms_command + " -r %(recomb)f %(rsites)i"
        if not rsites:
            rsites = theta*10
    sub_dict = {'total_chrom': numpy.sum(ns), 'iter': iter, 'theta': theta,
                'numpops': len(ns), 'sample_sizes': ' '.join(map(str, ns)),
                'core': core, 'recomb': recomb, 'rsites': rsites}

    ms_command = ms_command % sub_dict

    if seeds is not None:
        seed_command = " -seeds %i %i %i" % (seeds[0], seeds[1], seeds[2])
        ms_command = ms_command + seed_command
    
    return ms_command

def perturb_params(params, fold=1, lower_bound=None, upper_bound=None):
    """
    Generate a perturbed set of parameters.

    Each element of params is radomly perturbed <fold> factors of 2 up or down.
    fold: Number of factors of 2 to perturb by
    lower_bound: If not None, the resulting parameter set is adjusted to have 
                 all value greater than lower_bound.
    upper_bound: If not None, the resulting parameter set is adjusted to have 
                 all value less than upper_bound.
    """
    pnew = params * 2**(fold * (2*numpy.random.random(len(params))-1))
    if lower_bound is not None:
        for ii,bound in enumerate(lower_bound):
            if bound is None:
                lower_bound[ii] = -numpy.inf
        pnew = numpy.maximum(pnew, 1.01*numpy.asarray(lower_bound))
    if upper_bound is not None:
        for ii,bound in enumerate(upper_bound):
            if bound is None:
                upper_bound[ii] = numpy.inf
        pnew = numpy.minimum(pnew, 0.99*numpy.asarray(upper_bound))
    return pnew

def make_fux_table(fid, ts, Q, tri_freq):
    """
    Make file of 1-fux for use in ancestral misidentification correction.

    fid: Filename to output to.
    ts: Expected number of substitutions per site between ingroup and outgroup.
    Q: Trinucleotide transition rate matrix. This should be a 64x64 matrix, in
       which entries are ordered using the code CGTA -> 0,1,2,3. For example, 
       ACT -> 3*16+0*4+2*1=50. The transition rate from ACT to AGT is then 
       entry 50,54.
    tri_freq: Dictionary in which each entry maps a trinucleotide to its 
              ancestral frequency. e.g. {'AAA': 0.01, 'AAC':0.012...}
              Note that should be the frequency in the entire region scanned
              for variation, not just sites where there are SNPs.
    """
    # Ensure that the *columns* of Q sum to zero.
    # That is the correct condition when Q_{i,j} is the rate from i to j.
    # This indicates a typo in Hernandez, Williamson, and Bustamante.
    for ii in range(Q.shape[1]):
        s = Q[:,ii].sum() - Q[ii,ii]
        Q[ii,ii] = -s

    eQhalf = scipy.linalg.matfuncs.expm(Q * ts/2.)
    if not hasattr(fid, 'write'):
        newfile = True
        fid = open(fid, 'w')

    outlines = []
    for first_ii,first in enumerate(code):
        for x_ii,x in enumerate(code):
            for third_ii,third in enumerate(code):
                # This the index into Q and eQ
                xind = 16*first_ii+4*x_ii+1*third_ii
                for u_ii,u in enumerate(code):
                    # This the index into Q and eQ
                    uind = 16*first_ii+4*u_ii+1*third_ii

                    ## Note that the Q terms factor out in our final
                    ## calculation, because for both PMuUu and PMuUx the final
                    ## factor in Eqn 2 is P(S={u,x}|M=u).
                    #Qux = Q[uind,xind]
                    #denomu = Q[uind].sum() - Q[uind,uind]

                    PMuUu, PMuUx = 0,0
                    # Equation 2 in HWB. We have to generalize slightly to
                    # calculate PMuUx. In calculate PMuUx, we're summing over
                    # alpha the probability that the MRCA was alpha, and it
                    # substituted to x on the outgroup branch, and it
                    # substituted to u on the ingroup branch, and it mutated to
                    # x in the ingroup (conditional on it having mutated in the
                    # ingroup). Note that the mutation to x condition cancels
                    # in fux, so we don't bother to calculate it.
                    for aa,alpha in enumerate(code):
                        aind = 16*first_ii+4*aa+1*third_ii

                        pia = tri_freq[first+alpha+third]
                        Pau = eQhalf[aind,uind]
                        Pax = eQhalf[aind,xind]

                        PMuUu += pia * Pau*Pau
                        PMuUx += pia * Pau*Pax

                    # This is 1-fux. For a given SNP with actual ancestral state
                    # u and derived allele x, this is 1 minus the probability
                    # that the outgroup will have u.
                    # Eqn 3 in HWB.
                    res = 1 - PMuUu/(PMuUu + PMuUx)
                    # These aren't SNPs, so we can arbitrarily set them to 0
                    if u == x:
                        res = 0
                    
                    outlines.append('%c%c%c %c %.6f' % (first,x,third,u,res))

    fid.write(os.linesep.join(outlines))
    if newfile:
        fid.close()

def zero_diag(Q):
    """
    Copy of Q altered such that diagonal entries are all 0.
    """
    Q_nodiag = Q.copy()
    for ii in range(Q.shape[0]):
        Q_nodiag[ii,ii] = 0
    return Q_nodiag

def tri_freq_dict_to_array(tri_freq_dict):
    """
    Convert dictionary of trinucleotide frequencies to array in correct order.
    """
    tripi = numpy.zeros(64)
    for ii,left in enumerate(code):
        for jj,center in enumerate(code):
            for kk,right in enumerate(code):
                row = ii*16 + jj*4 + kk
                tripi[row] = tri_freq_dict[left+center+right]
    return tripi

def total_instantaneous_rate(Q, pi):
    """
    Total instantaneous substitution rate.
    """
    Qzero = zero_diag(Q)
    return numpy.dot(pi, Qzero).sum()

def make_data_dict(filename):
    """
    Parse SNP file and store info in a properly formatted dictionary.

    filename: Name of file to work with.

    This is specific to the particular data format described on the wiki. 
    Modification for other formats should be straightforward.

    The file can be zipped (extension .zip) or gzipped (extension .gz). If 
    zipped, there must be only a single file in the zip archive.
    """
    if os.path.splitext(filename)[1] == '.gz':
        import gzip
        f = gzip.open(filename)
    elif os.path.splitext(filename)[1] == '.zip':
        import zipfile
        archive = zipfile.ZipFile(filename)
        namelist = archive.namelist()
        if len(namelist) != 1:
            raise ValueError('Must be only a single data file in zip '
                             'archive: %s' % filename)
        f = archive.open(namelist[0])
    else:
        f = open(filename)

    # Skip to the header
    while True:
        header = f.readline()
        if not header.startswith('#'):
            break

    allele2_index = header.split().index('Allele2')

    # Pull out our pop ids
    pops = header.split()[3:allele2_index]

    # The empty data dictionary
    data_dict = {}

    # Now walk down the file
    for SNP_ii, line in enumerate(f):
        if line.startswith('#'):
            continue
        # Split the into fields by whitespace
        spl = line.split()

        data_this_snp = {}

        # We convert to upper case to avoid any issues with mixed case between
        # SNPs.
        data_this_snp['context'] = spl[0].upper()
        data_this_snp['outgroup_context'] = spl[1].upper()
        data_this_snp['outgroup_allele'] = spl[1][1].upper()
        data_this_snp['segregating'] = spl[2].upper(),spl[allele2_index].upper()

        calls_dict = {}
        for ii,pop in enumerate(pops):
            calls_dict[pop] = int(spl[3+ii]), int(spl[allele2_index+1+ii])
        data_this_snp['calls'] = calls_dict

        # We name our SNPs using the final columns
        snp_id = '_'.join(spl[allele2_index+1+len(pops):])
        if snp_id == '':
            snp_id = 'SNP_{0}'.format(SNP_ii)

        data_dict[snp_id] = data_this_snp

    return data_dict

def count_data_dict(data_dict, pop_ids):
    """
    Summarize data in data_dict by mapping SNP configurations to counts.

    data_dict: data_dict formatted as in Misc.make_data_dict
    pop_ids: IDs of populations to collect data for.

    Returns a dictionary with keys (successful_calls, derived_calls,
    polarized) mapping to counts of SNPs. Here successful_calls is a tuple
    with the number of good calls per population, derived_calls is a tuple
    of derived calls per pop, and polarized indicates whether that SNP was
    polarized using an ancestral state.
    """
    count_dict = collections.defaultdict(int)
    for snp, snp_info in data_dict.items():
        # Skip SNPs that aren't biallelic.
        if len(snp_info['segregating']) != 2:
            continue

        allele1,allele2 = snp_info['segregating']
        if 'outgroup_allele' in snp_info and snp_info['outgroup_allele'] != '-'\
            and snp_info['outgroup_allele'] in snp_info['segregating']:
            outgroup_allele = snp_info['outgroup_allele']
            this_snp_polarized = True
        else:
            outgroup_allele = allele1
            this_snp_polarized = False

        # Extract the allele calls for each population.
        allele1_calls = [snp_info['calls'][pop][0] for pop in pop_ids]
        allele2_calls = [snp_info['calls'][pop][1] for pop in pop_ids]
        # How many chromosomes did we call successfully in each population?
        successful_calls = [a1+a2 for (a1,a2)
                            in zip(allele1_calls, allele2_calls)]

        # Which allele is derived (different from outgroup)?
        if allele1 == outgroup_allele:
            derived_calls = allele2_calls
        elif allele2 == outgroup_allele:
            derived_calls = allele1_calls

        # Update count_dict
        count_dict[tuple(successful_calls),tuple(derived_calls),
                   this_snp_polarized] += 1
    return count_dict

def dd_from_SLiM_files(fnames, mut_types=None):
    """
    Create a data dictionary from a sequence of SLiM output files.

    It is assumed that each file corresponds to samples from a different
    population. For example, in SLiM:
        p1.outputSample(10, filePath='p1.slimout');
        p2.outputSample(10, filePath='p2.slimout');

    The populations will be named 0,1,2,... corresponding
    to their order in fnames.

    TODO: Add filtering by mutation type.
    """
    # Open all the files
    try:
        fids = [open(_) for _ in fnames]
    except TypeError:
        fids = fnames
    # For each population, we'll first map the mutation ids used in the file to
    # the simulation-level global mutation ids
    mut_dicts = [{} for _ in fids]
    for fid, mut_dict in zip(fids, mut_dicts):
        fid.readline(); fid.readline()
        line = fid.readline()
        while not line.startswith('Genomes:'):
            local_id, global_id, mut_type, _ = line.split(None, 3)
            if mut_types is None or mut_type in mut_types:
                mut_dict[local_id] = global_id
            line = fid.readline()

    # Now for each population we count each mutation
    mut_counts = [collections.defaultdict(int) for _ in fids]
    # We also use this pass to measure the sample size for each pop.
    sample_sizes = [0 for _ in fids]
    for pop_ii, (fid, mut_count) in enumerate(zip(fids, mut_counts)):
        # For each genome
        for line in fid:
            sample_sizes[pop_ii] += 1
            mutations = line.split()[2:]
            for m in mutations:
                mut_count[m] += 1

    # Now collect all mutations from all pops.
    all_muts = set()
    for mut_dict in mut_dicts:
        all_muts.update(mut_dict.values())

    # Create the empty data dictionary
    dd = {}
    for global_id in all_muts:
        dd[global_id] = {'segregating': [0,1],
                         'outgroup_allele': 0,
                         'calls': {}}
        for pop_id, n in enumerate(sample_sizes):
            # We initialize each entry to indicate that the allele is not
            # segregating in the population, since in this case it's
            # not reported in that population's file.
            dd[global_id]['calls'][pop_id] = [n, 0]

    # Now update the alleles that are segregating in each population.
    for pop_ii, (mut_count, mut_dict)\
            in enumerate(zip(mut_counts, mut_dicts)):
        for local_id, count in mut_count.items():
            try:
                # Lookup in mut_dict will fail if mutation isn't of appropriate
                # type to include in this run.
                global_id = mut_dict[local_id]
                dd[global_id]['calls'][pop_ii] = (sample_sizes[pop_ii]-count, count)
            except KeyError:
                pass

    return dd, sample_sizes

def combine_pops(fs, idx=[0,1]):
    """
    Combine the frequency spectra of two populations.
        fs:  Spectrum object (2D or 3D).
        idx: Indices for populations being collapsed. (defaul=[0,1])
    
    The function will always return the combined populations along the
    first axis of the sfs. The resulting spectrum is also returned
    as a numpy array, but can be converted to a Spectrum object
    using the dadi.Spectrum() function.
    """
    ns = fs.sample_sizes
    if len(ns) == 3:
        fs_tmp = numpy.array(fs)
        if idx == [0,1]:
            fs2 = numpy.zeros((ns[0]+ns[1]+1,ns[2]+1))
            for ii in range(ns[0]+1):
                for jj in range(ns[1]+1):
                    for kk in range(ns[2]+1):
                        fs2[ii+jj,kk] += fs_tmp[ii,jj,kk]
        elif idx == [0,2]:
            fs2 = numpy.zeros((ns[0]+ns[2]+1,ns[1]+1))
            for ii in range(ns[0]+1):
                for jj in range(ns[2]+1):
                    for kk in range(ns[1]+1):
                        fs2[ii+kk,jj] += fs_tmp[ii,jj,kk]
        elif idx == [1,2]:
            fs2 = numpy.zeros((ns[1]+ns[2]+1,ns[0]+1))
            for ii in range(ns[1]+1):
                for jj in range(ns[2]+1):
                    for kk in range(ns[0]+1):
                        fs2[jj+kk,ii] += fs_tmp[ii,jj,kk]
        else:
            print("Error: did not recognize population indices: {}".format(idx))
            exit(-1)
    elif len(ns) == 2:
        fs_tmp = numpy.array(fs)
        fs2    = numpy.zeros((ns[0]+ns[1]+1,))
        for ii in range(ns[0]+1):
            for jj in range(ns[1]+1):
                fs2[ii+jj] += fs_tmp[ii,jj]
    else:
        print("Error: could not combine populations.")
        exit(-1)
    return fs2
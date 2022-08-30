"""
Miscellaneous utility functions. Including ms simulation.
"""

import collections,os,sys,time, warnings
def simple_warning(message, category, filename, lineno, file=None, line=None):
    return '%s' % message
warnings.formatwarning = simple_warning

import numpy
import scipy.linalg

from .Spectrum_mod import Spectrum
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
        var_f_tmp = lambda t: var
    else:
        var_f_tmp = var
    # Wrapping arguments in float64 eases working with CUDA
    var_f = lambda t: numpy.float64(var_f_tmp(t))
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
    pnew = params * 2**(fold * (2*numpy.random.uniform(size=len(params))-1))
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
    for snp_info in data_dict.values():
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

def dd_from_SLiM_files(fnames, mut_types=None, chr='SLIM_'):
    """
    Create a data dictionary from a sequence of SLiM output files.

    Returns the data dictionary and a sequence of sample sizes.

    It is assumed that each file corresponds to samples from a different
    population. For example, in SLiM:
        p1.outputSample(10, filePath='p1.slimout');
        p2.outputSample(10, filePath='p2.slimout');

    The populations will be named 0,1,2,... corresponding
    to their order in fnames.

    fnames: Filenames to parse
    mut_types: Sequence of mutation types to include. If None, all mutations
               will be included.
    chr: Prefix to be used for indicating mutation locations.

    The keys in the resulting dictionary are of the form
    <chr>_<position>.<globalid>
    """
    # Open all the files
    try:
        fids = [open(_) for _ in fnames]
    except TypeError:
        fids = fnames
    # For each population, we'll first map the mutation ids used in the file to
    # the simulation-level global mutation ids
    mut_dicts = [{} for _ in fids]
    loc_dict = {}
    for fid, mut_dict in zip(fids, mut_dicts):
        fid.readline(); fid.readline()
        line = fid.readline()
        while not line.startswith('Genomes:'):
            local_id, global_id, mut_type, location, _ = line.split(None, 4)
            if mut_types is None or mut_type in mut_types:
                mut_dict[local_id] = global_id
                loc_dict[global_id] = location
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
        key = 'SLiM_'+loc_dict[global_id] + '.' + global_id
        dd[key] = {'segregating': [0,1],
                         'outgroup_allele': 0,
                         'calls': {}}
        for pop_id, n in enumerate(sample_sizes):
            # We initialize each entry to indicate that the allele is not
            # segregating in the population, since in this case it's
            # not reported in that population's file.
            dd[key]['calls'][pop_id] = [n, 0]

    # Now update the alleles that are segregating in each population.
    for pop_ii, (mut_count, mut_dict)\
            in enumerate(zip(mut_counts, mut_dicts)):
        for local_id, count in mut_count.items():
            try:
                # Lookup in mut_dict will fail if mutation isn't of appropriate
                # type to include in this run.
                global_id = mut_dict[local_id]
                key = 'SLiM_'+loc_dict[global_id] + '.' + global_id
                dd[key]['calls'][pop_ii] = (sample_sizes[pop_ii]-count, count)
            except KeyError:
                pass

    return dd, sample_sizes

def make_data_dict_vcf(vcf_filename, popinfo_filename, subsample=None, filter=True,
                       flanking_info=[None, None]):
    """
    Parse a VCF file containing genomic sequence information, along with a file
    identifying the population of each sample, and store the information in
    a properly formatted dictionary.

    Each file may be zipped (.zip) or gzipped (.gz). If a file is zipped,
    it must be the only file in the archive, and the two files cannot be zipped
    together. Both files must be present for the function to work.

    vcf_filename : Name of VCF file to work with. The function currently works
                   for biallelic SNPs only, so if REF or ALT is anything other
                   than a single base pair (A, C, T, or G), the allele will be
                   skipped. Additionally, genotype information must be present
                   in the FORMAT field GT, and genotype info must be known for
                   every sample, else the SNP will be skipped. If the ancestral
                   allele is known it should be specified in INFO field 'AA'.
                   Otherwise, it will be set to '-'.

    popinfo_filename : Name of file containing the population assignments for
                       each sample in the VCF. If a sample in the VCF file does
                       not have a corresponding entry in this file, it will be
                       skipped. See _get_popinfo for information on how this
                       file must be formatted.

    subsample : Dictionary with population names used in the popinfo_filename
                as keys and the desired sample size (in number of individuals)
                for subsampling as values. E.g., {"pop1": n1, "pop2": n2} for
                two populations.

    filter : If set to True, alleles will be skipped if they have not passed
             all filters (i.e. either 'PASS' or '.' must be present in FILTER
             column.

    flanking_info : Flanking information for the reference and/or ancestral
                    allele can be provided as field(s) in the INFO column. To
                    add this information to the dict, flanking_info should
                    specify the names of the fields that contain this info as a
                    list (e.g. ['RFL', 'AFL'].) If context info is given for
                    only one allele, set the other item in the list to None,
                    (e.g. ['RFL', None]). Information can be provided as a 3
                    base-pair sequence or 2 base-pair sequence, where the first
                    base-pair is the one immediately preceding the SNP, and the
                    last base-pair is the one immediately following the SNP.
    """
    do_subsampling = False
    if subsample is not None:
        do_subsampling = True
        warnings.warn('Note on subsampling: If you will be including inbreeding in your model, '
                      'do not project your data to smaller sample sizes in later steps of your analysis.')

    if os.path.splitext(popinfo_filename)[1] == '.gz':
        import gzip
        popinfo_file = gzip.open(popinfo_filename)
    elif os.path.splitext(popinfo_filename)[1] == '.zip':
        import zipfile
        archive = zipfile.ZipFile(popinfo_filename)
        namelist = archive.namelist()
        if len(namelist) != 1:
            raise ValueError("Must be only a single popinfo file in zip "
                             "archive: {}".format(popinfo_filename))
        popinfo_file = archive.open(namelist[0])
    else:
        popinfo_file = open(popinfo_filename)
    # pop_dict has key, value pairs of "SAMPLE_NAME" : "POP_NAME"
    try:
        popinfo_dict = _get_popinfo(popinfo_file)
    except:
        raise ValueError('Failed in parsing popinfo file.')
    popinfo_file.close()

    # Open VCF file
    if os.path.splitext(vcf_filename)[1] == '.gz':
        import gzip
        vcf_file = gzip.open(vcf_filename)
    elif os.path.splitext(vcf_filename)[1] == '.zip':
        import zipfile
        archive = zipfile.ZipFile(vcf_filename)
        namelist = archive.namelist()
        if len(namelist) != 1:
            raise ValueError("Must be only a single vcf file in zip "
                             "archive: {}".format(vcf_filename))
        vcf_file = archive.open(namelist[0])
    else:
        vcf_file = open(vcf_filename)

    data_dict = {}
    for line in vcf_file:
        # decoding lines for Python 3 - probably a better way to handle this
        try:
            line = line.decode()
        except AttributeError:
            pass
        # Skip metainformation
        if line.startswith('##'):
            continue
        # Read header
        if line.startswith('#'):
            header_cols = line.split()
            # Ensure there is at least one sample
            if len(header_cols) <= 9:
                raise ValueError("No samples in VCF file")
            # Use popinfo_dict to get the order of populations present in VCF
            poplist = [popinfo_dict[sample] if sample in popinfo_dict else None
                       for sample in header_cols[9:]]
            continue

        # Read SNP data
        # Data lines in VCF file are tab-delimited
        # See https://samtools.github.io/hts-specs/VCFv4.2.pdf
        cols = line.split("\t")
        snp_id = '_'.join(cols[:2]) # CHROM_POS
        snp_dict = {}

        # Skip SNP if filter is set to True and it fails a filter test
        if filter and cols[6] != 'PASS' and cols[6] != '.':
            continue

        # Add reference and alternate allele info to dict
        ref, alt = (allele.upper() for allele in cols[3:5])
        if ref not in ['A', 'C', 'G', 'T'] or alt not in ['A', 'C', 'G', 'T']:
            # Skip line if site is not an SNP
            continue
        snp_dict['segregating'] = (ref, alt)
        snp_dict['context'] = '-' + ref + '-'

        # Add ancestral allele information if available
        info = cols[7].split(';')
        for field in info:
            if field.startswith('AA=') or field.startswith('AA_ensembl=') or field.startswith('AA_chimp='):
                outgroup_allele = field.split('=')[1].upper()
                if outgroup_allele not in ['A','C','G','T']:
                    # Skip if ancestral not single base A, C, G, or T
                    outgroup_allele = '-'
                break
        else:
            outgroup_allele = '-'
        snp_dict['outgroup_allele'] = outgroup_allele
        snp_dict['outgroup_context'] = '-' + outgroup_allele + '-'

        # Add flanking info if it is present
        rflank, aflank = flanking_info
        for field in info:
            if rflank and field.startswith(rflank):
                flank = field[len(rflank)+1:].upper()
                if not (len(flank) == 2 or len(flank) == 3):
                    continue
                prevb, nextb = flank[0], flank[-1]
                if prevb not in ['A','C','T','G']:
                    prevb = '-'
                if nextb not in ['A','C','T','G']:
                    nextb = '-'
                snp_dict['context'] = prevb + ref + nextb
                continue
            if aflank and field.startswith(aflank):
                flank = field[len(aflank)+1:].upper()
                if not (len(flank) == 2 or len(flank) == 3):
                    continue
                prevb, nextb = flank[0], flank[-1]
                if prevb not in ['A','C','T','G']:
                    prevb = '-'
                if nextb not in ['A','C','T','G']:
                    nextb = '-'
                snp_dict['outgroup_context'] = prevb + outgroup_allele + nextb

        calls_dict = {}
        subsample_dict = {}
        gtindex = cols[8].split(':').index('GT')
        if do_subsampling:
            # Collect data for all genotyped samples
            for pop, sample in zip(poplist, cols[9:]):
                if pop is None:
                    continue
                gt = sample.split(':')[gtindex]
                if pop not in subsample_dict:
                    subsample_dict[pop] = []
                if '.' not in gt:
                    subsample_dict[pop].append(gt)

            # key-value pairs here are population names
            # and a list of genotypes to subsample from
            for pop, genotypes in subsample_dict.items():
                if pop not in calls_dict:
                    calls_dict[pop] = (0, 0)
                if len(genotypes) < subsample[pop]:
                    # Not enough calls for this SNP
                    break
                # Choose which individuals to use
                idx = numpy.random.choice([i for i in range(0,len(genotypes))], subsample[pop], replace=False)
                for ii in idx:
                    gt = subsample_dict[pop][ii]
                    refcalls, altcalls = calls_dict[pop]
                    refcalls += gt[::2].count('0')
                    altcalls += gt[::2].count('1')
                    calls_dict[pop] = (refcalls, altcalls)
            else:
                # Only runs if we didn't break out of this loop
                snp_dict['calls'] = calls_dict
                data_dict[snp_id] = snp_dict
        else:
            for pop, sample in zip(poplist, cols[9:]):
                if pop is None:
                    continue
                if pop not in calls_dict:
                    calls_dict[pop] = (0,0)
                # Genotype in VCF format 0|1|1|0:...
                gt = sample.split(':')[gtindex]
                #g1, g2 = gt[0], gt[2]
                #if g1 == '.' or g2 == '.':
                #    continue
                    #full_info = False
                    #break

                refcalls, altcalls = calls_dict[pop]
                #refcalls += int(g1 == '0') + int(g2 == '0')
                #altcalls += int(g1 == '1') + int(g2 == '1')

                # Assume biallelic variants
                refcalls += gt[::2].count('0')
                altcalls += gt[::2].count('1')
                calls_dict[pop] = (refcalls, altcalls)
            snp_dict['calls'] = calls_dict
            data_dict[snp_id] = snp_dict

    vcf_file.close()
    return data_dict

def _get_popinfo(popinfo_file):
    """
    Helper function for make_data_dict_vcf. Takes an open file that contains
    information on the population designations of each sample within a VCF file,
    and returns a dictionary containing {"SAMPLE_NAME" : "POP_NAME"} pairs.

    The file should be formatted as a table, with columns delimited by
    whitespace, and rows delimited by new lines. Lines beginning with '#' are
    considered comments and will be ignored. Each sample must appear on its own
    line. If no header information is provided, the first column will be assumed
    to be the SAMPLE_NAME column, while the second column will be assumed to be
    the POP_NAME column. If a header is present, it must be the first
    non-comment line of the file. The column positions of the words "SAMPLE" and
    "POP" (ignoring case) in this header will be used to determine proper
    positions of the SAMPLE_NAME and POP_NAME columns in the table.

    popinfo_file : An open text file of the format described above.
    """
    popinfo_dict = {}
    sample_col = 0
    pop_col = 1
    header = False

    # check for header info
    for line in popinfo_file:
        if line.startswith('#'):
            continue
        cols = [col.lower() for col in line.split()]
        if 'sample' in cols:
            header = True
            sample_col = cols.index('sample')
        if 'pop' in cols:
            header = True
            pop_col = cols.index('pop')
        break

    # read in population information for each sample
    popinfo_file.seek(0)
    for line in popinfo_file:
        if line.startswith('#') or not line.strip():
            continue
        cols = line.split()
        sample = cols[sample_col]
        pop = cols[pop_col]
        # avoid adding header to dict
        if (sample.lower() == 'sample' or pop.lower() == 'pop') and header:
            header = False
            continue
        popinfo_dict[sample] = pop

    return popinfo_dict

def annotate_from_annovar(dd, annovar_file, variant_type):
    """
    Return a data dictionary with only the sites of a requested type of variation based on an ANNOVAR '.exonic_variant_function' output file.

    dd: Data dictionary of sites of a requested type of annotation
    annovar_file: Output file from ANNOVAR with the '.exonic_variant_function' extension
    variant_type: The type of variant you want to make a data dictionary sites to contain
    """
    anno_list = []
    var_fid = open(annovar_file)
    for line in var_fid:
        variant = line.split('\t')[1]
        position = '_'.join(line.split()[4:6])
        if variant_type in variant:
            anno_list.append(position)
    var_fid.close()

    dd_anno = {}
    for key in dd:
        if key in anno_list:
            dd_anno[key] = dd[key]

    return dd_anno

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
                        fs2[ii+jj,kk] += fs_tmp[ii,kk,jj]
        elif idx == [1,2]:
            fs2 = numpy.zeros((ns[1]+ns[2]+1,ns[0]+1))
            for ii in range(ns[1]+1):
                for jj in range(ns[2]+1):
                    for kk in range(ns[0]+1):
                        fs2[ii+jj,kk] += fs_tmp[kk,ii,jj]
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
    fs2 = Spectrum(fs2)
    fs2.extrap_x = fs.extrap_x
    return fs2

def fragment_data_dict(dd, chunk_size):
    """
    Split data dictionary for bootstrapping.

    For bootstrapping, split the data dictionary in subdictionaries
    for each chunk of the genome. The chunk_size is given in
    basepairs.

    This method assumes that keys in the dictionary are
    chromosome_position[.additional_info]
    The [.additional_info] is optional, and can be used to distinguish 
    recurrent mutations at the same site.

    dd: Data dictionary to split
    chunk_size: Size of genomic chunks in basepairs

    Return: List of dictionaries corresponding to each chunk.
    """
    # split dictionary by chromosome name
    ndd = collections.defaultdict(list)
    for k in dd.keys():
        spl = k.split('.')[0].split('_')
        chrname, position = '_'.join(k.split('_')[:-1]), k.split('_')[-1]
        # Track additional_info
        if not '.' in position:
            add_info = None
        else:
            position, add_info = position.split('.',1)
        ndd[chrname].append((int(position), add_info))
            
    # generate chunks with given chunk size
    chunks_dict = collections.defaultdict(list)
    for chrname in ndd.keys():
        positions = sorted(ndd[chrname])
        end = chunk_size
        chunk_index = 0
        chunks_dict[chrname].append([])
        for p, add_info in positions:
            while p > end: 
                # Need a new chunk
                end += chunk_size
                chunk_index += 1
                chunks_dict[chrname].append([])
            chunks_dict[chrname][chunk_index].append((p, add_info))

    # Break data dictionary into dictionaries for each chunk
    new_dds = []
    for chrname, chunks in chunks_dict.items():
        for pos_list in chunks:
            new_dds.append({})
            for pos, add_info in pos_list:
                if not add_info:
                    key = '{0}_{1}'.format(chrname,pos)
                else:
                    key = '{0}_{1}.{2}'.format(chrname,pos,add_info)
                new_dds[-1][key] = dd[key]

    return new_dds

import numpy as np
import random
from .Spectrum_mod import Spectrum
import functools, operator

def bootstraps_from_dd_chunks(fragments, Nboot, pop_ids, projections, mask_corners=True, polarized=True):
    """
    Bootstrap frequency spectra from data dictionary fragments

    fragments: Fragmented data dictionary
    Nboot: Number of bootstrap spectra to generate
    Remaining arguments are as in Spectrum.from_data_dict
    """
    spectra = [Spectrum.from_data_dict(dd, pop_ids, projections, mask_corners, polarized)
               for dd in fragments]

    bootstraps = []
    for ii in range(Nboot):
        chosen = random.choices(spectra, k=len(spectra))
        bootstraps.append(functools.reduce(operator.add, chosen))

    bootstraps = [Spectrum(_, mask_corners=mask_corners, data_folded=not polarized, pop_ids=pop_ids)
                  for _ in bootstraps]
    return bootstraps

def bootstraps_subsample_vcf(vcf_filename, popinfo_filename, subsample, Nboot, chunk_size, pop_ids, filter=True,
                             flanking_info=[None, None], mask_corners=True, polarized=True):
    """
    Bootstrap frequency spectra from subsampling in VCF file

    This method is useful when you have subsampled individuals. You might do this
    because you're modeling inbreeding, or because you want to avoid composite likelihood
    complications (if your data is unlinked).

    Nboot: Number of boostrap spectra to generate
    chunk_size: Size of regions to divide genome into (in basepairs)
    Other arguments are as in make_data_dict_vcf and Spectrum.from_data_dict.
    """
    bootstraps = []
    projections = [subsample[pop]*2 for pop in pop_ids]
    for ii in range(Nboot):
        dd = make_data_dict_vcf(vcf_filename, popinfo_filename, subsample=subsample, filter=filter,
                                flanking_info=flanking_info)
        fragments = fragment_data_dict(dd, chunk_size)
        fs = bootstraps_from_dd_chunks(fragments, 1, pop_ids, projections, mask_corners, polarized)[0]
        bootstraps.append(fs)
    return bootstraps

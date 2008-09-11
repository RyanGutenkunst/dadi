import os
import numpy

def sfs_to_file(sfs, fid, precision=16, comment_lines = []):
    """
    Write a site-frequency spectrum to a file.

    sfs: the SFS array to write out.
    fid: string with file name to write to or an open file object.
    precision: precision with which to write out entries of the SFS. (They or
               formated via %.<p>g, where <p> is the precision.)
    comment lines: list of strings to be used as comment lines in the header of
                   the output file.
    """
    newfile = False
    if not hasattr(fid, 'write'):
        newfile = True
        fid = file(fid, 'w')

    for line in comment_lines:
        fid.write('# ')
        fid.write(line.strip())
        fid.write(os.linesep)

    for elem in sfs.shape:
        fid.write('%i ' % elem)
    fid.write(os.linesep)

    # Masked entries in the sfs will go in as 'nan'
    if numpy.ma.isMaskedArray(sfs):
        sfs = numpy.where(sfs.mask, numpy.nan, sfs.data)
    sfs.tofile(fid, ' ', '%%.%ig' % precision)
    fid.write(os.linesep)

    if newfile:
        fid.close()

def sfs_from_file(fid, mask_corners=True, return_comments=False):
    """
    Read a site-frequency-spectrum from a file.

    fid: string with file name to read from or an open file object.
    mask_corners: If True, return a masked array, in which the 0,0,0 and 
                  -1,-1,-1 entries are masked out. (These entries are 'absent 
                  in all pops' and 'fixed in all pops', respectively.)

    The file format is:
        # Any number of comment lines beginning with a '#'
        Followed by a single line containing N integers giving the dimensions
          of the sfs array. So this line would be '5 5 3' for an SFS that was
          5x5x3.
        Followed by the array elements, all on one line. The order of elements
          is e.g.: sfs[0,0,0] sfs[0,0,1]  sfs[0,0,2]  sfs[0,1,0]  sfs[0,1,1]...
    """
    newfile = False
    if not hasattr(fid, 'read'):
        newfile = True
        fid = file(fid, 'r')

    line = fid.readline()
    comments = []
    while line.startswith('#'):
        comments.append(line[1:].strip())
        line = fid.readline()
    shape = tuple([int(d) for d in line.split()])

    sfs = numpy.fromfile(fid, count=numpy.product(shape), sep=' ')
    sfs = sfs.reshape(*shape)

    if newfile:
        fid.close()

    if mask_corners:
        mask = numpy.ma.make_mask_none(sfs.shape)
        mask.flat[0] = mask.flat[-1] = True
        sfs = numpy.ma.masked_array(sfs, mask=mask)
                                   
    if not return_comments:
        return sfs
    else:
        return sfs,comments

def sfs_from_ms_file(input, average=False, report_sum=False, 
                     return_header=False):
    command = line = input.readline()
    command_terms = line.split()
    
    if command_terms[0].count('ms'):
        runs = int(command_terms[2])
        try:
            pop_flag = command_terms.index('-I')
            num_pops = int(command_terms[pop_flag+1])
            pop_samples = [int(command_terms[pop_flag+ii])
                           for ii in range(2, 2+num_pops)]
        except ValueError:
            num_pops = 1
            pop_samples = [int(command_terms[1])]
    elif command_terms[0].count('sfs_code'):
        runs = int(command_terms[2])
        num_pops = int(command_terms[1])
        # sfs_code default is 6 individuals, and here I'm assuming diploid
        pop_samples = [12] *  num_pops
        if '--sampSize' in command_terms or '-n' in command_terms:
            try:
                pop_flag = command_terms.index('--sampSize')
                pop_flag = command_terms.index('-n')
            except ValueError:
                pass
            pop_samples = [2*int(command_terms[pop_flag+ii])
                           for ii in range(1, 1+num_pops)]
    else:
        raise ValueError('Unrecognized command string: %s.' % command)
    
    total_samples = numpy.sum(pop_samples)
    sample_indices = numpy.cumsum([0] + pop_samples)
    bottom_l = sample_indices[:-1]
    top_l = sample_indices[1:]
    
    seeds = line = input.readline()
    while not line.startswith('//'):
        line = input.readline()
    
    spectra = []
    counts = numpy.zeros(len(pop_samples), numpy.int_)
    sfs_shape = numpy.asarray(pop_samples) + 1
    
    dimension = len(counts)
    
    if dimension > 1:
        bottom0 = bottom_l[0]
        top0 = top_l[0]
        bottom1 = bottom_l[1]
        top1 = top_l[1]
    if dimension > 2:
        bottom2 = bottom_l[2]
        top2 = top_l[2]
    
    #output.writelines([ms_command, seeds, '\n'])
    
    sfs = numpy.zeros(sfs_shape, numpy.int_)
    for ii in range(runs):
        line = input.readline()
        segsites = int(line.split()[-1])
        
        if segsites == 0:
            # Special case, need to read 3 lines to stay synced.
            for ii in range(3):
                line = input.readline()
            continue
        line = input.readline()
        while not line.startswith('positions'):
            line = input.readline()
    
        # Read the chromosomes in
        chromos = input.read((segsites+1)*total_samples)
    
        for snp in range(segsites):
            # Slice to get all the entries that refer to a particular SNP
            this_snp = chromos[snp::segsites+1]
            # Count SNPs per population, and record them.
            if dimension == 1:
                sfs[this_snp.count('1')] += 1
            elif dimension == 2:
                sfs[this_snp[bottom0:top0].count('1'), 
                    this_snp[bottom1:top1].count('1')] += 1
            elif dimension == 3:
                sfs[this_snp[bottom0:top0].count('1'), 
                    this_snp[bottom1:top1].count('1'),
                    this_snp[bottom2:top2].count('1')] += 1
            else:
                for ii in range(dimension):
                    bottom = bottom_l[ii]
                    top = top_l[ii]
                    counts[ii] = this_snp[bottom:top].count('1')
                sfs[tuple(counts)] += 1
    
        if not average and not report_sum:
            import scipy.io
            output.writelines(['//\n'])
            scipy.io.write_array(output, sfs, keep_open=True)
            output.writelines(['\n'])
            sfs = numpy.zeros(sfs_shape, numpy.int_)
    
        line = input.readline()
        line = input.readline()
    input.close()
    
    if average:
        sfs = sfs/(1.0 * runs)

    if not return_header:
        return sfs
    else:
        return sfs, (command,seeds)

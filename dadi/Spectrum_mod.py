"""
Contains Spectrum object, which represents frequency spectra.
"""
import os

import numpy
from numpy import newaxis as nuax
from scipy.integrate import trapz

from dadi.Numerics import reverse_array, _cached_projection, _lncomb

class Spectrum(numpy.ma.masked_array):
    """
    Represents a frequency spectrum.

    Spectra are represented by masked arrays. The masking allows us to ignore
    specific entries in the spectrum. Most often, these are the absent and fixed
    categories.
    """
    def __new__(cls, data, *args, **kwargs):
        """
        Overrides array.__new__ to set nicer defaults.
        """
        # We need to do this in __new__ rather than waiting for __init__ because
        # otherwise the masked_array __new__ will be used.
        return numpy.ma.masked_array.__new__(cls, data, copy=True, 
                                             dtype=float, fill_value=numpy.nan)
    def __init__(self, data, mask=None, mask_corners=True):
        """
        Construct a spectrum.

        data: Data for spectrum
        mask: Mask to use. If None, an empty mask is created.
        mask_corners: If True, the 'absent in all pops' and 'fixed in all pops'
                      entries are masked.
        """
        # Set the mask and the fill value
        if mask is None:
            mask = numpy.ma.make_mask_none(self.data.shape)
        self.mask = mask

        if mask_corners:
            self.mask_corners()

    def mask_corners(self):
        """
        Mask the 'seen in 0 samples' and 'seen in all samples' entries.
        """
        self.mask.flat[0] = self.mask.flat[-1] = True

    def unmask_all(self):
        """
        Unmask all values.
        """
        self.mask[[slice(None)]*self.Npop] = False

    def _get_sample_sizes(self):
        return numpy.asarray(self.shape) - 1
    sample_sizes = property(_get_sample_sizes)

    def _get_Npop(self):
        return self.ndim
    Npop = property(_get_Npop)

    def _ensure_dimension(self, Npop):
        """
        Ensure that fs has Npop dimensions.
        """
        if not self.Npop == Npop:
            raise ValueError('Only compatible with %id spectra.' % Npop)

    # Make from_file a static method, so we can use it without an instance.
    @staticmethod
    def from_file(fid, mask_corners=True, return_comments=False):
        """
        Read frequency spectrum from file.

        fid: string with file name to read from or an open file object.
        mask_corners: If True, mask the 'absent in all samples' and 'fixed in
                      all samples' entries.
        return_comments: If true, the return value is (fs, comments), where
                         comments is a list of strings containing the comments
                         from the file (without #'s).

        The file format is:
            # Any number of comment lines beginning with a '#'
            A single line containing N integers giving the dimensions of the fs
              array. So this line would be '5 5 3' for an SFS that was 5x5x3.
              (That would be 4x4x2 *samples*.)
            A single line giving the array elements. The order of elements is 
              e.g.: fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1] ...
        """
        newfile = False
        # Try to read from fid. If we can't, assume it's something that we can
        # use to open a file.
        if not hasattr(fid, 'read'):
            newfile = True
            fid = file(fid, 'r')

        line = fid.readline()
        # Strip out the comment
        comments = []
        while line.startswith('#'):
            comments.append(line[1:].strip())
            line = fid.readline()

        # Read the shape of the fs
        shape = tuple([int(d) for d in line.split()])

        data = numpy.fromfile(fid, count=numpy.product(shape), sep=' ')
        # fromfile returns a 1-d array. Reshape it to the proper form.
        data = data.reshape(*shape)

        # If we opened a new file, clean it up.
        if newfile:
            fid.close()

        # Convert to a fs object
        fs = Spectrum(data, mask_corners=mask_corners)
                                       
        if not return_comments:
            return fs
        else:
            return fs,comments

    def to_file(self, fid, precision=16, comment_lines = []):
        """
        Write frequency spectrum to file.
    
        fid: string with file name to write to or an open file object.
        precision: precision with which to write out entries of the SFS. (They 
                   are formated via %.<p>g, where <p> is the precision.)
        comment lines: list of strings to be used as comment lines in the header
                       of the output file.
        """
        # Open the file object.
        newfile = False
        if not hasattr(fid, 'write'):
            newfile = True
            fid = file(fid, 'w')
    
        # Write comments
        for line in comment_lines:
            fid.write('# ')
            fid.write(line.strip())
            fid.write(os.linesep)
    
        # Write out the shape of the fs
        for elem in self.shape:
            fid.write('%i ' % elem)
        fid.write(os.linesep)
    
        # Masked entries in the fs will go in as 'nan'
        data = self.filled()
        # Write to file
        data.tofile(fid, ' ', '%%.%ig' % precision)
        fid.write(os.linesep)
    
        # Close file
        if newfile:
            fid.close()
    # Overide the (perhaps confusing) original numpy tofile method.
    tofile = to_file

    def project(self, ns):
        """
        Project to smaller sample size.

        ns: Sample sizes for new spectrum.
        """
        if len(ns) != self.Npop:
            raise ValueError('Requested sample sizes not of same dimension '
                             'as spectrum. Perhaps you need to marginalize '
                             'over some populations first?')
        if numpy.any(numpy.asarray(ns) > numpy.asarray(self.sample_sizes)):
            raise ValueError('Cannot project to a sample size greater than '
                             'original. Original size is %s and requested size '
                             'is %s.' % (self.sample_sizes, ns))

        output = self.copy()
        # Iterate over each axis, applying the projection.
        for axis,proj in enumerate(ns):
            if proj != self.sample_sizes[axis]:
                output = output._project_one_axis(proj, axis)
        return output

    def _project_one_axis(self, n, axis=0):
        """
        Project along a single axis.
        """
        # This gets a little tricky with fancy indexing to make it work
        # for fs with arbitrary number of dimensions.
        if n > self.sample_sizes[axis]:
            raise ValueError('Cannot project to a sample size greater than '
                             'original. Called sizes were from %s to %s.' 
                             % (self.sample_sizes[axis], n))

        newshape = list(self.shape)
        newshape[axis] = n+1
        # Create a new empty fs that we'll fill in below.
        pfs = Spectrum(numpy.zeros(newshape), mask_corners=False)

        # Set up for our fancy indexes. These slices are currently like
        # [:,:,...]
        from_slice = [slice(None) for ii in range(self.Npop)]
        to_slice = [slice(None) for ii in range(self.Npop)]
        proj_slice = [nuax for ii in range(self.Npop)]

        proj_from = self.sample_sizes[axis]
        # For each possible number of hits.
        for hits in range(proj_from+1):
            # Adjust the slice in the array we're projecting from.
            from_slice[axis] = slice(hits, hits+1)
            # These are the least and most possible hits we could have in the
            #  projected fs.
            least, most = max(n - (proj_from - hits), 0), min(hits,n)
            to_slice[axis] = slice(least, most+1)
            # The projection weights.
            proj = _cached_projection(n, proj_from, hits)
            proj_slice[axis] = slice(least, most+1)
            # Warning: There are some subtleties (which may be numpy bugs) in
            # how multiplication of masked arrays works in this rather
            # complicated slicing scheme.
            # The commented line below does not work...
            #  pfs[to_slice] += self[from_slice] * proj[proj_slice]
            # A more step-by-step way to do this would be:
            #  pfs.data[to_slice] += self.data[from_slice] * proj[proj_slice]
            #  pfs.mask[to_slice] = numpy.logical_or(pfs.mask[to_slice],
            #                                        self.mask[from_slice])
            pfs[to_slice] += proj[proj_slice] * self[from_slice]
    
        return pfs

    def marginalize(self, over, mask_corners=True):
        """
        Reduced dimensionality spectrum summing over some populations.

        over: sequence of axes to sum over. For example (0,2) will sum over
              populations 0 and 2.
        mask_corners: If True, the typical corners of the resulting fs will be
                      masked
        """
        output = self.copy()
        for axis in sorted(over)[::-1]:
            output = output.sum(axis=axis)
        if mask_corners:
            output.mask_corners()
        return output

    def _counts_per_entry(self):
        """
        Counts per population for each entry in the fs.
        """
        ind = numpy.indices(self.shape)
        # Transpose the first access to the last, so ind[ii,jj,kk] = [ii,jj,kk]
        ind = ind.transpose(range(1,self.Npop+1)+[0])
        return ind

    def _total_per_entry(self):
        """
        Total derived alleles for each entry in the fs.
        """
        return numpy.sum(self._counts_per_entry(), axis=-1)

    def fold(self):
        """
        Folded frequency spectrum
    
        The folded fs assumes that information on which allele is ancestral or
        derived is unavailable. Thus the fs is in terms of minor allele 
        frequency.  Note that this makes the fs into a "triangular" array.
    
        Note that if a masked cell is folded into non-masked cell, the
        destination cell is masked as well.
        """
        # How many samples total do we have? The folded fs can only contain
        # entries up to total_samples/2 (rounded down).
        total_samples = numpy.sum(self.sample_sizes)

        total_per_entry = self._total_per_entry()
    
        # Here's where we calculate which entries are nonsense in the folded fs.
        where_folded_out = total_per_entry > int(total_samples/2)
    
        original_mask = self.mask
        # Here we create a mask that masks any values that were masked in
        # the original fs (or folded onto by a masked value).
        final_mask = numpy.logical_or(original_mask, 
                                      reverse_array(original_mask))
        
        # To do the actual folding, we take those entries that would be folded
        # out, reverse the array along all axes, and add them back to the
        # original fs.
        reversed = reverse_array(numpy.where(where_folded_out, self, 0))
        folded = self + reversed
    
        # Here's where we calculate which entries are nonsense in the folded fs.
        where_ambiguous = total_per_entry == int(total_samples/2)
        ambiguous = numpy.where(where_ambiguous, self, 0)
        folded += -0.5*ambiguous + 0.5*reverse_array(ambiguous)
    
        # Mask out the remains of the folding operation.
        final_mask = numpy.logical_or(final_mask, where_folded_out)
        folded = numpy.ma.masked_array(folded, mask=final_mask)
    
        return folded

    def sample(self):
        """
        Generate a Poisson-sampled fs from the current one.

        Note: Entries where the current fs is masked or 0 will be masked in the
              output sampled fs.
        """
        import scipy.stats
        # These are entries where the sampling has no meaning. Either the fs is
        # 0 there or masked. 
        bad_entries = numpy.logical_or(self == 0, self.mask)
        # We convert to a 1-d array for passing into the sampler
        means = self.ravel()
        # Filter out those bad entries.
        means[bad_entries.ravel()] = 1
        # Sample
        samp = scipy.stats.distributions.poisson.rvs(means, size=len(means))
        # Convert back to a properly shaped array
        samp = samp.reshape(self.shape)
        # Convert to a fs and mask the bad entries
        samp = Spectrum(samp, mask=bad_entries)
        return samp

    @staticmethod
    def from_ms_file(fid, average=True, mask_corners=True, return_header=False):
        """
        Read frequency spectrum from file of ms output.

        fid: string with file name to read from or an open file object.
        average: If True, the returned fs is the average over the runs in the ms
                 file. If False, the returned fs is the sum.
        mask_corners: If True, mask the 'absent in all samples' and 'fixed in
                      all samples' entries.
        return_header: If true, the return value is (fs, (command,seeds), where
                       command and seeds are strings containing the ms
                       commandline and the seeds used.
        """
        newfile = False
        # Try to read from fid. If we can't, assume it's something that we can
        # use to open a file.
        if not hasattr(fid, 'read'):
            newfile = True
            fid = file(fid, 'r')

        command = line = fid.readline()
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
        else:
            raise ValueError('Unrecognized command string: %s.' % command)
        
        total_samples = numpy.sum(pop_samples)
        sample_indices = numpy.cumsum([0] + pop_samples)
        bottom_l = sample_indices[:-1]
        top_l = sample_indices[1:]
        
        seeds = line = fid.readline()
        while not line.startswith('//'):
            line = fid.readline()
        
        counts = numpy.zeros(len(pop_samples), numpy.int_)
        fs_shape = numpy.asarray(pop_samples) + 1
        dimension = len(counts)
        
        if dimension > 1:
            bottom0 = bottom_l[0]
            top0 = top_l[0]
            bottom1 = bottom_l[1]
            top1 = top_l[1]
        if dimension > 2:
            bottom2 = bottom_l[2]
            top2 = top_l[2]
        if dimension > 3:
            bottom3 = bottom_l[3]
            top3 = top_l[3]
        
        data = numpy.zeros(fs_shape, numpy.int_)
        for ii in range(runs):
            line = fid.readline()
            segsites = int(line.split()[-1])
            
            if segsites == 0:
                # Special case, need to read 3 lines to stay synced.
                for ii in range(3):
                    line = fid.readline()
                continue
            line = fid.readline()
            while not line.startswith('positions'):
                line = fid.readline()
        
            # Read the chromosomes in
            chromos = fid.read((segsites+1)*total_samples)
        
            for snp in range(segsites):
                # Slice to get all the entries that refer to a particular SNP
                this_snp = chromos[snp::segsites+1]
                # Count SNPs per population, and record them.
                if dimension == 1:
                    data[this_snp.count('1')] += 1
                elif dimension == 2:
                    data[this_snp[bottom0:top0].count('1'), 
                        this_snp[bottom1:top1].count('1')] += 1
                elif dimension == 3:
                    data[this_snp[bottom0:top0].count('1'), 
                        this_snp[bottom1:top1].count('1'),
                        this_snp[bottom2:top2].count('1')] += 1
                elif dimension == 4:
                    data[this_snp[bottom0:top0].count('1'), 
                        this_snp[bottom1:top1].count('1'),
                        this_snp[bottom2:top2].count('1'),
                        this_snp[bottom3:top3].count('1')] += 1
                else:
                    # This is noticably slower, so we special case the cases
                    # above.
                    for ii in range(dimension):
                        bottom = bottom_l[ii]
                        top = top_l[ii]
                        counts[ii] = this_snp[bottom:top].count('1')
                    data[tuple(counts)] += 1
        
            line = fid.readline()
            line = fid.readline()

        if newfile:
            fid.close()
        
        fs = Spectrum(data, mask_corners=mask_corners)
        if average:
            fs /= runs

        if not return_header:
            return fs
        else:
            return fs, (command,seeds)

    def Fst(self):
        """
        Wright's Fst between the populations represented in the fs.
    
        This estimate of Fst assumes random mating, because we don't have
        heterozygote frequencies in the fs.
    
        Calculation is by the method of Weir and Cockerham _Evolution_ 38:1358.
        For a single SNP, the relevant formula is at the top of page 1363. To
        combine results between SNPs, we use the weighted average indicated by
        equation 10.
        """
        # This gets a little obscure because we want to be able to work with
        # spectra of arbitrary dimension.
    
        # First quantities from page 1360
        r = self.Npop
        ns = self.sample_sizes
        nbar = numpy.mean(ns)
        nsum = numpy.sum(ns)
        nc = (nsum - numpy.sum(ns**2)/nsum)/(r-1)
    
        # counts_per_pop is an r+1 dimensional array, where the last axis simply
        # records the indices of the entry. 
        # For example, counts_per_pop[4,19,8] = [4,19,8]
        counts_per_pop = numpy.indices(self.shape)
        counts_per_pop = numpy.transpose(counts_per_pop, axes=range(1,r+1)+[0])
    
        # The last axis of ptwiddle is now the relative frequency of SNPs in
        # that bin in each of the populations.
        ptwiddle = 1.*counts_per_pop/ns
    
        # Note that pbar is of the same shape as fs...
        pbar = numpy.sum(ns*ptwiddle, axis=-1)/nsum
    
        # We need to use 'this_slice' to get the proper aligment between
        # ptwiddle and pbar.
        this_slice = [slice(None)]*r + [numpy.newaxis]
        s2 = numpy.sum(ns * (ptwiddle - pbar[this_slice])**2, axis=-1)/((r-1)*nbar)
    
        # Note that this 'a' differs from equation 2, because we've used
        # equation 3 and b = 0 to solve for hbar.
        a = nbar/nc * (s2 - 1/(2*nbar-1) * (pbar*(1-pbar) - (r-1)/r*s2))
        d = 2*nbar/(2*nbar-1) * (pbar*(1-pbar) - (r-1)/r*s2)
    
        # The weighted sum over loci.
        asum = (self * a).sum()
        dsum = (self * d).sum()
    
        return asum/(asum+dsum)

    def S(self):
        """
        Segregating sites.
        """
        oldmask = self.mask.copy()
        self.mask_corners()
        S = self.sum()
        self.mask = oldmask
        return S
    
    def Watterson_theta(self):
        """
        Watterson's estimator of theta.
    
        Note that is only sensible for 1-dimensional spectra.
        """
        if self.Npop != 1:
            raise ValueError("Only defined on a one-dimensional fs.")
    
        n = self.sample_sizes[0]
        S = self.S()
        denom = numpy.sum(1./numpy.arange(1,n))
    
        return S/denom
    
    def pi(self):
        """
        Estimated expected heterozygosity.
    
        Note that this estimate assumes a randomly mating population.
        """
        if self.ndim != 1:
            raise ValueError("Only defined on a one-dimensional SFS.")
    
        n = self.sample_sizes[0]
        # sample frequencies p 
        p = 1.*numpy.arange(0,n+1)/n
        return n/(n-1) * 2*numpy.ma.sum(self*p*(1-p))
    
    def Tajima_D(self):
        """
        Tajima's D.
    
        Following Gillespie "Population Genetics: A Concise Guide" pg. 45
        """
        if not self.Npop == 1:
            raise ValueError("Only defined on a one-dimensional SFS.")
    
        S = self.S()
    
        n = 1.*self.sample_sizes[0]
        pihat = self.pi()
        theta = self.Watterson_theta()
    
        a1 = numpy.sum(1./numpy.arange(1,n))
        a2 = numpy.sum(1./numpy.arange(1,n)**2)
        b1 = (n+1)/(3*(n-1))
        b2 = 2*(n**2 + n + 3)/(9*n * (n-1))
        c1 = b1 - 1./a1
        c2 = b2 - (n+2)/(a1*n) + a2/a1**2
    
        C = numpy.sqrt((c1/a1)*S + c2/(a1**2 + a2) * S*(S-1))
    
        return (pihat - theta)/C

    @staticmethod
    def _from_phi_1D(n, xx, phi, mask_corners=True):
        data = numpy.zeros(n+1)
        for ii in range(0,n+1):
            factorx = comb(n,ii) * xx**ii * (1-xx)**(n-ii)
            data[ii] = trapz(factorx * phi, xx)
    
        return Spectrum(data, mask_corners=mask_corners)
    
    @staticmethod
    def _from_phi_2D(nx, ny, xx, yy, phi, mask_corners=True):
        # Calculate the 2D sfs from phi using the trapezoid rule for
        # integration.
        data = numpy.zeros((nx+1, ny+1))
        
        # Cache to avoid duplicated work.
        factorx_cache = {}
        for ii in range(0, nx+1):
            factorx = comb(nx, ii) * xx**ii * (1-xx)**(nx-ii)
            factorx_cache[nx,ii] = factorx
    
        dx, dy = numpy.diff(xx), numpy.diff(yy)
        for jj in range(0,ny+1):
            factory = comb(ny, jj) * yy**jj * (1-yy)**(ny-jj)
            integrated_over_y = trapz(factory[numpy.newaxis,:]*phi, dx=dy)
            for ii in range(0, nx+1):
                factorx = factorx_cache[nx,ii]
                data[ii,jj] = trapz(factorx*integrated_over_y, dx=dx)
    
        return Spectrum(data, mask_corners=mask_corners)
    
    @staticmethod
    def _from_phi_3D(nx, ny, nz, xx, yy, zz, phi, mask_corners=True):
        data = numpy.zeros((nx+1, ny+1, nz+1))
    
        dx, dy, dz = numpy.diff(xx), numpy.diff(yy), numpy.diff(zz)
        half_dx = dx/2.0
    
        # We cache these calculations...
        factorx_cache, factory_cache = {}, {}
        for ii in range(0, nx+1):
            factorx = comb(nx, ii) * xx**ii * (1-xx)**(nx-ii)
            factorx_cache[nx,ii] = factorx
        for jj in range(0, ny+1):
            factory = comb(ny, jj) * yy**jj * (1-yy)**(ny-jj)
            factory_cache[ny,jj] = factory[nuax,:]
    
        for kk in range(0, nz+1):
            factorz = comb(nz, kk) * zz**kk * (1-zz)**(nz-kk)
            over_z = trapz(factorz[nuax, nuax,:] * phi, dx=dz)
            for jj in range(0, ny+1):
                factory = factory_cache[ny,jj]
                over_y = trapz(factory * over_z, dx=dy)
                for ii in range(0, nx+1):
                    factorx = factorx_cache[nx,ii]
                    # It's faster here to do the trapezoid rule explicitly
                    # rather than using SciPy's more general routine.
                    integrand = factorx * over_y
                    ans = numpy.sum(half_dx * (integrand[1:]+integrand[:-1]))
                    data[ii,jj,kk] = ans
    
        return Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def from_phi(phi, ns, xxs, mask_corners=mask_corners):
        if not phi.ndim == len(ns) == len(xxs):
            raise ValueError('Dimensionality of phi and lengths of ns and xxs '
                             'do not all agree.')
        if phi.ndim == 1:
            return _from_phi_1D(ns[0], xxs[0], phi, mask_corners)
        elif phi.ndim == 2:
            return _from_phi_2D(ns[0], ns[1], xxs[0], xxs[1], phi, mask_corners)
        elif phi.ndim == 3:
            return _from_phi_3D(ns[0], ns[1], ns[2], xxs[0], xxs[1], xxs[2], 
                                phi, mask_corners)
        else:
            raise ValueError('Only implemented for dimensions 1,2 or 3.')

    def scramble_pop_ids(self, mask_corners=True):
        """
        Spectrum corresponding to scrambling individuals among populations.
        
        This is useful for assessing how diverged populations are.
        Essentially, it pools all the individuals represented in the fs and
        generates new populations of random individuals (without replacement)
        from that pool. If this fs is significantly different from the
        original, that implies population structure.
        """
        total_samp = numpy.sum(self.sample_sizes)
    
        # First generate a 1d sfs for the pooled population.
        combined = numpy.zeros(total_samp+1)
        # For each entry in the fs, this is the total number of derived alleles
        total_per_entry = self._total_per_entry()
        # Sum up to generate the equivalent 1-d spectrum.
        for derived,counts in zip(total_per_entry.ravel(), self.ravel()):
            combined[derived] += counts
    
        # Now resample back into a n-d spectrum
        # For each entry, this is the counts per popuation. 
        #  e.g. counts_per_entry[3,4,5] = [3,4,5]
        counts_per_entry = self._counts_per_entry()
        # Reshape it to be 1-d, so we can iterate over it easily.
        counts_per_entry = counts_per_entry.reshape(numpy.prod(self.shape), 
                                                    self.ndim)
        resamp = numpy.zeros(self.shape)
        for counts, derived in zip(counts_per_entry, total_per_entry.ravel()):
            # The probability here is 
            # (t1 choose d1)*(t2 choose d2)/(ntot choose derived)
            lnprob = sum(_lncomb(t,d) for t,d in zip(self.sample_sizes,counts))
            lnprob -= _lncomb(total_samp, derived)
            prob = numpy.exp(lnprob)
            # Assign result using the appropriate weighting
            resamp[tuple(counts)] += prob*combined[derived]

        return Spectrum(resamp, mask_corners=mask_corners)

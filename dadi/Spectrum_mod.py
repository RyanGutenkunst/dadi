"""
Contains Spectrum object, which represents frequency spectra.
"""
import logging
logging.basicConfig()
logger = logging.getLogger('Spectrum_mod')

import gzip, operator, os, sys

import numpy
from numpy import newaxis as nuax
import numpy as np
# Account for difference in scipy installations.
from scipy.special import comb
from scipy.integrate import trapz
from scipy.special import betainc

import dadi.Numerics
from dadi.Numerics import reverse_array, _cached_projection, _lncomb, BetaBinomConvolution

_dbeta_cache = {}
def cached_dbeta(nx, xx):
    key = nx, tuple(xx)

    xx = numpy.minimum(numpy.maximum(xx, 0), 1.0)
    if key not in _dbeta_cache:
        dbeta1 = np.empty((nx+1,len(xx)-1))
        dbeta2 = np.empty((nx+1,len(xx)-1))
        for ii in range(0, nx+1):
            b = betainc(ii+1,nx-ii+1,xx)
            dbeta1[ii] = b[1:]-b[:-1]
            b = betainc(ii+2,nx-ii+1,xx)
            dbeta2[ii] = b[1:]-b[:-1]
        _dbeta_cache[key] = dbeta1, dbeta2
    dbeta1, dbeta2 = _dbeta_cache[key]
    return dbeta1, dbeta2

_imported_demes = False

class Spectrum(numpy.ma.masked_array):
    """
    Represents a frequency spectrum.

    Spectra are represented by masked arrays. The masking allows us to ignore
    specific entries in the spectrum. Most often, these are the absent and fixed
    categories.

    The constructor has the format:
        fs = dadi.Spectrum(data, mask, mask_corners, data_folded, check_folding,
                           pop_ids, extrap_x)
        
        data: The frequency spectrum data
        mask: An optional array of the same size as data. 'True' entires in
              this array are masked in the Spectrum. These represent missing
              data categories. (For example, you may not trust your singleton
              SNP calling.)
        mask_corners: If True (default), the 'observed in none' and 'observed 
                      in all' entries of the FS will be masked. Typically these
                      entries are unobservable, and dadi cannot reliably
                      calculate them, so you will almost always want
                      mask_corners=True.g
        data_folded: If True, it is assumed that the input data is folded. An
                     error will be raised if the input data and mask are not
                     consistent with a folded Spectrum.
        check_folding: If True and data_folded=True, the data and mask will be
                       checked to ensure they are consistent with a folded
                       Spectrum. If they are not, a warning will be printed.
        pop_ids: Optional list of strings containing the population labels.
        extrap_x: Optional floating point value specifying x value to use
                  for extrapolation.
    """
    def __new__(subtype, data, mask=numpy.ma.nomask, mask_corners=True, 
                data_folded=None, check_folding=True, dtype=float, copy=True, 
                fill_value=numpy.nan, keep_mask=True, shrink=True, 
                pop_ids=None, extrap_x=None):
        data = numpy.asanyarray(data)

        if mask is numpy.ma.nomask:
            mask = numpy.ma.make_mask_none(data.shape)

        subarr = numpy.ma.masked_array(data, mask=mask, dtype=dtype, copy=copy,
                                       fill_value=fill_value, keep_mask=True, 
                                       shrink=True)
        subarr = subarr.view(subtype)

        if hasattr(data, 'folded'):
            if data_folded is None or data_folded == data.folded:
                subarr.folded = data.folded
            elif data_folded != data.folded:
                raise ValueError('Data does not have same folding status as '
                                 'was called for in Spectrum constructor.')
        elif data_folded is not None:
            subarr.folded = data_folded
        else:
            subarr.folded = False

        # Check that if we're declaring that the input data is folded, it
        # actually is, and the mask reflects this.
        if data_folded:
            total_samples = numpy.sum(subarr.sample_sizes)
            total_per_entry = subarr._total_per_entry()
            # Which entries are nonsense in the folded fs.
            where_folded_out = total_per_entry > int(total_samples/2)
            if check_folding\
               and not numpy.all(subarr.data[where_folded_out] == 0):
                logger.warning('Creating Spectrum with data_folded = True, but '
                            'data has non-zero values in entries which are '
                            'nonsensical for a folded Spectrum.')
            if check_folding\
               and not numpy.all(subarr.mask[where_folded_out]):
                logger.warning('Creating Spectrum with data_folded = True, but '
                            'mask is not True for all entries which are '
                            'nonsensical for a folded Spectrum.')

        if hasattr(data, 'pop_ids'):
            if pop_ids is None or pop_ids == data.pop_ids:
                subarr.pop_ids = data.pop_ids
            elif pop_ids != data.pop_ids:
                logger.warning('Changing population labels in construction of new '
                            'Spectrum.')
                if len(pop_ids) != subarr.ndim:
                    raise ValueError('pop_ids must be of length equal to '
                                     'dimensionality of Spectrum.')
                subarr.pop_ids = pop_ids
        else:
            if pop_ids is not None and len(pop_ids) != subarr.ndim:
                raise ValueError('pop_ids must be of length equal to '
                                 'dimensionality of Spectrum.')
            subarr.pop_ids = pop_ids

        if mask_corners:
            subarr.mask_corners()

        subarr.extrap_x = extrap_x

        return subarr

    # See http://www.scipy.org/Subclasses for information on the
    # __array_finalize__ and __array_wrap__ methods. I had to do some debugging
    # myself to discover that I also needed _update_from.
    # Also, see http://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    # Also, see http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    #
    # We need these methods to ensure extra attributes get copied along when
    # we do arithmetic on the FS.
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        numpy.ma.masked_array.__array_finalize__(self, obj)
        self.folded = getattr(obj, 'folded', 'unspecified')
        self.pop_ids = getattr(obj, 'pop_ids', None)
        self.extrap_x = getattr(obj, 'extrap_x', None)
    def __array_wrap__(self, obj, context=None):
        result = obj.view(type(self))
        result = numpy.ma.masked_array.__array_wrap__(self, obj, 
                                                      context=context)
        result.folded = self.folded
        result.pop_ids = self.pop_ids
        result.extrap_x = self.extrap_x
        return result
    def _update_from(self, obj):
        numpy.ma.masked_array._update_from(self, obj)
        if hasattr(obj, 'folded'):
            self.folded = obj.folded
        if hasattr(obj, 'pop_ids'):
            self.pop_ids = obj.pop_ids
        if hasattr(obj, 'extrap_x'):
            self.extrap_x = obj.extrap_x
    # masked_array has priority 15.
    __array_priority__ = 20

    def __repr__(self):
        return 'Spectrum(%s, folded=%s, pop_ids=%s)'\
                % (str(self), str(self.folded), str(self.pop_ids))

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
    def from_file(fname, mask_corners=True, return_comments=False):
        """
        Read frequency spectrum from file.

        fname: String with file name to read from. If it ends in .gz, gzip
               compression is assumed.
        mask_corners: If True, mask the 'absent in all samples' and 'fixed in
                      all samples' entries.
        return_comments: If true, the return value is (fs, comments), where
                         comments is a list of strings containing the comments
                         from the file (without #'s).

        See to_file method for details on the file format.
        """
        if fname.endswith('.gz'):
            fid = gzip.open(fname, 'rb')
        else:
            fid = open(fname, 'r')

        line = fid.readline()
        # Strip out the comments
        comments = []
        while line.startswith('#'):
            comments.append(line[1:].strip())
            line = fid.readline()

        # Read the shape of the data
        shape_spl = line.split()
        if 'folded' not in shape_spl and 'unfolded' not in shape_spl:
            # This case handles the old file format
            shape = tuple([int(d) for d in shape_spl])
            folded = False
            pop_ids = None
        else:
            # This case handles the new file format
            shape,next_ii = [int(shape_spl[0])], 1
            while shape_spl[next_ii] not in ['folded', 'unfolded']:
                shape.append(int(shape_spl[next_ii]))
                next_ii += 1
            folded = (shape_spl[next_ii] == 'folded')
            # Are there population labels in the file?
            if len(shape_spl) > next_ii + 1:
                pop_ids = line.split('"')[1::2]
            else:
                pop_ids = None

        data = numpy.fromstring(fid.readline().strip(), 
                                count=numpy.product(shape), sep=' ')
        # fromfile returns a 1-d array. Reshape it to the proper form.
        data = data.reshape(*shape)

        maskline = fid.readline().strip()
        if not maskline:
            # The old file format didn't have a line for the mask
            mask = None
        else:
            # This case handles the new file format
            mask = numpy.fromstring(maskline, 
                                    count=numpy.product(shape), sep=' ')
            mask = mask.reshape(*shape)

        fs = Spectrum(data, mask, mask_corners, data_folded=folded,
                      pop_ids=pop_ids)

        fid.close()
        if not return_comments:
            return fs
        else:
            return fs,comments

    fromfile = from_file

    def to_file(self, fname, precision=16, comment_lines = [], 
                foldmaskinfo=True):
        """
        Write frequency spectrum to file.

        fname: File name to write to.  If string ends in .gz, file will be saved
               with gzip compression.
        precision: precision with which to write out entries of the SFS. (They
                   are formated via %.<p>g, where <p> is the precision.)
        comment lines: list of strings to be used as comment lines in the header
                       of the output file.
        foldmaskinfo: If False, folding and mask and population label
                      information will not be saved. This conforms to the file
                      format for dadi versions prior to 1.3.0.

        The file format is:
            # Any number of comment lines beginning with a '#'
            A single line containing N integers giving the dimensions of the fs
              array. So this line would be '5 5 3' for an SFS that was 5x5x3.
              (That would be 4x4x2 *samples*.)
            On the *same line*, the string 'folded' or 'unfolded' denoting the
              folding status of the array
            On the *same line*, optional strings each containing the population
              labels in quotes separated by spaces, e.g. "pop 1" "pop 2"
            A single line giving the array elements. The order of elements is 
              e.g.: fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1] ...
            A single line giving the elements of the mask in the same order as
              the data line. '1' indicates masked, '0' indicates unmasked.
        """
        # Open the file object.
        if fname.endswith('.gz'):
            fid = gzip.open(fname, 'wb')
        else:
            fid = open(fname, 'w')

        # Write comments
        for line in comment_lines:
            fid.write('# ')
            fid.write(line.strip())
            fid.write('\n')

        # Write out the shape of the fs
        for elem in self.data.shape:
            fid.write('%i ' % elem)

        if foldmaskinfo:
            if not self.folded:
                fid.write('unfolded')
            else:
                fid.write('folded')
            if self.pop_ids is not None:
                for label in self.pop_ids:
                    fid.write(' "%s"' % label)

        fid.write('\n')

        # Write the data to the file. The obnoxious ravel call is to
        # ensure compatibility with old version that used self.data.tofile.
        numpy.savetxt(fid, [self.data.ravel()], delimiter=' ',
                      fmt='%%.%ig' % precision)

        if foldmaskinfo:
            # Write the mask to the file
            numpy.savetxt(fid, [numpy.asarray(self.mask, int).ravel()],
                          delimiter=' ', fmt='%d')

        fid.close()

    tofile = to_file

    ## Overide the (perhaps confusing) original numpy tofile method.
    #def tofile(self, *args,**kwargs):
    #    self.to_file(*args, **kwargs)

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

        original_folded = self.folded
        # If we started with an folded Spectrum, we need to unfold before
        # projecting.
        if original_folded:
            output = self.unfold()
        else:
            output = self.copy()

        # Iterate over each axis, applying the projection.
        for axis,proj in enumerate(ns):
            if proj != self.sample_sizes[axis]:
                output = output._project_one_axis(proj, axis)

        output.pop_ids = self.pop_ids
        output.extrap_x = self.extrap_x

        # Return folded or unfolded as original.
        if original_folded:
            return output.fold()
        else:
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
            # Do the multiplications
            pfs.data[tuple(to_slice)] += self.data[tuple(from_slice)] * proj[tuple(proj_slice)]
            pfs.mask[tuple(to_slice)] = numpy.logical_or(pfs.mask[tuple(to_slice)],
                                                         self.mask[tuple(from_slice)])

        return pfs

    def marginalize(self, over, mask_corners=True):
        """
        Reduced dimensionality spectrum summing over some populations.

        over: sequence of axes to sum over. For example (0,2) will sum over
              populations 0 and 2.
        mask_corners: If True, the typical corners of the resulting fs will be
                      masked
        """
        original_folded = self.folded
        # If we started with an folded Spectrum, we need to unfold before
        # marginalizing.
        if original_folded:
            output = self.unfold()
        else:
            output = self.copy()

        orig_mask = output.mask.copy()
        orig_mask.flat[0] = orig_mask.flat[-1] = False
        if numpy.any(orig_mask):
            logger.warning('Marginalizing a Spectrum with internal masked values. '
                        'This may not be a well-defined operation.')

        # Do the marginalization
        for axis in sorted(over)[::-1]:
            output = output.sum(axis=axis)
        pop_ids = None
        if self.pop_ids is not None:
            pop_ids = list(self.pop_ids)
            for axis in sorted(over)[::-1]:
                del pop_ids[axis]
        output.folded = False
        output.pop_ids = pop_ids
        output.extrap_x = self.extrap_x

        if mask_corners:
            output.mask_corners()

        # Return folded or unfolded as original.
        if original_folded:
            return output.fold()
        else:
            return output

    def combine_pops(self, tocombine):
        """
        Combine two or more populations in the fs, treating them as a single pop

        tocombine: Indices for populations being combined (starting from 1)

        The populations will alwasy be combined into the slot of the 
        population with the smallest index. For example, if the sample sizes of
        the spectrum are (1,2,3,4,5) and tocombine=[4,2,1], then the output spectrum
        will have sample_sizes (7,3,5) when populations 1, 2, and 4 are combined.

        The pop_ids of the new population will be the pop_ids of the combined
        populations with a '+' in between them.
        """
        tocombine = sorted(tocombine)
        result = self
        # Need to combine from highest to lowest index
        for right_pop in tocombine[1:][::-1]: 
            result = result.combine_two_pops([tocombine[0], right_pop])
        # Need to fix pop_id of combined pop
        if self.pop_ids:
            result.pop_ids[tocombine[0]-1] = '+'.join(self.pop_ids[_-1] for _ in tocombine)
        return result

    def combine_two_pops(self, tocombine):
        """
        Combine two populations in the fs, treating them as a single pop

        tocombine: Indices for populations being combined (starting from 1)

        The two populations will alwasy be combined into the slot of the 
        population with the smallest index. For example, if the sample sizes of
        the spectrum are (2,3,4,5) and tocombine=[4,2], then the output spectrum
        will have sample_sizes (2,8,4) when populations 2 and 4 are combined.

        The pop_ids of the new population will be the pop_ids of the two combined
        populations with a '+' in between them.
        """
        # Calculate new sample sizes
        tocombine = sorted([_-1 for _ in tocombine]) # Account for indexing from 1
        new_ns = list(self.sample_sizes)
        new_ns[tocombine[0]] = self.sample_sizes[tocombine[0]] + self.sample_sizes[tocombine[1]]
        del new_ns[tocombine[1]] # Remove pop being combined away

        # Create new pop ids
        new_pop_ids = None
        if self.pop_ids:
            new_pop_ids = list(self.pop_ids)
            new_pop_ids[tocombine[0]] = '{0}+{1}'.format(self.pop_ids[tocombine[0]], self.pop_ids[tocombine[1]])
            del new_pop_ids[tocombine[1]]

        # Create new spectrum
        new_data = np.zeros(shape=[n+1 for n in new_ns])
        new_fs = Spectrum(new_data, pop_ids=new_pop_ids)
        # Copy over extrapolation info
        new_fs.extrap_x = self.extrap_x

        # Fill new spectrum
        for index in np.ndindex(self.shape):
            new_index = list(index)
            new_index[tocombine[0]] = index[tocombine[0]] + index[tocombine[1]]
            del new_index[tocombine[1]]
            new_index = tuple(new_index)
            new_fs[new_index] += self[index]
            # Mask entry if any of the contributing entries are masked
            new_fs.mask[new_index] = (new_fs.mask[new_index] or self.mask[index])

        return new_fs

    def filter_pops(self, tokeep, mask_corners=True):
        """
        Filter Spectrum to keep only certain populations.

        Returns new Spectrum with len(tokeep) populations.
        Note: This is similar in practice to the marginalize operation. But here
              populations are numbered from 1, as in the majority of dadi.

        tokeep: Unordered set of population numbers to keep, numbering from 1.
        mask_corners: If True, the typical corners of the resulting fs will be
                      masked
        """
        toremove = list(range(0, self.ndim))
        for pop_ii in tokeep:
            # Apply -1 factor to account for indexing in marginalize
            toremove.remove(pop_ii-1)
        return self.marginalize(toremove)

    def _counts_per_entry(self):
        """
        Counts per population for each entry in the fs.
        """
        ind = numpy.indices(self.shape)
        # Transpose the first access to the last, so ind[ii,jj,kk] = [ii,jj,kk]
        ind = ind.transpose(list(range(1,self.Npop+1))+[0])
        return ind

    def _total_per_entry(self):
        """
        Total derived alleles for each entry in the fs.
        """
        return numpy.sum(self._counts_per_entry(), axis=-1)

    def log(self):
        """
        Return the natural logarithm of the entries of the frequency spectrum.

        Only necessary because numpy.ma.log now fails to propagate extra
        attributes after numpy 1.10.
        """
        logfs = numpy.ma.log(self)
        logfs.folded = self.folded
        logfs.pop_ids = self.pop_ids
        logfs.extrap_x = self.extrap_x
        return logfs

    def reorder_pops(self, neworder):
        """
        Get Spectrum with populations in new order

        Returns new Spectrum with same number of populations, but in a different order

        neworder: Integer list defining new order of populations, indexing the orginal
                  populations from 1. Must contain all integers from 1 to number of pops.
        """
        if sorted(neworder) != [_+1 for _ in range(self.ndim)]:
            raise(ValueError("neworder argument misspecified"))
        newaxes = [_-1 for _ in neworder]
        fs = self.transpose(newaxes)
        if self.pop_ids:
            fs.pop_ids = [self.pop_ids[_] for _ in newaxes]

        return fs

    def fold(self):
        """
        Folded frequency spectrum
    
        The folded fs assumes that information on which allele is ancestral or
        derived is unavailable. Thus the fs is in terms of minor allele 
        frequency.  Note that this makes the fs into a "triangular" array.
    
        Note that if a masked cell is folded into non-masked cell, the
        destination cell is masked as well.

        Note also that folding is not done in-place. The return value is a new
        Spectrum object.
        """
        if self.folded:
            raise ValueError('Input Spectrum is already folded.')

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
        folded = numpy.ma.masked_array(self.data + reversed)
        folded.data[where_folded_out] = 0
    
        # Deal with those entries where assignment of the minor allele is
        # ambiguous.
        where_ambiguous = (total_per_entry == total_samples/2.)
        ambiguous = numpy.where(where_ambiguous, self, 0)
        folded += -0.5*ambiguous + 0.5*reverse_array(ambiguous)
    
        # Mask out the remains of the folding operation.
        final_mask = numpy.logical_or(final_mask, where_folded_out)

        outfs = Spectrum(folded, mask=final_mask, data_folded=True,
                         pop_ids=self.pop_ids)
        outfs.extrap_x = self.extrap_x
        return outfs

    def unfold(self):
        """
        Unfolded frequency spectrum
    
        It is assumed that each state of a SNP is equally likely to be
        ancestral.

        Note also that unfolding is not done in-place. The return value is a new
        Spectrum object.
        """
        if not self.folded:
            raise ValueError('Input Spectrum is not folded.')

        # Unfolding the data is easy.
        reversed_data = reverse_array(self.data)
        newdata = (self.data + reversed_data)/2.

        # Unfolding the mask is trickier. We want to preserve masking of entries
        # that were masked in the original Spectrum.
        # Which entries in the original Spectrum were masked solely because
        # they are incompatible with a folded Spectrum?
        total_samples = numpy.sum(self.sample_sizes)
        total_per_entry = self._total_per_entry()
        where_folded_out = total_per_entry > int(total_samples/2)

        newmask = numpy.logical_xor(self.mask, where_folded_out)
        newmask = numpy.logical_or(newmask, reverse_array(newmask))
    
        outfs = Spectrum(newdata, mask=newmask, data_folded=False, 
                         pop_ids=self.pop_ids)
        outfs.extrap_x = self.extrap_x
        return outfs

    def fixed_size_sample(self, nsamples, only_nonmasked=False):
        """
        Generate a resampled fs from the current one.

        nsamples: Number of samples to include in the new FS.
        only_nonmasked: If True, only SNPs from non-masked will be resampled. 
                        Otherwise, all SNPs will be used.
        """
        flat = self.flatten()
        if only_nonmasked:
            pvals = flat.data/flat.sum()
            pvals[flat.mask] = 0
        else:
            pvals = flat.data/flat.data.sum()
    
        sample = numpy.random.multinomial(int(nsamples), pvals)
        sample = sample.reshape(self.shape)
    
        return dadi.Spectrum(sample, mask=self.mask, pop_ids=self.pop_ids)

    def sample(self):
        """
        Generate a Poisson-sampled fs from the current one.

        Note: Entries where the current fs is masked will be masked in the
              output sampled fs.
        """
        import scipy.stats
        # These are entries where the sampling has no meaning, b/c fs is masked.
        bad_entries = self.mask
        # We convert to a 1-d array for passing into the sampler
        means = self.ravel().copy()
        # Filter out those bad entries.
        means[bad_entries.ravel()] = 1
        # Sample
        samp = scipy.stats.distributions.poisson.rvs(means, size=len(means))
        # Replace bad entries with zero
        samp[bad_entries.ravel()] = 0
        # Convert back to a properly shaped array
        samp = samp.reshape(self.shape)
        # Convert to a fs and mask the bad entries
        samp = Spectrum(samp, mask=self.mask, data_folded=self.folded,
                        pop_ids = self.pop_ids)
        return samp

    @staticmethod
    def from_ms_file(fid, average=True, mask_corners=True, return_header=False,
                     pop_assignments=None, pop_ids=None, bootstrap_segments=1):
        """
        Read frequency spectrum from file of ms output.

        fid: string with file name to read from or an open file object.
        average: If True, the returned fs is the average over the runs in the ms
                 file. If False, the returned fs is the sum.
        mask_corners: If True, mask the 'absent in all samples' and 'fixed in
                      all samples' entries.
        return_header: If True, the return value is (fs, (command,seeds), where
                       command and seeds are strings containing the ms
                       commandline and the seeds used.
        pop_assignments: If None, the assignments of samples to populations is
                         done automatically, using the assignment in the ms
                         command line. To manually assign populations, pass a
                         list of the from [6,8]. This example places
                         the first 6 samples into population 1, and the next 8
                         into population 2.
        pop_ids: Optional list of strings containing the population labels.
                 If pop_ids is None, labels will be "pop0", "pop1", ...
        bootstrap_segments: If bootstrap_segments is an integer greater than 1,
                            the data will be broken up into that many segments
                            based on SNP position. Instead of single FS, a list
                            of spectra will be returned, one for each segment.
        """
        newfile = False
        # Try to read from fid. If we can't, assume it's something that we can
        # use to open a file.
        if not hasattr(fid, 'read'):
            newfile = True
            fid = open(fid, 'r')

        # Parse the commandline
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
        if pop_assignments:
            num_pops = len(pop_assignments)
            pop_samples = pop_assignments

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
        if dimension > 4:
            bottom4 = bottom_l[4]
            top4 = top_l[4]
        if dimension > 5:
            bottom5 = bottom_l[5]
            top5 = top_l[5]
        
        all_data = [numpy.zeros(fs_shape, numpy.int_)
                    for boot_ii in range(bootstrap_segments)]
        for run_ii in range(runs):
            line = fid.readline()
            segsites = int(line.split()[-1])
            
            if segsites == 0:
                # Special case, need to read 3 lines to stay synced.
                for _ in range(3):
                    line = fid.readline()
                continue
            line = fid.readline()
            while not line.startswith('positions'):
                line = fid.readline()

            # Read SNP positions for creating bootstrap segments
            positions = [float(_) for _ in line.split()[1:]]
            # Where we should break our interval to create our bootstraps
            breakpts = numpy.linspace(0, 1, bootstrap_segments+1)
            # The indices that correspond to those breakpoints
            break_iis = numpy.searchsorted(positions, breakpts)
            # Correct for searchsorted behavior if last position is 1,
            # to ensure all SNPs are captured
            break_iis[-1] = len(positions)
        
            # Read the chromosomes in
            chromos = fid.read((segsites+1)*total_samples)
        
            # For each bootstrap segment, relevant SNPs run from start_ii:end_ii
            for boot_ii, (start_ii, end_ii) \
                    in enumerate(zip(break_iis[:-1], break_iis[1:])):
                # Use the data array corresponding to this bootstrap segment
                data = all_data[boot_ii]
                for snp in range(start_ii, end_ii):
                    # Slice to get all the entries that refer to a given SNP
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
                    elif dimension == 5:
                        data[this_snp[bottom0:top0].count('1'), 
                             this_snp[bottom1:top1].count('1'),
                             this_snp[bottom2:top2].count('1'),
                             this_snp[bottom3:top3].count('1'),
                             this_snp[bottom4:top4].count('1')] += 1
                    elif dimension == 6:
                        data[this_snp[bottom0:top0].count('1'), 
                             this_snp[bottom1:top1].count('1'),
                             this_snp[bottom2:top2].count('1'),
                             this_snp[bottom3:top3].count('1'),
                             this_snp[bottom4:top4].count('1'),
                             this_snp[bottom5:top5].count('1')] += 1
                    else:
                        # This is noticably slower, so we special case the cases
                        # above.
                        for dim_ii in range(dimension):
                            bottom = bottom_l[dim_ii]
                            top = top_l[dim_ii]
                            counts[dim_ii] = this_snp[bottom:top].count('1')
                        data[tuple(counts)] += 1
        
            # Read to the next iteration
            line = fid.readline()
            line = fid.readline()

        if newfile:
            fid.close()

        all_fs = [Spectrum(data, mask_corners=mask_corners, pop_ids=pop_ids)
                  for data in all_data]
        if average:
            all_fs = [fs/runs for fs in all_fs]

        # If we aren't setting up for bootstrapping, return fs, rather than a
        # list of length 1. (This ensures backward compatibility.)
        if bootstrap_segments == 1:
            all_fs = all_fs[0]

        if not return_header:
            return all_fs
        else:
            return all_fs, (command,seeds)

    @staticmethod
    def from_sfscode_file(fid, sites='all', average=True, mask_corners=True, 
                          return_header=False, pop_ids=None):
        """
        Read frequency spectrum from file of sfs_code output.

        fid: string with file name to read from or an open file object.
        sites: If sites=='all', return the fs of all sites. If sites == 'syn',
               use only synonymous mutations. If sites == 'nonsyn', use
               only non-synonymous mutations.
        average: If True, the returned fs is the average over the runs in the 
                 file. If False, the returned fs is the sum.
        mask_corners: If True, mask the 'absent in all samples' and 'fixed in
                      all samples' entries.
        return_header: If true, the return value is (fs, (command,seeds), where
                       command and seeds are strings containing the ms
                       commandline and the seeds used.
        pop_ids: Optional list of strings containing the population labels.
                 If pop_ids is None, labels will be "pop0", "pop1", ...
        """
        newfile = False
        # Try to read from fid. If we can't, assume it's something that we can
        # use to open a file.
        if not hasattr(fid, 'read'):
            newfile = True
            fid = open(fid, 'r')

        if sites == 'all':
            only_nonsyn, only_syn = False, False
        elif sites == 'syn':
            only_nonsyn, only_syn = False, True
        elif sites == 'nonsyn':
            only_nonsyn, only_syn = True, False
        else:
            raise ValueError("'sites' argument must be one of ('all', 'syn', "
                             "'nonsyn').")
        
        command = fid.readline()
        command_terms = command.split()
        
        runs = int(command_terms[2])
        num_pops = int(command_terms[1])
        
        # sfs_code default is 6 individuals, and I assume diploid pop
        pop_samples = [12] *  num_pops
        if '--sampSize' in command_terms or '-n' in command_terms:
            try:
                pop_flag = command_terms.index('--sampSize')
                pop_flag = command_terms.index('-n')
            except ValueError:
                pass
            pop_samples = [2*int(command_terms[pop_flag+ii])
                           for ii in range(1, 1+num_pops)]
        
        pop_samples = numpy.asarray(pop_samples)
        pop_fixed_str = [',%s.-1' % i for i in range(num_pops)]
        pop_count_str = [',%s.' % i for i in range(num_pops)]
        
        seeds = fid.readline()
        line = fid.readline()
        
        data = numpy.zeros(numpy.asarray(pop_samples)+1, numpy.int_)
        
        # line = //iteration...
        line = fid.readline()
        for iter_ii in range(runs):
            for ii in range(5):
                line = fid.readline()
        
            # It is possible for a mutation to be listed several times in the
            # output.  To accomodate this, I keep a dictionary of identities
            # for those mutations, and hold off processing them until I've seen
            # all mutations listed for the iteration.
            mut_dict = {}
            
            # Loop until this iteration ends.
            while not line.startswith('//') and line != '':
                split_line = line.split(';')
                if split_line[-1] == '\n':
                    split_line = split_line[:-1]
        
                # Loop over mutations on this line.
                for mut_ii, mutation in enumerate(split_line):
                    counts_this_mut = numpy.zeros(num_pops, numpy.int_)
        
                    split_mut = mutation.split(',')
        
                    # Exclude synonymous mutations
                    if only_nonsyn and split_mut[7] == '0':
                        continue
                    # Exclude nonsynonymous mutations
                    if only_syn and split_mut[7] == '1':
                        continue
        
                    ind_start = len(','.join(split_mut[:12]))
                    by_individual = mutation[ind_start:]
        
                    mut_id = ','.join(split_mut[:4] + split_mut[5:11])
        
                    # Count mutations in each population
                    for pop_ii,fixed_str,count_str\
                            in zip(range(num_pops), pop_fixed_str, 
                                   pop_count_str):
                        if fixed_str in by_individual:
                            counts_this_mut[pop_ii] = pop_samples[pop_ii]
                        else:
                            counts_this_mut[pop_ii] =\
                                    by_individual.count(count_str)
        
                    # Initialize the list that will track the counts for this
                    # mutation. Using setdefault means that it won't overwrite
                    # if there's already a list stored there.
                    mut_dict.setdefault(mut_id, [0]*num_pops)
                    for ii in range(num_pops):
                        if counts_this_mut[ii] > 0 and mut_dict[mut_id][ii] > 0:
                            sys.stderr.write('Contradicting counts between '
                                             'listings for mutation %s in '
                                             'population %i.' 
                                             % (mut_id, ii))
                        mut_dict[mut_id][ii] = max(counts_this_mut[ii], 
                                                   mut_dict[mut_id][ii])
        
                line = fid.readline()
        
            # Now apply all the mutations with fixations that we deffered.
            for mut_id, counts in mut_dict.items():
                if numpy.any(numpy.asarray(counts) > pop_samples):
                    sys.stderr.write('counts_this_mut > pop_samples: %s > '
                                     '%s\n%s\n' % (counts, pop_samples, mut_id))
                    counts = numpy.minimum(counts, pop_samples)
                data[tuple(counts)] += 1
        
        if newfile:
            fid.close()
        
        fs = Spectrum(data, mask_corners=mask_corners, pop_ids=pop_ids)
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
    
        Calculation is by the method of Weir and Cockerham _Evolution_ 38:1358
        (1984).  For a single SNP, the relevant formula is at the top of page
        1363. To combine results between SNPs, we use the weighted average
        indicated by equation 10.
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
        counts_per_pop = numpy.transpose(counts_per_pop, axes=list(range(1,r+1))+[0])
    
        # The last axis of ptwiddle is now the relative frequency of SNPs in
        # that bin in each of the populations.
        ptwiddle = 1.*counts_per_pop/ns
    
        # Note that pbar is of the same shape as fs...
        pbar = numpy.sum(ns*ptwiddle, axis=-1)/nsum
    
        # We need to use 'this_slice' to get the proper aligment between
        # ptwiddle and pbar.
        this_slice = [slice(None)]*r + [numpy.newaxis]
        s2 = numpy.sum(ns * (ptwiddle - pbar[tuple(this_slice)])**2, axis=-1)/((r-1)*nbar)
    
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
        an = numpy.sum(1./numpy.arange(1,n))
    
        return S/an
    
    def theta_L(self):
        """
        theta_L as defined by Zeng et al. "Statistical Tests for Detecting
        Positive Selection by Utilizing High-Frequency Variants" (2006)
        Genetics
    
        Note that is only sensible for 1-dimensional spectra.
        """
        if self.Npop != 1:
            raise ValueError("Only defined on a one-dimensional fs.")
    
        n = self.sample_sizes[0]
        return numpy.sum(numpy.arange(1,n)*self[1:n])/(n-1)

    def Zengs_E(self):
        """
        Zeng et al.'s E statistic.

        From Zeng et al. "Statistical Tests for Detecting Positive Selection by
        Utilizing High-Frequency Variants" (2006) Genetics
        """
        num = self.theta_L() - self.Watterson_theta()

        n = self.sample_sizes[0]

        # See after Eq. 3
        an = numpy.sum(1./numpy.arange(1,n))
        # See after Eq. 9
        bn = numpy.sum(1./numpy.arange(1,n)**2)
        s = self.S()

        # See immediately after Eq. 12
        theta = self.Watterson_theta()
        theta_sq = s*(s-1.)/(an**2 + bn)

        # Eq. 14
        var = (n/(2.*(n-1.)) - 1./an) * theta\
                + (bn/an**2 + 2.*(n/(n-1.))**2 * bn - 2*(n*bn-n+1.)/((n-1.)*an)
                   - (3.*n+1.)/(n-1.)) * theta_sq

        return num/numpy.sqrt(var)
    
    def pi(self):
        r"""
        Estimated expected number of pairwise differences between two
        chromosomes in the population.
    
        Note that this estimate includes a factor of sample_size/(sample_size-1)
        to make E(\hat{pi}) = theta.
        """
        if self.ndim != 1:
            raise ValueError("Only defined for a one-dimensional SFS.")
    
        n = self.sample_sizes[0]
        # sample frequencies p 
        p = numpy.arange(0,n+1,dtype=float)/n
        # This expression derives from Gillespie's _Population_Genetics:_A
        # _Concise_Guide_, 2nd edition, section 2.6.
        return n/(n-1.) * 2*numpy.ma.sum(self*p*(1-p))
    
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
    def _from_phi_1D_direct(n, xx, phi, mask_corners=True,
                            het_ascertained=None):
        """
        Compute sample Spectrum from population frequency distribution phi.

        See from_phi for explanation of arguments.
        """
        data = numpy.zeros(n+1)
        for ii in range(0,n+1):
            factorx = comb(n,ii) * xx**ii * (1-xx)**(n-ii)
            if het_ascertained == 'xx':
                factorx *= xx*(1-xx)
            data[ii] = trapz(factorx * phi, xx)
    
        return Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def _from_phi_1D_direct_inbreeding(n, xx, phi, Fx, mask_corners=True,
                                       ploidyx=2,het_ascertained=None):
        """
        Compute sample Spectrum from population frequency distribution phi
        plus inbreeding.
        
        See from_phi_inbreeding for explanation of arguments.
        """
        if n % ploidyx == 0:
            nInd = n/ploidyx
        else:
            raise ValueError('Number of chromosomes {0} is not divisible '
                             'by ploidy {1}.'.format(str(n),str(ploidyx)))
        data = numpy.zeros(n+1)
        alphax = xx*((1.0-Fx)/Fx)
        alphax[0],alphax[-1] = 1.0e-20*((1.0-Fx)/Fx), (1.0-1.0e-20)*((1.0-Fx)/Fx)
        betax  = (1.0-xx)*((1.0-Fx)/Fx)
        betax[0],betax[-1] = (1.0-1.0e-20)*((1.0-Fx)/Fx), 1.0e-20*((1.0-Fx)/Fx)
        for ii in range(0,n+1):
            factorx = [BetaBinomConvolution(ii,nInd,alphax[j],betax[j],ploidy=ploidyx) for j in range(0,len(xx))]
            if het_ascertained == 'xx':
                factorx *= xx*(1-xx)
            data[ii] = trapz(factorx*phi,xx)
        
        return Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def _from_phi_1D_analytic(n, xx, phi, mask_corners=True, 
                              divergent=False):
        """
        Compute sample Spectrum from population frequency distribution phi.

        This function uses analytic formulae for integrating over a 
        piecewise-linear approximation to phi.

        See from_phi for explanation of arguments.

        divergent: If True, the interval from xx[0] to xx[1] is modeled as
                   phi[1] * xx[1]/x. This captures the typical 1/x
                   divergence at x = 0.
        """
        # This function uses the result that 
        # \int_0^y \Gamma(a+b)/\Gamma(a) \Gamma(b) x^{a-1) (1-x)^{b-1} 
        # is betainc(a,b,y)
        # So the integral in analytic for a piece-wise linear phi.
        data = numpy.zeros(n+1)

        # Values for xx just slighly (~1e-16) outside the range [0,1] can cause
        # betainc calculation to fail.
        xx = numpy.minimum(numpy.maximum(xx, 0), 1.0)

        # Slopes of our linear-segments
        s = (phi[1:]-phi[:-1])/(xx[1:]-xx[:-1])
        # For the integration of the "constant" term in the piecewise linear
        # approximation of phi from each interval to the next.
        c1 = (phi[:-1] - s*xx[:-1])/(n+1)
        for d in range(0,n+1):
            c2 = s*(d+1)/((n+1)*(n+2))
            beta1 = betainc(d+1,n-d+1,xx)
            beta2 = betainc(d+2,n-d+1,xx)
            # Each entry is the value of the integral from one value of xx to
            # the next.
            entries = c1*(beta1[1:]-beta1[:-1]) + c2*(beta2[1:]-beta2[:-1])
            if divergent:
                entries[0] = phi[1]*xx[1]/d * betainc(d,n-d+1,xx[1])
            data[d] = numpy.sum(entries)
        fs = dadi.Spectrum(data, mask_corners=mask_corners)
        return fs

    @staticmethod
    def _from_phi_2D_direct(nx, ny, xx, yy, phi, mask_corners=True, 
                            het_ascertained=None):
        """
        Compute sample Spectrum from population frequency distribution phi.

        See from_phi for explanation of arguments.
        """
        # Calculate the 2D sfs from phi using the trapezoid rule for
        # integration.
        data = numpy.zeros((nx+1, ny+1))

        # Cache to avoid duplicated work.
        factorx_cache = {}
        for ii in range(0, nx+1):
            factorx = comb(nx, ii) * xx**ii * (1-xx)**(nx-ii)
            if het_ascertained == 'xx':
                factorx *= xx*(1-xx)
            factorx_cache[nx,ii] = factorx

        dx, dy = numpy.diff(xx), numpy.diff(yy)
        for jj in range(0,ny+1):
            factory = comb(ny, jj) * yy**jj * (1-yy)**(ny-jj)
            if het_ascertained == 'yy':
                factory *= yy*(1-yy)
            integrated_over_y = trapz(factory[numpy.newaxis,:]*phi, dx=dy)
            for ii in range(0, nx+1):
                factorx = factorx_cache[nx,ii]
                data[ii,jj] = trapz(factorx*integrated_over_y, dx=dx)

        return Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def _from_phi_2D_direct_inbreeding(nx, ny, xx, yy, phi, Fx, Fy, mask_corners=True,
                                       ploidyx=2,ploidyy=2,het_ascertained=None):
        """
        Compute sample Spectrum from population frequency distribution phi plus
        inbreeding.
        
        See from_phi_inbreeding for explanation of arguments.
        """
        # Calculate the 2D sfs from phi using the trapezoid rule for
        # integration.
        if nx % ploidyx == 0:
            nIndx = nx/ploidyx
        else:
            raise ValueError('Number of chromosomes {0} is not divisible '
                             'by ploidy {1} for pop 1.'.format(str(nx),str(ploidyx)))
        
        if ny % ploidyy == 0:
            nIndy = ny/ploidyy
        else:
            raise ValueError('Number of chromosomes {0} is not divisible '
                             'by ploidy {1} for pop 2.'.format(str(ny),str(ploidyy)))
        
        data = numpy.zeros((nx+1, ny+1))
        alphax = xx*((1.0-Fx)/Fx)
        alphax[0],alphax[-1] = 1.0e-20*((1.0-Fx)/Fx), (1.0-1.0e-20)*((1.0-Fx)/Fx)
        betax = (1.0-xx)*((1.0-Fx)/Fx)
        betax[0],betax[-1] = (1.0-1.0e-20)*((1.0-Fx)/Fx), 1.0e-20*((1.0-Fx)/Fx)
        alphay = yy*((1.0-Fy)/Fy)
        alphay[0],alphay[-1] = 1.0e-20*((1.0-Fy)/Fy), (1.0-1.0e-20)*((1.0-Fy)/Fy)
        betay = (1.0-yy)*((1.0-Fy)/Fy)
        betay[0],betay[-1] = (1.0-1.0e-20)*((1.0-Fy)/Fy), 1.0e-20*((1.0-Fy)/Fy)
        # Cache to avoid duplicated work
        factorx_cache = {}
        for ii in range(0,nx+1):
            factorx = numpy.array([BetaBinomConvolution(ii,nIndx,alphax[j],betax[j],ploidy=ploidyx) for j in range(0,len(xx))])
            if het_ascertained == 'xx':
                factorx *= xx*(1-xx)
            factorx_cache[nx,ii] = factorx
        
        dx,dy = numpy.diff(xx), numpy.diff(yy)
        for jj in range(0,ny+1):
            factory = numpy.array([BetaBinomConvolution(jj,nIndy,alphay[j],betay[j],ploidy=ploidyy) for j in range(0,len(yy))])
            if het_ascertained == 'yy':
                factory *= yy*(1-yy)
            integrated_over_y = trapz(factory[numpy.newaxis,:]*phi, dx=dy)
            for ii in range(0,nx+1):
                factorx = factorx_cache[nx,ii]
                data[ii,jj] = trapz(factorx*integrated_over_y, dx=dx)
        
        return Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def _from_phi_2D_admix_props(nx, ny, xx, yy, phi, mask_corners=True, 
                                 admix_props=None):
        """
        Compute sample Spectrum from population frequency distribution phi.

        See from_phi for explanation of arguments.
        """
        xadmix = admix_props[0][0]*xx[:,nuax] + admix_props[0][1]*yy[nuax,:]
        yadmix = admix_props[1][0]*xx[:,nuax] + admix_props[1][1]*yy[nuax,:]

        # Calculate the 2D sfs from phi using the trapezoid rule for
        # integration.
        data = numpy.zeros((nx+1, ny+1))
        
        # Cache to avoid duplicated work.
        factorx_cache = {}
        for ii in range(0, nx+1):
            factorx = comb(nx, ii) * xadmix**ii * (1-xadmix)**(nx-ii)
            factorx_cache[nx,ii] = factorx
    
        dx, dy = numpy.diff(xx), numpy.diff(yy)
        for jj in range(0,ny+1):
            factory = comb(ny, jj) * yadmix**jj * (1-yadmix)**(ny-jj)
            for ii in range(0, nx+1):
                integrated_over_y = trapz(factorx_cache[nx,ii] * factory * phi,
                                          dx=dy)
                data[ii,jj] = trapz(integrated_over_y, dx=dx)
    
        fs = Spectrum(data, mask_corners=mask_corners)
        fs.extrap_x = xx[1]
        return fs

    #@staticmethod
    #def _from_phi_2D_analytic(nx, ny, xx, yy, phi, mask_corners=True):
    #    """
    #    Compute sample Spectrum from population frequency distribution phi.

    #    This function uses analytic formulae for integrating over a 
    #    piecewise-linear approximation to phi.

    #    See from_phi for explanation of arguments.
    #    """
    #    data = numpy.zeros((nx+1,ny+1))

    #    xx = numpy.minimum(numpy.maximum(xx, 0), 1.0)
    #    yy = numpy.minimum(numpy.maximum(yy, 0), 1.0)

    #    beta_cache_xx = {}
    #    for ii in range(0, nx+1):
    #        beta_cache_xx[ii+1,nx-ii+1] = betainc(ii+1,nx-ii+1,xx)
    #        beta_cache_xx[ii+2,nx-ii+1] = betainc(ii+2,nx-ii+1,xx)

    #    s_yy = (phi[:,1:]-phi[:,:-1])/(yy[nuax,1:]-yy[nuax,:-1])
    #    c1_yy = (phi[:,:-1] - s_yy*yy[nuax,:-1])/(ny+1)
    #    for jj in range(0, ny+1):
    #        c2_yy = s_yy*(jj+1)/((ny+1)*(ny+2))
    #        beta1_yy = betainc(jj+1,ny-jj+1,yy)
    #        beta2_yy = betainc(jj+2,ny-jj+1,yy)
    #        over_y = numpy.sum(c1_yy*(beta1_yy[nuax,1:]-beta1_yy[nuax,:-1])
    #                           + c2_yy*(beta2_yy[nuax,1:]-beta2_yy[nuax,:-1]),
    #                           axis=-1)

    #        s_xx = (over_y[1:]-over_y[:-1])/(xx[1:]-xx[:-1])
    #        c1_xx = (over_y[:-1] - s_xx*xx[:-1])/(nx+1)
    #        for ii in range(0, nx+1):
    #            c2_xx = s_xx*(ii+1)/((nx+1)*(nx+2))
    #            beta1_xx = beta_cache_xx[ii+1,nx-ii+1]
    #            beta2_xx = beta_cache_xx[ii+2,nx-ii+1]
    #            value = numpy.sum(c1_xx*(beta1_xx[1:]-beta1_xx[:-1])
    #                              + c2_xx*(beta2_xx[1:]-beta2_xx[:-1]))
    #            data[ii,jj] = value
    #    fs = dadi.Spectrum(data, mask_corners=mask_corners)
    #    return fs

    @staticmethod
    def _from_phi_2D_linalg(nx, ny, xx, yy, phi, mask_corners=True, raw=False):
        """
        Compute sample Spectrum from population frequency distribution phi.

        This function uses analytic formulae for integrating over a 
        piecewise-linear approximation to phi.

        See from_phi for explanation of arguments.

        raw: If True, return data as a numpy array, not a Spectrum object
        """
        if not (len(xx) == len(yy) and np.allclose(xx,yy)):
            raise ValueError('Must have xx==yy to use linear algebra calculation of FS from phi.')
        dbeta1_xx, dbeta2_xx = cached_dbeta(nx, xx)
        dbeta1_yy, dbeta2_yy = cached_dbeta(ny, yy)

        # For a somewhat less terse example of this code, see the
        # (archived) _from_phi_2D_analytic function.
        s_yy = (phi[:,1:] - phi[:,:-1])/(yy[nuax,1:] - yy[nuax,:-1])
        c1_yy = (phi[:,:-1] - s_yy*yy[nuax,:-1])/(ny+1)

        term1_yy = np.dot(dbeta1_yy,c1_yy.T)
        term2_yy = np.dot(dbeta2_yy, s_yy.T)
        term2_yy *= np.arange(1,ny+2)[:,np.newaxis]/((ny+1)*(ny+2))
        over_y_all = term1_yy + term2_yy

        s_xx_all = (over_y_all[:,1:] - over_y_all[:,:-1])/(xx[1:]-xx[:-1])
        c1_xx_all = (over_y_all[:,:-1] - s_xx_all*xx[:-1])/(nx+1)

        term1_all = np.dot(dbeta1_xx, c1_xx_all.T)
        term2_all = np.dot(dbeta2_xx, s_xx_all.T)
        term2_all *=  np.arange(1,nx+2)[:,np.newaxis]/((nx+1)*(nx+2))

        data = term1_all + term2_all

        if raw:
            return data
        else:
            return dadi.Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def _from_phi_3D_direct(nx, ny, nz, xx, yy, zz, phi, mask_corners=True,
                     het_ascertained=None):
        """
        Compute sample Spectrum from population frequency distribution phi.

        See from_phi for explanation of arguments.
        """
        data = numpy.zeros((nx+1, ny+1, nz+1))
    
        dx, dy, dz = numpy.diff(xx), numpy.diff(yy), numpy.diff(zz)
        half_dx = dx/2.0
    
        # We cache these calculations...
        factorx_cache, factory_cache = {}, {}
        for ii in range(0, nx+1):
            factorx = comb(nx, ii) * xx**ii * (1-xx)**(nx-ii)
            if het_ascertained == 'xx':
                factorx *= xx*(1-xx)
            factorx_cache[nx,ii] = factorx
        for jj in range(0, ny+1):
            factory = comb(ny, jj) * yy**jj * (1-yy)**(ny-jj)
            if het_ascertained == 'yy':
                factory *= yy*(1-yy)
            factory_cache[ny,jj] = factory[nuax,:]
    
        for kk in range(0, nz+1):
            factorz = comb(nz, kk) * zz**kk * (1-zz)**(nz-kk)
            if het_ascertained == 'zz':
                factorz *= zz*(1-zz)
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
    def _from_phi_4D_direct(nx, ny, nz, na, xx, yy, zz, aa, phi, mask_corners=True,
                     het_ascertained=None):
        """
        Compute sample Spectrum from population frequency distribution phi.

        See from_phi for explanation of arguments.
        """
        data = numpy.zeros((nx+1, ny+1, nz+1, na+1))
    
        dx, dy, dz, da = numpy.diff(xx), numpy.diff(yy), numpy.diff(zz), np.diff(aa)
        half_dx = dx/2.0
    
        # We cache these calculations...
        factorx_cache, factory_cache, factorz_cache = {}, {}, {}
        for ii in range(0, nx+1):
            factorx = comb(nx, ii) * xx**ii * (1-xx)**(nx-ii)
            if het_ascertained == 'xx':
                factorx *= xx*(1-xx)
            factorx_cache[nx,ii] = factorx
        for jj in range(0, ny+1):
            factory = comb(ny, jj) * yy**jj * (1-yy)**(ny-jj)
            if het_ascertained == 'yy':
                factory *= yy*(1-yy)
            factory_cache[ny,jj] = factory[nuax,:]
        for kk in range(0, nz+1):
            factorz = comb(nz, kk) * zz**kk * (1-zz)**(nz-kk)
            if het_ascertained == 'zz':
                factorz *= zz*(1-zz)
            factorz_cache[nz,kk] = factorz[nuax,nuax,:]
    
        for ll in range(0, na+1):
            factora = comb(na, ll) * aa**ll * (1-aa)**(na-ll)
            if het_ascertained == 'aa':
                factora *= aa*(1-aa)
            over_a = trapz(factora[nuax,nuax,nuax,:] * phi, dx=da)
            for kk in range(0, nz+1):
                factorz = factorz_cache[nz,kk]
                over_z = trapz(factorz * over_a, dx=dz)
                for jj in range(0, ny+1):
                    factory = factory_cache[ny,jj]
                    over_y = trapz(factory * over_z, dx=dy)
                    for ii in range(0, nx+1):
                        factorx = factorx_cache[nx,ii]
                        # It's faster here to do the trapezoid rule explicitly
                        # rather than using SciPy's more general routine.
                        integrand = factorx * over_y
                        ans = numpy.sum(half_dx * (integrand[1:]+integrand[:-1]))
                        data[ii,jj,kk,ll] = ans
    
        return Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def _from_phi_3D_direct_inbreeding(nx, ny, nz, xx, yy, zz, phi, Fx, Fy, Fz,
                                       mask_corners=True, ploidyx=2, ploidyy=2, ploidyz=2,
                                       het_ascertained=None):
        """
        Compute sample Spectrum from population frequency distribution phi
        plus inbreeding.
        
        See from_phi_inbreeding for explanation of arguments.
        """
        # Calculate the 2D sfs from phi using the trapezoid rule for
        # integration.
        if nx % ploidyx == 0:
            nIndx = nx/ploidyx
        else:
            raise ValueError('Number of chromosomes {0} is not divisible '
                             'by ploidy {1} for pop 1.'.format(str(nx),str(ploidyx)))
        
        if ny % ploidyy == 0:
            nIndy = ny/ploidyy
        else:
            raise ValueError('Number of chromosomes {0} is not divisible '
                             'by ploidy {1} for pop 2.'.format(str(ny),str(ploidyy)))
        
        if nz % ploidyz == 0:
            nIndz = nz/ploidyz
        else:
            raise ValueError('Number of chromosomes {0} is not divisible '
                             'by ploidy {1} for pop 3.'.format(str(nz),str(ploidyz)))
        
        data = numpy.zeros((nx+1, ny+1, nz+1))
        dx, dy, dz = numpy.diff(xx), numpy.diff(yy), numpy.diff(zz)
        half_dx = dx/2.0
        alphax = xx*((1.0-Fx)/Fx)
        alphax[0],alphax[-1] = 1.0e-20*((1.0-Fx)/Fx), (1.0-1.0e-20)*((1.0-Fx)/Fx)
        betax = (1.0-xx)*((1.0-Fx)/Fx)
        betax[0],betax[-1] = (1.0-1.0e-20)*((1.0-Fx)/Fx), 1.0e-20*((1.0-Fx)/Fx)
        alphay = yy*((1.0-Fy)/Fy)
        alphay[0],alphay[-1] = 1.0e-20*((1.0-Fy)/Fy), (1.0-1.0e-20)*((1.0-Fy)/Fy)
        betay = (1.0-yy)*((1.0-Fy)/Fy)
        betay[0],betay[-1] = (1.0-1.0e-20)*((1.0-Fy)/Fy), 1.0e-20*((1.0-Fy)/Fy)
        alphaz = zz*((1.0-Fz)/Fz)
        alphaz[0],alphaz[-1] = 1.0e-20*((1.0-Fz)/Fz), (1.0-1.0e-20)*((1.0-Fz)/Fz)
        betaz = (1.0-zz)*((1.0-Fz)/Fz)
        betaz[0],betaz[-1] = (1.0-1.0e-20)*((1.0-Fz)/Fz), 1.0e-20*((1.0-Fz)/Fz)
        
        # We cache these calculations...
        factorx_cache, factory_cache = {}, {}
        for ii in range(0, nx+1):
            factorx = numpy.array([BetaBinomConvolution(ii,nIndx,alphax[j],betax[j],ploidy=ploidyx) for j in range(0,len(xx))])
            if het_ascertained == 'xx':
                factorx *= xx*(1-xx)
            factorx_cache[nx,ii] = factorx
        for jj in range(0, ny+1):
            factory = numpy.array([BetaBinomConvolution(jj,nIndy,alphay[j],betay[j],ploidy=ploidyy) for j in range(0,len(yy))])
            if het_ascertained == 'yy':
                factory *= yy*(1-yy)
            factory_cache[ny,jj] = factory[nuax,:]
        
        for kk in range(0, nz+1):
            factorz = numpy.array([BetaBinomConvolution(kk,nIndz,alphaz[j],betaz[j],ploidy=ploidyz) for j in range(0,len(zz))])
            if het_ascertained == 'zz':
                factorz *= zz*(1-zz)
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
    def _from_phi_3D_admix_props(nx, ny, nz, xx, yy, zz, phi, mask_corners=True,
                                 admix_props=None):
        """
        Compute sample Spectrum from population frequency distribution phi.

        See from_phi for explanation of arguments.
        """
        if admix_props is None:
            admix_props = ((1,0,0), (0,1,0), (0,0,1))

        xadmix = admix_props[0][0]*xx[:,nuax,nuax]\
                + admix_props[0][1]*yy[nuax,:,nuax]\
                + admix_props[0][2]*zz[nuax,nuax,:]
        yadmix = admix_props[1][0]*xx[:,nuax,nuax]\
                + admix_props[1][1]*yy[nuax,:,nuax]\
                + admix_props[1][2]*zz[nuax,nuax,:]
        zadmix = admix_props[2][0]*xx[:,nuax,nuax]\
                + admix_props[2][1]*yy[nuax,:,nuax]\
                + admix_props[2][2]*zz[nuax,nuax,:]

        data = numpy.zeros((nx+1, ny+1, nz+1))
    
        dx, dy, dz = numpy.diff(xx), numpy.diff(yy), numpy.diff(zz)
    
        # We cache these calculations...
        factorx_cache, factory_cache, factorz_cache = {}, {}, {}
        for ii in range(0, nx+1):
            factorx = comb(nx, ii) * xadmix**ii * (1-xadmix)**(nx-ii)
            factorx_cache[nx,ii] = factorx

        for jj in range(0, ny+1):
            factory = comb(ny, jj) * yadmix**jj * (1-yadmix)**(ny-jj)
            factory_cache[ny,jj] = factory

        for kk in range(0, nz+1):
            factorz = comb(nz, kk) * zadmix**kk * (1-zadmix)**(nz-kk)
            factorz_cache[nz,kk] = factorz
    
        for kk in range(0, nz+1):
            factorz = factorz_cache[nz,kk]
            for jj in range(0, ny+1):
                factory = factory_cache[ny,jj]
                for ii in range(0, nx+1):
                    factorx = factorx_cache[nx,ii]
                    over_z = trapz(factorz * factory * factorx * phi, dx=dz)
                    over_y = trapz(over_z, dx=dy)
                    data[ii,jj,kk] = trapz(over_y, dx=dx)
    
        return Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def _from_phi_3D_linalg(nx, ny, nz, xx, yy, zz, phi, mask_corners=True, raw=False):
        """
        Compute sample Spectrum from population frequency distribution phi.

        This function uses analytic formulae for integrating over a 
        piecewise-linear approximation to phi.

        See from_phi for explanation of arguments.

        raw: If True, return data as a numpy array, not a Spectrum object
        """
        data = numpy.zeros((nx+1,ny+1,nz+1))

        dbeta1_zz, dbeta2_zz = cached_dbeta(nz, zz)

        # Quick testing suggests that doing the x direction first for better
        # memory alignment isn't worth much.
        s_zz = (phi[:,:,1:]-phi[:,:,:-1])/(zz[nuax,nuax,1:]-zz[nuax,nuax,:-1])
        c1_zz = (phi[:,:,:-1] - s_zz*zz[nuax,nuax,:-1])/(nz+1)
        # These calculations can be done without this for loop, but the
        # four-dimensional intermediate results consume massive amounts of RAM,
        # which makes the for loop faster for large systems.
        for kk in range(0, nz+1):
            # In testing, these two np.dot lines occupy 2/3 the time, so further
            # speedup will be difficult
            term1 = np.dot(c1_zz, dbeta1_zz[kk])
            term2 = np.dot(s_zz, dbeta2_zz[kk])
            term2 *= (kk+1)/((nz+1)*(nz+2))
            over_z = term1 + term2

            sub_fs = Spectrum._from_phi_2D_linalg(nx, ny, xx, yy, over_z, raw=True)
            data[:,:,kk] = sub_fs.data

        if raw:
            return data
        else:
            return dadi.Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def _from_phi_5D_linalg(nx, ny, nz, na, nb, xx, yy, zz, aa, bb, phi, mask_corners=True):
        """
        Compute sample Spectrum from population frequency distribution phi.

        This function uses analytic formulae for integrating over a 
        piecewise-linear approximation to phi.

        See from_phi for explanation of arguments.
        """
        data = numpy.zeros((nx+1,ny+1,nz+1,na+1,nb+1))

        dbeta1_bb, dbeta2_bb = cached_dbeta(nb, bb)

        s_bb = (phi[:,:,:,:,1:]-phi[:,:,:,:,:-1])/(bb[nuax,nuax,nuax,nuax,1:]-bb[nuax,nuax,nuax,nuax:-1])
        c1_bb = (phi[:,:,:,:,:-1] - s_bb*bb[nuax,nuax,nuax,nuax,:-1])/(nb+1)
        for mm in range(0, nb+1):
            term1 = np.dot(c1_bb, dbeta1_bb[mm])
            term2 = np.dot(s_bb, dbeta2_bb[mm])
            term2 *= (mm+1)/((nb+1)*(nb+2))
            over_b = term1 + term2

            sub_fs = Spectrum._from_phi_4D_linalg(nx, ny, nz, na, xx, yy, zz, aa, over_b, raw=True)
            data[:,:,:,:,mm] = sub_fs.data

        fs = dadi.Spectrum(data, mask_corners=mask_corners)
        return fs

    @staticmethod
    def _from_phi_4D_linalg(nx, ny, nz, na, xx, yy, zz, aa, phi, mask_corners=True, raw=False):
        """
        Compute sample Spectrum from population frequency distribution phi.

        This function uses analytic formulae for integrating over a 
        piecewise-linear approximation to phi.

        See from_phi for explanation of arguments.

        raw: If True, return data as a numpy array, not a Spectrum object
        """
        data = numpy.zeros((nx+1,ny+1,nz+1,na+1))

        dbeta1_aa, dbeta2_aa = cached_dbeta(na, aa)

        s_aa = (phi[:,:,:,1:]-phi[:,:,:,:-1])/(aa[nuax,nuax,nuax,1:]-aa[nuax,nuax,nuax:-1])
        c1_aa = (phi[:,:,:,:-1] - s_aa*aa[nuax,nuax,nuax,:-1])/(na+1)
        for ll in range(0, na+1):
            term1 = np.dot(c1_aa, dbeta1_aa[ll])
            term2 = np.dot(s_aa, dbeta2_aa[ll])
            term2 *= (ll+1)/((na+1)*(na+2))
            over_a = term1 + term2

            sub_fs = Spectrum._from_phi_3D_linalg(nx, ny, nz, xx, yy, zz, over_a, raw=True)
            data[:,:,:,ll] = sub_fs.data

        if raw:
            return data
        else:
            return dadi.Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def _from_phi_4D_admix_props(nx, ny, nz, na, xx, yy, zz, aa, phi, mask_corners=True,
                                 admix_props=None):
        """
        Compute sample Spectrum from population frequency distribution phi.

        See from_phi for explanation of arguments.
        """
        if admix_props is None:
            admix_props = ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1))

        xadmix = admix_props[0][0]*xx[:,nuax,nuax,nuax]\
                + admix_props[0][1]*yy[nuax,:,nuax,nuax]\
                + admix_props[0][2]*zz[nuax,nuax,:,nuax]\
                + admix_props[0][3]*aa[nuax,nuax,nuax,:]
        yadmix = admix_props[1][0]*xx[:,nuax,nuax,nuax]\
                + admix_props[1][1]*yy[nuax,:,nuax,nuax]\
                + admix_props[1][2]*zz[nuax,nuax,:,nuax]\
                + admix_props[1][3]*aa[nuax,nuax,nuax,:]
        zadmix = admix_props[2][0]*xx[:,nuax,nuax,nuax]\
                + admix_props[2][1]*yy[nuax,:,nuax,nuax]\
                + admix_props[2][2]*zz[nuax,nuax,:,nuax]\
                + admix_props[2][3]*aa[nuax,nuax,nuax,:]
        aadmix = admix_props[3][0]*xx[:,nuax,nuax,nuax]\
                + admix_props[3][1]*yy[nuax,:,nuax,nuax]\
                + admix_props[3][2]*zz[nuax,nuax,:,nuax]\
                + admix_props[3][3]*aa[nuax,nuax,nuax,:]

        data = numpy.zeros((nx+1, ny+1, nz+1, na+1))
    
        dx, dy, dz, da = numpy.diff(xx), numpy.diff(yy), numpy.diff(zz), numpy.diff(aa)
    
        # We cache these calculations...
        factorx_cache, factory_cache, factorz_cache, factora_cache = {}, {}, {}, {}
        for ii in range(0, nx+1):
            factorx = comb(nx, ii) * xadmix**ii * (1-xadmix)**(nx-ii)
            factorx_cache[nx,ii] = factorx

        for jj in range(0, ny+1):
            factory = comb(ny, jj) * yadmix**jj * (1-yadmix)**(ny-jj)
            factory_cache[ny,jj] = factory

        for kk in range(0, nz+1):
            factorz = comb(nz, kk) * zadmix**kk * (1-zadmix)**(nz-kk)
            factorz_cache[nz,kk] = factorz

        for ll in range(0, na+1):
            factora = comb(na, ll) * aadmix**ll * (1-aadmix)**(na-ll)
            factora_cache[na,ll] = factora
    
        for ll in range(0, na+1):
            factora = factora_cache[na,ll]
            for kk in range(0, nz+1):
                factorz = factorz_cache[nz,kk]
                for jj in range(0, ny+1):
                    factory = factory_cache[ny,jj]
                    for ii in range(0, nx+1):
                        factorx = factorx_cache[nx,ii]
                        over_a = trapz(factora * factorz * factory * factorx * phi, dx=da)
                        over_z = trapz(over_a, dx=dz)
                        over_y = trapz(over_z, dx=dy)
                        data[ii,jj,kk,ll] = trapz(over_y, dx=dx)
    
        return Spectrum(data, mask_corners=mask_corners)

    @staticmethod
    def from_phi(phi, ns, xxs, mask_corners=True, 
                 pop_ids=None, admix_props=None, het_ascertained=None, 
                 force_direct=False):
        """
        Compute sample Spectrum from population frequency distribution phi.

        phi: P-dimensional population frequency distribution.
        ns: Sequence of P sample sizes for each population.
        xxs: Sequence of P one-dimesional grids on which phi is defined.
        mask_corners: If True, resulting FS is masked in 'absent' and 'fixed'
                      entries.
        pop_ids: Optional list of strings containing the population labels.
                 If pop_ids is None, labels will be "pop0", "pop1", ...
        admix_props: Admixture proportions for sampled individuals. For example,
                     if there are two populations, and individuals from the
                     first pop are admixed with fraction f from the second
                     population, then admix_props=((1-f,f),(0,1)). For three
                     populations, the no-admixture setting is
                     admix_props=((1,0,0),(0,1,0),(0,0,1)). 
                     (Note that this option also forces direct integration,
                     which may be less accurate than the semi-analytic
                     method.)
        het_ascertained: If 'xx', then FS is calculated assuming that SNPs have
 	                 population 2 or 3, respectively.
                         been ascertained by being heterozygous in one
                         individual from population 1. (This individual is
                         *not* in the current sample.) If 'yy' or 'zz', it
                         assumed that the ascertainment individual came from
                         population 2 or 3, respectively.
                         (Note that this option also forces direct integration,
                         which may be less accurate than the semi-analytic
                         method. This could be fixed if there is interest. Note
                         also that this option cannot be used simultaneously
                         with admix_props.)
        force_direct: Forces integration to use older direct integration method,
                      rather than using analytic integration of sampling 
                      formula.
        """
        if admix_props and not numpy.allclose(numpy.sum(admix_props, axis=1),1):
            raise ValueError('Admixture proportions {0} must sum to 1 for all '
                             'populations.' .format(str(admix_props)))
        if not phi.ndim == len(ns) == len(xxs):
            raise ValueError('Dimensionality of phi and lengths of ns and xxs '
                             'do not all agree.')
        if het_ascertained and not het_ascertained in ['xx','yy','zz']:
            raise ValueError("If used, het_ascertained must be 'xx', 'yy', or "
                             "'zz'.")

        if admix_props and het_ascertained:
            error = """admix_props and het_ascertained options cannot be used 
            simultaneously. Instead, please use the PhiManip methods to 
            implement admixture. If this proves inappropriate for your use, 
            contact the the dadi developers, as it may be possible to support
            both options simultaneously in the future."""
            raise NotImplementedError(error)

        if phi.ndim == 1:
            if not het_ascertained and not force_direct:
                fs = Spectrum._from_phi_1D_analytic(ns[0], xxs[0], phi,
                                                    mask_corners)
            else:
                fs = Spectrum._from_phi_1D_direct(ns[0], xxs[0], phi, 
                                                  mask_corners, het_ascertained)
        elif phi.ndim == 2:
            if not het_ascertained and not admix_props and not force_direct:
                fs = Spectrum._from_phi_2D_linalg(ns[0], ns[1], 
                                                  xxs[0], xxs[1], phi,
                                                  mask_corners)
            elif not admix_props:
                fs = Spectrum._from_phi_2D_direct(ns[0], ns[1], xxs[0], xxs[1], 
                                                  phi, mask_corners, 
                                                  het_ascertained)
            else:
                fs = Spectrum._from_phi_2D_admix_props(ns[0], ns[1], 
                                                      xxs[0], xxs[1], 
                                                      phi, mask_corners, 
                                                      admix_props)
        elif phi.ndim == 3:
            if not het_ascertained and not admix_props and not force_direct:
                fs = Spectrum._from_phi_3D_linalg(ns[0], ns[1], ns[2], 
                                                  xxs[0], xxs[1], xxs[2],
                                                  phi, mask_corners)
            elif not admix_props:
                fs = Spectrum._from_phi_3D_direct(ns[0], ns[1], ns[2], 
                                                  xxs[0], xxs[1], xxs[2], 
                                                  phi, mask_corners, 
                                                  het_ascertained)
            else:
                fs = Spectrum._from_phi_3D_admix_props(ns[0], ns[1], ns[2], 
                                                       xxs[0], xxs[1], xxs[2], 
                                                       phi, mask_corners, 
                                                       admix_props)
        elif phi.ndim == 4:
            if not het_ascertained and not admix_props and not force_direct:
                fs = Spectrum._from_phi_4D_linalg(ns[0], ns[1], ns[2], ns[3],
                                                  xxs[0], xxs[1], xxs[2], xxs[3],
                                                  phi, mask_corners)
            elif not admix_props:
                fs = Spectrum._from_phi_4D_direct(ns[0], ns[1], ns[2], ns[3],
                                                  xxs[0], xxs[1], xxs[2], xxs[3],
                                                  phi, mask_corners, 
                                                  het_ascertained)
            else:
                fs = Spectrum._from_phi_4D_admix_props(ns[0], ns[1], ns[2], ns[3],
                                                       xxs[0], xxs[1], xxs[2], xxs[3],
                                                       phi, mask_corners, 
                                                       admix_props)
        elif phi.ndim == 5:
            if not het_ascertained and not admix_props and not force_direct:
                fs = Spectrum._from_phi_5D_linalg(ns[0], ns[1], ns[2], ns[3], ns[4],
                                                  xxs[0], xxs[1], xxs[2], xxs[3], xxs[4],
                                                  phi, mask_corners)
        else:
            raise ValueError('Only implemented for dimensions 1-5.')
        fs.pop_ids = pop_ids
        # Record value to use for extrapolation. This is the first grid point,
        # which is where new mutations are introduced. Note that extrapolation
        # will likely fail if grids differ between dimensions.
        fs.extrap_x = xxs[0][1]
        for xx in xxs[1:]:
            if not xx[1] == fs.extrap_x:
                logger.warning('Spectrum calculated from phi different grids for '
                            'different dimensions. Extrapolation may fail.')
        return fs

    @staticmethod
    def from_phi_inbreeding(phi, ns, xxs, Fs, ploidys, mask_corners=True, 
                            pop_ids=None, admix_props=None,
                            het_ascertained=None, force_direct=True):
        """
        Compute sample Spectrum from population frequency distribution phi
        plus inbreeding.
        
        phi: P-dimensional population frequency distribution.
        ns: Sequence of P sample sizes for each population.
        xxs: Sequence of P one-dimesional grids on which phi is defined.
        Fs: Sequence of P inbreeding coefficients for each population.
        ploidys: Sequence of P ploidy levels for each population.
        mask_corners: If True, resulting FS is masked in 'absent' and 'fixed'
                      entries.
        pop_ids: Optional list of strings containing the population labels.
                 If pop_ids is None, labels will be "pop0", "pop1", ...
        admix_props: Admixture proportions for sampled individuals. For example,
                     if there are two populations, and individuals from the
                     first pop are admixed with fraction f from the second
                     population, then admix_props=((1-f,f),(0,1)). For three
                     populations, the no-admixture setting is
                     admix_props=((1,0,0),(0,1,0),(0,0,1)). 
                     (Note that this option also forces direct integration,
                     which may be less accurate than the semi-analytic
                     method.)
        het_ascertained: If 'xx', then FS is calculated assuming that SNPs have
                         been ascertained by being heterozygous in one
                         individual from population 1. (This individual is
                         *not* in the current sample.) If 'yy' or 'zz', it
                         assumed that the ascertainment individual came from
                         population 2 or 3, respectively.
                         (Note that this option also forces direct integration,
                         which may be less accurate than the semi-analytic
                         method. This could be fixed if there is interest. Note
                         also that this option cannot be used simultaneously
                         with admix_props.)
        force_direct: Forces integration to use older direct integration method,
                      rather than using analytic integration of sampling 
                      formula.
        """
        if admix_props and not numpy.allclose(numpy.sum(admix_props, axis=1),1):
            raise ValueError('Admixture proportions {0} must sum to 1 for all '
                             'populations.' .format(str(admix_props)))
        if not phi.ndim == len(ns) == len(xxs) == len(Fs) == len(ploidys):
            raise ValueError('Dimensionality of phi and lengths of ns, xxs, ploidys, and Fs '
                             'do not all agree.')
        if het_ascertained and not het_ascertained in ['xx','yy','zz']:
            raise ValueError("If used, het_ascertained must be 'xx', 'yy', or "
                             "'zz'.")

        if admix_props and het_ascertained:
            error = """admix_props and het_ascertained options cannot be used 
            simultaneously. Instead, please use the PhiManip methods to 
            implement admixture. If this proves inappropriate for your use, 
            contact the the dadi developers, as it may be possible to support
            both options simultaneously in the future."""
            raise NotImplementedError(error)

        if phi.ndim == 1:
            fs = Spectrum._from_phi_1D_direct_inbreeding(ns[0], xxs[0], phi, Fs[0],
                                                         mask_corners, ploidys[0],
                                                         het_ascertained)
        elif phi.ndim == 2:
            fs = Spectrum._from_phi_2D_direct_inbreeding(ns[0], ns[1], xxs[0], xxs[1],
                                                         phi, Fs[0], Fs[1], mask_corners,
                                                         ploidys[0], ploidys[1],
                                                         het_ascertained)
        elif phi.ndim == 3:
            fs = Spectrum._from_phi_3D_direct_inbreeding(ns[0], ns[1], ns[2], 
                                                         xxs[0], xxs[1], xxs[2], 
                                                         phi, Fs[0], Fs[1], Fs[2],
                                                         mask_corners, ploidys[0],
                                                         ploidys[1], ploidys[2],
                                                         het_ascertained)
        else:
            raise ValueError('Only implemented for dimensions 1,2 or 3.')
        fs.pop_ids = pop_ids
        # Record value to use for extrapolation. This is the first grid point,
        # which is where new mutations are introduced. Note that extrapolation
        # will likely fail if grids differ between dimensions.
        fs.extrap_x = xxs[0][1]
        for xx in xxs[1:]:
            if not xx[1] == fs.extrap_x:
                logger.warning('Spectrum calculated from phi different grids for '
                            'different dimensions. Extrapolation may fail.')
        return fs

    def scramble_pop_ids(self, mask_corners=True):
        """
        Spectrum corresponding to scrambling individuals among populations.
        
        This is useful for assessing how diverged populations are.
        Essentially, it pools all the individuals represented in the fs and
        generates new populations of random individuals (without replacement)
        from that pool. If this fs is significantly different from the
        original, that implies population structure.
        """
        original_folded = self.folded
        # If we started with an folded Spectrum, we need to unfold before
        # projecting.
        if original_folded:
            self = self.unfold()

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

        resamp = Spectrum(resamp, mask_corners=mask_corners)
        if not original_folded:
            return resamp
        else:
            return resamp.fold()

    @staticmethod
    def from_data_dict(data_dict, pop_ids, projections, mask_corners=True,
                       polarized=True):
        """
        Spectrum from a dictionary of polymorphisms.

        pop_ids: list of which populations to make fs for.
        projections: list of sample sizes to project down to for each
                     population.
        mask_corners: If True (default), the 'observed in none' and 'observed 
                      in all' entries of the FS will be masked.
        polarized: If True, the data are assumed to be correctly polarized by 
                   `outgroup_allele'. SNPs in which the 'outgroup_allele'
                   information is missing or '-' or not concordant with the
                   segregating alleles will be ignored.
                   If False, any 'outgroup_allele' info present is ignored,
                   and the returned spectrum is folded.

        The data dictionary should be organized as:
            {snp_id:{'segregating': ['A','T'],
                     'calls': {'YRI': (23,3),
                                'CEU': (7,3)
                                },
                     'outgroup_allele': 'T'
                    }
            }
        The 'calls' entry gives the successful calls in each population, in the
        order that the alleles are specified in 'segregating'.
        Non-diallelic polymorphisms are skipped.
        """
        import dadi.Misc
        cd = dadi.Misc.count_data_dict(data_dict, pop_ids)
        fs = Spectrum._from_count_dict(cd, projections, polarized, pop_ids, 
                mask_corners=mask_corners)
        return fs

    @staticmethod
    def _from_count_dict(count_dict, projections, polarized=True, pop_ids=None,
            mask_corners=False):
        """
        Frequency spectrum from data mapping SNP configurations to counts.

        count_dict: Result of Misc.count_data_dict
        projections: List of sample sizes to project down to for each
                     population.
        polarized: If True, only include SNPs that count_dict marks as
                   polarized.
                   If False, include all SNPs and fold resulting Spectrum.
        pop_ids: Optional list of strings containing the population labels.
        mask_corners: If True (default), the 'observed in none' and 'observed 
                      in all' entries of the FS will be masked.
        """
        # create slices for projection calculation
        slices = [[numpy.newaxis] * len(projections) for ii in
                  range(len(projections))]
        for ii in range(len(projections)):
            slices[ii][ii] = slice(None,None,None)
        # Convert to tuples to avoid numpy error
        slices = [tuple(_) for _ in slices]

        fs_total = dadi.Spectrum(numpy.zeros(numpy.array(projections)+1),
                                 pop_ids=pop_ids, mask_corners=mask_corners)
        for (called_by_pop, derived_by_pop, this_snp_polarized), count\
                in count_dict.items():
            if polarized and not this_snp_polarized:
                continue
            pop_contribs = []
            iter = zip(projections, called_by_pop, derived_by_pop)
            for pop_ii, (p_to, p_from, hits) in enumerate(iter):
                contrib = _cached_projection(p_to,p_from,hits)[slices[pop_ii]]
                pop_contribs.append(contrib)
            fs_proj = pop_contribs[0]
            for contrib in pop_contribs[1:]:
                fs_proj = fs_proj*contrib

            # create slices for adding projected fs to overall fs
            fs_total += count * fs_proj
        if polarized:
            return fs_total
        else:
            return fs_total.fold()

    @staticmethod
    def _data_by_tri(data_dict):
        """
        Nest the data by derived context and outgroup base.

        The resulting dictionary contains only SNPs which are appropriate for
        use of Hernandez's ancestral misidentification correction. It is
        organized as {(derived_tri, outgroup_base): {snp_id: data,...}}
        """
        result = {}
        genetic_bases = 'ACTG'
        for snp, snp_info in data_dict.items():
            # Skip non-diallelic polymorphisms
            if len(snp_info['segregating']) != 2:
                continue
            allele1, allele2 = snp_info['segregating']
            # Filter out SNPs where we either non-constant ingroup or outgroup
            # context.
            try:
                ingroup_tri = snp_info['context']
                outgroup_tri = snp_info['outgroup_context']
            except KeyError:
                continue
            if not outgroup_tri[1] == snp_info['outgroup_allele']:
                raise ValueError('Outgroup context and allele are inconsistent '
                                 'for polymorphism: %s.' % snp)
            outgroup_allele = outgroup_tri[1]
    
            # These are all the requirements to apply the ancestral correction.
            # First 2 are constant context.
            # Next 2 are sensible context.
            # Next 1 is that outgroup allele is one of the segregating.
            # Next 2 are that segregating alleles are sensible.
            if outgroup_tri[0] != ingroup_tri[0]\
               or outgroup_tri[2] != ingroup_tri[2]\
               or ingroup_tri[0] not in genetic_bases\
               or ingroup_tri[2] not in genetic_bases\
               or outgroup_allele not in [allele1, allele2]\
               or allele1 not in genetic_bases\
               or allele2 not in genetic_bases:
                continue
    
            if allele1 == outgroup_allele:
                derived_allele = allele2
            elif allele2 == outgroup_allele:
                # In this case, the second allele is non_outgroup
                derived_allele = allele1
            derived_tri = ingroup_tri[0] + derived_allele + ingroup_tri[2]
            result.setdefault((derived_tri, outgroup_allele), {})
            result[derived_tri, outgroup_allele][snp] = snp_info
        return result
    
    @staticmethod
    def from_data_dict_corrected(data_dict, pop_ids, projections,
                                 fux_filename, force_pos=True,
                                 mask_corners=True):
        """
        Spectrum from a dictionary of polymorphisms, corrected for ancestral
        misidentification.

        The correction is based upon:
            Hernandez, Williamson & Bustamante _Mol_Biol_Evol_ 24:1792 (2007)

        force_pos: If the correction is too agressive, it may leave some small
                   entries in the fs less than zero. If force_pos is true,
                   these entries will be set to zero, in such a way that the
                   total number of segregating SNPs is conserved.
        fux_filename: The name of the file containing the 
                   misidentification probabilities.
                   The file is of the form:
                       # Any number of comments lines beginning with #
                       AAA T 0.001
                       AAA G 0.02
                       ...
                   Where every combination of three + one bases is considered
                   (order is not important).  The triplet is the context and
                   putatively derived allele (x) in the reference species. The
                   single base is the base (u) in the outgroup. The numerical
                   value is 1-f_{ux} in the notation of the paper.

        The data dictionary should be organized as:
            {snp_id:{'segregating': ['A','T'],
                     'calls': {'YRI': (23,3),
                                'CEU': (7,3)
                                },
                     'outgroup_allele': 'T',
                     'context': 'CAT',
                     'outgroup_context': 'CAT'
                    }
            }
        The additional entries are 'context', which includes the two flanking
        bases in the species of interest, and 'outgroup_context', which
        includes the aligned bases in the outgroup.

        This method skips entries for which the correction cannot be applied.
        Most commonly this is because of missing or non-constant context.
        """
        # Read the fux file into a dictionary.
        fux_dict = {}
        f = open(fux_filename)
        for line in f.readlines():
            if line.startswith('#'):
                continue
            sp = line.split()
            fux_dict[(sp[0], sp[1])] = 1-float(sp[2])
        f.close()
    
        # Divide the data into classes based on ('context', 'outgroup_allele')
        by_context = Spectrum._data_by_tri(data_dict)
    
        fs = numpy.zeros(numpy.asarray(projections)+1)
        while by_context:
            # Each time through this loop, we eliminate two entries from the 
            # data dictionary. These correspond to one class and its
            # corresponding misidentified class.
            (derived_tri, out_base), nomis_data = by_context.popitem()

            # The corresponding bases if the ancestral state had been
            # misidentifed.
            mis_out_base = derived_tri[1]
            mis_derived_tri = derived_tri[0] + out_base + derived_tri[2]
            # Get the data for that case. Note that we default to an empty
            # dictionary if we don't have data for that class.
            mis_data = by_context.pop((mis_derived_tri, mis_out_base), {})
    
            fux = fux_dict[(derived_tri, out_base)]
            fxu = fux_dict[(mis_derived_tri, mis_out_base)]
    
            # Get the spectra for these two cases
            Nux = Spectrum.from_data_dict(nomis_data, pop_ids, projections)
            Nxu = Spectrum.from_data_dict(mis_data, pop_ids, projections)
    
            # Equations 5 & 6 from the paper.
            Nxu_rev = reverse_array(Nxu)
            Rux = (fxu*Nux - (1-fxu)*Nxu_rev)/(fux+fxu-1)
            Rxu = reverse_array((fux*Nxu_rev - (1-fux)*Nux)/(fux+fxu-1))
    
            fs += Rux + Rxu
    
        # Here we take the negative entries, and flip them back, so they end up
        # zero and the total number of SNPs is conserved.
        if force_pos:
            negative_entries = numpy.minimum(0, fs)
            fs -= negative_entries
            fs += reverse_array(negative_entries)
    
        return Spectrum(fs, mask_corners=mask_corners, pop_ids=pop_ids)

    @staticmethod
    def from_demes(
        g, sampled_demes, sample_sizes, pts, log_extrap=False, sample_times=None, Ne=None):
        """
        Takes a deme graph and computes the SFS. ``demes`` is a package for
        specifying demographic models in a user-friendly, human-readable YAML
        format. This function automatically parses the demographic description
        and returns a SFS for the specified populations and sample sizes.

        This function is new in version 1.1.0. Future developments will allow for
        inference using ``demes``-based demographic descriptions.

        :param g: A ``demes`` DemeGraph from which to compute the SFS. The DemeGraph
            can either be specified as a YAML file, in which case `g` is a string,
            or as a ``DemeGraph`` object.
        :type g: str or :class:`demes.DemeGraph`
        :param sampled_demes: A list of deme IDs to take samples from. We can repeat
            demes, as long as the sampling of repeated deme IDs occurs at distinct
            times.
        :type sampled_demes: list of strings
        :param sample_sizes: A list of the same length as ``sampled_demes``,
            giving the sample sizes for each sampled deme.
        :type sample_sizes: list of ints
        :param sample_times: If None, assumes all sampling occurs at the end of the
            existence of the sampled deme. If there are
            ancient samples, ``sample_times`` must be a list of same length as
            ``sampled_demes``, giving the sampling times for each sampled
            deme. Sampling times are given in time units of the original deme graph,
            so might not necessarily be generations (e.g. if ``g.time_units`` is years)
        :type sample_times: list of floats, optional
        :param Ne: reference population size. If none is given, we use the initial
            size of the root deme.
        :type Ne: float, optional
        :return: A ``dadi`` site frequency spectrum, with dimension equal to the
            length of ``sampled_demes``, and shape equal to ``sample_sizes`` plus one
            in each dimension, indexing the allele frequency in each deme from 0
            to n[i], where i is the deme index.
        :rtype: :class:`dadi.Spectrum`
        """
        global _imported_demes
        if not _imported_demes:
            try:
                global demes
                global Demes
                import demes
                import dadi.Demes as Demes

                _imported_demes = True
            except ImportError:
                raise ImportError("demes is not installed, need to `pip install demes`")

        if isinstance(g, str):
            dg = demes.load(g)
        else:
            dg = g

        func_ex = dadi.Numerics.make_extrap_func(Demes.SFS)

        fs = func_ex(
            dg,
            sampled_demes,
            sample_sizes,
            sample_times,
            Ne,
            pts,
        )
        return fs

    # The code below ensures that when I do arithmetic with Spectrum objects,
    # it is not done between a folded and an unfolded array. If it is, I raise
    # a ValueError.

    # While I'm at it, I'm also fixing the annoying behavior that if a1 and a2
    # are masked arrays, and a3 = a1 + a2. Then wherever a1 or a2 was masked,
    # a3.data ends up with the a1.data values, rather than a1.data + a2.data.
    # Note that this fix doesn't work for operation by numpy.ma.exp and 
    # numpy.ma.log. Guess I can't have everything.

    # I'm using exec here to avoid copy-pasting a dozen boiler-plate functions.
    # The calls to check_folding_equal ensure that we don't try to combine
    # folded and unfolded Spectrum objects.

    # I set check_folding = False in the constructor because it raises useless
    # warnings when, for example, I do (model + 1). 

    # These functions also ensure that the pop_ids and extrap_x attributes
    # get properly copied over.

    # This is pretty advanced Python voodoo, so don't fret if you don't
    # understand it at first glance. :-)
    for method in ['__add__','__radd__','__sub__','__rsub__','__mul__',
                   '__rmul__','__div__','__rdiv__','__truediv__','__rtruediv__',
                   '__floordiv__','__rfloordiv__','__rpow__','__pow__']:
        exec("""
def %(method)s(self, other):
    self._check_other_folding(other)
    if isinstance(other, numpy.ma.masked_array):
        newdata = self.data.%(method)s (other.data)
        newmask = numpy.ma.mask_or(self.mask, other.mask)
    else:
        newdata = self.data.%(method)s (other)
        newmask = self.mask
    newpop_ids = self.pop_ids
    if hasattr(other, 'pop_ids'):
        if other.pop_ids is None:
            newpop_ids = self.pop_ids
        elif self.pop_ids is None:
            newpop_ids = other.pop_ids
        elif other.pop_ids != self.pop_ids:
            logger.warning('Arithmetic between Spectra with different pop_ids. '
                        'Resulting pop_id may not be correct.')
    if hasattr(other, 'extrap_x') and self.extrap_x != other.extrap_x:
        extrap_x = None
    else:
        extrap_x = self.extrap_x
    outfs = self.__class__.__new__(self.__class__, newdata, newmask, 
                                   mask_corners=False, data_folded=self.folded,
                                   check_folding=False, pop_ids=newpop_ids,
                                   extrap_x=extrap_x)
    return outfs
""" % {'method':method})

    # Methods that modify the Spectrum in-place.
    for method in ['__iadd__','__isub__','__imul__','__idiv__',
                   '__itruediv__','__ifloordiv__','__ipow__']:
        exec("""
def %(method)s(self, other):
    self._check_other_folding(other)
    if isinstance(other, numpy.ma.masked_array):
        self.data.%(method)s (other.data)
        self.mask = numpy.ma.mask_or(self.mask, other.mask)
    else:
        self.data.%(method)s (other)
    if hasattr(other, 'pop_ids') and other.pop_ids is not None\
             and other.pop_ids != self.pop_ids:
        logger.warning('Arithmetic between Spectra with different pop_ids. '
                    'Resulting pop_id may not be correct.')
    if hasattr(other, 'extrap_x') and self.extrap_x != other.extrap_x:
        self.extrap_x = None
    return self
""" % {'method':method})

    def _check_other_folding(self, other):
        """
        Ensure other Spectrum has same .folded status
        """
        if isinstance(other, self.__class__)\
           and other.folded != self.folded:
            raise ValueError('Cannot operate with a folded Spectrum and an '
                             'unfolded one.')

# Allow spectrum objects to be pickled. 
# See http://effbot.org/librarybook/copy-reg.htm
try:
    import copyreg
except:
    # For Python 2.x compatibility
    import copy_reg as copyreg
def Spectrum_unpickler(data, mask, data_folded, pop_ids, extrap_x):
    return dadi.Spectrum(data, mask, mask_corners=False, data_folded=data_folded, check_folding=False, pop_ids=pop_ids, extrap_x=extrap_x)
def Spectrum_pickler(fs):
    return Spectrum_unpickler, (fs.data, fs.mask, fs.folded, fs.pop_ids, fs.extrap_x)
copyreg.pickle(Spectrum, Spectrum_pickler, Spectrum_unpickler)

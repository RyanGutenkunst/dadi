import os

import numpy
from numpy import newaxis as nuax
import scipy.stats

from dadi.SFS import cached_projection
from dadi.Numerics import reverse_array

class Spectrum(numpy.ma.MaskedArray):
    def __init__(self, data, mask=None, mask_corners=True):
        data = numpy.asarray(data)
        numpy.ma.masked_array.__init__(self, data, copy=True)

        # Set the mask and the fill value
        if mask is None:
            mask = numpy.ma.make_mask_none(data.shape)
        self.mask = mask
        self.set_fill_value(numpy.nan)

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


import unittest

import os

import numpy
import dadi

class SpectrumTestCase(unittest.TestCase):

    comments = ['comment 1', 'comment 2']
    filename = 'test.fs'
    data = numpy.random.rand(3,3)
    def test_to_file(self, remove=True):
        """
        Saving spectrum to file.
        """
        fs = dadi.Spectrum(self.data)
        fs.to_file(self.filename, comment_lines=self.comments)
        if remove:
            os.remove(self.filename)

    def test_from_file(self):
        """
        Loading spectrum from file.
        """
        # Make sure we have a file to read.
        self.test_to_file(remove=False)
        # Read the file.
        fs,comments = dadi.Spectrum.from_file(self.filename, 
                                              return_comments=True)
        # Ensure that fs was read correctly.
        orig = dadi.Spectrum(self.data)
        # We have to use filled here because we can't compare the masked values.
        self.assert_(numpy.allclose(fs.filled(0), orig.filled(0)))
        self.assert_(numpy.all(fs.mask == orig.mask))
        # Ensure comments were read correctly.
        for ii,line in enumerate(comments):
            self.assertEqual(line, self.comments[ii])

    def test_folding(self):
        """
        Folding a 2D spectrum.
        """
        data = numpy.reshape(numpy.arange(12), (3,4))
        fs = dadi.Spectrum(data)
        ff = fs.fold()

        # Ensure no SNPs have gotten lost.
        self.assertAlmostEqual(fs.sum(), ff.sum(), 6)
        self.assertAlmostEqual(fs.data.sum(), ff.data.sum(), 6)
        # Ensure that the empty entries are actually empty.
        self.assert_(numpy.all(ff.data[::-1] == numpy.tril(ff.data[::-1])))

        # This turns out to be the correct result.
        correct = numpy.tri(4)[::-1][-3:]*11
        self.assert_(numpy.allclose(correct, ff.data))

    def test_ambiguous_folding(self):
        """
        Test folding when the minor allele is ambiguous.
        """
        data = numpy.zeros((4,4))
        # Both these entries correspond to a an allele seen in 3 of 6 samples.
        # So the minor allele is ambiguous. In this case, we average the two
        # possible assignments.
        data[0,3] = 1
        data[3,0] = 3
        fs = dadi.Spectrum(data)
        ff = fs.fold()

        correct = numpy.zeros((4,4))
        correct[0,3] = correct[3,0] = 2
        self.assert_(numpy.allclose(correct, ff.data))

    def test_masked_folding(self):
        """
        Test folding when the minor allele is ambiguous.
        """
        data = numpy.zeros((5,6))
        fs = dadi.Spectrum(data)
        # This folds to an entry that will already be masked.
        fs.mask[1,2] = True
        # This folds to (1,1), which needs to be masked.
        fs.mask[3,4] = True
        ff = fs.fold()
        # Ensure that all those are masked.
        for entry in [(1,2), (3,4), (1,1)]:
            self.assert_(ff.mask[entry])

suite = unittest.TestLoader().loadTestsFromTestCase(SpectrumTestCase)

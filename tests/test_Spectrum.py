import unittest

import os

import numpy
from dadi.Spectrum_mod import Spectrum

class SpectrumTestCase(unittest.TestCase):

    comments = ['comment 1', 'comment 2']
    filename = 'test.fs'
    data = numpy.random.rand(3,3)
    def test_to_file(self):
        """
        Saving spectrum to file.
        """
        fs = Spectrum(self.data)
        fs.to_file(self.filename, comment_lines=self.comments)
    def test_from_file(self):
        """
        Loading spectrum from file.
        """
        # Make sure we have a file to read.
        self.test_to_file()
        # Read the file.
        fs,comments = Spectrum.from_file(self.filename, return_comments=True)
        # Ensure that fs was read correctly.
        orig = Spectrum(self.data)
        # We have to use filled here because we can't compare the masked values.
        self.assert_(numpy.allclose(fs.filled(0), orig.filled(0)))
        self.assert_(numpy.all(fs.mask == orig.mask))
        # Ensure comments were read correctly.
        for ii,line in enumerate(comments):
            self.assertEqual(line, self.comments[ii])

    def tearDown(self):
        # Remove the test file we created
        os.remove(self.filename)

suite = unittest.TestLoader().loadTestsFromTestCase(SpectrumTestCase)

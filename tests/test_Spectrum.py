import os
import unittest

import numpy
import scipy.special
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

    def test_folded_slices(self):
        ns = (3,4)
        fs1 = dadi.Spectrum(numpy.random.rand(*ns))
        folded1 = fs1.fold()

        self.assert_(fs1[:].folded == False)
        self.assert_(folded1[:].folded == True)

        self.assert_(fs1[0].folded == False)
        self.assert_(folded1[1].folded == True)

        self.assert_(fs1[:,0].folded == False)
        self.assert_(folded1[:,1].folded == True)

    def test_folded_arithmetic(self):
        """
        Test that arithmetic operations respect and propogate .folded attribute.
        """
        # Disable logging of warnings because arithmetic may generate Spectra
        # with entries < 0, but we don't care at this point.
        import logging
        dadi.Spectrum_mod.logger.setLevel(logging.ERROR)

        ns = (3,4)
        fs1 = dadi.Spectrum(numpy.random.uniform(size=ns))
        fs2 = dadi.Spectrum(numpy.random.uniform(size=ns))

        folded1 = fs1.fold()
        folded2 = fs2.fold()

        # We'll iterate through each of these arithmetic functions.
        from operator import add,sub,mul,div,truediv,floordiv,pow,abs,pos,neg,\
                iadd,isub,imul,idiv,itruediv,ifloordiv,ipow

        arr = numpy.random.uniform(size=ns)
        marr = numpy.random.uniform(size=ns)

        for op in [add,sub,mul,div,truediv,floordiv,pow]:
            # Check that binary operations propogate folding status.
            # Need to check cases both on right-hand-side of operator and
            # left-hand-side

            # Note that numpy.power(2.0,fs2) does not properly propagate type
            # or status. I'm not sure how to fix this.

            result = op(fs1,fs2)
            self.assertFalse(result.folded)
            result = op(fs1,2.0)
            self.assertFalse(result.folded)
            result = op(2.0,fs2)
            self.assertFalse(result.folded)
            result = op(fs1,arr)
            self.assertFalse(result.folded)
            result = op(arr,fs2)
            self.assertFalse(result.folded)
            result = op(fs1,marr)
            self.assertFalse(result.folded)
            result = op(marr,fs2)
            self.assertFalse(result.folded)

            result = op(folded1,folded2)
            self.assertTrue(result.folded)
            result = op(folded1,2.0)
            self.assertTrue(result.folded)
            result = op(2.0,folded2)
            self.assertTrue(result.folded)
            result = op(folded1,arr)
            self.assertTrue(result.folded)
            result = op(arr,folded2)
            self.assertTrue(result.folded)
            result = op(folded1,marr)
            self.assertTrue(result.folded)
            result = op(marr,folded2)
            self.assertTrue(result.folded)

            # Check that exceptions are properly raised when folding status 
            # differs
            self.assertRaises(ValueError, op, fs1, folded2)
            self.assertRaises(ValueError, op, folded1, fs2)

        for op in [abs,pos,neg,numpy.ma.log,numpy.ma.exp,numpy.ma.sqrt,
                   scipy.special.gammaln]:
            # Check that unary operations propogate folding status.
            result = op(fs1)
            self.assertFalse(result.folded)
            result = op(folded1)
            self.assertTrue(result.folded)

        for op in [iadd,isub,imul,idiv,itruediv,ifloordiv,ipow]:
            # Check that in-place operations preserve folding status.
            op(fs1,fs2)
            self.assertFalse(fs1.folded)
            op(fs1,2.0)
            self.assertFalse(fs1.folded)
            op(fs1,arr)
            self.assertFalse(fs1.folded)
            op(fs1,marr)
            self.assertFalse(fs1.folded)

            op(folded1,folded2)
            self.assertTrue(folded1.folded)
            op(folded1,2.0)
            self.assertTrue(folded1.folded)
            op(folded1,arr)
            self.assertTrue(folded1.folded)
            op(folded1,marr)
            self.assertTrue(folded1.folded)

            # Check that exceptions are properly raised.
            self.assertRaises(ValueError, op, fs1, folded2)
            self.assertRaises(ValueError, op, folded1, fs2)

        # Restore logging of warnings
        dadi.Spectrum_mod.logger.setLevel(logging.WARNING)
    
    def test_unfolding(self):
        ns = (3,4)

        # We add some unusual masking.
        fs = dadi.Spectrum(numpy.random.uniform(size=ns))
        fs.mask[0,1] = fs.mask[1,1] = True

        folded = fs.fold()
        unfolded = folded.unfold()

        # Check that it was properly recorded
        self.assertFalse(unfolded.folded)

        # Check that no data was lost
        self.assertAlmostEqual(fs.data.sum(), folded.data.sum())
        self.assertAlmostEqual(fs.data.sum(), unfolded.data.sum())

        # Note that fs.sum() need not be equal to folded.sum(), if fs had
        # some masked values.
        self.assertAlmostEqual(folded.sum(), unfolded.sum())

        # Check that the proper entries are masked.
        self.assertTrue(unfolded.mask[0,1])
        self.assertTrue(unfolded.mask[(ns[0]-1),(ns[1]-1)-1])
        self.assertTrue(unfolded.mask[1,1])
        self.assertTrue(unfolded.mask[(ns[0]-1)-1,(ns[1]-1)-1])

    def test_marginalize(self):
        ns = (7,8,6)

        fs = dadi.Spectrum(numpy.random.uniform(size=ns))
        folded = fs.fold()

        marg1 = fs.marginalize([1])
        # Do manual marginalization.
        manual = dadi.Spectrum(fs.data.sum(axis=1))

        # Check that these are equal in the unmasked entries.
        self.assert_(numpy.allclose(numpy.where(marg1.mask, 0, marg1.data),
                                    numpy.where(manual.mask, 0, manual.data)))

        # Check folded Spectrum objects. I should get the same result if I
        # marginalize then fold, as if I fold then marginalize.
        mf1 = marg1.fold()
        mf2 = folded.marginalize([1])
        self.assert_(numpy.allclose(mf1,mf2))

    def test_projection(self):
        # Test that projecting a multi-dimensional Spectrum succeeds
        ns = (7,8,6)
        fs = dadi.Spectrum(numpy.random.uniform(size=ns))
        p = fs.project([3,4,5])
        # Also that we don't lose any data
        self.assertAlmostEqual(fs.data.sum(), p.data.sum())

        # Check that when I project an equilibrium spectrum, I get back an
        # equilibrium spectrum
        fs = dadi.Spectrum(1./numpy.arange(100))
        p = fs.project([17])
        self.assert_(numpy.allclose(p[1:-1], 1./numpy.arange(1,len(p)-1)))

        # Check that masked values are propagated correctly.
        fs = dadi.Spectrum(1./numpy.arange(20))
        # All values with 3 or fewer observed should be masked.
        fs.mask[3] = True
        p = fs.project([10])
        self.assert_(numpy.all(p.mask[:4]))

        # Check that masked values are propagated correctly.
        fs = dadi.Spectrum(1./numpy.arange(20))
        fs.mask[-3] = True
        # All values with 3 or fewer observed should be masked.
        p = fs.project([10])
        self.assert_(numpy.all(p.mask[-3:]))

        # A more complicated two dimensional projection problem...
        fs = dadi.Spectrum(numpy.random.uniform(size=(9,7)))
        fs.mask[2,3] = True
        p = fs.project([4,4])
        self.assert_(numpy.all(p.mask[:3,1:4]))

        # Test that projecting a folded multi-dimensional Spectrum succeeds
        # Should get the same result if I fold then project as if I project
        # then fold.
        ns = (7,8,6)
        fs = dadi.Spectrum(numpy.random.uniform(size=ns))
        fs.mask[2,3,1] = True
        folded = fs.fold()

        p = fs.project([3,4,5])
        pf1 = p.fold()
        pf2 = folded.project([3,4,5])

        # Check equality
        self.assert_(numpy.all(pf1.mask == pf2.mask))
        self.assert_(numpy.allclose(pf1.data, pf2.data))

suite = unittest.TestLoader().loadTestsFromTestCase(SpectrumTestCase)

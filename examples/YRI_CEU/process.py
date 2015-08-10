import glob
import dadi

for ii, fname in enumerate(glob.glob('/Users/rgutenk/Desktop/Documents/Research/2009/fit3/conventional_boot/*.fs')):
    fsin = dadi.Spectrum.from_file(fname)
    fsout = fsin.marginalize([2])
    dadi.Spectrum.tofile(fsout, 'bootstraps/{0:02d}.fs'.format(ii))

# Importing these adds a 'bdist_mpkg' option that allows building binary
# packages on OS X.
try:
    import setuptools
    import bdist_mpkg
except ImportError:
    pass

import os

import numpy.distutils.core as core

# Configure our C modules that are built with f2py.
tridiag = core.Extension(name = 'dadi.tridiag',
                         sources = ['dadi/tridiag.pyf', 'dadi/tridiag.c'])
int_c = core.Extension(name = 'dadi.integration_c',
                       sources = ['dadi/integration_c.pyf', 
                                  'dadi/integration1D.c',
                                  'dadi/integration2D.c', 
                                  'dadi/integration3D.c',
                                  'dadi/integration_shared.c',
                                  'dadi/tridiag.c'])

# If we're building a distribution, try to update svnversion. Note that this
# fails silently.
for arg in os.sys.argv:    
    if arg.count('sdist') or arg.count('bdist'):
        os.system("svn up")
        os.system("svn info > dadi/svnversion")

core.setup(name='dadi',
           version='1.2.3',
           author='Ryan Gutenkunst',
           author_email='rng7@cornell.edu',
           url='http://dadi.googlecode.com',
           ext_modules = [tridiag, int_c],
           scripts=['scripts/ms_jsfs.py'],
           packages=['dadi'], 
           package_data = {'dadi':['svnversion'],
                           'tests':['IM.fs']},
           license='BSD'
           )

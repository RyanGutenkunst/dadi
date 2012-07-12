# Importing these adds a 'bdist_mpkg' option that allows building binary
# packages on OS X.
try:
    import setuptools
    import bdist_mpkg
except ImportError:
    pass

import os,sys

import numpy.distutils.core as core

#
# Microsoft Visual C++ only supports C up to the version iso9899:1990 (C89).
# gcc by default supports much more. To ensure MSVC++ compatibility when using
# gcc, we need to add extra compiler args. This code tries to ensure such
# arguments are added *only* when we're using gcc.
#
import numpy.distutils
compiler = numpy.distutils.ccompiler.get_default_compiler()
for arg in sys.argv:
    if arg.startswith('--compiler'):
        compiler = arg.split('=')[1]
if compiler in ['unix','mingw32','cygwin']:
    extra_compile_args = []
    # RNG: This seems to cause problems on some machines. To test for
    # compatibility with VC++, uncomment this line.
    #extra_compile_args = ['-std="iso9899:1990"', '-pedantic-errors']
else:
    extra_compile_args = []


# Configure our C modules that are built with f2py.
tridiag = core.Extension(name = 'dadi.tridiag',
                         sources = ['dadi/tridiag.pyf', 'dadi/tridiag.c'],
                         extra_compile_args=extra_compile_args)
int_c = core.Extension(name = 'dadi.integration_c',
                       sources = ['dadi/integration_c.pyf', 
                                  'dadi/integration1D.c',
                                  'dadi/integration2D.c', 
                                  'dadi/integration3D.c',
                                  'dadi/integration_shared.c',
                                  'dadi/tridiag.c'],
                         extra_compile_args=extra_compile_args)

# If we're building a distribution, try to update svnversion. Note that this
# fails silently.
if any(arg.count('dist') for arg in os.sys.argv):    
    os.system("svn up")
    os.system("svn info > dadi/svnversion")

core.setup(name='dadi',
           version='1.6.3',
           author='Ryan Gutenkunst',
           author_email='rgutenk@email.arizona.edu',
           url='http://dadi.googlecode.com',
           ext_modules = [tridiag, int_c],
           scripts=['scripts/ms_jsfs.py'],
           packages=['dadi'], 
           package_data = {'dadi':['svnversion'],
                           'tests':['IM.fs']},
           license='BSD'
           )

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
                                  'dadi/integration4D.c',
                                  'dadi/integration5D.c',
                                  'dadi/integration_shared.c',
                                  'dadi/tridiag.c'],
                         extra_compile_args=extra_compile_args)
pdfs = core.Extension(name = 'dadi.DFE.PDFs_c',
                      sources = ['dadi/DFE/PDFs.pyf', 'dadi/DFE/PDFs.c'],
                      extra_compile_args=extra_compile_args)

if '--cython' in sys.argv:
    # Remove extra argument, so distutils doesn't complain
    sys.argv.remove('--cython')

    # Configure our C modules that are built with Cython.
    # This needs to be done in a separate setup step from the remainder of dadi,
    # to get f2py and Cython to play nicely together.
    #http://numpy-discussion.10968.n7.nabble.com/cython-and-f2py-td26250.html
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Distutils import build_ext
    
    tri_modules = ['transition1', 'transition2', 'transition12', 'transition1D']
    tri_extensions = [core.Extension(name='dadi.Triallele.{0}'.format(_), sources=['dadi/Triallele/{0}.pyx'.format(_)]) for _ in tri_modules]
    
    setup(ext_modules = tri_extensions,
          include_dirs = [numpy.get_include()],
          cmdclass = {'build_ext': build_ext},
          # Note that we build the extension modules in place
          script_args = ['build_ext', '--inplace'], 
          )
    
    two_locus_modules = ['projection_genotypes', 'surface_interaction', 'transition1', 'transition2', 'transition3', 'transition12', 'transition13', 'transition23', 'transition1D']
    two_locus_extensions = [core.Extension(name='dadi.TwoLocus.{0}'.format(_), sources=['dadi/TwoLocus/{0}.pyx'.format(_)]) for _ in two_locus_modules]
    
    setup(ext_modules = two_locus_extensions,
          include_dirs = [numpy.get_include()],
          cmdclass = {'build_ext': build_ext},
          # Note that we build the extension modules in place
          script_args = ['build_ext', '--inplace'], 
          )

with open("README.md", "r") as fh:
    long_description = fh.read()

numpy.distutils.core.setup(name='dadi',
                           version='2.2.0',
                           author='Ryan Gutenkunst',
                           author_email='rgutenk@arizona.edu',
                           url='https://bitbucket.org/gutenkunstlab/dadi',
                           ext_modules = [tridiag, int_c, pdfs],
                           packages=setuptools.find_packages(),
                           description="Fit population genetic models of demography and selection using diffusion approximations to the allele frequency spectrum",
                           long_description_content_type="text/markdown",
                           long_description=long_description,
                           package_data = {'tests':['IM.fs'],
                                           'dadi.cuda':['kernels.cu'],
                           #                # Copy Triallele extension modules
                                           'dadi.Triallele':['*.so', '*.pyd'],
                           #                # Copy TwoLocus extension modules
                                           'dadi.TwoLocus':['*.so', '*.pyd'],
                           #                # Copy DFE extension modules,
                                           'dadi.DFE':['*.so', '*.pyd']},
                           install_requires=['scipy', 'numpy', 'matplotlib', 'nlopt'],
                           classifiers=[
                               "Programming Language :: Python :: 3",
                               "License :: OSI Approved :: BSD License",
                               "Operating System :: OS Independent",
                               "Development Status :: 5 - Production/Stable",
                               "Intended Audience :: Science/Research",
                               "Natural Language :: English",
                               "Topic :: Scientific/Engineering :: Bio-Informatics"
                           ]
                           )
# To build completely
# rm -rf build dist */*.so */*/*.so */*module.c */*/*module.c
# python setup.py build_ext --inplace --cython
# To build API documention (https://pdoc3.github.io/pdoc/). 
# Remember to push the updated documentation to the repository
# rm -rf doc/api; pdoc -f --html -o doc/api dadi
# To build test mkdocs documentation in site/index.hml
# python -m mkdocs build --clean --no-directory-urls --config-file mkdocs.yml
# To distribute to PyPI
# rm -rf dist; python3 setup.py sdist bdist_wheel
# (Testing) python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# (Final) python3 -m twine upload dist/*   # Username is RyanGutenkunst
# To distribute to Conda
# A pull request will automatically be created to update https://github.com/conda-forge/dadi-feedstock
# Update recipe meta.yaml in dadi-feedstock/recipe with new version number and sha256. 
#  (To generate sha256, use openssl dgst -sha256 dist/dadi-<version>.tar.gz
# Then "git add meta.yaml" and "git commit" then "git push"
# Then update the pull request

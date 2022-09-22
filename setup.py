from setuptools import Extension, setup
import numpy

from Cython.Build import cythonize

extensions = [Extension(name='dadi.tridiag_cython', sources=['dadi/tridiag_cython.pyx', 'dadi/tridiag.c'])]
extensions.append(Extension(name = 'dadi.integration_c',
                            sources=['dadi/integration_c.pyx', 'dadi/integration1D.c',
                                     'dadi/integration2D.c', 'dadi/integration3D.c', 'dadi/integration4D.c',
                                     'dadi/integration5D.c', 'dadi/integration_shared.c',
                                     'dadi/tridiag.c']))
extensions.append(Extension(name='dadi.DFE.PDFs_cython', sources=['dadi/DFE/PDFs_cython.pyx']))

tri_modules = ['transition1', 'transition2', 'transition12', 'transition1D']
two_locus_modules = ['projection_genotypes', 'surface_interaction', 'transition1', 'transition2', 'transition3', 'transition12', 'transition13', 'transition23', 'transition1D']
extensions.extend([Extension(name='dadi.Triallele.{0}'.format(_), sources=['dadi/Triallele/{0}.pyx'.format(_)]) for _ in tri_modules])
extensions.extend([Extension(name='dadi.TwoLocus.{0}'.format(_), sources=['dadi/TwoLocus/{0}.pyx'.format(_)]) for _ in two_locus_modules])

setup(ext_modules=cythonize(extensions),
      include_dirs=[numpy.get_include()],
      package_data = {'dadi.cuda':['kernels.cu']})

#with open("README.md", "r") as fh:
#    long_description = fh.read()
#
#setuptools.setup(name='dadi',
#                           version='devel',
#                           author='Ryan Gutenkunst',
#                           author_email='rgutenk@arizona.edu',
#                           url='https://bitbucket.org/gutenkunstlab/dadi',
#                           #ext_modules = [tridiag, int_c, pdfs],
#                           packages=setuptools.find_packages(),
#                           description="Fit population genetic models of demography and selection using diffusion approximations to the allele frequency spectrum",
#                           long_description_content_type="text/markdown",
#                           #long_description=long_description,
#                           #package_data = {'tests':['IM.fs'],
#                           #                'dadi.cuda':['kernels.cu'],
#                           #                # Copy Triallele extension modules
#                           #                'dadi.Triallele':['*.so', '*.pyd'],
#                           ##                # Copy TwoLocus extension modules
#                           #                'dadi.TwoLocus':['*.so', '*.pyd'],
#                           ##                # Copy DFE extension modules,
#                           #                'dadi.DFE':['*.so', '*.pyd'],
#                           ##                # Copy DFE extension modules,
#                           #                'dadi':['*.so', '*.pyd']},
#                           install_requires=['scipy', 'numpy', 'matplotlib', 'nlopt'],
#                           classifiers=[
#                               "Programming Language :: Python :: 3",
#                               "License :: OSI Approved :: BSD License",
#                               "Operating System :: OS Independent",
#                               "Development Status :: 5 - Production/Stable",
#                               "Intended Audience :: Science/Research",
#                               "Natural Language :: English",
#                               "Topic :: Scientific/Engineering :: Bio-Informatics"
#                           ]
#                           )
## To build completely
## rm -rf build dist */*.so */*/*.so */*module.c */*/*module.c
## python setup.py build_ext --inplace --cython
## To build API documention (https://pdoc3.github.io/pdoc/)
## rm -rf doc/api; pdoc -f --html -o doc/api dadi
## To build test mkdocs documentation in site/index.hml
## python -m mkdocs build --clean --no-directory-urls --config-file mkdocs.yml
## To distribute to PyPI
## rm -rf dist; python3 setup.py sdist bdist_wheel
## (Testing) python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
## (Final) python3 -m twine upload dist/*   # Username is RyanGutenkunst
## To distribute to Conda
## A pull request will automatically be created to update https://github.com/conda-forge/dadi-feedstock
## Update recipe meta.yaml in dadi-feedstock/recipe with new version number and sha256. 
##  (To generate sha256, use openssl dgst -sha256 dist/dadi-<version>.tar.gz
## Then "git add meta.yaml" and "git commit" then "git push"
## Then update the pull request
#
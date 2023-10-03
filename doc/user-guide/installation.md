# Installation

### Dependencies

dadi depends on a number of Python libraries. The absolute dependencies are

- Python 3
- NumPy
- SciPy
- nlopt

It is also recommended that you install

- matplotlib
- IPython

The easiest way to obtain all these dependencies to install the [Anaconda Python Distribution](https://www.anaconda.com/download). You'll need to separately [install nlopt from conda-forge](https://anaconda.org/conda-forge/nlopt). And the easiest way to install dadi is then via `conda`, using the command `conda install -c conda-forge dadi`. dadi can also be installed via `pip`, using the command `python3 -m pip install dadi`.

### GPU computing

dadi can be sped up substantially by running on a CUDA-enabled Nvidia GPU. 
To enable this functionality, you will need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
After install the CUDA Toolkit, you will then need to install PyCUDA and scikit-cuda. 
Both of these can be installed from the Python Package Index using pip, `python3 -m pip install pycuda` and `python3 -m pip install scikit-cuda`.

### Installing from source

dadi can be easily installed from [source code](https://bitbucket.org/gutenkunstlab/dadi/src/master/), as long as you have an appropriate C compiler installed. (On OS X, you'll need to install the Developer Tools to get gcc. On Windows, you'll need to install the Microsoft Visual Studio to get C/C++ builder.) To do so, first unpack the source code tarball, `unzip dadi-<version>.zip` In the `dadi-<version>` directory, run `python setup.py install`. This will compile the C modules dadi uses and install those plus all dadi Python files in your Python installation's `site-packages` directory. A (growing) series of tests can be run in the `tests` directory, via `python run_tests.py`.

If you want the latest and greatest from dadi, clone the [Bitbucket repository](https://bitbucket.org/gutenkunstlab/dadi). You can then create a local install using `python setup.py develop`. This will ensure that when you edit pull revisions or edit the source code, changes are reflected immediately, without requiring a separate install step.

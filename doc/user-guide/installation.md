# Installation

### Dependencies

dadi depends on a number of Python libraries. The absolute dependencies are

- Python 3
- NumPy
- SciPy

It is also recommended that you install

- matplotlib
- IPython

The easiest way to obtain all these dependencies to install the [Anaconda Python Distribution](https://www.anaconda.com/distribution/). And the easiest way to install dadi is then via `conda`, using the command `conda install -c bioconda dadi`. dadi can also be installed via `pip`, using the command `python3 -m pip install dadi`.

### Installing from source

dadi can be easily installed from source code, as long as you have an appropriate C compiler installed. (On OS X, you'll need to install the Developer Tools to get gcc. On Windows, you'll need to install the Microsoft Visual Studio to get C/C++ builder.) To do so, first unpack the source code tarball, `unzip dadi-<version>.zip` In the `dadi-<version>` directory, run `python setup.py install`. This will compile the C modules dadi uses and install those plus all dadi Python files in your Python installation's `site-packages` directory. A (growing) series of tests can be run in the `tests` directory, via `python run_tests.py`.

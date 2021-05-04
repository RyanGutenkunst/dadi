# GPU computing

dadi can be sped up substantially by running on a CUDA-enabled Nvidia GPU. 

### Installation

To enable GPU computing, you will first need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
After installing the Toolkit, you will then need to install [PyCUDA](https://documen.tician.de/pycuda/) and [scikit-cuda](https://scikit-cuda.readthedocs.io/). 
Both of these can be installed from the Python Package Index using pip, `python3 -m pip install pycuda` and `python3 -m pip install scikit-cuda`.

### Usage

GPU computing is implemented for integration of population spectra, `phi`.
To enable GPU computing, run the command `dadi.cuda_enabled(True)` in your script, before you carry out any model simulations or optimizations.
To disable GPU computing, run the command `dadi.cuda_enabled(False)` in your script.

Combined CPU and GPU computing is enabled for dadi.DFE.Cache2D.
To use this functionality, set `mp=True`.

### RAM limitations

Generally, memory is more limited on GPUs than GPUs. To estimate how many gigabytes of RAM your analysis will need,
use the command `dadi.pts_to_RAM(pts, P)`. Here `pts` is the largest grid points setting you'll be using, 
and `P` is the largest number of simultaneous populations in your mdoels.

### Benchmarking

To benchmark your compute environment, use the script at `examples/CUDA/benchmark.py`. 
For example, running it as `python benchmark.py --cuda --RAM 2.0` with run the benchmarks on the GPU, with maximum
`pts` setting to use roughly 2 GB of RAM.
To test on your CPU, omit the `--cuda` argument.
Note that benchmark runs can get very long for large RAM settings, so you may want to begin with smaller settings
and work your way up to the maximum you can use.

### Warnings

When you run GPU dadi code, you may see warnings like:
"UserWarning: creating CUBLAS context to get version number"
and
"DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead".
These warnings come from the PyCUDA and scikit-cuda libraries, and they will not affect your results.
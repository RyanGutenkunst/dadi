# Bootstrapping

Because dadi's likelihood function treats all variants as independent, and they are often not, standard likelihood theory should not be used to estimate parameter uncertainties and significance levels for hypothesis test. To do such tests, one can bootstrap. For estimating parameter uncertainties, one can use a nonparameteric bootstrap, i.e. sampling with replacement from independent units of your data (genes or chromosomes) to generate new data sets to fit. For hypothesis tests, the parametric bootstrap is preferred. This involves using a coalescent simulator (such as *ms*) to generate simulated data sets. Care must be taken to simulate the sequencing strategy as closely as possible.

### Interacting with *ms*

dadi provides several methods to ease interaction with *ms*. The method `Spectrum.from_ms_file` will generate an FS from *ms* output. The method `Misc.ms_command` will generate the command line for *ms* corresponding to a particular simulation. As an example:

	import os
	
	core = "-n 1 -n 2 -ej 0.3 2 1"
	command = dadi.Misc.ms_command(theta = 1000, ns = (20, 20), core, 1000, recomb = 0.3)
	ms_fs = dadi.Spectrum.from_ms_file(os.popen(command))

Here the `os.open` command lets us read the *ms* output straight from the command, without writing an intermediate file to disk. If you'd like to actually write the file, you could do 

	os.system("%s > temp.msout" % command)
	ms_fs = dadi.Spectrum.from_ms_file("temp.msout")

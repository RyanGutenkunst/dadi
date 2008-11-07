"""
Miscellaneous utility functions. Including ms simulation.
"""

import sys,time

import numpy

#: Storage for times at which each stream was flushed.
__times_last_flushed = {}
def delayed_flush(stream=sys.stdout, delay=1):
    """
    Flush a stream, ensuring that it is only flushed every 'delay' *minutes*.
    Note that upon the first call to this method, the stream is not flushed.

    stream: The stream to flush. For this to work with simple 'print'
            statements, the stream should be sys.stdout.
    delay: Minimum time *in minutes* between flushes.

    This function is useful to prevent I/O overload on the cluster.
    """
    global __times_last_flushed

    curr_time = time.time()
    # If this is the first time this method has been called with this stream,
    # we need to fill in the times_last_flushed dict. setdefault will do this
    # without overwriting any entry that may be there already.
    if stream not in __times_last_flushed:
        __times_last_flushed[stream] = curr_time
    last_flushed = __times_last_flushed[stream]

    # Note that time.time() returns values in seconds, hence the factor of 60.
    if (curr_time - last_flushed) >= delay*60:
        stream.flush()
        __times_last_flushed[stream] = curr_time

def ensure_1arg_func(var):
    """
    Ensure that var is actually a one-argument function.

    This is primarily used to convert arguments that are constants into
    trivial functions of time for use in integrations where parameters are
    allowed to change over time.
    """
    if numpy.isscalar(var):
        # If a constant was passed in, use lambda to make it a nice
        #  simple function.
        var_f = lambda t: var
    else:
        var_f = var
    if not callable(var_f):
        raise ValueError('Argument is not a constant or a function.')
    try:
        var_f(0.0)
    except TypeError:
        raise ValueError('Argument is not a constant or a one-argument '
                         'function.')
    return var_f

def ms_command(theta, ns, core, iter, recomb=0, rsites=None):
    """
    Generate ms command for simulation from core.

    theta: Assumed theta
    ns: Sample sizes
    core: Core of ms command that specifies demography.
    iter: Iterations to run ms
    recomb: Assumed recombination rate
    rsites: Sites for recombination. If None, default is 10*theta.
    """
    ms_command = "ms %(total_chrom)i %(iter)i -t %(theta)f -I %(numpops)i "\
            "%(sample_sizes)s %(core)s"
    if recomb:
        ms_command = ms_command + " -r %(recomb)f %(rsites)i"
        if not rsites:
            rsites = theta*10
    sub_dict = {'total_chrom': numpy.sum(ns), 'iter': iter, 'theta': theta,
                'numpops': len(ns), 'sample_sizes': ' '.join(map(str, ns)),
                'core': core, 'recomb': recomb, 'rsites': rsites}

    return ms_command % sub_dict

import sys,time

import numpy

# Storage for times at which each stream was flushed.
__times_last_flushed = {}
default_flush_delay = 30./60
def delayed_flush(stream=sys.stdout):
    """
    Flush a stream, ensuring that it is only flushed every 'delay' *minutes*.
    Note that upon the first call to this method, the stream is not flushed.

    stream: The stream to flush. For this to work with simple 'print'
            statements, the stream should be sys.stdout.
    delay: Minimum time *in minutes* between flushes.

    This function is useful to prevent I/O overload on the cluster.
    """
    global __times_last_flushed

    delay = default_flush_delay

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
    except:
        raise ValueError('Argument is not a constant or a one-argument '
                         'function.')
    return var_f

if __name__ == '__main__':
    # Testing code. To test, run 'python misc.py >& test.out &' and monitor
    #  test.out using 'cat'. It should only change once every 10 seconds.
    secs_to_run = 60
    sleep_interval = 1.0
    start_time = curr_time = time.time()
    while curr_time - start_time < secs_to_run:
        time.sleep(sleep_interval)
        curr_time = time.time()
        print int(curr_time-start_time),
        delayed_flush(sys.stdout, 10/60.)

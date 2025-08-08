import numpy

def ensure_1arg_func_vectorized(vars_list):
    """
    Version of dadi.Misc.ensure_1ar_func that returns a 
    single vectorized function that can handle multiple parameters at once.
    This is useful for the selection parameters in polyploidy models.
    
    vars_list: List of variables to be passed to the function.
    
    Returns:
        A function that takes t and returns a numpy array of results
    """
    processed_funcs = []
    
    for var in vars_list:
        if numpy.isscalar(var):
            var_f_tmp = lambda t, v=var: v
        else:
            var_f_tmp = var
        
        var_f = lambda t, f=var_f_tmp: numpy.float64(f(t))
        
        if not callable(var_f):
            raise ValueError('Argument is not a constant or a function.')
        try:
            var_f(0.0)
        except TypeError:
            raise ValueError('Argument is not a constant or a one-argument function.')
        
        processed_funcs.append(var_f)
    
    # Return a single function that evaluates all at once
    def vectorized_func(t):
        return numpy.array([f(t) for f in processed_funcs])
    
    return vectorized_func
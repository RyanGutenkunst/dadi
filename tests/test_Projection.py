import numpy
import dadi

def test_project_up():
    """
    Saving spectrum to file.
    """
    fixed_params = [0.1,None,None]
    params_up = dadi.Inference._project_params_up([0.2,0.3], fixed_params)
    assert(numpy.allclose(params_up, [0.1,0.2,0.3]))

    fixed_params = [0.1,0.2,None]
    params_up = dadi.Inference._project_params_up([0.3], fixed_params)
    assert(numpy.allclose(params_up, [0.1,0.2,0.3]))

    fixed_params = [0.1,0.2,None]
    params_up = dadi.Inference._project_params_up(0.3, fixed_params)
    assert(numpy.allclose(params_up, [0.1,0.2,0.3]))

"""
Tests for dadi.Godambe.

Tier 1 (this file, for now): the finite-difference engine, get_grad and
get_hess, checked against functions whose derivatives are known in closed
form. These run in milliseconds and involve none of dadi's PDE machinery, so
a failure here is unambiguously a bug in the differentiation code.

These tests are written to characterize the *existing* hand-rolled routines
before they are replaced by scipy.differentiate. Anything asserted here is
behavior the replacement must reproduce.

A note on eps: get_grad and get_hess take a *fractional* stepsize. The step
for parameter i is eps*p[i], except that when abs(eps*p[i]) < 1e-6 (including
p[i] == 0) the step falls back to the absolute value eps and that parameter
switches to a one-sided difference.
"""
import numpy

from dadi.Godambe import get_grad, get_hess

# A quadratic in p, f = c + b.p + 0.5 p'Ap, with A symmetric.
# Every finite-difference branch in get_grad/get_hess is exact for a
# quadratic, up to floating-point roundoff, which makes these assertions
# tight rather than merely indicative.
QUAD_C = 1.5
QUAD_B = numpy.array([0.7, -1.3, 2.0])
QUAD_A = numpy.array([[2.0, 0.5, -0.3],
                      [0.5, 1.0, 0.8],
                      [-0.3, 0.8, 3.0]])


def quad(p):
    p = numpy.asarray(p, dtype=float)
    return QUAD_C + QUAD_B.dot(p) + 0.5*p.dot(QUAD_A).dot(p)


def quad_grad(p):
    return QUAD_B + QUAD_A.dot(numpy.asarray(p, dtype=float))


def test_grad_quadratic_exact():
    """Central differences recover a quadratic's gradient to roundoff."""
    p0 = [1.0, 2.0, 0.5]
    grad = get_grad(quad, p0, 0.01)
    # get_grad returns a column vector.
    assert grad.shape == (3, 1)
    assert numpy.allclose(grad.ravel(), quad_grad(p0), rtol=1e-8)


def test_hess_quadratic_exact():
    """Second differences recover a quadratic's Hessian, which is just A."""
    p0 = [1.0, 2.0, 0.5]
    hess = get_hess(quad, p0, 0.01)
    assert hess.shape == (3, 3)
    assert numpy.allclose(hess, QUAD_A, rtol=1e-6)


def test_hess_is_symmetric():
    """get_hess mirrors the upper triangle; the result must be symmetric."""
    hess = get_hess(quad, [1.0, 2.0, 0.5], 0.01)
    assert numpy.allclose(hess, hess.T, rtol=0, atol=0)


def test_grad_and_hess_accept_extra_args():
    """The args= passthrough reaches the function."""
    def scaled(p, factor):
        return factor*quad(p)

    p0 = [1.0, 2.0, 0.5]
    grad = get_grad(scaled, p0, 0.01, args=[3.0])
    hess = get_hess(scaled, p0, 0.01, args=[3.0])
    assert numpy.allclose(grad.ravel(), 3.0*quad_grad(p0), rtol=1e-8)
    assert numpy.allclose(hess, 3.0*QUAD_A, rtol=1e-6)


def test_grad_nonquadratic_converges_at_second_order():
    """
    For a non-quadratic function the central difference carries O(h^2)
    truncation error. Halving eps should cut the error by roughly four.
    """
    def f(p):
        return numpy.exp(p[0])*numpy.sin(p[1])

    p0 = [0.3, 0.7]
    exact = numpy.array([numpy.exp(p0[0])*numpy.sin(p0[1]),
                         numpy.exp(p0[0])*numpy.cos(p0[1])])

    err_coarse = numpy.abs(get_grad(f, p0, 0.02).ravel() - exact).max()
    err_fine = numpy.abs(get_grad(f, p0, 0.01).ravel() - exact).max()
    # Second-order convergence, with slack for the constant factor.
    assert err_fine < err_coarse/3.0


def test_hess_nonquadratic_accurate():
    def f(p):
        return numpy.exp(p[0])*numpy.sin(p[1])

    p0 = [0.3, 0.7]
    e0, s1, c1 = numpy.exp(p0[0]), numpy.sin(p0[1]), numpy.cos(p0[1])
    exact = numpy.array([[e0*s1, e0*c1],
                         [e0*c1, -e0*s1]])
    assert numpy.allclose(get_hess(f, p0, 1e-3), exact, rtol=1e-4, atol=1e-6)


# The branches below are the ones with no direct scipy.differentiate
# equivalent: scipy.differentiate.jacobian has step_direction, but
# scipy.differentiate.hessian has none. They are pinned deliberately.

def test_grad_at_zero_parameter_is_one_sided():
    """
    A parameter of exactly zero takes an absolute step and a one-sided
    difference. That difference is only first-order accurate, so the
    gradient carries a predictable O(eps) bias -- this is characterizing
    existing behavior, not endorsing it.
    """
    eps = 1e-4
    p0 = [0.0, 2.0, 0.5]
    grad = get_grad(quad, p0, eps).ravel()
    exact = quad_grad(p0)

    # The zero parameter's forward difference on a quadratic is exactly
    # the true derivative plus 0.5*A_ii*eps.
    assert numpy.isclose(grad[0], exact[0] + 0.5*QUAD_A[0, 0]*eps, rtol=1e-6)
    # The other parameters are untouched by the fallback and stay exact.
    assert numpy.allclose(grad[1:], exact[1:], rtol=1e-8)


def test_hess_at_zero_parameter_still_exact_for_quadratic():
    """
    The one-sided second difference (f(p+2h) - 2f(p+h) + f(p))/h**2 is
    still exact for a quadratic, so the zero-parameter fallback costs
    nothing here even though the gradient's does.
    """
    p0 = [0.0, 2.0, 0.5]
    assert numpy.allclose(get_hess(quad, p0, 1e-4), QUAD_A, rtol=1e-5)


def test_tiny_parameter_triggers_absolute_step():
    """
    When eps*p < 1e-6 the step size switches from fractional to absolute.
    With p=1e-9 and eps=1e-3 the product is 1e-12, so the step becomes
    1e-3 and the parameter goes one-sided.
    """
    eps = 1e-3
    p0 = [1e-9, 2.0, 0.5]
    grad = get_grad(quad, p0, eps).ravel()
    exact = quad_grad(p0)
    assert numpy.isclose(grad[0], exact[0] + 0.5*QUAD_A[0, 0]*eps, rtol=1e-4)


def test_negative_parameters_are_centered():
    """
    Regression test for the step-size guard. It once read

        if pval*eps_in < 1e-6:

    a *signed* comparison where magnitude was intended, so every negative
    parameter satisfied it and was silently demoted to a first-order
    one-sided difference with an absolute step. A negative parameter is
    not a small parameter; it gets the same centered fractional step as a
    positive one, and so a quadratic comes back exact.
    """
    p0 = [-1.0, -2.5, 0.5]
    assert numpy.allclose(get_grad(quad, p0, 0.01).ravel(), quad_grad(p0),
                          rtol=1e-8)
    assert numpy.allclose(get_hess(quad, p0, 0.01), QUAD_A, rtol=1e-6)


def test_log_space_parameters_below_one_are_centered():
    """
    The reach of that guard bug, and the reason it was worth fixing.
    get_godambe calls these routines on numpy.log(p0) when log=True, so
    every parameter below 1 arrives here negative -- and nu=0.5, T=0.1 are
    entirely routine in dadi. Each one used to land on the degraded path.

    Exercised through the same values log=True would produce.
    """
    p0 = list(numpy.log([0.5, 0.1, 2.0]))
    assert all(v < 0 for v in p0[:2]), 'setup: first two must be negative'
    assert numpy.allclose(get_grad(quad, p0, 0.01).ravel(), quad_grad(p0),
                          rtol=1e-8)


def test_small_magnitude_guard_still_fires_for_negatives():
    """
    Fixing the sign must not disarm the guard it was protecting. A
    genuinely tiny negative parameter should still fall back to an
    absolute step, exactly as its positive counterpart does.
    """
    eps = 1e-3
    for sign in (+1, -1):
        p0 = [sign*1e-9, 2.0, 0.5]
        grad = get_grad(quad, p0, eps).ravel()
        exact = quad_grad(p0)
        # One-sided forward difference, hence the O(eps) bias.
        assert numpy.isclose(grad[0], exact[0] + 0.5*QUAD_A[0, 0]*eps,
                             rtol=1e-4), f'sign={sign}'


def test_gradient_of_linear_function_is_constant():
    """A linear function has a constant gradient and a zero Hessian."""
    def lin(p):
        return 3.0 + 2.0*p[0] - 5.0*p[1]

    p0 = [4.0, 7.0]
    assert numpy.allclose(get_grad(lin, p0, 0.01).ravel(), [2.0, -5.0],
                          rtol=1e-9)
    assert numpy.allclose(get_hess(lin, p0, 0.01), numpy.zeros((2, 2)),
                          atol=1e-6)


def test_log_uncerts_consistent_with_linear_uncerts():
    """
    End-to-end guard on the step-size bug, at the level users see it.

    For a log-parametrized model the delta method gives
    sigma_log = sigma_linear / p, exactly at the MLE. This asserts that
    FIM_uncert agrees with itself across log=True and log=False.

    The bug inflated the log=True uncertainties by 50-75%, so the
    tolerance below does not need to be tight to catch it. It is loose on
    purpose: the spectrum is evaluated at the generating parameters rather
    than at a fitted optimum, so the gradient is not exactly zero and the
    delta-method identity holds only approximately.
    """
    import dadi

    numpy.random.seed(42)
    func_ex = dadi.Numerics.make_extrap_log_func(dadi.Demographics1D.two_epoch)
    ns, pts_l = (20,), [40, 50, 60]
    p_true = [0.5, 0.1]          # both below 1, so both were affected
    data = (10000*func_ex(p_true, ns, pts_l)).sample()

    lin = dadi.Godambe.FIM_uncert(func_ex, pts_l, p_true, data, log=False)
    log = dadi.Godambe.FIM_uncert(func_ex, pts_l, p_true, data, log=True)

    # Compare only the model parameters; theta is appended by multinom=True
    # and is not on the same footing.
    predicted = lin[:len(p_true)]/numpy.asarray(p_true)
    assert numpy.allclose(log[:len(p_true)], predicted, rtol=0.15), (
        f'log-space uncerts {log[:len(p_true)]} disagree with the delta-method '
        f'prediction {predicted} from the linear-space uncerts')


def test_single_parameter():
    """A one-parameter problem should not be a special case."""
    def f(p):
        return 2.0*p[0]**2 + 3.0*p[0]

    grad = get_grad(f, [1.5], 0.01)
    hess = get_hess(f, [1.5], 0.01)
    assert grad.shape == (1, 1)
    assert hess.shape == (1, 1)
    assert numpy.isclose(grad[0, 0], 4.0*1.5 + 3.0, rtol=1e-8)
    assert numpy.isclose(hess[0, 0], 4.0, rtol=1e-6)

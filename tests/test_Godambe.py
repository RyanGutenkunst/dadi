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


# ---------------------------------------------------------------------------
# Tier 2: statistical and structural properties of the public entry points.
#
# The statistical assertions below rest on one setup detail that is easy to
# get wrong. The Bartlett identity (J == H, so GIM == FIM) holds when the
# score is evaluated at the parameters that actually generated the bootstrap
# replicates. Evaluating at a theta fitted to one realization instead leaves
# a non-zero mean score, and J picks up an extra cU*cU' term that inflates
# trace(J H^-1) by a factor of two or more. So these tests use an explicit
# theta with multinom=False, fixed at the generating value.
# ---------------------------------------------------------------------------
import pytest

import dadi
import dadi.Godambe


NS = (20,)
PTS_L = [40, 50, 60]
P_TRUE = [0.5, 0.1]        # nu, T
THETA = 5e4
NBOOT = 100


@pytest.fixture(autouse=True)
def _clear_godambe_cache():
    """
    dadi.Godambe.cache is module-global and never evicted. Clear it around
    each test so results cannot depend on execution order.
    """
    dadi.Godambe.cache.clear()
    yield
    dadi.Godambe.cache.clear()


@pytest.fixture(scope='module')
def setup():
    """Data and unlinked bootstraps from a two_epoch model."""
    func_ex = dadi.Numerics.make_extrap_log_func(dadi.Demographics1D.two_epoch)
    numpy.random.seed(0)
    model = func_ex(P_TRUE, NS, PTS_L)
    data = (THETA*model).sample()
    # Independent Poisson replicates: unlinked by construction, which is the
    # regime where the Godambe correction should reduce to the Fisher one.
    numpy.random.seed(1)
    all_boot = [(THETA*model).sample() for _ in range(NBOOT)]

    # Explicit-theta wrapper, so the evaluation point is exactly the
    # parameter set that generated the bootstraps.
    def func_theta(p, ns, pts):
        return p[-1]*func_ex(p[:-1], ns, pts)

    return {'func_ex': func_ex, 'func_theta': func_theta, 'model': model,
            'data': data, 'all_boot': all_boot,
            'p_ext': list(P_TRUE) + [THETA],
            'll': dadi.Inference.ll_multinom(model, data)}


def test_gim_approximates_fim_for_unlinked_data(setup):
    """
    The headline statistical check. For unlinked data the composite
    likelihood is the true likelihood, so J == H and the Godambe
    information collapses to the Fisher information.

    The tolerance is loose because J is estimated from a finite number of
    bootstraps and H is observed rather than expected information. It is
    still far tighter than the failures it is meant to catch: a transposed
    matrix, a missing inverse or a sign error moves these by orders of
    magnitude, not by tens of percent.
    """
    fim = dadi.Godambe.FIM_uncert(setup['func_theta'], PTS_L, setup['p_ext'],
                                  setup['data'], multinom=False)
    dadi.Godambe.cache.clear()
    gim = dadi.Godambe.GIM_uncert(setup['func_theta'], PTS_L, setup['all_boot'],
                                  setup['p_ext'], setup['data'], multinom=False)
    ratio = gim/fim
    assert numpy.all(ratio > 1/1.5) and numpy.all(ratio < 1.5), (
        f'GIM/FIM = {ratio}, expected near 1 for unlinked data')


def test_effective_dimension_near_parameter_count(setup):
    """
    For unlinked data the effective number of parameters should be close
    to the actual number, here three (nu, T, theta).
    """
    eff = dadi.Godambe.effective_dimension(
        setup['func_theta'], PTS_L, setup['all_boot'], setup['p_ext'],
        setup['data'], multinom=False)
    assert 2.0 < eff < 5.0, f'effective dimension {eff}, expected near 3'


def test_lrt_adjust_order_unity_for_unlinked_data(setup):
    """
    The LRT adjustment factor corrects for composite-likelihood inflation.
    With unlinked data there is nothing to correct, so it should sit near
    one rather than near zero or ten.
    """
    adj = dadi.Godambe.LRT_adjust(setup['func_theta'], PTS_L, setup['all_boot'],
                                  setup['p_ext'], setup['data'],
                                  nested_indices=[1], multinom=False)
    assert 0.3 < adj < 2.0, f'LRT adjustment {adj}, expected near 1'


def test_mean_score_is_small_at_generating_parameters(setup):
    """
    cU, the mean score over bootstraps, should be near zero when evaluated
    at the generating parameters. This is the quantity whose neglect
    inflates J, so it is worth asserting directly.
    """
    GIM, H, J, cU = dadi.Godambe.get_godambe(
        setup['func_theta'], PTS_L, setup['all_boot'], setup['p_ext'],
        setup['data'], eps=0.01, log=False)
    # Compare the mean score to the spread of the scores it averages.
    scale = numpy.sqrt(numpy.diag(J))
    assert numpy.all(numpy.abs(cU.ravel())/scale < 0.5), (
        f'mean score {cU.ravel()} large relative to score spread {scale}')


def test_hessian_is_symmetric_and_positive_definite(setup):
    """-H at a good fit is a covariance-like matrix: symmetric, positive definite."""
    H = dadi.Godambe.get_godambe(setup['func_theta'], PTS_L, [], setup['p_ext'],
                                 setup['data'], eps=0.01, log=False,
                                 just_hess=True)
    assert numpy.allclose(H, H.T, rtol=0, atol=0)
    assert numpy.all(numpy.linalg.eigvals(H) > 0), (
        f'H has non-positive eigenvalues {numpy.linalg.eigvals(H)}')


# --- Exact identities. These carry no statistical slack and should hold to
# --- floating-point precision, so they are the sharpest tests in the file.

def test_multinom_matches_explicit_theta(setup):
    """
    multinom=True appends the optimal theta and wraps the model. Doing that
    by hand must give identical uncertainties.
    """
    theta_opt = dadi.Inference.optimal_sfs_scaling(setup['model'], setup['data'])
    auto = dadi.Godambe.FIM_uncert(setup['func_ex'], PTS_L, P_TRUE,
                                   setup['data'], multinom=True)
    dadi.Godambe.cache.clear()
    manual = dadi.Godambe.FIM_uncert(setup['func_theta'], PTS_L,
                                     list(P_TRUE) + [theta_opt], setup['data'],
                                     multinom=False)
    assert numpy.allclose(auto, manual, rtol=1e-10)


def test_claic_matches_its_definition(setup):
    """CLAIC = -2*ll + 2*effective_dimension."""
    eff = dadi.Godambe.effective_dimension(
        setup['func_theta'], PTS_L, setup['all_boot'], setup['p_ext'],
        setup['data'], multinom=False)
    dadi.Godambe.cache.clear()
    claic = dadi.Godambe.CLAIC(setup['ll'], setup['func_theta'], PTS_L,
                               setup['all_boot'], setup['p_ext'], setup['data'],
                               multinom=False)
    assert numpy.isclose(claic, -2*setup['ll'] + 2*eff, rtol=1e-10)


def test_clbic_matches_its_definition(setup):
    """CLBIC = -2*ll + effective_dimension*log(number of segregating sites)."""
    eff = dadi.Godambe.effective_dimension(
        setup['func_theta'], PTS_L, setup['all_boot'], setup['p_ext'],
        setup['data'], multinom=False)
    dadi.Godambe.cache.clear()
    clbic = dadi.Godambe.CLBIC(setup['ll'], setup['func_theta'], PTS_L,
                               setup['all_boot'], setup['p_ext'], setup['data'],
                               multinom=False)
    expected = -2*setup['ll'] + eff*numpy.log(setup['data'].sum())
    assert numpy.isclose(clbic, expected, rtol=1e-10)


def test_effective_dimension_equals_trace_of_H_GIM_inverse(setup):
    """effective_dimension is trace(H GIM^-1) by construction."""
    GIM, H, J, cU = dadi.Godambe.get_godambe(
        setup['func_theta'], PTS_L, setup['all_boot'], setup['p_ext'],
        setup['data'], eps=0.01, log=False)
    expected = numpy.trace(numpy.matmul(H, numpy.linalg.inv(GIM)))
    dadi.Godambe.cache.clear()
    eff = dadi.Godambe.effective_dimension(
        setup['func_theta'], PTS_L, setup['all_boot'], setup['p_ext'],
        setup['data'], multinom=False)
    assert numpy.isclose(eff, expected, rtol=1e-10)


def test_gim_uncerts_are_sqrt_diag_of_inverse_GIM(setup):
    uncerts, GIM, H = dadi.Godambe.GIM_uncert(
        setup['func_theta'], PTS_L, setup['all_boot'], setup['p_ext'],
        setup['data'], multinom=False, return_GIM=True)
    assert numpy.allclose(uncerts,
                          numpy.sqrt(numpy.diag(numpy.linalg.inv(GIM))),
                          rtol=1e-12)
    assert GIM.shape == (3, 3) and H.shape == (3, 3)


def test_fim_uncerts_are_sqrt_diag_of_inverse_FIM(setup):
    uncerts, FIM = dadi.Godambe.FIM_uncert(
        setup['func_theta'], PTS_L, setup['p_ext'], setup['data'],
        multinom=False, return_FIM=True)
    assert numpy.allclose(uncerts,
                          numpy.sqrt(numpy.diag(numpy.linalg.inv(FIM))),
                          rtol=1e-12)


def test_wald_stat_is_zero_at_the_null(setup):
    """Testing a parameter against its own value gives exactly zero."""
    w = dadi.Godambe.Wald_stat(setup['func_theta'], PTS_L, setup['all_boot'],
                               setup['p_ext'], setup['data'],
                               nested_indices=[1], full_params=setup['p_ext'],
                               multinom=False)
    assert w == 0.0


def test_wald_stat_grows_with_departure_from_the_null(setup):
    """A larger parameter offset must give a larger Wald statistic."""
    def wald(T):
        dadi.Godambe.cache.clear()
        return dadi.Godambe.Wald_stat(
            setup['func_theta'], PTS_L, setup['all_boot'], setup['p_ext'],
            setup['data'], nested_indices=[1],
            full_params=[P_TRUE[0], T, THETA], multinom=False)

    assert 0 < wald(0.11) < wald(0.12) < wald(0.13)


def test_wald_adj_and_org_returns_both(setup):
    adj, org = dadi.Godambe.Wald_stat(
        setup['func_theta'], PTS_L, setup['all_boot'], setup['p_ext'],
        setup['data'], nested_indices=[1],
        full_params=[P_TRUE[0], 0.12, THETA], multinom=False,
        adj_and_org=True)
    # The adjustment rescales by GIM vs H; both are positive here.
    assert adj > 0 and org > 0 and adj != org


def test_wald_stat_rejects_mismatched_full_params(setup):
    with pytest.raises(KeyError):
        dadi.Godambe.Wald_stat(setup['func_theta'], PTS_L, setup['all_boot'],
                               setup['p_ext'], setup['data'],
                               nested_indices=[1], full_params=[0.1, 0.2, 0.3, 0.4],
                               multinom=False)


def test_boot_theta_adjusts_rejected_with_multinom(setup):
    """theta is fitted internally when multinom=True, so adjusting it is ill-posed."""
    with pytest.raises(ValueError):
        dadi.Godambe.GIM_uncert(setup['func_ex'], PTS_L, setup['all_boot'],
                                P_TRUE, setup['data'], multinom=True,
                                boot_theta_adjusts=[1.0]*NBOOT)
    with pytest.raises(ValueError):
        dadi.Godambe.LRT_adjust(setup['func_ex'], PTS_L, setup['all_boot'],
                                P_TRUE, setup['data'], nested_indices=[1],
                                multinom=True, boot_theta_adjusts=[1.0]*NBOOT)


# --- sum_chi2_ppf

def test_sum_chi2_ppf_matches_chi2_survival():
    """
    Despite the name, this returns an upper-tail probability, not an
    inverse cdf -- and that is what callers want, since the documented
    usage is pval = sum_chi2_ppf(D, weights). With weights=(0,1) it is
    exactly the one-degree-of-freedom chi-squared survival function.
    """
    import scipy.stats.distributions as ssd
    for x in (0.5, 2.0, 5.0):
        assert numpy.isclose(dadi.Godambe.sum_chi2_ppf(x, weights=(0, 1)),
                             ssd.chi2.sf(x, 1), rtol=1e-10)


def test_sum_chi2_ppf_boundary_weights():
    """
    The half-half mixture used for a single parameter on a boundary, the
    case the YRI_CEU example exercises.
    """
    import scipy.stats.distributions as ssd
    x = 3.0
    expected = 1 - (0.5 + 0.5*ssd.chi2.cdf(x, 1))
    assert numpy.isclose(dadi.Godambe.sum_chi2_ppf(x, weights=(0.5, 0.5)),
                         expected, rtol=1e-10)


def test_sum_chi2_ppf_accepts_array():
    """
    Array input, which the function's own comment says it should handle.
    """
    xs = [2.0, 3.0, 4.0]
    got = dadi.Godambe.sum_chi2_ppf(xs, weights=(0, 1))
    one_at_a_time = [dadi.Godambe.sum_chi2_ppf(x, weights=(0, 1)) for x in xs]
    assert numpy.allclose(got, one_at_a_time, rtol=1e-12)
    assert numpy.ndim(got) == 1


def test_sum_chi2_ppf_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError):
        dadi.Godambe.sum_chi2_ppf(2.0, weights=(0.3, 0.3))


# ---------------------------------------------------------------------------
# Tier 3: regression pins.
#
# These exist to measure drift, not to assert correctness. When the
# differentiation backend is swapped to scipy.differentiate these values are
# expected to move; the point is that the move is visible and quantified
# rather than silent. Update them deliberately, never reflexively.
#
# rtol is 1e-4 rather than machine precision because the underlying PDE
# solve can differ slightly across platforms and BLAS builds.
# ---------------------------------------------------------------------------

def test_regression_uncertainties(setup):
    pinned = {
        'FIM_explicit': [1.1002743886e-02, 7.5080134966e-03, 2.3945697647e+02],
        'GIM_explicit': [1.1675923484e-02, 8.7351496435e-03, 2.4773749864e+02],
    }
    fim = dadi.Godambe.FIM_uncert(setup['func_theta'], PTS_L, setup['p_ext'],
                                  setup['data'], multinom=False)
    assert numpy.allclose(fim, pinned['FIM_explicit'], rtol=1e-4)
    dadi.Godambe.cache.clear()
    gim = dadi.Godambe.GIM_uncert(setup['func_theta'], PTS_L, setup['all_boot'],
                                  setup['p_ext'], setup['data'], multinom=False)
    assert numpy.allclose(gim, pinned['GIM_explicit'], rtol=1e-4)


def test_regression_multinom_uncertainties(setup):
    """
    Pinned with multinom=True as well, since that path appends theta and is
    what most users call.
    """
    fim = dadi.Godambe.FIM_uncert(setup['func_ex'], PTS_L, P_TRUE, setup['data'])
    assert numpy.allclose(
        fim, [1.0142985818e-02, 6.8397582683e-03, 2.3912629534e+02], rtol=1e-4)
    dadi.Godambe.cache.clear()
    fim_log = dadi.Godambe.FIM_uncert(setup['func_ex'], PTS_L, P_TRUE,
                                      setup['data'], log=True)
    assert numpy.allclose(fim_log, [0.0196837079, 0.066385044, 0.0047450551],
                          rtol=1e-4)


def test_regression_information_criteria(setup):
    eff = dadi.Godambe.effective_dimension(
        setup['func_theta'], PTS_L, setup['all_boot'], setup['p_ext'],
        setup['data'], multinom=False)
    assert numpy.isclose(eff, 3.5042890411, rtol=1e-4)
    dadi.Godambe.cache.clear()
    claic = dadi.Godambe.CLAIC(setup['ll'], setup['func_theta'], PTS_L,
                               setup['all_boot'], setup['p_ext'], setup['data'],
                               multinom=False)
    assert numpy.isclose(claic, 220.7118514858, rtol=1e-4)
    dadi.Godambe.cache.clear()
    clbic = dadi.Godambe.CLBIC(setup['ll'], setup['func_theta'], PTS_L,
                               setup['all_boot'], setup['p_ext'], setup['data'],
                               multinom=False)
    assert numpy.isclose(clbic, 255.4637324591, rtol=1e-4)


def test_regression_test_statistics(setup):
    adj = dadi.Godambe.LRT_adjust(setup['func_theta'], PTS_L, setup['all_boot'],
                                  setup['p_ext'], setup['data'],
                                  nested_indices=[1], multinom=False)
    assert numpy.isclose(adj, 0.6829407530, rtol=1e-4)
    dadi.Godambe.cache.clear()
    score = dadi.Godambe.score_stat(setup['func_theta'], PTS_L,
                                    setup['all_boot'], setup['p_ext'],
                                    setup['data'], nested_indices=[1],
                                    multinom=False)
    assert numpy.isclose(score, 0.0009001733, rtol=1e-3)
    dadi.Godambe.cache.clear()
    wald = dadi.Godambe.Wald_stat(setup['func_theta'], PTS_L, setup['all_boot'],
                                  setup['p_ext'], setup['data'],
                                  nested_indices=[1],
                                  full_params=[0.5, 0.12, THETA],
                                  multinom=False)
    assert numpy.isclose(wald, 80.8155504301, rtol=1e-4)

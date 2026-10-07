"""Source controls for MATLAB minimax rational p/q output conversion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.minimax import MinimaxRationalResult, minimax

_ANALYTIC_ATOL = 5e-13


def _assert64(actual, expected, *, atol=_ANALYTIC_ATOL):
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=0.0, atol=atol
    )


def _result(
    support,
    w_n,
    w_d,
    *,
    m,
    n,
    domain=(-1.0, 1.0),
    polynomial_coeffs=None,
    success=True,
    r=None,
):
    return MinimaxRationalResult(
        r=(lambda x: jnp.zeros_like(jnp.asarray(x))) if r is None else r,
        err=0.0,
        xk=jnp.asarray([]),
        delta=0.0,
        iter=0,
        m=m,
        n=n,
        support=jnp.asarray(support),
        wN=jnp.asarray(w_n),
        wD=jnp.asarray(w_d),
        poles=jnp.asarray([]),
        zeros=jnp.asarray([]),
        domain=domain,
        success=success,
        polynomial_coeffs=(
            None if polynomial_coeffs is None else jnp.asarray(polynomial_coeffs)
        ),
    )


def test_source_node_conversion_returns_analytic_p_and_q_separately():
    result = _result([-1.0, 1.0], [2.0, 3.0], [4.0, 5.0], m=1, n=1)
    p, q = result.as_chebfuns()
    x = jnp.asarray([-1.0, 0.0, 1.0])
    # MATLAB's nodal construction gives p(x)=5*x+1 and q(x)=-(9*x+1).
    _assert64(p(x), [-4.0, 1.0, 6.0])
    _assert64(q(x), [8.0, -1.0, -10.0])
    _assert64(p(x) / q(x), [-0.5, -1.0, -0.6])


def test_source_conversion_preserves_noncanonical_domain():
    result = _result(
        [2.0, 4.0], [2.0, 3.0], [4.0, 5.0],
        m=1, n=1, domain=(2.0, 4.0),
    )
    p, q = result.as_chebfuns()
    x = jnp.asarray([2.0, 3.0, 4.0])
    _assert64(p(x), [-4.0, 1.0, 6.0])
    _assert64(q(x), [8.0, -1.0, -10.0])


def test_source_scale_is_carried_by_numerator_weights_only():
    result = _result([-1.0, 1.0], [14.0, 21.0], [4.0, 5.0], m=1, n=1)
    p, q = result.as_chebfuns()
    x = jnp.asarray([-1.0, 0.0, 1.0])
    _assert64(p(x), [-28.0, 7.0, 42.0])
    _assert64(q(x), [8.0, -1.0, -10.0])


def test_odd_degree_reduction_returns_source_zero_over_one_pair():
    result = _result([], [], [], m=-1, n=2)
    p, q = result.as_chebfuns()
    x = jnp.asarray([-1.0, 0.0, 1.0])
    _assert64(p(x), [0.0, 0.0, 0.0])
    _assert64(q(x), [1.0, 1.0, 1.0])


def test_zero_target_without_support_returns_source_zero_over_one_pair():
    result = _result([], [], [], m=2, n=2)
    p, q = result.as_chebfuns()
    x = jnp.asarray([-1.0, 0.0, 1.0])
    _assert64(p(x), [0.0, 0.0, 0.0])
    _assert64(q(x), [1.0, 1.0, 1.0])


def test_degree_zero_denominator_keeps_polynomial_result():
    coeffs = jnp.asarray([1.0, 2.0, 0.5, 0.0, 0.0])
    result = _result(
        [], [], [], m=2, n=0, polynomial_coeffs=coeffs
    )
    p, q = result.as_chebfuns()
    x = jnp.asarray([-1.0, 0.0, 1.0])
    _assert64(p(x), [-0.5, 0.5, 3.5])
    _assert64(q(x), [1.0, 1.0, 1.0])
    assert len(p) == 3


def test_solver_result_pair_matches_stable_handle_and_target_scaling():
    f = lambda x: 1.0 / (2.0 + x)  # noqa: E731
    f_scaled = lambda x: 3.0 / (2.0 + x)  # noqa: E731
    base = minimax(f, 1, rational=True, denom=1)
    scaled = minimax(f_scaled, 1, rational=True, denom=1)
    x = jnp.asarray([-0.75, -0.25, 0.25, 0.75])
    for result in (base, scaled):
        p, q = result.as_chebfuns()
        _assert64(result.r(x), p(x) / q(x), atol=2e-11)
    _assert64(scaled.r(x), 3.0 * base.r(x), atol=2e-8)


def test_unsuccessful_empty_rational_result_is_not_misread_as_zero():
    result = _result([], [], [], m=1, n=1, success=False)
    with pytest.raises(ValueError, match="no valid barycentric representation"):
        result.as_chebfuns()


@pytest.mark.parametrize("disable_jit", [False, True])
def test_conversion_never_uses_concrete_numpy_transform_mirrors(
    disable_jit, monkeypatch
):
    from chebfunjax.utils import transforms

    def forbidden(_values):
        raise AssertionError("NumPy transform mirror must not be called")

    monkeypatch.setattr(transforms, "_vals2coeffs_np", forbidden)
    monkeypatch.setattr(transforms, "_coeffs2vals_np", forbidden)
    result = _result([-1.0, 1.0], [2.0, 3.0], [4.0, 5.0], m=1, n=1)
    with jax.disable_jit(disable_jit):
        p, q = result.as_chebfuns()
        x = jnp.asarray([-1.0, 0.0, 1.0])
        _assert64(p(x), [-4.0, 1.0, 6.0])
        _assert64(q(x), [8.0, -1.0, -10.0])


def test_unequal_degree_pair_handles_every_sample_at_support():
    result = _result(
        [-1.0, 0.0, 1.0], [0.5, -1.0, 0.5], [-1.5, 2.0, -1.5],
        m=0, n=2,
    )
    p, q = result.as_chebfuns()
    x = jnp.asarray([-1.0, 0.0, 1.0])
    # Every source Chebyshev sample collides with a support. The analytic
    # source limit should yield p(x)=1 and q(x)=x**2+2.
    _assert64(p(x), [1.0, 1.0, 1.0])
    _assert64(q(x), [3.0, 2.0, 3.0])

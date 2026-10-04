"""Independent analytic controls for the source Brent inverse callback.

Provenance
----------
MATLAB source : @chebfun/inv.m, local fInverseBrent
Chebfun commit: 7574c77

These controls supplement tests/chebfun/test_inv.m; they are not new MATLAB
assertion groups. Empty inputs and the explicit safety cap are Python adapters.
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.inverse import _brent

EPS = float(jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize("slope,offset,domain", [
    (1.0, 0.0, (-1.0, 1.0)),
    (-2.0, 1.0, (-1.0, 1.0)),
    (0.5, -1.0, (2.0, 6.0)),
])
def test_affine_inverse_with_endpoint_and_vector_targets(slope, offset, domain):
    a, b = domain
    exact = jnp.linspace(a, b, 19)
    targets = slope * exact + offset
    actual = _brent(lambda x: slope * x + offset, targets, a, b)
    assert float(jnp.max(jnp.abs(actual - exact))) < 100 * EPS * max(abs(a), abs(b))


def test_nonlinear_inverse_matches_logarithm():
    targets = jnp.linspace(jnp.exp(-1.0), jnp.exp(1.0), 100)
    actual = _brent(jnp.exp, targets, -1.0, 1.0)
    assert float(jnp.max(jnp.abs(actual - jnp.log(targets)))) < 100 * EPS


def test_scalar_target_preserves_scalar_shape():
    actual = _brent(jnp.exp, jnp.asarray(1.5), -1.0, 1.0)
    assert actual.shape == ()
    assert abs(float(actual - jnp.log(1.5))) < 100 * EPS


def test_compiled_callback_uses_changed_chebfun_coefficients():
    # Identical PyTree shapes, different array leaves: a callback closed over
    # the first function would return the wrong inverse on the second call.
    first = cj.Chebfun.from_coeffs(jnp.asarray([0.0, 1.0]))
    second = cj.Chebfun.from_coeffs(jnp.asarray([1.0, 2.0]))
    targets = jnp.linspace(-0.9, 0.9, 17)
    first_actual = _brent(first, targets, -1.0, 1.0)
    second_actual = _brent(second, targets, -1.0, 1.0)
    assert float(jnp.max(jnp.abs(first_actual - targets))) < 100 * EPS
    assert float(jnp.max(jnp.abs(second_actual - (targets - 1.0) / 2))) < 100 * EPS


def test_empty_target_shape_adapter():
    actual = _brent(jnp.exp, jnp.empty((0,)), -1.0, 1.0)
    assert actual.shape == (0,)
    assert actual.dtype == jnp.float64


def test_unconverged_safety_cap_raises_instead_of_returning_partial_result():
    with pytest.raises(RuntimeError, match="still unconverged"):
        actual = _brent(jnp.exp, jnp.asarray([1.5]), -1.0, 1.0,
                        max_iterations=0)
        actual.block_until_ready()


def test_callback_retains_explicit_breakpoint_values_inside_compiled_loop():
    # Source's first trial for identity and target zero is exactly zero.
    # Assigning f(0)=0.25 must leave a nonzero residual after that trial.
    # This isolates callback state semantics; it does not claim a monotonic
    # inverse exists for the function with this isolated point assignment.
    original = cj.Chebfun.identity()
    unchanged = _brent(original, jnp.asarray([0.0]), -1.0, 1.0,
                       max_iterations=1)
    assert float(unchanged[0]) == 0.0
    assigned = original.define_point(0.0, 0.25)
    with pytest.raises(RuntimeError, match="still unconverged"):
        result = _brent(assigned, jnp.asarray([0.0]), -1.0, 1.0,
                        max_iterations=1)
        result.block_until_ready()

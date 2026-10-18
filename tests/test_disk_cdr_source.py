"""Pinned source CDR weights and zero-pivot callers, CPU binary64.

The full16native comparison is retained in evidence. Two inherited finite
subnormal failures remain explicit: maxfinite reciprocal and half-minnormal
input. The14qualified rows below do not establish universal IEEE/CDR parity.
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech

_FIXTURES = Path(__file__).parent / "fixtures"
_IEEE = json.loads((_FIXTURES / "disk_cdr_ieee_native_20261018.json").read_text())
_DEFERRED = {"7fefffffffffffff", "0008000000000000"}
_QUALIFIED = [r for r in _IEEE["rows"] if r["input"] not in _DEFERRED]
_ZERO = json.loads((_FIXTURES / "disk_zero_poles_native_20261018.json").read_text())


def _number(word):
    return jax.lax.bitcast_convert_type(jnp.uint64(int(word, 16)), jnp.float64)


def _constant(pivots, col=1.0):
    pivots = jnp.atleast_1d(pivots)
    c = Chebtech2.from_coeffs(jnp.array([col]))
    r = Trigtech.from_coeffs(jnp.ones(1), is_real=True)
    return Diskfun(
        cols=[c] * pivots.size,
        rows=[r] * pivots.size,
        pivots=pivots,
        idx_plus=tuple(range(pivots.size)),
        idx_minus=(),
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("case", _QUALIFIED, ids=lambda c: c["input"])
def test_native_cdr_qualified_words(case, compiled):
    f = _constant(_number(case["input"]))

    def op(obj):
        return obj.cdr()[1][0, 0]

    actual = jax.jit(op)(f) if compiled else op(f)
    assert jax.lax.bitcast_convert_type(actual, jnp.uint64) == jnp.uint64(int(case["cdr"], 16))


@pytest.mark.parametrize("case", [1, 2], ids=["rescaled", "tiny_T2"])
@pytest.mark.parametrize("compiled", [False, True])
def test_native_zero_pivot_callers(case, compiled):
    native = _ZERO["cases"][case]["output"]
    assert native["pivot"]["real"] == ["0000000000000000"]
    assert native["cols"]["real"] == ["0000000000000000"]
    assert native["rows"]["real"] == ["3ff0000000000000"]
    f = _constant(_number(native["pivot"]["real"][0]), col=0.0)
    x, y = jnp.array([-0.2, 0.0, 0.3, 0.9]), jnp.array([0.1, 0.0, -0.2, 0.0])

    # The Cartesian domain guard is eager. Validate it normally, then
    # qualify the documented JIT-safe polar evaluator on the same points.
    theta, radius = jnp.arctan2(y, x), jnp.hypot(x, y)

    def op(obj):
        return obj(theta, radius)

    actual = jax.jit(op)(f) if compiled else op(f)
    assert jnp.array_equal(actual, jnp.zeros(4))
    assert jnp.array_equal(f.feval_cart(x, y), jnp.zeros(4))
    assert jnp.array_equal(f.coeffs2(), jnp.zeros((1, 1)))
    assert jnp.array_equal(f.sample(5, 5), jnp.zeros((5, 5)))
    assert f.iszero() and not f.isempty()
    assert f.numerical_rank() == 0
    assert f.vscale() == 0 and f.norm() == 0
    assert f.sum2() == 0 and f.mean2() == 0
    assert f.integral("unitcircle") == 0
    for dim in (1, 2):
        assert jnp.array_equal(f.sum(dim=dim)(jnp.array([0.2, 0.5])), jnp.zeros(2))


def test_iszero_keeps_source_slice_test_for_nonzero_zero_pivot_factors():
    # Native iszero uses raw reciprocals before FEVAL and the final slice test.
    # CDR-zero weights alone do not imply that its slice test returnsTrue.
    f = _constant(jnp.array([0.0]))
    assert jnp.array_equal(f(jnp.array([0.2]), jnp.array([0.3])), jnp.zeros(1))
    assert not f.iszero()


def test_nan_weight_is_not_silently_zeroed_in_coeffs2():
    f = _constant(jnp.array([jnp.nan]))
    assert jnp.all(jnp.isnan(f.coeffs2()))


def test_finite_pivot_first_and_second_derivatives():
    x = jnp.array([-4.0, -0.5, 0.5, 4.0])
    f = _constant(jnp.ones(4))

    def weights(p):
        return jnp.diag(eqx.tree_at(lambda obj: obj.pivots, f, p).cdr()[1])

    value, tangent = jax.jvp(jax.jit(weights), (x,), (jnp.ones_like(x),))
    assert jnp.array_equal(value, 1 / x)
    assert jnp.allclose(tangent, -1 / x**2, rtol=2e-15, atol=0)
    second = jax.jacfwd(jax.jacfwd(weights))(x)
    assert jnp.allclose(
        second[jnp.arange(4), jnp.arange(4), jnp.arange(4)], 2 / x**3, rtol=2e-15, atol=0
    )


def test_zero_field_coordinate_derivatives():
    f = _constant(jnp.array([0.0]), col=0.0)

    def polar(z):
        return f(z[0], z[1])

    for point in [jnp.array([0.2, 0.3]), jnp.zeros(2)]:
        assert jnp.array_equal(jax.jit(jax.grad(polar))(point), jnp.zeros(2))
        assert jnp.array_equal(jax.jit(jax.hessian(polar))(point), jnp.zeros((2, 2)))

    # Cartesian origin differentiation of hypot is a separate mapping issue.
    def cart(z):
        # Compose the conversion with polar FEVAL; feval_cart's eager
        # rejection guard is not an AD/JIT contract being added here.
        return f(jnp.arctan2(z[1], z[0]), jnp.hypot(z[0], z[1]))

    assert jnp.array_equal(jax.jit(jax.grad(cart))(jnp.array([0.2, -0.3])), jnp.zeros(2))

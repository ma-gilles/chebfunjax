"""Bounded CPU checks derived from MATLAB Chebfun power assertion groups 19–20.

Provenance
----------
MATLAB source : tests/chebfun/test_power.m, @chebfun/power.m, @chebfun/compose.m
Chebfun commit: 7574c77

Source tests: MATLAB tests/chebfun/test_power.m (7574c77), passes
19–20, with unchanged ``10*eps*vscale(h)`` bounds. The test file defines its
own local normest(f,dom): it samples ``sum(dom)*rand(10,1)-dom(1)`` after
seedRNG(6178). NumPy RandomState(6178) below is a deterministic MT19937
adapter, not a claim of MATLAB stream equivalence. Matrix infinity norm uses
maximum row sum (NumPy's ``ord=inf``), not flatten-maximum.

The independent metadata controls below are JAX/chebfunjax source-contract
controls, not additional MATLAB ``test_power.m`` pass claims. They check row
orientation, explicit breakpoint values, overlap breakpoints, and empty
operands. Adaptive construction is eager; constructed evaluation supports JAX AD.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
DOM = (0.1, 2.0)


def _local_test_normest_adapter(f, dom=DOM):
    """Translate the test file's literal random-sample helper formula.

    MATLAB helper at test_power.m:406–418 reseeds 6178 per call. This uses
    NumPy MT19937 with that integer as a reproducible but different stream.
    For vector-valued evaluations, MATLAB ``norm(...,inf)`` is the induced
    matrix infinity norm, the largest row sum.
    """
    rng = np.random.RandomState(6178)
    x = sum(dom) * rng.random_sample(10) - dom[0]
    values = np.asarray(f(jnp.asarray(x)))
    if values.ndim == 1:
        return float(np.max(np.abs(values)))
    return float(np.linalg.norm(values, ord=np.inf))


def test_source_pass19_scalar_x_to_x():
    x = cj.chebfun(lambda t: t, domain=DOM)
    result = x ** x
    expected = cj.chebfun(lambda t: t ** t, domain=DOM)
    error = _local_test_normest_adapter(result - expected, DOM)
    assert error < 10 * EPS * float(expected.vscale)


def test_source_pass20_array_valued_x_to_x():
    x = cj.chebfun(
        lambda t: jnp.stack([t, jnp.exp(1j * t)], axis=-1), domain=DOM
    )
    result = x ** x
    expected = cj.chebfun(
        lambda t: jnp.stack([t ** t, jnp.exp(1j * t) ** jnp.exp(1j * t)], axis=-1),
        domain=DOM,
    )
    error = _local_test_normest_adapter(result - expected, DOM)
    assert error < 10 * EPS * float(expected.vscale)


def test_binary_power_preserves_scalar_row_orientation():
    base = cj.chebfun(lambda t: 1.5 + 0.1 * t, domain=(-1.0, 1.0)).T
    exponent = cj.chebfun(lambda t: 0.5 + 0.1 * t, domain=(-1.0, 1.0)).T
    result = base ** exponent
    assert result.is_transposed
    assert result.size() == base.size()
    xs = jnp.linspace(-0.9, 0.9, 31)
    np.testing.assert_allclose(
        np.asarray(result(xs)), np.asarray(base(xs) ** exponent(xs)),
        rtol=20 * EPS, atol=20 * EPS,
    )


def test_binary_power_preserves_row_array_orientation_and_pointvalues():
    domain = (-1.0, 0.0, 1.0)
    base = cj.chebfun(
        lambda t: jnp.stack([2.0 + 0.1 * t, 3.0 + 0.2 * t], axis=-1),
        domain=domain,
    ).T.set_point_values(jnp.asarray([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
    exponent = cj.chebfun(
        lambda t: jnp.stack([1.0 + 0.1 * t, 1.5 + 0.1 * t], axis=-1),
        domain=domain,
    ).T.set_point_values(jnp.asarray([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]))
    result = base ** exponent
    assert result.is_transposed and result.size() == base.size()
    expected_points = jnp.power(base._breakpoint_values(), exponent._breakpoint_values())
    np.testing.assert_allclose(
        np.asarray(result._breakpoint_values()), np.asarray(expected_points),
        rtol=20 * EPS, atol=20 * EPS,
    )


def test_binary_power_uses_stored_mean_at_jump_before_power():
    base = cj.chebfun([jnp.ones_like, lambda t: 3*jnp.ones_like(t)],
                      domain=(-1.0, 0.0, 1.0))
    exponent = cj.chebfun(lambda t: 2.0 + 0.0*t)
    result = base ** exponent
    # Source compose maps mean(1,3)^2 = 4, rather than mean(1^2,3^2) = 5.
    np.testing.assert_array_equal(np.asarray(result.point_values), [1.0, 4.0, 9.0])
    assert float(result(jnp.asarray(0.0))) == 4.0


def test_binary_power_merges_distinct_breaks_and_maps_pointvalues():
    base = cj.chebfun(
        [lambda t: 1.0 + 0.1 * t, lambda t: 2.0 + 0.1 * t],
        domain=(-1.0, 0.0, 1.0),
    ).set_point_values(jnp.asarray([0.9, 1.5, 2.1]))
    exponent = cj.chebfun(
        [lambda t: 1.5 + 0.1 * t, lambda t: 2.0 + 0.1 * t],
        domain=(-1.0, 0.25, 1.0),
    ).set_point_values(jnp.asarray([1.4, 1.8, 2.1]))
    result = base ** exponent
    merged = jnp.asarray([-1.0, 0.0, 0.25, 1.0])
    assert np.allclose(np.asarray(result.domain.breakpoints), np.asarray(merged),
                       rtol=0.0, atol=0.0)
    expected = jnp.power(base(merged), exponent(merged))
    np.testing.assert_allclose(
        np.asarray(result._breakpoint_values()), np.asarray(expected),
        rtol=20 * EPS, atol=20 * EPS,
    )


def test_binary_power_empty_operand_returns_empty():
    empty = Chebfun.empty()
    f = cj.chebfun(lambda t: 1.0 + t, domain=(-1.0, 1.0))
    assert (empty ** f).isempty()
    assert (f ** empty).isempty()


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_bounded_binary_power_value_and_jvp_on_each_chebtech_kind(Tech):
    domain = Domain((-1.0, 1.0))
    base_tech = Tech.from_coeffs(jnp.asarray([2.0, 0.1]))
    exponent_tech = Tech.from_coeffs(jnp.asarray([0.3, 0.2]))
    base = Chebfun(
        funs=[_Piece(tech=base_tech, interval=(-1.0, 1.0))], domain=domain
    )
    exponent = Chebfun(
        funs=[_Piece(tech=exponent_tech, interval=(-1.0, 1.0))], domain=domain
    )
    result = base ** exponent

    xs = jnp.asarray([-0.8, -0.1, 0.25, 0.9])
    fvals = 2.0 + 0.1 * xs
    gvals = 0.3 + 0.2 * xs
    expected = fvals ** gvals
    actual = result(xs)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=20 * EPS, atol=20 * EPS
    )

    _, derivative = jax.jvp(
        lambda x: result(x), (xs,), (jnp.ones_like(xs),)
    )
    expected_derivative = expected * (
        0.2 * jnp.log(fvals) + 0.1 * gvals / fvals
    )
    np.testing.assert_allclose(
        np.asarray(derivative), np.asarray(expected_derivative),
        rtol=128 * EPS, atol=128 * EPS,
    )


@pytest.mark.parametrize("array_base", [True, False])
def test_binary_power_broadcasts_internal_pointvalues(array_base):
    array = cj.chebfun(lambda t: jnp.stack([2+.1*t, 3+.2*t], axis=-1),
                       domain=(-1., 0., 1.)).T
    array = array.set_point_values(jnp.asarray([[2., 3.], [4., 5.], [6., 7.]]))
    scalar = cj.chebfun(lambda t: .5+.1*t, domain=(-1., 0., 1.)).T
    scalar = scalar.set_point_values(jnp.asarray([.5, 2., 3.]))
    base, exponent = (array, scalar) if array_base else (scalar, array)
    result = base ** exponent
    assert result.is_transposed and result.size() == array.size()
    if array_base:
        expected = array._breakpoint_values() ** scalar._breakpoint_values()[:, None]
    else:
        expected = scalar._breakpoint_values()[:, None] ** array._breakpoint_values()
    np.testing.assert_allclose(result._breakpoint_values(), expected,
                               rtol=20*EPS, atol=20*EPS)

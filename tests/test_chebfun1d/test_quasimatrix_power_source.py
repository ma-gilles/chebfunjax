"""Bounded checks derived from Quasimatrix cases from MATLAB test_power.m.

These checks use NumPy RandomState(MT19937) with seed 6178 to make
the source test's local random sampling reproducible in Python; MATLAB's
``seedRNG(6178)`` stream equivalence is unverified.

Provenance
----------
MATLAB source : tests/chebfun/test_power.m, passes 6, 8, 10, 12, 16, 18, 21
Chebfun commit: 7574c77
MATLAB behavior: @chebfun/power.m and @chebfun/dimCheck.m
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


def _source_sample(seed: int, *, domain=None):
    """Apply the local MATLAB test_power.m sample formula with Python MT19937.

    For a one-argument local normest call the MATLAB helper uses
    ``2*rand(100,1)-1``. With a domain it uses the literal, unusual expression
    ``sum(dom)*rand(10,1)-dom(1)`` rather than interval length scaling.
    """
    rng = np.random.RandomState(seed)
    if domain is None:
        return 2.0 * rng.rand(100) - 1.0
    dom = np.asarray(domain, dtype=np.float64)
    return float(dom.sum()) * rng.rand(10) - float(dom[0])


def _matrix_inf_norm(values):
    """MATLAB norm(A,inf): maximum absolute row sum for matrix outputs."""
    arr = np.asarray(values)
    if arr.ndim <= 1:
        return float(np.max(np.abs(arr), initial=0.0))
    return float(np.linalg.norm(arr, ord=np.inf))


def _q_minus_array_norm(q, f, x):
    """Evaluate a Quasimatrix-minus-array-Chebfun comparison at sample x."""
    return _matrix_inf_norm(np.asarray(q(jnp.asarray(x))) - np.asarray(f(jnp.asarray(x))))


def _array3(x):
    return jnp.stack([jnp.sin(x), jnp.cos(x), 1j * jnp.exp(x)], axis=-1)


def _array_chebfun():
    return cj.chebfun(_array3)


def _to_quasimatrix(f, ncols=3):
    # Source constructor path: split the array-valued Chebfun into scalar
    # columns, then construct a Quasimatrix from those columns.
    return cj.cell2quasi(f.mat2cell([1] * ncols))


class TestQuasimatrixPowerSourceCases:
    """Source computations and bounds with a declared Python sample adapter."""

    def test_source_pass06_quasimatrix_zero_power(self):
        f = _array_chebfun()
        fq = _to_quasimatrix(f)
        g = f ** 0
        err = _q_minus_array_norm(fq ** 0, g, _source_sample(6178))
        assert err < EPS * float(g.vscale)

    def test_source_pass08_quasimatrix_one_power(self):
        f = _array_chebfun()
        fq = _to_quasimatrix(f)
        g = f ** 1
        err = _q_minus_array_norm(fq ** 1, g, _source_sample(6178))
        assert err < EPS * float(g.vscale)

    def test_source_pass10_quasimatrix_square(self):
        f = _array_chebfun()
        fq = _to_quasimatrix(f)
        g = f ** 2
        err = _q_minus_array_norm(fq ** 2, g, _source_sample(6178))
        assert err < 10 * EPS

    def test_source_pass12_quasimatrix_cube(self):
        f = _array_chebfun()
        fq = _to_quasimatrix(f)
        g = f ** 3
        err = _q_minus_array_norm(fq ** 3, g, _source_sample(6178))
        assert err < EPS * float(g.vscale)

    def test_source_pass16_one_to_quasimatrix(self):
        f = _array_chebfun()
        fq = _to_quasimatrix(f)
        g = 1.0 ** f
        err = _q_minus_array_norm(1.0 ** fq, g, _source_sample(6178))

        # Literal source test_power.m pass(16) uses vscale(h), where h is the
        # scalar expected Chebfun left over from pass(14): (2i).^sin(x).
        h = cj.chebfun(lambda x: (2.0j) ** jnp.sin(x))
        assert err < 10 * EPS * float(h.vscale)

    def test_source_pass18_complex_base_to_quasimatrix(self):
        f = _array_chebfun()
        fq = _to_quasimatrix(f)
        g = (2.0j) ** f
        err = _q_minus_array_norm((2.0j) ** fq, g, _source_sample(6178))

        # Literal source pass(18) uses h from pass(17), the independently
        # constructed three-column expected array-valued Chebfun.
        h = cj.chebfun(
            lambda x: jnp.stack(
                [
                    (2.0j) ** jnp.sin(x),
                    (2.0j) ** jnp.cos(x),
                    (2.0j) ** (1j * jnp.exp(x)),
                ],
                axis=-1,
            )
        )
        assert err < 10 * EPS * float(h.vscale)

    def test_source_pass21_quasimatrix_pairwise_power(self):
        x = cj.chebfun(lambda t: jnp.stack([t, jnp.exp(1j * t)], axis=-1), domain=(0.1, 2.0))
        xq = _to_quasimatrix(x, ncols=2)
        g = x ** x
        gq = xq ** xq
        sample = _source_sample(6178, domain=(0.1, 2.0))
        err = _q_minus_array_norm(gq, g, sample)

        # Preserve pass(21)'s h and tolerance from pass(20), not a recomputed
        # norm estimate of the Quasimatrix result.
        h = cj.chebfun(
            lambda t: jnp.stack(
                [t ** t, jnp.exp(1j * t) ** jnp.exp(1j * t)], axis=-1
            ),
            domain=(0.1, 2.0),
        )
        assert err < 10 * EPS * float(h.vscale)


def test_quasimatrix_power_retains_source_column_order_and_count():
    """Additional structural control for the 3-column source pairing cases."""
    f = _array_chebfun()
    q = _to_quasimatrix(f) ** 1
    assert len(q.cols) == 3
    x = jnp.asarray([-0.5, 0.0, 0.5])
    np.testing.assert_allclose(
        np.asarray(q(x)), np.asarray(f(x)), rtol=0.0, atol=0.0
    )


def test_quasimatrix_pairing_order_uses_distinguishable_columns():
    """Independent source-shaped pairing-order control, not a new tolerance."""
    f = cj.chebfun(lambda x: jnp.stack([1.0 + x, 2.0 + x], axis=-1), domain=(0.1, 1.0))
    q = _to_quasimatrix(f, ncols=2)
    g = cj.chebfun(lambda x: jnp.stack([(1.0 + x) ** (1.0 + x), (2.0 + x) ** (2.0 + x)], axis=-1), domain=(0.1, 1.0))
    sample = jnp.asarray([0.2, 0.6, 0.9])
    assert _q_minus_array_norm(q ** q, g, sample) < 20 * EPS * float(g.vscale)


def test_quasimatrix_scalar_base_and_scalar_exponent_keep_each_column():
    """Additional supported scalar dispatch controls, separate from source bounds."""
    f = cj.chebfun(lambda x: jnp.stack([1.0 + x, 2.0 + x], axis=-1), domain=(0.1, 1.0))
    q = _to_quasimatrix(f, ncols=2)
    sample = jnp.asarray([0.2, 0.6, 0.9])

    powered = q ** 2
    expected = cj.chebfun(lambda x: jnp.stack([(1.0 + x) ** 2, (2.0 + x) ** 2], axis=-1), domain=(0.1, 1.0))
    assert _q_minus_array_norm(powered, expected, sample) < 20 * EPS * float(expected.vscale)

    reverse = 2.0 ** q
    expected_reverse = cj.chebfun(lambda x: jnp.stack([2.0 ** (1.0 + x), 2.0 ** (2.0 + x)], axis=-1), domain=(0.1, 1.0))
    assert _q_minus_array_norm(reverse, expected_reverse, sample) < 20 * EPS * float(expected_reverse.vscale)


@pytest.mark.parametrize("singleton_base", [True, False])
def test_quasimatrix_pairwise_singleton_column_expansion(singleton_base):
    many = cj.cell2quasi([cj.chebfun(lambda x: 2+.1*x),
                          cj.chebfun(lambda x: 3+.2*x)])
    one = cj.cell2quasi([cj.chebfun(lambda x: .5+.1*x)])
    base, exponent = (one, many) if singleton_base else (many, one)
    result = base ** exponent
    points = jnp.asarray([-.7, .1, .8])
    f = np.asarray(base(points))
    b = np.asarray(exponent(points))
    assert result.n_cols == 2 and result.domain == many.domain
    np.testing.assert_allclose(result(points), f ** b,
                               rtol=20*EPS, atol=20*EPS)


def test_quasimatrix_pairwise_incompatible_column_counts():
    q2 = cj.cell2quasi([cj.chebfun(2.), cj.chebfun(3.)])
    q3 = cj.cell2quasi([cj.chebfun(2.), cj.chebfun(3.), cj.chebfun(4.)])
    with pytest.raises(ValueError, match="column counts"):
        _ = q2 ** q3


def test_quasimatrix_pairwise_mismatched_domains():
    q1 = cj.cell2quasi([cj.chebfun(2., domain=(0., 1.))])
    q2 = cj.cell2quasi([cj.chebfun(3., domain=(0., 2.))])
    with pytest.raises(ValueError, match="Domain"):
        _ = q1 ** q2


@pytest.mark.parametrize("operation", ["__pow__", "__rpow__"])
def test_quasimatrix_numeric_array_defers_unsupported_dispatch(operation):
    q = cj.cell2quasi([cj.chebfun(2.), cj.chebfun(3.)])
    assert getattr(q, operation)(jnp.asarray([2., 3.])) is NotImplemented

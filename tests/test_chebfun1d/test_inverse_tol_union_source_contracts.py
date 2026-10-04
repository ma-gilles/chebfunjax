"""Source contracts for MATLAB tolerance unions used by inverse domains.

Provenance
----------
MATLAB source : @chebfun/tolUnion.m, @chebfun/inv.m (lines 62-68 and 231-268)
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.inverse import (
    _inverse_domain,
    _newton_or_roots,
    tol_union,
)


def test_tol_union_sorts_exact_duplicates_and_uses_strict_threshold():
    result = tol_union(jnp.array([2.0, 0.0, 1.0]), jnp.array([1.0, 4.0]), tol=1.0)
    # Gaps equal to tol are retained because the MATLAB condition is diff(C)<tol.
    np.testing.assert_array_equal(np.asarray(result), [0.0, 1.0, 2.0, 4.0])
    np.testing.assert_array_equal(
        np.asarray(Chebfun.tol_union([2.0, 0.0, 1.0], [1.0, 4.0], 1.0)),
        np.asarray(result),
    )
    np.testing.assert_array_equal(
        np.asarray(Chebfun.tolUnion([2.0, 0.0, 1.0], [1.0, 4.0], 1.0)),
        np.asarray(result),
    )


def test_tol_union_default_is_100_eps_times_larger_infinity_norm():
    eps = np.finfo(np.float64).eps
    left = 1.0
    right = 1.0 + 50 * eps
    result = tol_union(jnp.array([left]), jnp.array([right]))
    assert result.shape == (1,)
    assert float(result[0]) == 0.5 * (left + right)


def test_tol_union_empty_vectors_and_exact_duplicate():
    np.testing.assert_array_equal(np.asarray(tol_union([], [])), np.empty((0,)))
    np.testing.assert_array_equal(
        np.asarray(tol_union([1.0, 2.0], [1.0, 2.0])), [1.0, 2.0]
    )


def test_tol_union_chain_merges_adjacent_pairs_simultaneously():
    eps = np.finfo(np.float64).eps
    values = jnp.asarray([1.0, 1.0 + 2 * eps, 1.0 + 4 * eps, 1.0 + 6 * eps])
    result = tol_union(values[:2], values[2:], tol=3 * eps)
    # MATLAB marks all three adjacent pairs before deletion. Pair means after
    # the first are deleted; it does not recursively merge the chain.
    assert result.shape == (1,)
    assert float(result[0]) == 0.5 * (float(values[0]) + float(values[1]))


def test_inverse_domain_uses_extrema_then_two_source_unions():
    class PiecewiseStub:
        class _Domain:
            breakpoints = (-1.0, 0.0, 1.0)

        domain = _Domain()

        def minandmax(self):
            # Values deliberately differ from the endpoint images.
            return ((-0.5, -0.25), (1.0, 2.0))

        def __call__(self, points, side):
            assert side in ("left", "right")
            value = 0.2 if side == "left" else 0.8
            return jnp.full_like(jnp.asarray(points), value, dtype=jnp.float64)

    result = _inverse_domain(PiecewiseStub())
    assert result == (-0.25, 0.2, 0.8, 2.0)


def test_inverse_domain_union_stages_keep_exact_second_stage_threshold():
    eps = np.finfo(np.float64).eps

    class BreakpointStub:
        class _Domain:
            breakpoints = (-1.0, 0.0, 1.0)

        domain = _Domain()

        def minandmax(self):
            return ((-1.0, 0.0), (1.0, 1.0))

        def __call__(self, points, side):
            value = 1.0 - (120 if side == "left" else 80) * eps
            return jnp.full_like(jnp.asarray(points), value, dtype=jnp.float64)

    result = _inverse_domain(BreakpointStub())
    expected_break = 1.0 - 100 * eps
    np.testing.assert_array_equal(
        np.asarray(result), np.asarray([0.0, expected_break, 1.0])
    )


def test_sparse_newton_fallback_receives_source_tolerance(monkeypatch):
    import chebfunjax.chebfun1d.inverse as inverse

    observed = {}

    def fake_roots(_f, y, tol):
        observed["tol"] = tol
        return y

    monkeypatch.setattr(inverse, "_roots", fake_roots)
    result = _newton_or_roots(
        object(), object(), jnp.asarray([0.1, 0.2]), -1.0, 1.0,
        1.0e-4, forward_length=10,
    )
    assert result.shape == (2,)
    assert observed["tol"] == 2.0e-5

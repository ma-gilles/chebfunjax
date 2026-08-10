"""Port of MATLAB Chebfun tests/chebtech/test_prolong.m (Opus 4.8; marker
audit Fable 5).

The MATLAB test loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; we
parametrize over ``[Chebtech1, Chebtech2]``.

Every MATLAB assertion (pass 1-7) is ported on BOTH tech kinds; there are no
gaps.  All sub-tests except pass 1 operate on array-valued techs
(``[F(x), -F(x)]``, and pass 7's ``v = [1 2 3]`` which MATLAB reads as three
constant columns, cf. ``repmat([1 2 3], 5, 1)``); chebfunjax Chebtech supports
(n, m) coefficient matrices and ``prolong`` acts column-wise, so those are
real tests.

Provenance
----------
MATLAB source : tests/chebtech/test_prolong.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)

BOTH = [Chebtech1, Chebtech2]


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtechProlong:
    @pytest.mark.parametrize("Tech", BOTH)
    def test_prolong_to_length_one(self, Tech):
        # pass(n, 1): prolong(sin, 1) -> length 1, coeffs ~ 0 (sin is odd).
        f = Tech.from_function(lambda x: jnp.sin(x))
        g = f.prolong(1)
        assert g.n == 1
        assert _ninf(g.coeffs) < 10 * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): pass 2-7 port now
    # that techs support (n, m) coefficient matrices.
    @pytest.mark.parametrize("Tech", BOTH)
    def test_array_valued_prolong_to_one(self, Tech):
        # pass(n, 2): prolong([sin, -sin], 1) -> length 1, coeffs ~ 0.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), -jnp.sin(x)], axis=-1))
        g = f.prolong(1)
        assert g.n == 1
        assert _ninf(g.coeffs) < 10 * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_array_valued_prolong_identity(self, Tech):
        # pass(n, 3): prolong to the same length is exact.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), -jnp.sin(x)], axis=-1))
        g = f.prolong(f.n)
        assert np.array_equal(np.asarray(Tech.coeffs2vals(f.coeffs)),
                              np.asarray(Tech.coeffs2vals(g.coeffs)))

    @pytest.mark.parametrize("Tech", BOTH)
    @pytest.mark.parametrize("k", [32, 100])
    def test_array_valued_prolong_longer(self, Tech, k):
        # pass(n, 4)-(5): zero-padding preserves the sampled values.
        from chebfunjax.utils.quadrature import chebpts

        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), -jnp.sin(x)], axis=-1))
        g = f.prolong(k)
        kind = 1 if Tech is Chebtech1 else 2
        x = chebpts(k, kind=kind)
        values = np.asarray(Tech.coeffs2vals(g.coeffs))
        exact = np.stack([np.sin(np.asarray(x)),
                          -np.sin(np.asarray(x))], axis=-1)
        assert np.max(np.abs(values - exact)) < 10 * g.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_array_valued_highfreq_truncate(self, Tech):
        # pass(n, 6): truncating [sin(1000x), -sin(1000x)] to length 1.
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(1000 * x), -jnp.sin(1000 * x)], axis=-1))
        g = f.prolong(1)
        assert g.n == 1
        assert _ninf(g.coeffs) < 1e2 * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_constant_columns_prolong(self, Tech):
        # pass(n, 7): three constant columns [1 2 3] prolonged to 5 points.
        f = Tech.from_values(jnp.asarray([[1.0, 2.0, 3.0]]))
        g = f.prolong(5)
        values = np.asarray(Tech.coeffs2vals(g.coeffs))
        exact = np.tile([1.0, 2.0, 3.0], (5, 1))
        assert np.max(np.abs(values - exact)) < 10 * g.vscale * EPS

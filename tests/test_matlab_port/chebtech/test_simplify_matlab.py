"""Port of MATLAB Chebfun tests/chebtech/test_simplify.m (Opus 4.8).

The MATLAB test loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; we
parametrize over ``[Chebtech1, Chebtech2]``.

Notes on gaps (see the report):
* The array-valued sub-tests (pass 10-12) are skipped.
* ``iszero`` is not a chebfunjax method; we check ``max(|coeffs|) == 0``.
* The unhappy ``sqrt(x)`` sub-test (pass 2) uses complex ``sqrt`` (matching
  MATLAB, where ``sqrt`` of a negative real is complex); either way it does not
  resolve, so it is unhappy on both tech kinds and simplify leaves it alone.

Provenance
----------
MATLAB source : tests/chebtech/test_simplify.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))

# MATLAB: simptol = 1e-6.
SIMPTOL = 1e-6

BOTH = [Chebtech1, Chebtech2]


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtechSimplify:
    @pytest.mark.parametrize("Tech", BOTH)
    def test_empty_left_alone(self, Tech):
        # pass(n, 1): simplify of an empty tech is unchanged (isequal(f, g)).
        f = Tech.from_coeffs(jnp.array([]))
        g = f.simplify()
        assert g.n == f.n == 0
        assert g.ishappy == f.ishappy

    @pytest.mark.parametrize("Tech", BOTH)
    def test_unhappy_left_alone(self, Tech):
        # pass(n, 2): an unhappy tech (sqrt has a branch point at 0) is left
        # alone by simplify.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Tech.from_function(lambda x: jnp.sqrt(x.astype(jnp.complex128)))
        g = f.simplify()
        assert not f.ishappy
        assert g.n == f.n
        assert _ninf(f.coeffs - g.coeffs) == 0.0

    @pytest.mark.parametrize("Tech", BOTH)
    def test_last_coeff_nonzero(self, Tech):
        # pass(n, 3)
        f = Tech.from_function(lambda x: jnp.sin(100 * (x + 0.1)))
        g = f.simplify(SIMPTOL)
        assert float(jnp.abs(g.coeffs[-1])) != 0.0

    @pytest.mark.parametrize("Tech", BOTH)
    def test_shortened(self, Tech):
        # pass(n, 4): length(g) < length(f)
        f = Tech.from_function(lambda x: jnp.sin(100 * (x + 0.1)))
        g = f.simplify(SIMPTOL)
        assert len(g) < len(f)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_accuracy_preserved(self, Tech):
        # pass(n, 5)
        f = Tech.from_function(lambda x: jnp.sin(100 * (x + 0.1)))
        g = f.simplify(SIMPTOL)
        assert _ninf(f(X) - g(X)) < 1e2 * SIMPTOL * f.vscale

    @pytest.mark.parametrize("Tech", BOTH)
    def test_scaling_down_coeffs_above_floor(self, Tech):
        # pass(n, 6): after scaling by 1e-8, every retained coeff >= eps*vscale
        f = Tech.from_function(lambda x: jnp.sin(100 * (x + 0.1)))
        g1 = (1e-8 * f).simplify(SIMPTOL)
        assert bool(np.all(np.abs(np.asarray(g1.coeffs)) >= EPS * g1.vscale))

    @pytest.mark.parametrize("Tech", BOTH)
    def test_scaling_down_length_invariant(self, Tech):
        # pass(n, 7): length(g1) == length(g)
        f = Tech.from_function(lambda x: jnp.sin(100 * (x + 0.1)))
        g = f.simplify(SIMPTOL)
        g1 = (1e-8 * f).simplify(SIMPTOL)
        assert len(g1) == len(g)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_scaling_up_coeffs_above_floor(self, Tech):
        # pass(n, 8): after scaling by 1e8, every retained coeff >= eps*vscale
        f = Tech.from_function(lambda x: jnp.sin(100 * (x + 0.1)))
        g2 = (1e8 * f).simplify(SIMPTOL)
        assert bool(np.all(np.abs(np.asarray(g2.coeffs)) >= EPS * g2.vscale))

    @pytest.mark.parametrize("Tech", BOTH)
    def test_scaling_up_length_invariant(self, Tech):
        # pass(n, 9): length(g2) == length(g)
        f = Tech.from_function(lambda x: jnp.sin(100 * (x + 0.1)))
        g = f.simplify(SIMPTOL)
        g2 = (1e8 * f).simplify(SIMPTOL)
        assert len(g2) == len(g)

    # FIXED (Fable 5, Big-Three array-valued epic): pass 10-12 port now
    # that techs support (n, m) coefficient matrices.
    @pytest.mark.parametrize("Tech", BOTH)
    def test_array_valued(self, Tech):
        # pass(n, 10)-(12): array-valued simplify keeps a nonzero leading
        # row, shortens the series, and stays accurate.
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(100 * (x + 0.1)), jnp.cos(100 * (x + 0.1)),
                 jnp.exp(x)], axis=-1))
        g = f.simplify(SIMPTOL)
        assert bool(np.any(np.abs(np.asarray(g.coeffs)[0, :]) != 0))
        assert len(g) < len(f)
        err = float(jnp.max(jnp.abs(f(X) - g(X))))
        assert err < 10 * SIMPTOL * f.vscale

    @pytest.mark.parametrize("Tech", BOTH)
    def test_contrived_length_one(self, Tech):
        # pass(n, 13): simplify with a huge tol leaves a length-1 tech.
        f = Tech.from_function(lambda x: jnp.sin(100 * (x + 0.1)))
        g = f.simplify(1e20)
        assert len(g) == 1

    @pytest.mark.parametrize("Tech", BOTH)
    def test_long_zero_simplifies(self, Tech):
        # pass(n, 14): a long identically-zero tech simplifies to length 1.
        # MATLAB uses struct('fixedLength', 8); chebfunjax builds a length-8
        # zero tech directly.  ``iszero`` -> max(|coeffs|) == 0.
        f = Tech.from_function(lambda x: 0.0 * x, n=8)
        g = f.simplify()
        assert _ninf(g.coeffs) == 0.0
        assert len(g) == 1

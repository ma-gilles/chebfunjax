"""Port of MATLAB Chebfun tests/chebtech/test_turbo.m (Opus 4.8).

MATLAB's "turbo" construction (``pref.useTurbo = true``) computes twice as many
Chebyshev coefficients using contour integrals over a Bernstein ellipse.
chebfunjax now exposes this as ``from_function(..., turbo=True)`` (a port of
``@chebtech/constructorTurbo.m``): the plain construction stays adaptive and the
coefficients are recomputed to high accuracy.  ``n=<fixedLength>`` selects the
fixedLength variant.  The scalar assertions run; the array-valued assertions
are skipped (chebfunjax Chebtech is scalar-valued).

Provenance
----------
MATLAB source : tests/chebtech/test_turbo.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import iv as besseli

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

EPS = float(np.finfo(np.float64).eps)

_NO_TURBO_ARRAY = "chebfunjax Chebtech is scalar-valued; no array-valued turbo"


def _exp_exact(k):
    c = 2.0 * besseli(k, 1)
    c[0] = c[0] / 2.0
    return c


class TestChebtechTurbo:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_length_doubles(self, Tech, kind):
        # pass(n, 1): length(f_turbo) == 2*length(f_plain).
        f_plain = Tech.from_function(jnp.exp)
        f_turbo = Tech.from_function(jnp.exp, turbo=True)
        assert len(f_turbo) == 2 * len(f_plain)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_isreal(self, Tech, kind):
        # pass(n, 2): isreal(f_turbo).
        f_turbo = Tech.from_function(jnp.exp, turbo=True)
        assert not np.iscomplexobj(np.asarray(f_turbo.coeffs))

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_coeff_accuracy(self, Tech, kind):
        # pass(n, 3): coeffs match 2*besseli(k,1) to tol.
        f_plain = Tech.from_function(jnp.exp)
        f_turbo = Tech.from_function(jnp.exp, turbo=True)
        rho = np.exp(abs(np.log(EPS)) / len(f_plain)) ** (2.0 / 3.0)
        k = np.arange(len(f_turbo))
        err = np.abs(np.asarray(f_turbo.coeffs) - _exp_exact(k))
        tol = 1e2 * EPS * rho ** (-k)
        assert np.all(err < tol)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_array_length(self, Tech, kind):
        # pass(n, 4)
        pytest.skip(_NO_TURBO_ARRAY)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_array_isreal(self, Tech, kind):
        # pass(n, 5)
        pytest.skip(_NO_TURBO_ARRAY)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_array_coeff_accuracy(self, Tech, kind):
        # pass(n, 6)
        pytest.skip(_NO_TURBO_ARRAY)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_fixedlength(self, Tech, kind):
        # pass(n, 7): length(f_turbo) == 75 with fixedLength.
        f_turbo = Tech.from_function(jnp.exp, n=75, turbo=True)
        assert len(f_turbo) == 75

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_fixedlength_accuracy(self, Tech, kind):
        # pass(n, 8)
        f_plain = Tech.from_function(jnp.exp)
        f_turbo = Tech.from_function(jnp.exp, n=75, turbo=True)
        rho = np.exp(abs(np.log(EPS)) / len(f_plain)) ** (2.0 / 3.0)
        k = np.arange(len(f_turbo))
        err = np.abs(np.asarray(f_turbo.coeffs) - _exp_exact(k))
        tol = 1e2 * EPS * rho ** (-k)
        assert np.all(err < tol)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_fixedlength_array(self, Tech, kind):
        # pass(n, 9)
        pytest.skip(_NO_TURBO_ARRAY)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_fixedlength_array_accuracy(self, Tech, kind):
        # pass(n, 10)
        pytest.skip(_NO_TURBO_ARRAY)

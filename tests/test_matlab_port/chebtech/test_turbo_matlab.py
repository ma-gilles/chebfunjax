"""Port of MATLAB Chebfun tests/chebtech/test_turbo.m (Opus 4.8).

MATLAB's "turbo" construction (``pref.useTurbo = true``) computes twice as many
Chebyshev coefficients using contour integrals over a Bernstein ellipse.
chebfunjax now exposes this as ``from_function(..., turbo=True)`` (a port of
``@chebtech/constructorTurbo.m``): the plain construction stays adaptive and the
coefficients are recomputed to high accuracy.  ``n=<fixedLength>`` selects the
fixedLength variant.  Array-valued techs landed in 2026-07, so every MATLAB
assertion (scalar and array-valued) is ported at MATLAB's tolerances.

No gaps: all ten MATLAB passes are exercised.

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

def _exp_exact(k):
    c = 2.0 * besseli(k, 1)
    c[0] = c[0] / 2.0
    return c


def _recip_exact(k):
    # MATLAB: c = (1/sqrt(6))*((-1).^k)./((5 + sqrt(24)).^k); c(1) = 1/(2*sqrt(6)).
    c = (1.0 / np.sqrt(6.0)) * ((-1.0) ** k) / ((5.0 + np.sqrt(24.0)) ** k)
    c[0] = 1.0 / (2.0 * np.sqrt(6.0))
    return c


def _array_op(x):
    """MATLAB ``@(x) [exp(x) 1./(x + 5)]``."""
    return jnp.stack([jnp.exp(x), 1.0 / (x + 5.0)], axis=-1)


def _array_exact(k):
    return np.stack([_exp_exact(k), _recip_exact(k)], axis=-1)


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
        # pass(n, 4): length doubles for an array-valued build too.
        f_plain = Tech.from_function(_array_op)
        f_turbo = Tech.from_function(_array_op, turbo=True)
        assert len(f_turbo) == 2 * len(f_plain)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_array_isreal(self, Tech, kind):
        # pass(n, 5): isreal(f_turbo).
        f_turbo = Tech.from_function(_array_op, turbo=True)
        assert not np.iscomplexobj(np.asarray(f_turbo.coeffs))

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_array_coeff_accuracy(self, Tech, kind):
        # pass(n, 6): both columns match their exact coefficients.
        f_plain = Tech.from_function(_array_op)
        f_turbo = Tech.from_function(_array_op, turbo=True)
        rho = np.exp(abs(np.log(EPS)) / len(f_plain)) ** (2.0 / 3.0)
        k = np.arange(len(f_turbo))
        err = np.abs(np.asarray(f_turbo.coeffs) - _array_exact(k))
        tol = (1e2 * EPS * rho ** (-k))[:, None]
        assert np.all(err < tol)

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
        # pass(n, 9): fixedLength = 75 with an array-valued input.
        f_turbo = Tech.from_function(_array_op, n=75, turbo=True)
        assert len(f_turbo) == 75

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_turbo_fixedlength_array_accuracy(self, Tech, kind):
        # pass(n, 10)
        f_plain = Tech.from_function(_array_op)
        f_turbo = Tech.from_function(_array_op, n=75, turbo=True)
        rho = np.exp(abs(np.log(EPS)) / len(f_plain)) ** (2.0 / 3.0)
        k = np.arange(len(f_turbo))
        err = np.abs(np.asarray(f_turbo.coeffs) - _array_exact(k))
        tol = (1e2 * EPS * rho ** (-k))[:, None]
        assert np.all(err < tol)

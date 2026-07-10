"""Port of MATLAB Chebfun tests/singfun/test_singfun_constructor.m (Opus 4.8).

Covers the calling sequence where the user provides the exponents explicitly
(pass 1-12).  The auto-detection cases (pass 13-18) and the smoothfun/double
construction + iszero cases (pass 19-24) are xfailed: chebfunjax's
``Singfun.from_function`` requires explicit exponents and has no exponent
auto-detection, no smoothfun class, and no ``iszero``.

MATLAB's ``singType`` (pole/root/sing/none) is a redundant hint that
chebfunjax ignores; constructing with or without it is identical here.

Provenance
----------
MATLAB source : tests/singfun/test_singfun_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

# Arbitrary values used for exponents (from the MATLAB test).
A = 0.338745372057174
B = 0.561224728136042
A_INT = 2
B_INT = 5

X = jnp.asarray(np.sort(np.linspace(-0.99, 0.99, 100)))
XX = X[19:80]  # interior slice, as MATLAB's x(20:80)


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _isequal(f, g):
    if tuple(f.exponents) != tuple(g.exponents):
        return False
    cf, cg = f.coeffs, g.coeffs
    return cf.shape == cg.shape and bool(jnp.array_equal(cf, cg))


class TestSingfunConstructorNegativeFractional:
    def setup_method(self):
        def fh(x):
            return jnp.sin(x) / ((1 + x) ** A * (1 - x) ** B)

        self.fh = fh
        self.f = _sf(self.fh, (-A, -B))
        self.g = _sf(self.fh, (-A, -B))

    def test_isequal(self):
        assert _isequal(self.f, self.g)

    def test_f_exponents(self):
        assert not np.any(np.asarray(self.f.exponents) + np.array([A, B]))

    def test_g_exponents(self):
        assert not np.any(np.asarray(self.g.exponents) + np.array([A, B]))

    def test_feval(self):
        assert _ninf(self.fh(X) - self.f(X)) < 1e2 * EPS


class TestSingfunConstructorPositiveFractional:
    def setup_method(self):
        self.fh = lambda x: jnp.sin(x) * (1 + x) ** A * (1 - x) ** B
        self.f = _sf(self.fh, (A, B))
        self.g = _sf(self.fh, (A, B))

    def test_isequal(self):
        assert _isequal(self.f, self.g)

    def test_f_exponents(self):
        assert not np.any(np.asarray(self.f.exponents) - np.array([A, B]))

    def test_g_exponents(self):
        assert not np.any(np.asarray(self.g.exponents) - np.array([A, B]))

    def test_feval(self):
        assert _ninf(self.fh(X) - self.f(X)) < 1e1 * EPS


class TestSingfunConstructorNegativeInteger:
    def setup_method(self):
        self.fh = lambda x: jnp.exp(x) / ((1 + x) ** A_INT * (1 - x) ** B_INT)
        self.f = _sf(self.fh, (-A_INT, -B_INT))
        self.g = _sf(self.fh, (-A_INT, -B_INT))

    def test_isequal(self):
        assert _isequal(self.f, self.g)

    def test_f_exponents(self):
        assert not np.any(np.asarray(self.f.exponents) + np.array([A_INT, B_INT]))

    def test_g_exponents(self):
        assert not np.any(np.asarray(self.g.exponents) + np.array([A_INT, B_INT]))

    def test_feval(self):
        # don't check near the end-points
        assert _ninf(self.fh(XX) - self.f(XX)) < 1e2 * EPS


class TestSingfunConstructorAutoDetect:
    """MATLAB detects the exponents when the user passes none (singfun(fh))."""

    @pytest.mark.xfail(
        reason="chebfunjax Singfun.from_function requires explicit exponents; "
        "it has no exponent auto-detection",
        strict=True,
    )
    def test_negative_fractional_autodetect(self):
        fh = lambda x: jnp.exp(jnp.sin(x)) / ((1 + x) ** A * (1 - x) ** B)
        f = Singfun.from_function(fh)  # missing required exponents -> TypeError
        assert _ninf(np.asarray(f.exponents) + np.array([A, B])) < 1e-11

    @pytest.mark.xfail(
        reason="chebfunjax has no exponent auto-detection",
        strict=True,
    )
    def test_positive_fractional_autodetect(self):
        fh = lambda x: jnp.sin(jnp.exp(jnp.cos(x))) * (1 + x) ** A * (1 - x) ** B
        f = Singfun.from_function(fh)
        assert _ninf(np.asarray(f.exponents) - np.array([A, B])) < 1e-11

    @pytest.mark.xfail(
        reason="chebfunjax has no exponent auto-detection",
        strict=True,
    )
    def test_negative_integer_autodetect(self):
        fh = lambda x: jnp.exp(jnp.sin(x ** 2)) / ((1 + x) ** A_INT * (1 - x) ** B_INT)
        f = Singfun.from_function(fh)
        assert _ninf(np.asarray(f.exponents) + np.array([A_INT, B_INT])) < 1e-11


class TestSingfunConstructorFromSmoothfunOrDouble:
    @pytest.mark.xfail(
        reason="chebfunjax has no smoothfun class / smoothfun input constructor "
        "and no iszero",
        strict=True,
    )
    def test_from_smoothfun(self):
        from chebfunjax.tech.chebtech import Chebtech2

        f = Chebtech2.from_function(lambda x: jnp.sin(x))
        s = Singfun(f)  # missing required exponents -> TypeError
        assert bool(jnp.all((f - s.smoothPart).coeffs == 0))

    @pytest.mark.xfail(
        reason="chebfunjax Singfun.from_function needs a callable; there is no "
        "double->Singfun constructor nor iszero",
        strict=True,
    )
    def test_from_double(self):
        f = Singfun(42)  # not a valid constructor -> error
        assert bool(jnp.all((f - 42).coeffs == 0))

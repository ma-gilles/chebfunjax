"""Port of MATLAB Chebfun tests/chebtech/test_restrict.m (Opus 4.8; marker
audit Fable 5).

Self-validating: each restriction is checked against the analytic function
on the sub-interval, remapped to [-1, 1], at the SAME tolerance MATLAB uses
(1e3 * vscale(g) * eps).  The MATLAB file loops ``for n = 1:2`` over
``{chebtech1(), chebtech2()}``.

``restrict`` now exists on BOTH tech classes; every method is exercised on
Chebtech1 and Chebtech2.  ``restrict(f, [a b])`` maps the sub-interval
[a, b] onto [-1, 1] via ``t -> (2/(b-a))*(t-a) - 1``.

Every MATLAB assertion (pass 1-11) is ported; there are no gaps:

* Breakpoint VECTORS are supported.  ``f.restrict([a, b, c, ...])`` returns a
  LIST of techs, one per sub-interval (MATLAB returns a cell array);
  ``f.restrict([a, b])`` and ``f.restrict(a, b)`` both return a single tech.
  This covers pass 5 (non-monotone vector -> badInterval) and pass 10
  (multi-subinterval output), which were previously skipped.
* Array-valued restriction (pass 11) is supported: Chebtech coefficients may
  be an (n, m) matrix (one function per column), and ``restrict`` acts
  column-wise.
* chebfunjax raises ``ValueError`` where MATLAB raises the identifier
  ``CHEBFUN:CHEBTECH:restrict:badInterval``; chebfunjax has no MATLAB error
  identifiers, so the ported tests assert the exception type only.

Provenance
----------
MATLAB source : tests/chebtech/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _spotcheck_restrict(Tech, fun, a, b):
    """Restrict fun to [a, b] and return (error, tol) vs the analytic exact."""
    f = Tech.from_function(fun)
    g = f.restrict(a, b)
    x = jnp.asarray(np.linspace(a, b, 100))
    # Map [a, b] -> [-1, 1].
    mapx = (2.0 / (b - a)) * (x - a) - 1.0
    err = _ninf(fun(x) - g(mapx))
    tol = 1e3 * g.vscale * EPS
    return err, tol


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechRestrict:
    def test_restrict_empty(self, Tech):
        # pass(n, 1): restricting an empty tech stays empty.
        f = Tech.from_coeffs(jnp.asarray([]))
        g = f.restrict(-0.5, 0.5)
        assert g.n == 0

    def test_restrict_full_interval(self, Tech):
        # pass(n, 2): restrict to [-1, 1] returns an equal function.
        f = Tech.from_function(lambda x: jnp.sin(x))
        g = f.restrict(-1.0, 1.0)
        assert g.n == f.n
        assert _ninf(g.coeffs - f.coeffs) == 0.0

    def test_restrict_badinterval_right(self, Tech):
        # pass(n, 3): restrict(f, [-1, 3]) -> badInterval error.
        f = Tech.from_function(lambda x: jnp.sin(x))
        with pytest.raises(ValueError):
            f.restrict(-1.0, 3.0)

    def test_restrict_badinterval_left(self, Tech):
        # pass(n, 4): restrict(f, [-2, 1]) -> badInterval error.
        f = Tech.from_function(lambda x: jnp.sin(x))
        with pytest.raises(ValueError):
            f.restrict(-2.0, 1.0)

    def test_restrict_badinterval_nonmonotone(self, Tech):
        # pass(n, 5): restrict(f, [-1 -0.25 0.3 0.1 1]) -> badInterval.
        # The vector is not increasing (0.3 > 0.1), which MATLAB rejects via
        # any(diff(s) <= 0) in @chebtech/restrict.m.
        f = Tech.from_function(lambda x: jnp.sin(x))
        with pytest.raises(ValueError):
            f.restrict([-1.0, -0.25, 0.3, 0.1, 1.0])

    def test_restrict_spotcheck_exp(self, Tech):
        # pass(n, 6): exp(x) - 1 on [-0.2, 0.1].
        err, tol = _spotcheck_restrict(Tech, lambda x: jnp.exp(x) - 1.0, -0.2, 0.1)
        assert err < tol

    def test_restrict_spotcheck_runge(self, Tech):
        # pass(n, 7): 1/(1 + x^2) on [-0.7, 0.9].
        err, tol = _spotcheck_restrict(Tech, lambda x: 1.0 / (1.0 + x**2), -0.7, 0.9)
        assert err < tol

    def test_restrict_spotcheck_highfreq(self, Tech):
        # pass(n, 8): cos(1e3 x) on [0.1, 0.5].
        err, tol = _spotcheck_restrict(Tech, lambda x: jnp.cos(1e3 * x), 0.1, 0.5)
        assert err < tol

    def test_restrict_spotcheck_complex_sinh(self, Tech):
        # pass(n, 9): sinh(t*exp(2*pi*1i/6)) on [-0.4, 1] (complex-valued).
        err, tol = _spotcheck_restrict(
            Tech,
            lambda t: jnp.sinh(t * jnp.exp(2 * jnp.pi * 1j / 6)), -0.4, 1.0
        )
        assert err < tol

    def test_restrict_multiple_subintervals(self, Tech):
        # pass(n, 10), first assignment: restrict(f, [-0.7 0.3 0.8]) returns
        # a cell of two techs matching the two pairwise restrictions.
        f = Tech.from_function(lambda x: jnp.sin(x) + jnp.sin(x ** 2))
        g = f.restrict([-0.7, 0.3, 0.8])
        h1 = f.restrict(-0.7, 0.3)
        h2 = f.restrict(0.3, 0.8)
        assert len(g) == 2
        x = jnp.asarray(np.linspace(-1.0, 1.0, 100))
        tol = 10 * EPS
        assert _ninf((g[0] - h1)(x)) < tol
        assert _ninf((g[1] - h2)(x)) < tol

    def test_restrict_multiple_subintervals_array_valued(self, Tech):
        # pass(n, 10), second assignment (MATLAB overwrites the index):
        # array-valued [sin cos] restricted over [-0.6 0.1 1].
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = f.restrict([-0.6, 0.1, 1.0])
        h1 = f.restrict(-0.6, 0.1)
        h2 = f.restrict(0.1, 1.0)
        assert len(g) == 2
        assert g[0].coeffs.shape[1] == 2
        x = jnp.asarray(np.linspace(-1.0, 1.0, 100))
        tol = 10 * EPS
        assert _ninf((g[0] - h1)(x)) < tol
        assert _ninf((g[1] - h2)(x)) < tol

    def test_restrict_array_valued(self, Tech):
        # pass(n, 11): restrict of the array-valued [sin cos exp] on [-1, -0.7].
        # FIXED (Fable 5, Big-Three array-valued epic): Chebtech now supports
        # (n, m) coeffs; restrict acts column-wise. vscale(g) is the scalar
        # global max, matching MATLAB max(vscale(g)*eps) for the scalar tol.
        a, b = -1.0, -0.7

        def fun(x):
            return jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)

        f = Tech.from_function(fun)
        g = f.restrict(a, b)
        x = jnp.asarray(np.linspace(a, b, 100))
        mapx = (2.0 / (b - a)) * (x - a) - 1.0
        err = _ninf(fun(x) - g(mapx))
        tol = 1e3 * g.vscale * EPS
        assert err < tol

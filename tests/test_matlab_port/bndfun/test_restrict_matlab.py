"""Port of MATLAB Chebfun tests/bndfun/test_restrict.m (Opus 4.8).

Self-validating: restrictions are sampled on the sub-interval and compared to
the original function at the SAME tolerance MATLAB uses.  chebfunjax's
``Bndfun.restrict(a, b)`` takes two scalar endpoints (a single sub-interval);
MATLAB additionally supports restricting to a *partition* (returning a cell
array of funs), which chebfunjax does not -- those assertions are xfail.

Provenance
----------
MATLAB source : tests/bndfun/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))


def _bf(f, n=None):
    # xfail cases pass a small fixed n so a non-converging build stays fast.
    return Bndfun.from_function(f, DOM, n=n)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _spotcheck_restrict(fun_op, subint, tol_factor=8e4, n=None):
    f = _bf(fun_op, n=n)
    a, b = subint
    g = f.restrict(a, b)
    xr = np.linspace(a, b, 100)
    y_exact = np.asarray(fun_op(jnp.asarray(xr)))
    y_approx = np.asarray(g(jnp.asarray(xr)))
    return float(np.max(np.abs(y_exact - y_approx))) < tol_factor * f.vscale * EPS


class TestBndfunRestrict:
    def test_empty_input(self):
        # pass(1): restrict of an empty bndfun is empty
        f = Bndfun.empty()
        g = f.restrict(-0.5, 0.5)
        assert g.isempty()

    def test_restrict_whole_domain_is_identity(self):
        f = _bf(jnp.sin)
        g = f.restrict(DOM.a, DOM.b)
        assert g.domain == f.domain
        assert bool(np.array_equal(np.asarray(g.coeffs), np.asarray(f.coeffs)))

    def test_restrict_superset_raises(self):
        f = _bf(jnp.sin)
        with pytest.raises(ValueError):
            f.restrict(DOM.a - 1, DOM.b + 1)

    def test_restrict_infinite_raises(self):
        f = _bf(jnp.sin)
        with pytest.raises(ValueError):
            f.restrict(-np.inf, 1.0)

    @pytest.mark.xfail(
        reason="chebfunjax Bndfun.restrict takes two scalar endpoints only; a "
        "non-monotonic multi-breakpoint partition [-1 -0.25 0.3 0.1 1] cannot "
        "be passed (no partition/cell-array restrict)."
    )
    def test_restrict_nonmonotonic_partition_raises(self):
        f = _bf(jnp.sin)
        with pytest.raises(ValueError):
            f.restrict([-1, -0.25, 0.3, 0.1, 1])  # noqa

    def test_restrict_produces_correct_domain(self):
        f = _bf(jnp.sin)
        g = f.restrict(2.0, 3.0)
        assert (g.domain.a, g.domain.b) == (2.0, 3.0)

    def test_spotcheck_exp(self):
        assert _spotcheck_restrict(lambda x: jnp.exp(x) - 1, (-2.0, 4.0))

    def test_spotcheck_runge(self):
        assert _spotcheck_restrict(lambda x: 1.0 / (1 + x ** 2), (-0.7, 0.9))

    def test_spotcheck_high_freq_cos(self):
        assert _spotcheck_restrict(lambda x: jnp.cos(1e3 * x), (0.1, 0.5))

    def test_spotcheck_complex_sinh(self):
        z = np.exp(2 * np.pi * 1j / 6)
        assert _spotcheck_restrict(lambda t: jnp.sinh(t * z), (-0.4, 1.0))

    @pytest.mark.xfail(
        reason="chebfunjax has no multi-interval (partition) restrict returning "
        "a list of funs."
    )
    def test_multi_subinterval_restriction(self):
        f = _bf(lambda x: jnp.sin(x) + jnp.sin(x ** 2))
        g = f.restrict([-1.7, 2.3, 6.8])  # noqa
        h1 = f.restrict(-1.7, 2.3)
        h2 = f.restrict(2.3, 6.8)
        xr = np.linspace(-1.0, 1.0, 100)
        err1 = _ninf((g[0] - h1)(jnp.asarray(xr)))
        err2 = _ninf((g[1] - h2)(jnp.asarray(xr + 4)))
        assert err1 < 1e3 * EPS and err2 < 1e3 * EPS

    @pytest.mark.xfail(
        reason="chebfunjax has no multi-interval (partition) restrict "
        "returning a list of funs."
    )
    def test_multi_subinterval_domains(self):
        f = _bf(lambda x: jnp.sin(x) + jnp.sin(x ** 2))
        g = f.restrict([2.0, 3.0, 5.0])  # noqa
        assert (g[0].domain.a, g[0].domain.b) == (2.0, 3.0)
        assert (g[1].domain.a, g[1].domain.b) == (3.0, 5.0)

    def test_array_valued_spotcheck(self):
        # pass(13): restrict of array-valued [sin cos exp] to [-1, -0.7].
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun; restrict
        # acts column-wise.  Same 8e4*vscale*eps tolerance as the scalar
        # spot-checks (see _spotcheck_restrict).
        assert _spotcheck_restrict(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1),
            (-1.0, -0.7),
        )

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (blowup) Bndfun: (x-a)^-0.5 sin(x)."
    )
    def test_singular_spotcheck(self):
        pow_ = -0.5

        def op(x):
            return (x - DOM.a) ** pow_ * jnp.sin(x)

        assert _spotcheck_restrict(op, (-1.0, -0.7), n=17)

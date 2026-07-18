"""Port of MATLAB Chebfun tests/classicfun/test_roots.m (Opus 4.8).

Self-validating: rootfinding on a Bndfun (and, where supported, Unbndfun) is
checked against analytic / high-precision roots at the SAME tolerances MATLAB
uses.  Bessel and Airy operators are evaluated with SciPy inside the
(non-traced) constructor sampling; this is test-only and does not violate the
library's JAX-only rule.

Provenance
----------
MATLAB source : tests/classicfun/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))


def _bf(op):
    return Bndfun.from_function(op, DOM)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestClassicfunRoots:
    def test_bessel_j0(self):
        # f(x) = besselj(0, map(x)), map(x) = (x+2)*100/9 -> range [0, 100]
        mp = lambda x: (x + 2) * 100.0 / 9.0
        f = _bf(lambda x: jnp.asarray(sp.jv(0, np.asarray(mp(x)))))
        r = np.sort(np.asarray(mp(np.asarray(f.roots()))))
        exact = sp.jn_zeros(0, r.shape[0])
        assert r.shape[0] == 32
        assert _ninf(r - exact) < 10 * f.n * EPS

    def test_oscillatory_sine(self):
        k = 100
        f = _bf(lambda x: jnp.sin(np.pi * k * x))
        r = np.sort(np.asarray(f.roots()))
        exact = np.arange(-2 * k, 7 * k + 1) / k
        assert _ninf(r - exact) < 10 * EPS * f.vscale

    def test_perturbed_quartic(self):
        f = _bf(lambda x: (x - 0.1) * (x + 0.9) * x * (x - 0.9) + 1e-14 * x ** 5)
        r = f.roots()
        assert r.shape[0] == 4
        assert _ninf(f(r)) < 100 * EPS * f.vscale

    def test_linear(self):
        f = _bf(lambda x: x)
        r = f.roots()
        assert _ninf(r) < EPS * f.vscale

    def test_even_parabola_from_values(self):
        # MATLAB: bndfun([20.25 ; 0 ; 20.25]) -- a numeric column is treated as
        # VALUES at the Chebyshev points, giving an even parabola with a double
        # root at 0.
        t = Chebtech2.from_values(jnp.asarray([20.25, 0.0, 20.25]))
        f = Bndfun.from_chebtech(t, Domain((-1.0, 1.0)))
        r = f.roots()
        assert r.shape[0] == 2
        assert _ninf(r) < EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax roots() returns only real roots; the 'complex' "
        "option (roots(f,'complex',1)) is not implemented."
    )
    def test_complex_roots_1_plus_x2(self):
        f = _bf(lambda x: 1 + x ** 2)
        r = f.roots()  # would need roots(f, 'complex', 1) -> [i, -i]
        assert _ninf(np.sort_complex(np.asarray(r)) - np.array([-1j, 1j])) < EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax roots() has no 'complex'/'prune' options."
    )
    def test_complex_roots_pruned(self):
        f = Bndfun.from_function(
            lambda x: (1 + 25 * x ** 2) * jnp.exp(x), Domain((-1.0, 1.0))
        )
        r = f.roots()
        assert _ninf(np.sort_complex(np.asarray(r)) - np.array([-1j, 1j]) / 5) < 10 * EPS * f.vscale

    def test_complex_roots_recurse(self):
        # MATLAB pass(8): numel(roots(f,'complex',1)) >=
        # numel(roots(f,'complex',1,'recurse',0)).  chebfunjax
        # Classicfun.roots takes no options, so the recurse=0 baseline
        # cannot be expressed.  (The vacuous r1==r2 transcription left by
        # the Opus 4.8 port xpassed; replaced by an honest skip in the
        # Fable 5 audit.)
        pytest.skip("Classicfun.roots has no 'recurse' option")

    def test_array_valued_roots(self):
        # pass(9): roots of [sin(pi x), cos(pi x), x^2+1] -> NaN-padded per-column
        # roots matching [-2:7 ; -1.5:6.5 ; NaN(1,11)].
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun.
        f = _bf(lambda x: jnp.stack(
            [jnp.sin(np.pi * x), jnp.cos(np.pi * x), x ** 2 + 1], axis=-1))
        r = np.asarray(f.roots())
        tol = 1e1 * EPS * f.vscale
        c0 = np.sort(r[:, 0][~np.isnan(r[:, 0])])
        c1 = np.sort(r[:, 1][~np.isnan(r[:, 1])])
        assert _ninf(c0 - np.arange(-2, 8)) < tol
        assert _ninf(c1 - np.arange(-1.5, 7.0)) < tol
        assert np.sum(~np.isnan(r[:, 2])) == 0  # x^2+1 has no real roots

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (exponents) Bndfun: (x-a)^-0.5*cos(x) "
        "cannot be constructed."
    )
    def test_singular_roots(self):
        raise NotImplementedError("singular Bndfun roots")

    @pytest.mark.xfail(
        reason="chebfunjax Unbndfun has no roots() method, and lacks the "
        "blowup (exponents [2 2]) representation this test requires."
    )
    def test_unbndfun_roots(self):
        raise NotImplementedError("Unbndfun roots")

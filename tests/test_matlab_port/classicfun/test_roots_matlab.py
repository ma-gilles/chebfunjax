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
import scipy.special as sp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun
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
        r = np.asarray(mp(f.roots()))
        exact = np.asarray([
            2.40482555769577276862163, 5.52007811028631064959660,
            8.65372791291101221695437, 11.7915344390142816137431,
            14.9309177084877859477626, 18.0710639679109225431479,
            21.2116366298792589590784, 24.3524715307493027370579,
            27.4934791320402547958773, 30.6346064684319751175496,
            33.7758202135735686842385, 36.9170983536640439797695,
            40.0584257646282392947993, 43.1997917131767303575241,
            46.3411883716618140186858, 49.4826098973978171736028,
            52.6240518411149960292513, 55.7655107550199793116835,
            58.9069839260809421328344, 62.0484691902271698828525,
            65.1899648002068604406360, 68.3314693298567982709923,
            71.4729816035937328250631, 74.6145006437018378838205,
            77.7560256303880550377394, 80.8975558711376278637723,
            84.0390907769381901578795, 87.1806298436411536512617,
            90.3221726372104800557177, 93.4637187819447741711905,
            96.6052679509962687781216, 99.7468198586805964702799,
        ])
        assert r.shape[0] == 32
        assert _ninf(r - exact) < 10 * f.n * EPS

    def test_oscillatory_sine(self):
        k = 100
        f = _bf(lambda x: jnp.sin(np.pi * k * x))
        r = np.asarray(f.roots())
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

    def test_complex_roots_1_plus_x2(self):
        # roots(f, 'complex', 1) -> [i, -i]; Classicfun forwards the option
        # to the tech rootfinder and the affine map carries complex roots.
        f = _bf(lambda x: 1 + x ** 2)
        r = f.roots(complex_roots=True)
        assert _ninf(np.asarray(r) - np.array([1j, -1j])) < EPS * f.vscale

    def test_complex_roots_pruned(self):
        # roots(f, 'complex', 1) prunes spurious roots -> +/- i/5.
        f = Bndfun.from_function(
            lambda x: (1 + 25 * x ** 2) * jnp.exp(x), Domain((-1.0, 1.0))
        )
        r = f.roots(complex_roots=True, prune=True)
        assert _ninf(np.asarray(r) - np.array([1j, -1j]) / 5) < 10 * EPS * f.vscale

    def test_complex_roots_recurse(self):
        # MATLAB pass(8): numel(roots(f,'complex',1)) >=
        # numel(roots(f,'complex',1,'recurse',0)).  FIXED (Fable 5):
        # Classicfun.roots forwards the 'recurse' option to the tech, so the
        # recurse=0 baseline can be expressed and compared.
        f = _bf(lambda x: jnp.sin(10 * np.pi * x))
        r1 = np.asarray(f.roots(complex_roots=True, recurse=False))
        r2 = np.asarray(f.roots(complex_roots=True))
        assert r2.size >= r1.size

    def test_array_valued_roots(self):
        # pass(9): roots of [sin(pi x), cos(pi x), x^2+1] -> NaN-padded per-column
        # roots matching [-2:7 ; -1.5:6.5 ; NaN(1,11)].
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun.
        f = _bf(lambda x: jnp.stack(
            [jnp.sin(np.pi * x), jnp.cos(np.pi * x), x ** 2 + 1], axis=-1))
        r = np.asarray(f.roots())
        tol = 1e1 * EPS * f.vscale
        # MATLAB's literal column-major, signed comparison (including padding).
        expected = np.concatenate([np.arange(-2, 8), np.arange(-1.5, 7), np.full(11, np.nan)])
        flat = r.ravel(order="F")
        assert flat.shape == expected.shape
        assert np.all((flat - expected < tol) | np.isnan(expected))
        # Independent absolute-error and padding checks supplement the source.
        assert _ninf(r[:, 0] - np.arange(-2, 8)) < tol
        assert _ninf(r[:-1, 1] - np.arange(-1.5, 7)) < tol
        assert np.isnan(r[-1, 1]) and np.all(np.isnan(r[:, 2]))

    def test_singular_roots(self):
        op = lambda x: (x - DOM.a) ** -0.5 * jnp.cos(x)
        f = Bndfun.from_function(op, DOM, exponents=(-0.5, 0.0))
        r = f.roots()
        exact = jnp.asarray([-0.5, 0.5, 1.5]) * jnp.pi
        assert r.shape == exact.shape
        assert _ninf(r - exact) < 100 * f.vscale * EPS

    def test_unbndfun_roots(self):
        op = lambda x: x ** 2 * (1 - jnp.exp(-x ** 2)) - 2
        f = Unbndfun.from_function(op, Domain((-np.inf, np.inf)), exps=(2.0, 2.0))
        r = f.roots()
        exact = jnp.asarray([-1.4962104914103104707, 1.4962104914103104707])
        assert r.shape == exact.shape
        assert _ninf(r - exact) < 100 * EPS * f.vscale

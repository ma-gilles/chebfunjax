"""Port of MATLAB Chebfun tests/chebtech/test_roots.m (Opus 4.8).

Self-validating: roots are checked against analytic exacts at the SAME
tolerance MATLAB uses (multiples of length(f)*eps).  The MATLAB file loops
``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; here each assertion is
parametrized over both classes.

Option surface (all ported):
- ``roots(f, 'complex', 1)`` -> ``roots(complex_roots=True)``;
  ``'prune'`` -> ``prune=``; ``'recurse'`` -> ``recurse=``.  MATLAB's
  ``'complex'`` is exactly ``all_roots=True, prune=True``.
- ``roots(f, 'qz', 1)`` -> ``roots(qz=True)``.
- array-valued ``roots``: roots() loops the colleague matrix per column and
  NaN-pads, so pass(n, 9) ports directly.

Provenance
----------
MATLAB source : tests/chebtech/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)

# 32 positive zeros of J0 in [0, 100] (MATLAB besselj reference).
_BESSEL_J0_ZEROS = np.array(
    [
        2.40482555769577276862163,
        5.52007811028631064959660,
        8.65372791291101221695437,
        11.7915344390142816137431,
        14.9309177084877859477626,
        18.0710639679109225431479,
        21.2116366298792589590784,
        24.3524715307493027370579,
        27.4934791320402547958773,
        30.6346064684319751175496,
        33.7758202135735686842385,
        36.9170983536640439797695,
        40.0584257646282392947993,
        43.1997917131767303575241,
        46.3411883716618140186858,
        49.4826098973978171736028,
        52.6240518411149960292513,
        55.7655107550199793116835,
        58.9069839260809421328344,
        62.0484691902271698828525,
        65.1899648002068604406360,
        68.3314693298567982709923,
        71.4729816035937328250631,
        74.6145006437018378838205,
        77.7560256303880550377394,
        80.8975558711376278637723,
        84.0390907769381901578795,
        87.1806298436411536512617,
        90.3221726372104800557177,
        93.4637187819447741711905,
        96.6052679509962687781216,
        99.7468198586805964702799,
    ]
)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechRoots:
    def test_roots_bessel(self, Tech):
        # pass(n, 1): roots of besselj(0, 50*(x+1)) mapped back to [0, 100].
        f = Tech.from_function(lambda x: sp.jv(0, 50.0 * (np.asarray(x) + 1.0)))
        r = np.sort(np.asarray(f.roots()))
        rm = (r + 1.0) * 50.0
        assert rm.size == _BESSEL_J0_ZEROS.size
        assert _ninf(rm - _BESSEL_J0_ZEROS) < 1e1 * f.n * EPS

    def test_roots_oscillatory(self, Tech):
        # pass(n, 2): roots of sin(pi*k*x) are -k/k, ..., k/k.
        k = 500
        f = Tech.from_function(lambda x: jnp.sin(jnp.pi * k * x))
        r = np.sort(np.asarray(f.roots()))
        exact = np.arange(-k, k + 1) / k
        assert r.size == exact.size
        assert _ninf(r - exact) < f.n * EPS

    def test_roots_perturbed_poly(self, Tech):
        # pass(n, 3): perturbed quartic has exactly 4 roots.
        f = Tech.from_function(
            lambda x: (x - 0.1) * (x + 0.9) * x * (x - 0.9) + 1e-14 * x**5
        )
        r = f.roots()
        assert len(r) == 4
        assert _ninf(f(jnp.asarray(r))) < 1e2 * f.n * EPS

    def test_roots_simple_linear(self, Tech):
        # pass(n, 4): values [-1; 1] -> linear f, single root at 0.
        f = Tech.from_values(jnp.asarray([-1.0, 1.0]))
        r = np.asarray(f.roots())
        assert r.size >= 1 and np.all(r == 0.0)

    def test_roots_simple_quadratic(self, Tech):
        # pass(n, 5): values [1; 0; 1] -> f ~ x^2, double root at 0.
        f = Tech.from_values(jnp.asarray([1.0, 0.0, 1.0]))
        r = np.asarray(f.roots())
        assert r.size == 2 and _ninf(r) < EPS

    def test_roots_complex_pair(self, Tech):
        # pass(n, 6): roots(1 + 25 x^2, 'complex', 1) -> +/- i/5.
        f = Tech.from_function(lambda x: 1 + 25 * x**2)
        r = np.asarray(f.roots(complex_roots=True))
        # MATLAB compares against [1i; -1i]/5; the eig order is arbitrary, so
        # sort by descending imaginary part (a faithful set/order comparison).
        r = r[np.argsort(np.imag(r))[::-1]]
        assert r.size == 2
        assert _ninf(r - np.array([1j, -1j]) / 5) < 10 * EPS

    def test_roots_complex_prune(self, Tech):
        # pass(n, 7): roots((1+25x^2)exp(x), 'complex', 1, 'prune', 1).
        f = Tech.from_function(lambda x: (1 + 25 * x**2) * jnp.exp(x))
        r = np.asarray(f.roots(complex_roots=True, prune=True))
        r = r[np.argsort(np.imag(r))[::-1]]
        assert r.size == 2
        assert _ninf(r - np.array([1j, -1j]) / 5) < 10 * f.n * EPS

    def test_roots_complex_recurse(self, Tech):
        # pass(n, 8): roots(sin(100 pi x), 'complex', 1, 'recurse', 0/1).
        f = Tech.from_function(lambda x: jnp.sin(100 * jnp.pi * x))
        r1 = f.roots(complex_roots=True, recurse=False)
        r2 = f.roots(complex_roots=True)
        assert np.asarray(r1).size == 201 and np.asarray(r2).size >= 213

    # FIXED (Fable 5, Big-Three array-valued epic): roots() now loops
    # the colleague matrix per column and NaN-pads (both tech classes).
    def test_roots_array_valued(self, Tech):
        # pass(n, 9): roots of [sin(pi x), cos(pi x)] (array-valued);
        # MATLAB compares the flattened NaN-padded matrix.
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1))
        r = np.asarray(f.roots())
        r2 = np.array([[-1.0, -0.5], [0.0, 0.5], [1.0, np.nan]])
        ok = (np.abs(r - r2) < 10 * f.n * EPS) | np.isnan(r2)
        assert bool(np.all(ok))

    # roots(f, 'qz', 1): colleague-matrix pencil solved by the QZ /
    # generalized-eigenvalue algorithm (chebfunjax roots(qz=True)).
    def test_roots_qz_nonempty(self, Tech):
        # pass(n, 10): roots(1e-10 x^3 + x^2 - 1e-12, 'qz', 1) non-empty.
        f = Tech.from_function(lambda x: 1e-10 * x ** 3 + x ** 2 - 1e-12)
        r = f.roots(qz=True)
        assert np.asarray(r).size > 0

    def test_roots_qz_feval(self, Tech):
        # pass(n, 11): feval at the 'qz' roots is ~0.
        f = Tech.from_function(lambda x: 1e-10 * x ** 3 + x ** 2 - 1e-12)
        r = f.roots(qz=True)
        assert _ninf(f(jnp.asarray(r))) < 10 * EPS

    def test_roots_qz_lowdegree(self, Tech):
        # pass(n, 12): roots((x-.5)(x-1/3), 'qz', 1).
        f = Tech.from_function(lambda x: (x - 0.5) * (x - 1.0 / 3.0))
        r = f.roots(qz=True)
        assert _ninf(f(jnp.asarray(r))) < EPS

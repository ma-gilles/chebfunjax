"""Port of MATLAB Chebfun tests/chebfun/test_polyfitL1.m (Fable 5).

The L1-optimality condition int T_k * sign(f - p) = 0 is evaluated
EXACTLY via closed-form Chebyshev segment integrals between the
bisection-refined sign changes (MATLAB integrates chebfuns exactly;
a discrete-grid proxy only reaches O(grid spacing)).

FIXED in the Fable 5 audit: polyfitL1's previous 'Watson update' was a
relaxation onto a fixed interpolant (not L1-optimal); now a proper
Watson-Newton iteration on the coefficients with the SPD Jacobian
J_kj = 2 sum_r T_k(r) T_j(r)/|e'(r)|.

Provenance
----------
MATLAB source : tests/chebfun/test_polyfitL1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as C

import chebfunjax as cj

TOL = 1e-8


def _int_T(k, lo, hi):
    if k == 0:
        return hi - lo
    if k == 1:
        return 0.5 * (hi ** 2 - lo ** 2)

    def A(t):
        return 0.5 * (C.chebval(t, np.eye(k + 2)[k + 1]) / (k + 1)
                      - C.chebval(t, np.eye(k)[k - 1]) / (k - 1))
    return A(hi) - A(lo)


def _optimality(f_np, p, n):
    sfine = np.linspace(-1.0, 1.0, 4001)
    e = f_np(sfine) - np.asarray(p(jnp.asarray(sfine)))
    idx = np.nonzero(np.diff(np.sign(e)) != 0)[0]
    roots = []
    for i in idx:
        lo, hi = sfine[i], sfine[i + 1]
        flo = e[i]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm = f_np(np.array([mid]))[0] - float(p(jnp.asarray(mid)))
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        roots.append(0.5 * (lo + hi))
    seg = np.concatenate([[-1.0], np.asarray(roots), [1.0]])
    mids = 0.5 * (seg[:-1] + seg[1:])
    sig = np.sign(f_np(mids) - np.asarray(p(jnp.asarray(mids))))
    return max(abs(sum(sig[j] * _int_T(k, seg[j], seg[j + 1])
                       for j in range(len(sig))))
               for k in range(n + 1))


class TestChebfunPolyfitL1:
    def test_l1_optimality_condition(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(10 * x))
        n = 5
        p = f.polyfitL1(n)
        opt = _optimality(lambda s: np.exp(s) * np.sin(10 * s), p, n)
        assert opt < TOL

    def test_known_best_l1_of_abs_plus_x(self):
        # MATLAB pass(1): best degree-1 L1 approx of |x| + x is 0.5 + x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = cj.chebfun(lambda x: jnp.abs(x) + x, splitting=True)
        p = g.polyfitL1(1)
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 19))
        err = jnp.abs(p(xs) - (0.5 + xs))
        assert float(jnp.max(err)) < 1e-10

    def test_beats_l2_projection_in_l1(self):
        from chebfunjax.utils.quadrature import legpts
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(10 * x))
        p1 = f.polyfitL1(5)
        p2 = f.polyfit(5)
        x, w = (np.asarray(v) for v in legpts(3000))
        fx = np.exp(x) * np.sin(10 * x)
        e1 = np.dot(w, np.abs(fx - np.asarray(p1(jnp.asarray(x)))))
        e2 = np.dot(w, np.abs(fx - np.asarray(p2(jnp.asarray(x)))))
        assert e1 < e2

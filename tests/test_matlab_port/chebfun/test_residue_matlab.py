"""Port of MATLAB Chebfun tests/chebfun/test_residue.m (Fable 5).

FIXED: residue (partial fractions, both directions) added in the
Fable 5 audit (chebfunjax.residue, scipy.signal-backed).

Provenance
----------
MATLAB source : tests/chebfun/test_residue.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunResidue:
    def test_all_matlab_assertions(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(lambda x: (x - 1.1) * (x ** 2 + 1)
                           * (x - 10j))
            g = cj.chebfun(lambda x: x ** 5)
            r, p, k = cj.residue(g, f)
            G, F = cj.residue(r, p, k)

        tol = 100 * np.finfo(float).eps
        p = np.asarray(p)
        pexact = np.array([10j, 1.1, -1j, 1j])
        # pass(1): poles match (real/imag parts sorted separately)
        err = (np.linalg.norm(np.sort(p.real) - np.sort(pexact.real))
               + np.linalg.norm(np.sort(p.imag) - np.sort(pexact.imag)))
        assert err < 10 * tol

        # pass(2): g*F == G*f up to normalization
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 33))
        gF = np.asarray(g(xs)) * np.asarray(F(xs))
        Gf = np.asarray(G(xs)) * np.asarray(f(xs))
        kv = np.asarray(k(xs))
        assert np.max(np.abs(gF - Gf)) / np.max(np.abs(gF)) \
            < 10 * tol * np.max(np.abs(kv))

        # pass(3): quotient k == x + 10i + 1.1
        assert np.max(np.abs(kv - (np.asarray(xs) + 10j + 1.1))) \
            < 10 * tol * 11

        # pass(4): empty third argument treated as zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            B, A = cj.residue(np.array([1.0, 1.0]),
                              np.array([1.0, -1.0]), None)
        assert np.max(np.abs(np.asarray(B(xs)) - 2 * np.asarray(xs))) \
            < tol
        assert np.max(np.abs(np.asarray(A(xs))
                             - (np.asarray(xs) ** 2 - 1))) < tol

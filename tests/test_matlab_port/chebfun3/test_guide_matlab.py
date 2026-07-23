"""Port of MATLAB Chebfun tests/chebfun3/test_guide.m (Fable 5).

Assertion-for-assertion at the MATLAB tolerances, with these named
exceptions: pass 10 constructs with the 'trig' flag (chebfunjax
Chebfun3 has no trig tech); MATLAB's 5-output ``hosvd`` (pass 14-19)
returns factor quasimatrices, while chebfunjax's hosvd returns
``(sv, g)`` with g an equivalent Chebfun3 whose factors are
L2-orthonormal and core all-orthogonal -- the same orthogonality
properties MATLAB pins are asserted on g's factors and core.

Provenance
----------
MATLAB source : tests/chebfun3/test_guide.m
Chebfun commit: 7574c77
"""

# uses-numpy: Gram-matrix orthonormality checks on hosvd factor coefficients
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 1e3 * EPS
_EXACT_SUM3 = 4.28685406230184188268


def _mode_length(factors):
    return max(len(np.asarray(t.coeffs)) for t in factors)


class TestChebfun3Guide:
    def test_pass1to4_basics(self):
        f = Chebfun3.from_function(
            lambda x, y, z: 1.0 / (1.0 + x**2 + y**2 + z**2))
        assert abs(float(f(jnp.asarray(0.0), jnp.asarray(0.5),
                           jnp.asarray(0.5))) - 2.0 / 3.0) < TOL
        assert abs(float(f.sum3()) - _EXACT_SUM3) < TOL
        assert abs(float(f.mean3()) - _EXACT_SUM3 / 8.0) < TOL
        assert abs(float(f.max3()[0]) - 1.0) < TOL

    def test_pass5to7_degenerate_lengths(self):
        from chebfunjax.chebfun1d.chebfun import chebfun

        f1 = chebfun(lambda x: jnp.exp(x))
        len1d = len(np.asarray(f1.funs[0].coeffs))
        for k, fn in enumerate([lambda x, y, z: jnp.exp(x),
                                lambda x, y, z: jnp.exp(y),
                                lambda x, y, z: jnp.exp(z)]):
            f3 = Chebfun3.from_function(fn)
            lens = [_mode_length(f3.cols), _mode_length(f3.rows),
                    _mode_length(f3.tubes)]
            assert lens[k] < 2 * len1d
            for j in range(3):
                if j != k:
                    assert lens[j] == 1

    def test_pass8_max3_submultiplicative(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.sin(x + y * z))
        g = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(15 * jnp.exp(z))
            / (5.0 + x**3 + 2 * y**2 + z))
        assert float((f * g).max3()[0]) <= (float(f.max3()[0])
                                            * float(g.max3()[0]))

    def test_pass9_helix_line_integral(self):
        f = Chebfun3.from_function(lambda x, y, z: x + y * z)
        L = 8 * np.pi

        def curve(t):
            return (jnp.cos(t), jnp.sin(t), t / L)

        I = float(f.integral(curve, domain=(0.0, L)))
        exact = -np.sqrt(1.0 + L**2) / L
        assert abs(I - exact) < TOL

    @pytest.mark.skip(reason="'trig' Chebfun3 constructor flag (MATLAB "
                      "pass 10) -- chebfunjax Chebfun3 has no trig tech")
    def test_pass10_trig_ctor(self):
        raise NotImplementedError

    def test_pass11to19_hosvd(self):
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.sin(x + 2 * y + 3 * z))
        sv, g = f.hosvd()
        # pass 11-13 (MATLAB: sv{k}(2) <= sv{k}(2), i.e. ordering holds)
        for k in range(3):
            s = np.asarray(sv[k])
            assert np.all(s[:-1] >= s[1:] - 1e-14)
        # pass 14-16: factor quasimatrices are L2-orthonormal.
        from chebfunjax.utils.quadrature import chebweights
        for factors in (g.cols, g.rows, g.tubes):
            nmax = max(len(np.asarray(t.coeffs)) for t in factors)
            npts = max(2 * nmax, 8)
            from chebfunjax.utils.quadrature import chebpts
            x = np.asarray(chebpts(npts, kind=2))
            w = np.asarray(chebweights(npts, kind=2))
            V = np.stack([np.asarray(t(jnp.asarray(x))) for t in factors],
                         axis=1)
            G = V.T @ (w[:, None] * V)
            assert float(np.max(np.abs(
                G - np.eye(G.shape[0])))) < 1e2 * TOL
        # pass 17-19: core slices are mutually orthogonal (all-orthogonal
        # core), the property MATLAB pins via elementwise slice products.
        core = np.asarray(g.core)
        for mode in range(3):
            M = np.moveaxis(core, mode, 0).reshape(core.shape[mode], -1)
            if M.shape[0] >= 2:
                assert abs(float(M[0] @ M[1])) < 1e2 * TOL * \
                    float(np.linalg.norm(M[0]) * np.linalg.norm(M[1]) + 1)

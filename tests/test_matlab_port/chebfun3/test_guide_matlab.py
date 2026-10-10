"""Chebfun3 guide tests, with native HOSVD assertions11--19 restored.

MATLAB source: tests/chebfun3/test_guide.m, Chebfun7574c77.
HOSVD uses all five outputs, native matrix norms, and unrelaxed tolerance.
Earlier guide predicates are retained; their broader audit remains open.
"""

# uses-numpy: inherited earlier guide test array helpers.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref

EPS = float(np.finfo(np.float64).eps)
TOL = 1e3 * ChebfunPref().cheb3Prefs.chebfun3eps
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

    def test_pass10_trig_ctor(self):
        from chebfunjax.chebfun3d.chebfun3 import chebfun3
        ff = lambda x, y, z: jnp.tanh(3 * jnp.sin(x)) - jnp.sin(y + 0.5) ** 2 + jnp.cos(6 * z)  # noqa: E731
        dom = (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi)
        m, n, p = chebfun3(ff, dom, trig=True).length()
        mc, nc, pc = chebfun3(ff, dom).length()
        assert m <= mc and n <= nc and p <= pc                               # pass(10)

    def test_pass11to19_hosvd(self):
        # Literal native assertions11--19, including its three tautologies.
        f = Chebfun3.from_function(lambda x, y, z: jnp.sin(x+2*y+3*z))
        values, core, cols, rows, tubes = f.hosvd(return_factors=True)
        for mode in values:
            assert mode[1] <= mode[1]
        for panel in (cols, rows, tubes):
            gram = panel.ctranspose() @ panel
            assert jnp.linalg.norm(jnp.eye(gram.shape[0])-gram, ord=2) < TOL
        assert jnp.linalg.norm(core[0, :, :]*core[1, :, :], ord=2) < TOL
        assert jnp.linalg.norm(core[:, 0, :]*core[:, 1, :], ord=2) < TOL
        assert jnp.linalg.norm(core[:, :, 0]*core[:, :, 1], ord=2) < TOL

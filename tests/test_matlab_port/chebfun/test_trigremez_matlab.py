"""Port of MATLAB Chebfun tests/chebfun/test_trigremez.m (Fable 5).

FIXED: trigremez (periodic Remez, Javed & Trefethen; polynomial
case) added in the Fable 5 audit (Big-Three trig-rational
directive).  The rational case remains a documented gap.

Provenance
----------
MATLAB source : tests/chebfun/test_trigremez.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

TOL = 1e-10
TT = jnp.asarray(np.linspace(0.01, 1.99, 60))


class TestChebfunTrigremez:
    def test_best_approx_and_equioscillation(self):
        f = cj.chebfun(lambda x: jnp.cos(4 * np.pi * x) + 1,
                       domain=(0.0, 2.0), trig=True)
        # pass(2)-(3): degree 1 -> the constant 1, equioscillating
        p, errmax, status = cj.trigremez(f, 1)
        assert float(jnp.max(jnp.abs(p(TT) - 1.0))) < 100 * TOL
        xk = jnp.asarray(status["xk"])
        err_ref = np.abs(np.asarray(f(xk)) - np.asarray(p(xk)))
        assert float(np.max(np.abs(err_ref - errmax))) < 100 * TOL

    def test_degree_4_reproduces(self):
        # pass(4)-(5) analogue: degree 4 reproduces the function
        f = cj.chebfun(lambda x: jnp.cos(4 * np.pi * x) + 1,
                       domain=(0.0, 2.0), trig=True)
        p, errmax, _ = cj.trigremez(f, 4)
        assert float(jnp.max(jnp.abs(p(TT) - f(TT)))) < 100 * TOL
        assert errmax < 100 * TOL


class TestTrigremezRational:
    """MATLAB passes 8-13: the rational (m, n) mode."""

    def test_abs_x_rational_equioscillation(self):
        # passes 8-11: f = |x|, type (2,2): reference errors
        # equioscillate with alternating signs.
        f = cj.chebfun(lambda x: jnp.abs(x), splitting=True)
        p, q, r, err, status = cj.trigremez(f, 2, 2)
        assert len(p) == 5 and len(q) == 5
        xk = np.asarray(status["xk"])
        equi = np.asarray(f(jnp.asarray(xk))) - np.asarray(r(xk))
        assert float(np.std(np.abs(equi))) < 1e-8
        if equi[0] < 0:
            equi = -equi
        assert np.all(np.sign(equi[0::2]) == 1)
        assert np.all(np.sign(equi[1::2]) == -1)

    def test_exp_sin_pi_x_type_6_6(self):
        # passes 12-13: f = exp(sin(pi x)), type (6,6): near-exact.
        f = cj.chebfun(lambda x: jnp.exp(jnp.sin(np.pi * x)), trig=True)
        p, q, r, err, status = cj.trigremez(f, 6, 6)
        xs = np.linspace(-1, 1, 1500)
        e = np.max(np.abs(np.asarray(f(jnp.asarray(xs)))
                          - np.asarray(r(xs))))
        assert float(e) < 1e-8
        assert len(p) == 13 and len(q) == 13

    def test_trigcf_example_reference_error(self):
        # The TrigCFExample page's punchline value: exp(sin t), type
        # (2,1) has minimax error 0.001789066754500 (13 digits here).
        f = cj.chebfun(lambda t: jnp.exp(jnp.sin(t)),
                       domain=[-np.pi, np.pi], trig=True)
        _, _, _, err, _ = cj.trigremez(f, 2, 1)
        assert abs(err - 0.001789066754500) < 5e-12

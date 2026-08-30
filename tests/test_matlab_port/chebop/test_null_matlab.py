"""Port of MATLAB Chebfun tests/chebop/test_null.m (Fable 5).

The chebcolloc2 discretization loop entry maps to the two altdisc
discretizations chebfunjax assembles (ultraS, chebcolloc1).

Provenance
----------
MATLAB source : tests/chebop/test_null.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-7


def _n(f, d, k=33):
    xs = jnp.linspace(d[0] + 1e-6, d[1] - 1e-6, k)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


def _ip(u, v):
    return float(jnp.asarray(u.inner(v)))


class TestChebopNull:
    def test_all_matlab_assertions(self):
        for disc in ("ultraS", "chebcolloc1"):
            # Test 1: null of d/dx on [0, pi] = constants.
            N = Chebop(lambda u: u.diff(), domain=(0.0, math.pi))
            V = N.null(disc)
            assert len(V) == 1                              # pass(2)
            assert _n(N(V[0]), (0.0, math.pi)) < TOL        # pass(1)
            assert abs(_ip(V[0], V[0]) - 1.0) < TOL         # pass(3)

            # Test 2: 0.2 u''' - sin(3x) u' with u(1) = const.
            N = Chebop(lambda x, u: 0.2 * u.diff(3)
                       - (3.0 * x).sin() * u.diff())
            N.rbc = 1.0
            V = N.null(disc)
            assert len(V) == 2                              # pass(5)
            for v in V:
                assert _n(N(v), (-1.0, 1.0)) < 1e5 * TOL    # pass(4)
                assert abs(float(v(jnp.asarray(1.0)))) < TOL  # pass(7)
            G = [[_ip(a, b) for b in V] for a in V]
            assert abs(G[0][0] - 1) + abs(G[1][1] - 1) \
                + abs(G[0][1]) < TOL                        # pass(6)

            # Test 3: first-order system.
            L = Chebop(lambda x, u, v: [u.diff() + v, v.diff() + u])
            V = L.null(disc)
            assert len(V) == 2                              # dim
            for comp in V:
                out = L(comp[0], comp[1])
                assert max(_n(out[0], (-1.0, 1.0)),
                           _n(out[1], (-1.0, 1.0))) < TOL   # pass(8)

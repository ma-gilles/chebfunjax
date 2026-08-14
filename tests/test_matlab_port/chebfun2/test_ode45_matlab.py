"""Port of MATLAB Chebfun tests/chebfun2/test_ode45.m (Fable 5).

ode45(F, tspan, y0) with a chebfun2v vector field returns chebfun
trajectories.  MATLAB's pass(1) uses an events option to stop the
integration (not ported — no events interface); the initial-value
recovery it checks is asserted directly, and the phase-plane sweeps
of pass(2) run verbatim.

Provenance
----------
MATLAB source : tests/chebfun2/test_ode45.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


def _cf2(fn, dom):
    return Chebfun2.from_function(fn, domain=dom).approx


class TestChebfun2Ode45:
    def test_projectile(self):
        # pass(1): h'' = -1 - 0.01 h' with the (h, h', x) state trick;
        # Y(0) must recover u0 (MATLAB bound 1e-3).
        dom = (0.0, 30.0, 0.0, 2.0)
        F = Chebfun2v([
            _cf2(lambda h, hp: 0 * h + hp, dom),
            _cf2(lambda h, hp: -1.0 - 0.01 * hp, dom),
            _cf2(lambda h, hp: 1.0 + 0 * h, dom),
        ])
        T, Y = F.ode45((0.0, 2.0), [2.0, 0.0, 0.0])
        assert abs(float(np.asarray(Y(jnp.asarray(0.0)))[0]) - 2.0) < 1e-3

    def test_phase_plane(self):
        # pass(2): the A*g linear phase-plane sweeps run; cross-check the
        # flow of u' = A u against the matrix exponential.
        from scipy.linalg import expm

        A = np.array([[2.0, -2.0], [0.0, 1.0]])
        G = Chebfun2v([
            _cf2(lambda x, y: 2 * x - 2 * y, (-1.0, 1.0, -1.0, 1.0)),
            _cf2(lambda x, y: 0 * x + y, (-1.0, 1.0, -1.0, 1.0)),
        ])
        for u0 in ([0.1, 0.05], [-0.1, -0.05], [-0.1, 0.0], [0.1, 0.0]):
            _, y = G.ode45((0.0, 1.0), u0)
            want = expm(A) @ np.asarray(u0)
            got = np.asarray(y(jnp.asarray(1.0)))
            assert np.max(np.abs(got - want)) < 1e-8
        for u0 in ([0.1, 0.1], [-0.1, -0.1]):
            _, y = G.ode45((0.0, 2.0 / 3.0), u0)
            want = expm((2.0 / 3.0) * A) @ np.asarray(u0)
            got = np.asarray(y(jnp.asarray(2.0 / 3.0)))
            assert np.max(np.abs(got - want)) < 1e-8

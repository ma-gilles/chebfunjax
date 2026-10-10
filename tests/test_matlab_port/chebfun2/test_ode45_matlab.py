"""Native default-options phase-plane predicate plus separate Python controls.

Native pass(2): all seven source calls, including the duplicated initial state,
with default options and successful completion as the only predicate.
The shortened projectile and tight analytic endpoint tests are independent
Python regressions. The separate test_native45_projectile_event.py preserves
native pass(1), including [0,30], RelTol=100*eps and its terminal event callback.
Independent crossing controls qualify the locator; the native domain mask
can prevent the projectile from reaching the event.

Provenance
----------
MATLAB source: tests/chebfun2/test_ode45.m; Chebfun commit7574c77.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


def _cf2(fn, dom):
    return Chebfun2.from_function(fn, domain=dom).approx


class TestChebfun2Ode45:
    def test_shortened_projectile_python_control(self):
        # Independent shortened no-event control; full native pass(1) is tested separately: h'' = -1 - 0.01 h' with the (h, h', x) state trick;
        # Y(0) must recover u0 (MATLAB bound 1e-3).
        dom = (0.0, 30.0, 0.0, 2.0)
        F = Chebfun2v([
            _cf2(lambda h, hp: 0 * h + hp, dom),
            _cf2(lambda h, hp: -1.0 - 0.01 * hp, dom),
            _cf2(lambda h, hp: 1.0 + 0 * h, dom),
        ])
        T, Y = F.ode45((0.0, 2.0), [2.0, 0.0, 0.0])
        assert abs(float(np.asarray(Y(jnp.asarray(0.0)))[0]) - 2.0) < 1e-3

    def test_native_phase_plane_default_success(self):
        g = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        A = [[2., -2.], [0., 1.]]
        components = [Chebfun2(approx=c) for c in g.components]
        G = Chebfun2v([(A[j][0]*components[0]+A[j][1]*components[1]).approx
                       for j in range(2)])
        for u0 in ([.1, .05], [-.1, -.05], [-.1, -.05], [-.1, 0.], [.1, 0.]):
            G.ode45((0., 1.), u0)
        for u0 in ([.1, .1], [-.1, -.1]):
            G.ode45((0., 2./3.), u0)

    def test_tight_endpoint_python_control(self):
        # Existing extra analytic check with its historical Python tolerances;
        # this is separate from the native default-options success predicate.
        from scipy.linalg import expm

        A = np.array([[2.0, -2.0], [0.0, 1.0]])
        G = Chebfun2v([
            _cf2(lambda x, y: 2 * x - 2 * y, (-1.0, 1.0, -1.0, 1.0)),
            _cf2(lambda x, y: 0 * x + y, (-1.0, 1.0, -1.0, 1.0)),
        ])
        for u0 in ([0.1, 0.05], [-0.1, -0.05], [-0.1, 0.0], [0.1, 0.0]):
            _, y = G.ode45((0.0, 1.0), u0, rtol=1e-10, atol=1e-12)
            want = expm(A) @ np.asarray(u0)
            value = complex(y(jnp.asarray(1.0)))
            got = np.asarray([value.real, value.imag])
            assert np.max(np.abs(got - want)) < 1e-8
        for u0 in ([0.1, 0.1], [-0.1, -0.1]):
            _, y = G.ode45((0.0, 2.0 / 3.0), u0, rtol=1e-10, atol=1e-12)
            want = expm((2.0 / 3.0) * A) @ np.asarray(u0)
            value = complex(y(jnp.asarray(2.0 / 3.0)))
            got = np.asarray([value.real, value.imag])
            assert np.max(np.abs(got - want)) < 1e-8

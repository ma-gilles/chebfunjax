"""Port of MATLAB Chebfun tests/chebfun2v/test_constructor2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_constructor2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * 2.220446049250313e-16


def _maxdiff(F, fns, dom=(-1.0, 1.0, -1.0, 1.0)):
    xs = jnp.linspace(dom[0] + 1e-9, dom[1] - 1e-9, 9)
    ys = jnp.linspace(dom[2] + 1e-9, dom[3] - 1e-9, 9)
    X, Y = jnp.meshgrid(xs, ys)
    worst = 0.0
    for c, fn in zip(F.components, fns):
        f2 = Chebfun2(approx=c)
        worst = max(worst, float(jnp.max(jnp.abs(
            jnp.asarray(f2(X, Y)) - fn(X, Y)))))
    return worst


class TestChebfun2vConstructor2:
    def test_vectorize_flag(self):
        import math  # noqa: F401
        f2 = lambda x, y: math.prod([float(x), float(y)])
        target = lambda X, Y: X * Y
        # pass(1): 2 components.
        H2 = Chebfun2v.from_functions(f2, f2, vectorize=True)
        assert _maxdiff(H2, [target, target]) < TOL
        # pass(2): with a domain.
        dom = (-2.0, 3.0, -1.0, 0.0)
        H2 = Chebfun2v.from_functions(f2, f2, vectorize=True,
                                      domain=dom)
        assert _maxdiff(H2, [target, target], dom) < TOL
        # pass(3)/(4): 3 components.
        H2 = Chebfun2v.from_functions(f2, f2, f2, vectorize=True)
        assert _maxdiff(H2, [target, target, target]) < TOL
        H2 = Chebfun2v.from_functions(f2, f2, f2, vectorize=True,
                                      domain=dom)
        assert _maxdiff(H2, [target, target, target], dom) < TOL

    def test_surface_example(self):
        # pass(5)/(6): Moebius-strip tangents are orthogonal, and
        # nComponents reports 3.
        dom = (0.0, 2 * np.pi, -1.0, 1.0)
        x = Chebfun2.from_function(
            lambda u, v: (1 + 0.5 * v * jnp.cos(u / 2)) * jnp.cos(u),
            domain=dom)
        y = Chebfun2.from_function(
            lambda u, v: (1 + 0.5 * v * jnp.cos(u / 2)) * jnp.sin(u),
            domain=dom)
        z = Chebfun2.from_function(
            lambda u, v: 0.5 * v * jnp.sin(u / 2), domain=dom)
        r = Chebfun2v([x.approx, y.approx, z.approx])
        ru = r.diff(1, dim=1)
        rv = r.diff(1, dim=2)
        ip = ru.dot(rv)
        xs = jnp.linspace(0.1, 2 * np.pi - 0.1, 9)
        ys = jnp.linspace(-0.9, 0.9, 9)
        X, Y = jnp.meshgrid(xs, ys)
        assert float(jnp.max(jnp.abs(jnp.asarray(
            Chebfun2(approx=ip.approx if hasattr(ip, "approx")
                     else ip)(X, Y))))) < 10 * TOL
        assert r.n_components == 3

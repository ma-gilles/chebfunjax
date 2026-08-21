"""Port of MATLAB Chebfun tests/chebfun2v/test_subsref.m (Fable 5).

Component indexing maps to .components; the composition passes go
through Chebfun2v.compose / Chebfun3v.compose.

Provenance
----------
MATLAB source : tests/chebfun2v/test_subsref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

jax.config.update("jax_enable_x64", True)

TOL = 1000 * 2.220446049250313e-16


def _dev2(F, fns, dom=(-1.0, 1.0, -1.0, 1.0)):
    xs = jnp.linspace(dom[0] + 1e-9, dom[1] - 1e-9, 7)
    ys = jnp.linspace(dom[2] + 1e-9, dom[3] - 1e-9, 7)
    X, Y = jnp.meshgrid(xs, ys)
    worst = 0.0
    comps = F.components
    for c, fn in zip(comps, fns):
        worst = max(worst, float(jnp.max(jnp.abs(
            jnp.asarray(Chebfun2(approx=c)(X, Y)) - fn(X, Y)))))
    return worst


class TestChebfun2vSubsref:
    def test_component_indexing(self):
        # pass(1): F(1) recovers the first component with its pivots.
        f = Chebfun2.from_function(lambda x, y: jnp.sin(x * y))
        F = Chebfun2v([f.approx, f.approx, f.approx])
        G = F.components[0]
        assert np.allclose(np.asarray(G.pivots),
                           np.asarray(f.approx.pivots))

    def test_compose_with_two_chebfun2(self):
        # pass(2)-(4): G(f1, f2) for 2- and 3-component G.
        dom = (0.0, 2.0, -2.0, 2.0)
        f1 = Chebfun2.from_function(lambda x, y: x + 1)
        f2 = Chebfun2.from_function(lambda x, y: 2 * y)
        F12 = Chebfun2v([f1.approx, f2.approx])

        G = Chebfun2v.from_functions(
            lambda x, y: x + y, lambda x, y: x - y, domain=dom)
        H = F12.compose(G)
        assert _dev2(H, [lambda X, Y: X + 2 * Y + 1,
                         lambda X, Y: X - 2 * Y + 1]) < TOL
        vals = H(jnp.asarray(1.0), jnp.asarray(1.0))
        assert np.allclose(np.asarray(vals).ravel(),
                           [4.0, 0.0], atol=TOL)

        G3 = Chebfun2v.from_functions(
            lambda x, y: x + y, lambda x, y: x - y,
            lambda x, y: x * y, domain=dom)
        H3 = F12.compose(G3)
        assert _dev2(H3, [lambda X, Y: X + 2 * Y + 1,
                          lambda X, Y: X - 2 * Y + 1,
                          lambda X, Y: 2 * (X + 1) * Y]) < TOL

    def test_compose_with_chebfun2v(self):
        # pass(6)/(7): G(F) with F the identity field.
        F = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        G = Chebfun2v.from_functions(lambda x, y: x + y,
                                     lambda x, y: x - y)
        H = F.compose(G)
        assert _dev2(H, [lambda X, Y: X + Y,
                         lambda X, Y: X - Y]) < TOL
        G3 = Chebfun2v.from_functions(lambda x, y: x + y,
                                      lambda x, y: x - y,
                                      lambda x, y: x * y)
        H3 = F.compose(G3)
        assert _dev2(H3, [lambda X, Y: X + Y, lambda X, Y: X - Y,
                          lambda X, Y: X * Y]) < TOL

    def test_compose_with_chebfun3_inputs(self):
        # pass(5): G(f1, f2) with chebfun3 inputs -> Chebfun3v.
        f1 = Chebfun3.from_function(lambda x, y, z: x + 0 * y)
        f2 = Chebfun3.from_function(lambda x, y, z: y + z + 0 * x)
        F = Chebfun3v([f1, f2] if not hasattr(Chebfun3v, "from_list")
                      else [f1, f2])
        G = Chebfun2v.from_functions(
            lambda x, y: x, lambda x, y: y,
            domain=(-1.0, 1.0, -2.0, 2.0))
        H = F.compose(G)
        xs = jnp.linspace(-0.9, 0.9, 5)
        X, Y, Z = jnp.meshgrid(xs, xs, xs)
        want = [X, Y + Z]
        for c, w in zip(H.components, want):
            assert float(jnp.max(jnp.abs(
                jnp.asarray(c(X, Y, Z)) - w))) < 10 * TOL

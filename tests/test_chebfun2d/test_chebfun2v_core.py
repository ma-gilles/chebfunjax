"""Core coverage for Chebfun2v laplacian/isreal/shape and complex handles."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


class TestChebfun2vCore:
    def test_laplacian_harmonic_is_zero(self):
        # x^2 - y^2 is harmonic: componentwise Laplacian vanishes.
        F = Chebfun2v.from_functions(lambda x, y: x ** 2 - y ** 2,
                                     lambda x, y: 2 * x * y)
        L = F.laplacian()
        for c in L.components:
            f = Chebfun2(approx=c)
            assert abs(float(np.asarray(f(0.4, -0.3)))) < 1e-11

    def test_lap_alias(self):
        F = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                     lambda x, y: jnp.sin(y))
        a = Chebfun2(approx=F.lap().components[0])
        b = Chebfun2(approx=F.laplacian().components[0])
        assert abs(float(np.asarray(a(0.2, 0.1)))
                   - float(np.asarray(b(0.2, 0.1)))) == 0.0

    def test_shape_property(self):
        F = Chebfun2v.from_functions(lambda x, y: x + 0 * y,
                                     lambda x, y: y + 0 * x)
        assert F.shape == (2, float("inf"), float("inf"))

    def test_isreal_and_complex_construction(self):
        # from_functions must keep purely-imaginary components (bug #35:
        # direct SeparableApprox construction dropped them).
        G = Chebfun2v.from_functions(lambda x, y: x + 0 * y,
                                     lambda x, y: 1j * (y + 0 * x))
        assert not G.isreal()
        v = complex(np.asarray(Chebfun2(approx=G.components[1])(0.3, 0.5)))
        assert abs(v - 0.5j) < 1e-13

    def test_compose_scalar_and_vector(self):
        # g(F) for scalar g and componentwise for vector G.
        F = Chebfun2v.from_functions(lambda x, y: x + 0 * y,
                                     lambda x, y: y + 0 * x)
        from chebfunjax.chebfun2d.chebfun2 import chebfun2
        g = chebfun2(lambda x, y: x * y)
        h = F.compose(g)
        assert abs(float(np.asarray(h(0.3, 0.5))) - 0.15) < 1e-12
        G = Chebfun2v.from_functions(lambda x, y: x - y,
                                     lambda x, y: x + y)
        H = F.compose(G)
        h0 = Chebfun2(approx=H.components[0])
        assert abs(float(np.asarray(h0(0.3, 0.5))) - (-0.2)) < 1e-12

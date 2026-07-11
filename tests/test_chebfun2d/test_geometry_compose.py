"""Composition ops + Spherefun/Diskfun arithmetic (Fable 5 gap fixes)."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.spherefun.spherefun import Spherefun


class TestGeometryCompose:
    def test_chebfun2_named_ops(self):
        f = Chebfun2.from_function(lambda x, y: x * y)
        X, Y = jnp.asarray(0.3), jnp.asarray(-0.4)
        npt.assert_allclose(float(f.exp()(X, Y)), np.exp(-0.12),
                            atol=1e-13)
        npt.assert_allclose(float((f + 2).sqrt()(X, Y)),
                            np.sqrt(1.88), atol=1e-13)

    def test_chebfun3_named_ops(self):
        f = Chebfun3.from_function(lambda x, y, z: x * y * z)
        p = jnp.asarray(0.3), jnp.asarray(-0.4), jnp.asarray(0.5)
        npt.assert_allclose(float(f.cos()(*p)), np.cos(-0.06),
                            atol=1e-13)

    def test_spherefun_arithmetic_and_compose(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Spherefun.from_function(lambda l, t: jnp.cos(t))
            g = f + 2 * f * f
            e = f.exp()
        L, T = jnp.asarray(0.7), jnp.asarray(1.1)
        c = np.cos(1.1)
        npt.assert_allclose(float(g(L, T)), c + 2 * c * c, atol=1e-13)
        npt.assert_allclose(float(e(L, T)), np.exp(c), atol=1e-13)

    def test_diskfun_arithmetic_and_compose(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Diskfun.from_function(lambda t, r: r * jnp.cos(t))
            h = f * f - f + 1.0
            s = f.sin()
        T0, R0 = jnp.asarray(0.6), jnp.asarray(0.7)
        x0 = 0.7 * np.cos(0.6)
        npt.assert_allclose(float(h(T0, R0)), x0 * x0 - x0 + 1,
                            atol=1e-13)
        npt.assert_allclose(float(s(T0, R0)), np.sin(x0), atol=1e-13)

    def test_ballfun_compose(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        r0, l0, t0 = (jnp.asarray(0.6), jnp.asarray(0.7),
                      jnp.asarray(1.1))
        xb = 0.6 * np.cos(0.7) * np.sin(1.1)
        npt.assert_allclose(float(f.exp()(r0, l0, t0)), np.exp(xb),
                            atol=1e-12)
        npt.assert_allclose(float((f + 2).sqrt()(r0, l0, t0)),
                            np.sqrt(xb + 2), atol=1e-12)

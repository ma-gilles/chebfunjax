"""Core unit tests for Spherefunv vector-field methods (Fable 5).

Exercises the audit additions — real/imag/conj and iszero — without a
MATLAB golden reference.

Note: chebfunjax's Spherefun uses a real BMC representation, so a
Spherefun (and hence a Spherefunv) is real-valued.  For a real field
``real(u) == u``, ``conj(u) == u`` and ``imag(u) == 0``, matching MATLAB
for real inputs.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv


def _pts():
    lam = jnp.asarray(np.linspace(-3.0, 3.0, 12))
    th = jnp.asarray(np.linspace(0.1, 3.0, 12))
    return lam, th


class TestSpherefunvComplexParts:
    def test_real_conj_are_identity_imag_zero(self):
        u = Spherefunv.from_functions(
            lambda lam, th: jnp.sin(th) * jnp.cos(lam),
            lambda lam, th: jnp.cos(th))
        lam, th = _pts()
        for method in ("real", "conj"):
            rf, rg = getattr(u, method)()(lam, th)
            uf, ug = u(lam, th)
            np.testing.assert_allclose(np.asarray(rf), np.asarray(uf),
                                       atol=1e-12)
            np.testing.assert_allclose(np.asarray(rg), np.asarray(ug),
                                       atol=1e-12)
        i_f, i_g = u.imag()(lam, th)
        np.testing.assert_allclose(np.asarray(i_f), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(i_g), 0.0, atol=1e-12)

    def test_iszero(self):
        z = Spherefun.from_function(lambda lam, th: 0.0 * lam)
        assert Spherefunv(z, z).iszero()
        u = Spherefunv.from_functions(
            lambda lam, th: jnp.cos(th),
            lambda lam, th: jnp.sin(th) * jnp.cos(lam))
        assert not u.iszero()

"""Port of MATLAB Chebfun tests/chebfun3/test_tucker.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun3.tucker()`` now
returns ``(core, cols, rows, tubes)`` with the mode factors as 1D
Chebfuns.  MATLAB's ``chebfun3.txm`` (tensor times matrix) contraction
is written here as an ``np.einsum``; MATLAB's single-output
``core = tucker(f)`` is ``f.tucker()[0]``.

Provenance
----------
MATLAB source : tests/chebfun3/test_tucker.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


def _quasi(cols, t):
    """MATLAB's C(t, :): the (len(t), r) matrix of factor values."""
    return np.stack([np.asarray(c(jnp.asarray(t))) for c in cols], axis=-1)


def _reconstruct(core, C, R, T, x, y, z):
    """core x_1 C(x) x_2 R(y) x_3 T(z), i.e. three chained txm calls."""
    return np.einsum("ijk,ai,bj,ck->abc", np.asarray(core),
                     _quasi(C, x), _quasi(R, y), _quasi(T, z))


class TestChebfun3Tucker:
    def test_tucker_reconstructs_on_unit_cube(self):
        # pass(1): the Tucker factors reproduce f on a fine grid.
        f = chebfun3(lambda x, y, z: jnp.cos(x * y * z))
        core, C, R, T = f.tucker()
        x = np.linspace(-1.0, 1.0, 24)
        xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
        exact = np.asarray(f(jnp.asarray(xx), jnp.asarray(yy),
                             jnp.asarray(zz)))
        got = _reconstruct(core, C, R, T, x, x, x)
        assert float(np.linalg.norm((got - exact).ravel())) < TOL

    def test_tucker_reconstructs_on_a_box(self):
        # pass(2): the same on a non-cubic box, where each factor lives
        # on its own interval.
        dom = (-1.0, 1.0, 0.0, 2.0, 2.0, 4.0)
        f = chebfun3(lambda x, y, z: jnp.cos(x * y * z), domain=dom)
        core, C, R, T = f.tucker()
        x = np.linspace(dom[0], dom[1], 16)
        y = np.linspace(dom[2], dom[3], 16)
        z = np.linspace(dom[4], dom[5], 16)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        exact = np.asarray(f(jnp.asarray(xx), jnp.asarray(yy),
                             jnp.asarray(zz)))
        got = _reconstruct(core, C, R, T, x, y, z)
        assert float(np.linalg.norm((got - exact).ravel())) < TOL

    def test_core_only(self):
        # pass(3): MATLAB's single-output tucker(f) returns just the core.
        f = chebfun3(lambda x, y, z: jnp.cos(x * y * z))
        core, _, _, _ = f.tucker()
        assert float(np.linalg.norm(
            np.asarray(core).ravel() - np.asarray(f.core).ravel())) < TOL

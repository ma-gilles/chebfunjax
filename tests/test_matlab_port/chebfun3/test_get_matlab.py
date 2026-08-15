"""Port of MATLAB Chebfun tests/chebfun3/test_get.m (Fable 5).

Property access (f.domain, f.core, f.cols, ...) versus tucker().

Provenance
----------
MATLAB source : tests/chebfun3/test_get.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

jax.config.update("jax_enable_x64", True)

TOL = 1e2 * 1e-14


class TestChebfun3Get:
    def test_all_matlab_assertions(self):
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x * y * z))
        core, cols, rows, tubes = f.tucker()

        # pass(1)
        assert np.linalg.norm(
            np.asarray([-1, 1, -1, 1, -1, 1], dtype=float)
            - np.asarray(f.domain, dtype=float)) < TOL
        # pass(2)
        assert float(jnp.max(jnp.abs(
            jnp.ravel(core) - jnp.ravel(f.core)))) < TOL
        # pass(3)-(5): the tucker() factors carry the same data as the
        # stored cols/rows/tubes techs.
        for quasi, techs in ((cols, f.cols), (rows, f.rows),
                             (tubes, f.tubes)):
            assert len(quasi) == len(techs)
            for q, t in zip(quasi, techs):
                xs = jnp.linspace(-0.97, 0.97, 21)
                a, b = (float(q.domain.breakpoints[0]),
                        float(q.domain.breakpoints[-1]))
                ts = (2 * xs - (a + b)) / (b - a)
                assert float(jnp.max(jnp.abs(
                    jnp.asarray(q(xs)) - jnp.asarray(t(ts))))) < TOL

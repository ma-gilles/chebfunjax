"""Port of MATLAB Chebfun tests/chebfun3/test_feval.m (Fable 5).

Ported through the scalar/vector evaluation checks; chebfun-path
evaluation f(c(t)) is skipped (no composition).

Provenance
----------
MATLAB source : tests/chebfun3/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, ninf

TOL = 1e3 * EPS
DOM = (-1.0, 2.0, -np.pi / 2, np.pi, -3.0, 1.0)
P = (np.pi / 6, np.pi / 12, -1.0)


class TestChebfun3Feval:
    @pytest.mark.parametrize("which,exact0,exactP", [
        (0, 0.0, P[0]), (1, 0.0, P[1]), (2, 0.0, P[2])])
    def test_coordinate_functions(self, which, exact0, exactP):
        fns = [lambda x, y, z: x, lambda x, y, z: y, lambda x, y, z: z]
        f = Chebfun3.from_function(fns[which], domain=DOM)
        vs = max(abs(v) for v in DOM)
        z = jnp.asarray(0.0)
        assert abs(float(f(z, z, z)) - exact0) < TOL * vs
        pt = tuple(jnp.asarray(v) for v in P)
        assert abs(float(f(*pt)) - exactP) < TOL * vs

    def test_smooth_function_random_points(self):
        def fa(x, y, z):
            return jnp.cos(x) + jnp.sin(x * y) + jnp.sin(z * x)
        g = Chebfun3.from_function(fa)
        rng = np.random.default_rng(42)
        pts = rng.uniform(-1, 1, (3, 10))
        X, Y, Z = (jnp.asarray(pts[i]) for i in range(3))
        assert ninf(fa(X, Y, Z) - g(X, Y, Z)) < 10 * TOL

    def test_eval_along_chebfun_path(self):
        pytest.skip("Chebfun3 cannot compose with a Chebfun path")

"""Port of MATLAB Chebfun tests/chebfun3/test_fold_unfold.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_fold_unfold.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

jax.config.update("jax_enable_x64", True)


class TestChebfun3Foldunfold:
    def test_all_matlab_assertions(self):
        m, n, p = 5, 4, 6
        rng = np.random.RandomState(0)
        T = jnp.asarray(rng.rand(m, n, p))

        M1 = Chebfun3.unfold(T, 1)
        assert M1.shape == (m, n * p)  # pass(1)
        T_new = Chebfun3.fold(M1, (m, n, p), 1, (2, 3))
        assert float(jnp.linalg.norm(
            jnp.ravel(T_new - T))) == 0.0  # pass(2)

        M2 = Chebfun3.unfold(T, 2)
        assert M2.shape == (n, m * p)  # pass(3)
        T_new = Chebfun3.fold(M2, (m, n, p), 2, (1, 3))
        assert float(jnp.linalg.norm(
            jnp.ravel(T_new - T))) == 0.0  # pass(4)

        M3 = Chebfun3.unfold(T, 3)
        assert M3.shape == (p, m * n)  # pass(5)
        T_new = Chebfun3.fold(M3, (m, n, p), 3, (1, 2))
        assert float(jnp.linalg.norm(
            jnp.ravel(T_new - T))) == 0.0  # pass(6)

        # The unfolding uses MATLAB's column-major ordering: check one
        # entry mapping explicitly for mode 1.
        assert float(M1[2, 1 + n * 3] - T[2, 1, 3]) == 0.0

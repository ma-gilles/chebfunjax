"""Port of MATLAB Chebfun tests/chebfun3/test_construnctorsyntax.m
(Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_construnctorsyntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.utils.matlab_expr import matlab_expression

jax.config.update("jax_enable_x64", True)


class TestChebfun3Construnctorsyntax:
    def test_syntaxes(self):
        def f(x, y, z):
            return jnp.cos(x) + jnp.sin(x * y) + z

        g1 = Chebfun3.from_function(f)
        g2 = Chebfun3.from_function(
            matlab_expression("cos(x) + sin(x.*y) + z",
                              ("x", "y", "z")))
        g3 = Chebfun3.from_function(
            lambda x, y, z: np.cos(x) + np.sin(x * y) + z,
            vectorize=True)
        p = (jnp.asarray(0.3), jnp.asarray(0.4), jnp.asarray(-0.2))
        want = float(np.cos(0.3) + np.sin(0.12) - 0.2)
        for g in (g1, g2, g3):
            assert abs(float(g(*p)) - want) < 1e-12

"""Supplemental controls for the Python Dirichlet Hodge decomposition.

The lowercase helmholtz_decomposition extension returns a scalar potential,
curl-free field and divergence-free remainder. It is not the native
@ballfunv/HelmholtzDecomposition.m API, whose scalar F/P/T outputs are tested
separately in test_HelmholtzDecomposition_matlab.py (Chebfun7574c77).
The formerly cited lowercase native test file does not exist.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.ballfun.ballfunv import Ballfunv

R0, L0, T0 = jnp.asarray(0.55), jnp.asarray(0.8), jnp.asarray(1.2)


class TestBallfunvHelmholtzDecomposition:
    def test_parts_sum_to_field(self):
        F = Ballfunv.from_functions(lambda x, y, z: x + y * z,
                                    lambda x, y, z: y - x * z,
                                    lambda x, y, z: z * z)
        phi, cf, df = F.helmholtz_decomposition()
        for orig, a, b in zip(F.components, cf.components,
                              df.components):
            got = float(a(R0, L0, T0)) + float(b(R0, L0, T0))
            want = float(orig(R0, L0, T0))
            assert abs(got - want) < 1e-6
        # div-free part has (near-)zero divergence
        assert abs(float(df.div()(R0, L0, T0))) < 1e-5

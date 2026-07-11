"""Port of MATLAB Chebfun tests/ballfunv/test_helmholtz_decomposition.m
(Fable 5).

F = curl-free + div-free; the curl-free part is grad(poisson(div F)).

Provenance
----------
MATLAB source : tests/ballfunv/test_helmholtz_decomposition.m
Chebfun commit: 7574c77
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

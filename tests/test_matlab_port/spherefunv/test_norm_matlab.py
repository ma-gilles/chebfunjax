"""Spherefunv norm contract: MATLAB method plus pointwise magnitude extension.

Provenance
----------
MATLAB source : @spherefunv/norm.m, tests/spherefunv/test_empty.m
Chebfun commit: 7574c77

The pinned MATLAB tree has no tests/spherefunv/test_norm.m. The source method
returns a scalar global Frobenius norm; the pointwise field is a separate
chebfunjax extension named ``magnitude``.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

L0, T0 = jnp.asarray(0.7), jnp.asarray(1.1)


class TestSpherefunvNorm:
    def test_global_frobenius_norm(self):
        # MATLAB source: sqrt(sum_j norm(F_j, 2)^2). Each constant c on the
        # unit sphere has L2 norm |c|*sqrt(4*pi), giving 13*sqrt(4*pi).
        def constant(c):
            return Spherefun.from_function(
                lambda lam, th: c * jnp.ones_like(lam))

        F = Spherefunv(constant(3.0), constant(4.0), constant(12.0))
        n = F.norm()
        expected = 13.0 * (4.0 * float(jnp.pi)) ** 0.5
        # The source formula performs three component norm evaluations and a
        # short sum/square-root reduction. Allow 256 binary64 epsilons at the
        # expected scale for representation and reduction roundoff; this is a
        # derived control bound, not a MATLAB source-test tolerance.
        bound = (256.0 * float(jnp.finfo(jnp.float64).eps)
                 * max(1.0, abs(expected)))
        assert jnp.ndim(n) == 0
        assert abs(float(n) - expected) <= bound

    def test_empty_norm_matches_source_empty_value(self):
        # @spherefunv/norm.m returns [] for empty input; the Python empty array
        # represents that empty MATLAB value.
        n = Spherefunv.empty().norm()
        assert n.shape == (0,)

    def test_pointwise_magnitude_extension(self):
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        g = Spherefun.from_function(lambda lam, th: jnp.sin(lam)
                                    * jnp.sin(th))
        F = Spherefunv(f, g)
        n = F.magnitude()
        exact = (float(f(L0, T0)) ** 2 + float(g(L0, T0)) ** 2) ** 0.5
        got = float(n(L0, T0))
        assert abs(got - exact) < 1e-9

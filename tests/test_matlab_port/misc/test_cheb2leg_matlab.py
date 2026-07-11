"""Port of MATLAB Chebfun tests/misc/test_cheb2leg.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_cheb2leg.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.transforms import cheb2leg, leg2cheb

TOL = 5e-12


class TestCheb2leg:
    def test_leading_coefficient_only(self):
        N = 20
        c_cheb = jnp.zeros(N + 1, dtype=jnp.float64).at[0].set(1.0)
        # MATLAB stores coeffs highest-first in this old test; the
        # library convention here is lowest-first: T_0 == P_0.
        c_leg = cheb2leg(c_cheb)
        assert float(jnp.max(jnp.abs(c_leg - c_cheb))) < TOL

    def test_decaying_coefficients_reference(self):
        # MATLAB: c_cheb = 1./(N:-1:1)'.^2; c_cheb(2:2:end) negated
        # (low-degree-first vector, entries (-1)^k/(N-k)^2).
        N = 20
        c = 1.0 / np.arange(N, 0, -1, dtype=float) ** 2
        c[1::2] *= -1
        c_leg = np.asarray(cheb2leg(jnp.asarray(c)))
        # MATLAB reference: c_leg(2) = 0.011460983274163
        assert abs(c_leg[1] - 0.011460983274163) / 0.011460983274163 < TOL

    def test_roundtrip(self):
        rng = np.random.default_rng(3)
        c = jnp.asarray(rng.standard_normal(40) / (np.arange(1, 41) ** 2))
        back = leg2cheb(cheb2leg(c))
        assert float(jnp.max(jnp.abs(back - c))) < TOL

    def test_large_n_leading(self):
        N = 1000
        c_cheb = jnp.zeros(N + 1, dtype=jnp.float64).at[0].set(1.0)
        c_leg = cheb2leg(c_cheb)
        assert float(jnp.max(jnp.abs(c_leg - c_cheb))) < TOL

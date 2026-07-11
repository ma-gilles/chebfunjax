"""Port of MATLAB Chebfun tests/misc/test_coeffs2vals.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_coeffs2vals.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.chebtech import _coeffs_to_values, _values_to_coeffs
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)


class TestCoeffs2vals:
    def test_roundtrip(self):
        rng = np.random.default_rng(1)
        c = jnp.asarray(rng.standard_normal(17))
        back = _values_to_coeffs(_coeffs_to_values(c))
        assert float(jnp.max(jnp.abs(back - c))) < 100 * EPS

    def test_single_mode_is_chebyshev_poly(self):
        n = 12
        x = np.asarray(chebpts(n, kind=2))
        for k in [0, 3, 7]:
            c = jnp.zeros(n, dtype=jnp.float64).at[k].set(1.0)
            v = np.asarray(_coeffs_to_values(c))
            exact = np.cos(k * np.arccos(np.clip(x, -1, 1)))
            assert float(np.max(np.abs(v - exact))) < 100 * EPS

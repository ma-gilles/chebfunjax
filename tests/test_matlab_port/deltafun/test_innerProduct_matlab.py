"""Port of MATLAB Chebfun tests/deltafun/test_innerProduct.m (Fable 5).

FIXED: Deltafun.innerProduct added in the Fable 5 audit
(distributional pairing <f + sum m_k delta_k, g>).

Provenance
----------
MATLAB source : tests/deltafun/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

D = Domain((-1.0, 1.0))


class TestDeltafunInnerProduct:
    def test_delta_sifts(self):
        # <delta_{x0}, g> = g(x0)
        zero = Bndfun.from_function(lambda x: jnp.zeros_like(x), D)
        d = Deltafun.from_fun_and_deltas(zero, jnp.asarray([0.3]),
                                         jnp.asarray([[1.0]]))
        g = Bndfun.from_function(jnp.cos, D)
        assert abs(float(d.innerProduct(g)) - np.cos(0.3)) < 1e-13

    def test_smooth_plus_delta(self):
        f = Deltafun.from_fun_and_deltas(
            Bndfun.from_function(jnp.sin, D), jnp.asarray([0.25]),
            jnp.asarray([[2.0]]))
        g = Bndfun.from_function(jnp.cos, D)
        # <sin, cos> = 0 (odd*even) + 2 cos(0.25)
        assert abs(float(f.innerProduct(g)) - 2 * np.cos(0.25)) < 1e-13

    def test_overlapping_deltas_rejected(self):
        zero = Bndfun.from_function(lambda x: jnp.zeros_like(x), D)
        d1 = Deltafun.from_fun_and_deltas(zero, jnp.asarray([0.3]),
                                          jnp.asarray([[1.0]]))
        with pytest.raises(ValueError):
            d1.innerProduct(d1)

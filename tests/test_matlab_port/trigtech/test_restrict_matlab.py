"""Port of MATLAB Chebfun tests/trigtech/test_restrict.m (Fable 5).

FIXED: Trigtech.restrict added in the Fable 5 audit (Chebyshev output,
as MATLAB's restriction of a periodic function is not periodic).

Provenance
----------
MATLAB source : tests/trigtech/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


class TestTrigtechRestrict:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty trigtech construction "
                    "via make()")

    def test_bad_interval_raises(self):
        f = Trigtech.from_function(lambda x: jnp.sin(2 * jnp.pi * x))
        with pytest.raises(ValueError):
            f.restrict(-1.0, 3.0)

    def test_values_on_subinterval(self):
        f = Trigtech.from_function(lambda x: jnp.sin(2 * jnp.pi * x))
        r = f.restrict(-0.5, 0.25)
        ts = jnp.asarray(np.linspace(-0.99, 0.99, 40))
        x = -0.5 + 0.75 * (np.asarray(ts) + 1) / 2
        err = jnp.abs(r(ts) - jnp.sin(2 * jnp.pi * jnp.asarray(x)))
        assert float(jnp.max(err)) < 100 * f.vscale * EPS

    def test_array_valued(self):
        pytest.skip("chebfunjax has no array-valued trigtech")

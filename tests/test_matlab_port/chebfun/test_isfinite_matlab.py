"""Port of MATLAB Chebfun tests/chebfun/test_isfinite.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_isfinite.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj


class TestChebfunIsfinite:
    def test_smooth_is_finite(self):
        f = cj.chebfun(jnp.sin)
        assert not bool(f.isinf())
        assert not bool(f.isnan())

    def test_blowup_cases(self):
        pytest.skip("chebfun-level blowup (exps) not implemented; "
                    "singular funs tested in singfun ports")

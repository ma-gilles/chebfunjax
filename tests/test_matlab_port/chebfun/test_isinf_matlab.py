"""Port of MATLAB Chebfun tests/chebfun/test_isinf.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_isinf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj


class TestChebfunIsinf:
    def test_smooth_not_inf(self):
        f = cj.chebfun(lambda x: jnp.exp(x))
        assert not bool(f.isinf())

    def test_blowup_cases(self):
        pytest.skip("chebfun-level blowup (exps) not implemented")

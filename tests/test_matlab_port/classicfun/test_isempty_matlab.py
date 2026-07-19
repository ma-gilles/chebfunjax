"""Port of MATLAB Chebfun tests/classicfun/test_isempty.m (Fable 5).

The empty Bndfun (MATLAB ``bndfun()``) is ``isempty``; a constructed one is
not.  The array-valued and quasimatrix cases and the UNBNDFUN cases have no
scalar-valued analogue here.

Provenance
----------
MATLAB source : tests/classicfun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

DOM = Domain((-2.0, 7.0))


class TestClassicfunIsempty:
    def test_empty_is_empty(self):
        # pass(1): isempty(bndfun())
        assert Bndfun.empty().isempty()

    def test_constructed_not_empty(self):
        # pass(2): ~isempty(bndfun(@sin))
        assert not Bndfun.from_function(jnp.sin, DOM).isempty()

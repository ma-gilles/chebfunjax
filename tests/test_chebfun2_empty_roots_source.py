"""Empty-root dispatch from the original superclass.

Provenance
----------
MATLAB source: @separableApprox/roots.m, lines 33-37
Chebfun commit: 7574c77
"""
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


def test_empty_scalar_roots():
    assert len(Chebfun2.empty().roots()) == 0


@pytest.mark.parametrize("method", [None, "ms", "resultant", "invalid"])
def test_empty_common_roots_before_method_dispatch(method):
    # Source exits before examining a second operand or optional method.
    empty = Chebfun2.empty()
    assert len(empty.roots(empty, method=method)) == 0

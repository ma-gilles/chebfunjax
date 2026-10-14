"""Supplemental source controls; no native test_size.m exists.

Provenance
----------
MATLAB source : @chebfun3/size.m
Chebfun commit: 7574c77
"""

import math

import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


@pytest.mark.parametrize("empty", [False, True])
def test_continuous_size_and_source_dimensions(empty):
    f = Chebfun3.empty() if empty else Chebfun3.from_function(lambda x, y, z: x)
    assert f.size() == (math.inf, math.inf, math.inf)
    m, n, p = f.size()
    assert m == n == p == math.inf
    for dim in (1, 2, 3):
        assert f.size(dim) == math.inf
    for dim in (0, 4, -1, 1.5):
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:size:outputs"):
            f.size(dim)

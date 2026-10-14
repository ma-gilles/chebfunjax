"""Literal source squeeze edge branches and nonsymmetric orientation.

Provenance
----------
MATLAB source : @chebfun3/squeeze.m, @chebfun2/outerProduct.m,
    @chebfun2/constructor.m
Chebfun commit: 7574c77
"""

import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import Chebfun3

DOMAIN = (0., 1., 2., 3., 4., 6.)


def test_empty_zero_and_constant_source_priority():
    out = Chebfun3.empty().squeeze()
    assert isinstance(out, Chebfun) and out.isempty()
    zero = Chebfun3.from_function(lambda x, y, z: 0*x, domain=DOMAIN)
    assert zero.rank == (0, 0, 0)
    assert zero.squeeze() is zero
    constant = Chebfun3.from_function(lambda x, y, z: 2+0*x, domain=DOMAIN).squeeze()
    assert isinstance(constant, Chebfun)
    assert tuple(constant.domain.breakpoints) == (4., 6.)
    assert abs(float(constant(5.))-2.) < 1e-13


def test_rank_one_nonconstant_branch_preserves_source_error():
    f = Chebfun3.from_function(lambda x, y, z: y*z, domain=DOMAIN)
    assert f.rank == (1, 1, 1)
    with pytest.raises(ValueError, match="MATLAB:unassignedOutputs"):
        f.squeeze()


def test_nonsymmetric_outer_product_keeps_source_domain_order():
    f = Chebfun3.from_function(lambda x, y, z: y+2*z, domain=DOMAIN)
    g = f.squeeze()
    assert isinstance(g, Chebfun2)
    assert tuple(g.domain) == (4., 6., 2., 3.)
    assert abs(float(g(5., 2.5))-12.5) < 1e-13


def test_small_real_dependency_is_not_dropped_by_sampling():
    f = Chebfun3.from_function(lambda x, y, z: x+1e-12*y+z, domain=DOMAIN)
    assert f.rank == (2, 2, 2)
    assert f.squeeze() is f

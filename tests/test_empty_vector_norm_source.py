"""Native empty vector norms, Chebfun commit 7574c77.

Provenance: @chebfun2v/norm.m, @chebfun3v/norm.m, @diskfunv/norm.m.
Each returns [] before computing component norms. Python maps native empty
numeric arrays to shape (0,), consistent with the existing diskfunv contract.
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
from chebfunjax.diskfun.diskfunv import Diskfunv


@pytest.mark.parametrize("factory", [Chebfun2v.empty, Chebfun3v, Diskfunv.empty])
def test_native_empty_norm_returns_empty_numeric_array(factory):
    result = factory().norm()
    assert isinstance(result, jax.Array)
    assert result.shape == (0,)
    assert result.dtype == jnp.float64

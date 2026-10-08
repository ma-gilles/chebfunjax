"""All six source predicates for real and complex Chebfun3 norms.

Provenance
----------
MATLAB source : tests/chebfun3/test_norm.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


@pytest.fixture(scope="module", params=[False, True], ids=["real", "complex"])
def source_function(request):
    factor = 1j if request.param else 1.
    return Chebfun3.from_function(lambda x, y, z: factor*x)


@pytest.mark.parametrize("p,exact", [(None, 2*jnp.sqrt(2/3)),
                                     ("inf", 1.), (4, (8/5)**.25)],
                         ids=["fro", "inf", "four"])
def test_original_norm_predicate(source_function, p, exact):
    # Use the source optimization-toolbox bound for both real and complex;
    # do not relax complex infinity norm to its toolbox-absent fallback .004.
    value = source_function.norm() if p is None else source_function.norm(p)
    assert jnp.abs(value-exact) < 1000*jnp.finfo(jnp.float64).eps

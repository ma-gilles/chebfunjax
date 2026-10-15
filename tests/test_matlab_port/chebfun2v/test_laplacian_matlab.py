"""Original component norms and 50*eps bounds for the vector Laplacian.

The native file evaluates two predicates, then overwrites slots1/2 in its
three-component case. Both groups are retained as independent checks here.

Provenance
----------
MATLAB source: tests/chebfun2v/test_laplacian.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

TOL = 50 * ChebfunPref().cheb2Prefs.chebfun2eps


@pytest.mark.parametrize("count", [2, 3])
def test_original_component_laplacians(count):
    functions = [lambda x,y: jnp.cos(x), lambda x,y: jnp.sin(y),
                 lambda x,y: x*y+y**2]
    field = Chebfun2v.from_functions(*functions[:count])
    actual = field.laplacian()
    for component, result in zip(field.components, actual.components):
        f = Chebfun2(approx=component)
        reference = f.diff(dim=1, k=2)+f.diff(dim=2, k=2)
        assert float((reference-Chebfun2(approx=result)).norm()) < TOL

"""Additional actual-empty output contracts derived from native source.

Provenance
----------
MATLAB source : @chebfun3/{mean,mean3,std3,sum,sum2,sum3,norm,integral,
    minandmax3,max3,min3,tucker,diff,power,permute}.m
Chebfun commit: 7574c77
The min3 indexing error is source-derived, not a captured MATLAB error text.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


@pytest.mark.parametrize('method', ['sum', 'sum2', 'sum3', 'integral', 'norm', 'mean3', 'std3'])
def test_numeric_empty_reductions(method):
    result = getattr(Chebfun3.empty(), method)()
    assert result.shape == (0,)
    assert result.dtype == jnp.float64


@pytest.mark.parametrize('method,count', [('minandmax3', 2), ('max3', 2), ('tucker', 4)])
def test_empty_output_arity_adapter(method, count):
    result = getattr(Chebfun3.empty(), method)()
    assert isinstance(result, tuple) and len(result) == count
    for value in result:
        assert value.shape == (0,)
        assert value.dtype == jnp.float64


@pytest.mark.parametrize('method', ['mean', 'diff', 'permute'])
def test_object_empty_routes(method):
    result = getattr(Chebfun3.empty(), method)()
    assert isinstance(result, Chebfun3) and result.isempty()


@pytest.mark.parametrize('method,args', [('sum', (99,)), ('sum2', ((99, 100),)),
                                        ('mean', (99,)), ('diff', (99,))])
def test_empty_precedes_dimension_use(method, args):
    result = getattr(Chebfun3.empty(), method)(*args)
    if method in ('sum', 'sum2'):
        assert result.shape == (0,)
    else:
        assert result.isempty()


def test_min3_preserves_source_empty_indexing_error():
    with pytest.raises(IndexError):
        Chebfun3.empty().min3()


def test_empty_power_operand_routes():
    f = Chebfun3.empty()
    for result in (f**2, 2**f, f**f, f**f+f):
        assert isinstance(result, Chebfun3) and result.isempty()

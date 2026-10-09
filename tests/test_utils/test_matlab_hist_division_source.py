"""Independent division and overflow source-operation controls."""
import math
import struct

import jax.numpy as jnp
import pytest

from chebfunjax.utils.matlab_hist import _source_divide, _source_linspace


def words(values):
    return [struct.unpack('Q', struct.pack('d', float(x)))[0] for x in values]


@pytest.mark.parametrize('shape', [(), (1,), (3, 4)])
@pytest.mark.parametrize('divisor', [36, 36., jnp.asarray(36, dtype=jnp.int64)])
def test_scalar_division_shape_dtype_and_source_words(shape, divisor):
    count=math.prod(shape) if shape else 1
    values=[(17+k)*(1.91-(-.73)) for k in range(count)]
    source=jnp.asarray(values,dtype=jnp.float64).reshape(shape)
    actual=_source_divide(source,divisor)
    assert actual.shape==shape and actual.dtype==jnp.float64
    assert words(actual.reshape(-1).tolist())==words([x/36 for x in values])


@pytest.mark.parametrize('dtype', [jnp.int32,jnp.int64,jnp.float32,jnp.float64])
def test_input_conversion_matches_source_double_histogram(dtype):
    source=jnp.asarray([-17,0,19,36],dtype=dtype)
    actual=_source_divide(source,36)
    assert actual.dtype==jnp.float64
    assert words(actual.tolist())==words([-17/36,0.,19/36,1.])


def test_private_divisor_contract_is_scalar():
    with pytest.raises(ValueError,match='scalar divisor'):
        _source_divide(jnp.ones(3),jnp.ones(3))


@pytest.mark.parametrize('left,right,count,branch', [(-1.2e308,1.1e308,7,'span_overflow'),
                                                     (1.e308,1.1e308,37,'product_overflow')])
def test_independent_overflow_grid_words(left,right,count,branch):
    delta=right-left
    if branch=='span_overflow':
        assert math.isinf(delta)
        expected=[left+(right/count)*k-(left/count)*k for k in range(count+1)]
    else:
        assert math.isfinite(delta) and math.isinf(delta*(count-1))
        expected=[left+k*(delta/count) for k in range(count+1)]
    expected[0],expected[-1]=left,right
    actual=_source_linspace(jnp.asarray(left),jnp.asarray(right),count)
    assert words(actual.tolist())==words(expected)
    assert all(math.isfinite(x) for x in actual.tolist())

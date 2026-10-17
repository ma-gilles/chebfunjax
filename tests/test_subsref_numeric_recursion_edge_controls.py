"""Authored controls for record-only numeric result adapters.

No original MATLAB predicates or captured native cases are added here.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.fixture(scope='module')
def f():
    return cj.chebfun(lambda x: jnp.stack((x, x*x), axis=-1), domain=[-2., 3.])


def test_unsupported_python_index_without_dtype(f):
    # NumPy namespace objects are not necessarily numeric arrays. Inspecting
    # their metadata must reach the existing unsupported-index category.
    with pytest.raises(ValueError) as caught:
        f.subsref([{'type': '()', 'subs': [jnp.array([[.25], [.5]])]},
                   {'type': '()', 'subs': [np.sin]}])
    assert caught.value.identifier == 'MATLAB:matrix:unableToUseTypeAsIndex'


@pytest.mark.parametrize('prop', ['lval', 'rval'])
def test_row_endpoint_get_numeric_records_preserve_source_column(f, prop):
    # Native GET calls FEVAL(f, scalar), whose row-Chebfun result is a column.
    # Preserve public Python GET's current rank-one return convention.
    row = f.T
    before = row.get(prop)
    assert before.shape == (2,)
    actual = row.subsref([{'type': '.', 'subs': prop},
                         {'type': '()', 'subs': [[2, 1]]}])
    assert actual.shape == (2, 1)
    np.testing.assert_array_equal(actual, jnp.asarray(before)[jnp.array([1, 0])].reshape(2, 1))
    np.testing.assert_array_equal(row.get(prop), before)

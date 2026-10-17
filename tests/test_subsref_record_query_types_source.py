"""Source ISNUMERIC query dispatch; authored controls, not native captures.

@chebfun/subsref.m (7574c77): selector errors precede ISNUMERIC (line96),
registered composition/colon follow, then nonnumeric (line130). Python numeric
lists model numeric arrays; object arrays model cells for this adapter.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

_ID = 'CHEBFUN:CHEBFUN:subsref:nonnumeric'


class ArrayConversionOnly:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Foreign array conversion must not be invoked.')


class JaxConversionOnly:
    def __jax_array__(self):
        raise AssertionError('Foreign JAX conversion must not be invoked.')


@pytest.fixture(scope='module')
def f():
    return cj.chebfun(lambda x: jnp.stack((x, x*x), axis=-1))


_REJECT = {
    'python_true': lambda: True,
    'python_false': lambda: False,
    'numpy_scalar_logical': lambda: np.bool_(True),
    'jax_scalar_logical': lambda: jnp.asarray(True),
    'numpy_logical_row': lambda: np.array([[True, False]]),
    'jax_logical_column': lambda: jnp.array([[True], [False]]),
    'logical_matrix': lambda: jnp.array([[True, False], [False, True]]),
    'empty_logical': lambda: jnp.empty((0, 2), dtype=bool),
    'logical_list': lambda: [True, False],
    'logical_nested_list': lambda: [[True], [False]],
    'text': lambda: 'left',
    'numpy_text': lambda: np.array(['left', 'right']),
    'bytes': lambda: b':',
    'object_cell': lambda: np.array([0.], dtype=object),
    'dict': lambda: {'value': 0.},
    'callable': lambda: jnp.sin,
    'none': lambda: None,
    'array_conversion': ArrayConversionOnly,
    'jax_conversion': JaxConversionOnly,
    'foreign_sequence': lambda: [ArrayConversionOnly()],
}


@pytest.mark.parametrize('case', list(_REJECT))
def test_source_nonnumeric_query_id(f, case):
    with pytest.raises(ValueError) as caught:
        f.subsref({'type': '()', 'subs': [_REJECT[case]()]})
    assert caught.value.identifier == _ID


@pytest.mark.parametrize('case', ['python_true', 'object_cell', 'array_conversion'])
def test_selector_error_precedes_query_type(f, case):
    with pytest.raises(ValueError) as caught:
        f.subsref({'type': '()', 'subs': [_REJECT[case](), 3]})
    assert caught.value.identifier == 'CHEBFUN:CHEBFUN:subsref:badsubscript'


_ACCEPT = {
    'python_integer': lambda: 1,
    'python_real': lambda: .25,
    'python_complex': lambda: .25 + .125j,
    'numpy_integer': lambda: np.int32(0),
    'numpy_unsigned': lambda: np.uint32(1),
    'numpy_real': lambda: np.float64(.25),
    'numpy_complex': lambda: np.complex128(.25 + .125j),
    'numpy_array': lambda: np.array([[.25], [.5]]),
    'jax_integer': lambda: jnp.array([[0], [1]], dtype=jnp.int32),
    'jax_unsigned': lambda: jnp.array([[0], [1]], dtype=jnp.uint32),
    'jax_real': lambda: jnp.array([[.25], [.5]]),
    'jax_complex': lambda: jnp.array([[.25 + .125j], [.5 - .125j]]),
    'numeric_list': lambda: [.25, .5],
    'numeric_nested_list': lambda: [[.25], [.5]],
    'mixed_numeric_logical_adapter': lambda: [True, 0.],
    'empty_numeric': lambda: [],
}


@pytest.mark.parametrize('case', list(_ACCEPT))
def test_source_numeric_query_evaluates_actual_full_feval(f, case):
    query = _ACCEPT[case]()
    canonical = jnp.asarray(query)
    if canonical.ndim == 0:
        canonical = canonical.reshape(1, 1)
    elif canonical.ndim == 1:
        canonical = canonical.reshape(1, -1)
    actual = f.subsref({'type': '()', 'subs': [query]})
    expected = cj.feval(f, canonical)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('dtype', [jnp.int32, jnp.float64, jnp.complex128])
def test_traced_numeric_query_dispatch(f, dtype):
    x = jnp.array([[0], [1]], dtype=dtype)
    evaluate = jax.jit(lambda q: f.subsref({'type': '()', 'subs': [q]}))
    np.testing.assert_array_equal(evaluate(x), cj.feval(f, x))


def test_dynamic_real_query_derivative(f):
    def evaluate(q):
        return f.subsref({'type': '()', 'subs': [q]})
    x = jnp.array([[.25], [.5]])
    np.testing.assert_array_equal(jax.jacfwd(evaluate)(x), jax.jacfwd(lambda q: cj.feval(f, q))(x))


def test_traced_logical_query_rejects_source_id(f):
    evaluate = jax.jit(lambda q: f.subsref({'type': '()', 'subs': [q]}))
    with pytest.raises(ValueError) as caught:
        evaluate(jnp.array([[True], [False]]))
    assert caught.value.identifier == _ID


@pytest.mark.parametrize('query', [True, False, jnp.array([True, False])])
def test_direct_python_logical_evaluation_unchanged(f, query):
    np.testing.assert_array_equal(f(query), cj.feval(f, query))


def test_colon_and_registered_function_query_unchanged(f):
    assert f.subsref({'type': '()', 'subs': [':']}).isequal(f)
    g = cj.chebfun(lambda x: x/2)
    actual = f.subsref({'type': '()', 'subs': [g]})
    expected = f(g)
    x = jnp.linspace(-1., 1., 31)
    np.testing.assert_array_equal(actual(x), expected(x))

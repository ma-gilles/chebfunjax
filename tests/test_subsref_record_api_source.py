"""Additional public-record adapter controls; not original MATLAB clauses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize('row', [False, True])
def test_numeric_records_one_based_columns_and_orientation(row):
    f = cj.chebfun(lambda x: jnp.stack((jnp.sin(x), jnp.cos(x)), axis=-1))
    x = jnp.asarray([[-.4, .3], [.1, .7]])
    obj = f.T if row else f
    subs = [[2], x] if row else [x, [2]]
    actual = obj.subsref({'type': '()', 'subs': subs})
    expected = np.asarray(f(x, columns=[1]))
    if row:
        expected = expected.T
    np.testing.assert_array_equal(actual, expected)
    colon = [[2], ':'] if row else [':', [2]]
    selected = obj.subsref({'type': '()', 'subs': colon})
    assert selected.isequal(f[:, [1]].T if row else f[:, [1]])
    assert obj(':', [1]).isequal(selected)
    assert obj(':', columns=[1]).isequal(selected)


@pytest.mark.parametrize('text', ['X', '???'])
def test_direct_subsref_and_explicit_feval_distinct_identifiers(text):
    f = cj.chebfun(lambda x: x*x)
    for call, identifier in (
        (lambda: f(text), 'CHEBFUN:CHEBFUN:subsref:nonnumeric'),
        (lambda: f.subsref({'type': '()', 'subs': [text]}),
         'CHEBFUN:CHEBFUN:subsref:nonnumeric'),
        (lambda: cj.feval(f, text), 'CHEBFUN:CHEBFUN:feval:strInput'),
    ):
        with pytest.raises(ValueError) as caught:
            call()
        assert caught.value.identifier == identifier
    assert float(f('left')) == float(cj.feval(f, 'left'))


def test_brace_record_follows_native_nonrecursive_branch():
    # @chebfun/subsref.m recurses only in () and . branches. The {} branch
    # returns its own result, including when handed further index records.
    f = cj.chebfun(lambda x: x*x)
    records = [{'type': '{}', 'subs': [':']}, {'type': '[]', 'subs': []}]
    assert f.subsref(records).isequal(f)


def test_traced_record_transpose_stays_on_jax_device():
    f = cj.chebfun(lambda x: jnp.stack((x, x*x), axis=-1)).T
    def evaluate(x):
        return f.subsref({'type': '()', 'subs': [x]})
    x = jnp.asarray([[-.4, .3], [.1, .7]])
    actual = jax.jit(evaluate)(x)
    assert isinstance(actual, jax.Array)
    np.testing.assert_array_equal(actual, f(x))
    derivative = jax.jacfwd(evaluate)(x)
    np.testing.assert_allclose(derivative, jax.jacfwd(lambda t: f(t))(x),
                               rtol=0, atol=0)

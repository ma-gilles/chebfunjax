"""Source public parse/vector/metadata traces; Chebfun7574c77."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d._construction import (
    OMITTED,
    endpoint_limit,
    prepare_preferences,
    values_at_breakpoints,
    vector_check,
)
from chebfunjax.chebpref import ChebfunPref


def test_near_endpoint_order_and_no_later_shape_probes():
    calls = []

    def op(x):
        calls.append(x)
        return x*x

    wrapped = vector_check(op, (2., 6.))
    assert len(calls) == 1
    assert jnp.array_equal(calls[0], jnp.asarray([2., 6.])+jnp.asarray([1., -1.])*4/200)
    x = jnp.asarray([2.5, 3., 5.])
    assert jnp.array_equal(wrapped(x), x*x) and len(calls) == 2


@pytest.mark.parametrize('columns', [1, 2, 3])
def test_constant_scalar_or_explicit_row_expansion(columns):
    calls = []

    def op(x):
        calls.append(jnp.shape(x))
        return 7. if columns == 1 else jnp.arange(columns)[None, :]

    wrapped = vector_check(op, (-1., 1.))
    assert calls == ([(2,), ()] if columns == 2 else [(2,)])
    result = wrapped(jnp.ones(4))
    expected = jnp.full(4, 7.) if columns == 1 else jnp.tile(jnp.arange(columns), (4, 1))
    assert jnp.array_equal(result, expected)


def test_square_transpose_requires_source_scalar_probe():
    calls = []

    def op(x):
        calls.append(jnp.shape(x))
        return jnp.stack((jnp.atleast_1d(x), 2*jnp.atleast_1d(x)))

    with pytest.warns(UserWarning, match='vectorCheck:transpose'):
        wrapped = vector_check(op, (-1., 1.))
    assert calls == [(2,), ()]
    x = jnp.asarray([.1, .2, .3])
    assert jnp.array_equal(wrapped(x), jnp.stack((x, 2*x), axis=1))


@pytest.mark.parametrize('array', [False, True])
def test_source_retry_scalar_and_array_wrapper_call_order(array):
    calls = []

    def op(x):
        calls.append(jnp.shape(x))
        if jnp.ndim(x):
            raise ValueError('scalar only')
        return jnp.asarray([x, 2*x]) if array else x

    wrapped = vector_check(op, (-1., 1.))
    # The two-column vec output at the two endpoints is square. Native
    # vectorCheck then calls the array wrapper once on a scalar; that adds
    # its column-count probe and one row call (two further scalar calls).
    assert calls == ([(2,), (), (), (), (), (), ()] if array else [(2,), (), (), ()])
    calls.clear()
    result = wrapped(jnp.asarray([.1, .2, .3]))
    assert calls == ([(), (), (), ()] if array else [(), (), ()])
    assert result.shape == ((3, 2) if array else (3,))


def test_final_callback_exception_propagates_without_later_fallback():
    calls = []

    def op(x):
        calls.append(jnp.shape(x))
        raise RuntimeError('source sentinel')

    with pytest.raises(RuntimeError, match='source sentinel'):
        vector_check(op, (-1., 1.))
    assert calls == [(2,), ()]


def test_explicit_vectorize_probes_scalar_before_endpoints():
    calls = []

    def op(x):
        calls.append(jnp.shape(x))
        return x

    vector_check(op, (-1., 1.), True)
    assert calls == [(), (), ()]


@pytest.mark.parametrize('domain', [(1., 1.+2.**-40), (-jnp.inf, jnp.inf)])
def test_source_probe_arithmetic_for_narrow_and_infinite_domain(domain):
    calls = []

    def op(x):
        calls.append(x)
        return jnp.ones_like(x)

    vector_check(op, domain)
    ends = jnp.asarray(domain)
    expected = ends+jnp.asarray([1., -1.])*jnp.diff(ends)/200
    assert len(calls) == 1
    assert jnp.array_equal(calls[0], expected, equal_nan=True)


def test_private_preferences_exact_default_and_periodic_order():
    original = ChebfunPref({'tech': 'trigtech', 'maxLength': 65537, 'domain': [2., 8.]})
    pref, domain, data, explicit = prepare_preferences(OMITTED, original)
    assert pref.maxLength == 65537 and domain == (2., 8.) and not explicit
    assert data['hscale'] == 8
    pref.maxLength = 33
    assert original.maxLength == 65537
    with pytest.raises(ValueError, match='doubleLengthSplitting'):
        prepare_preferences(pref={'splitting': True},
                            keywords={'trig': True, 'doubleLength': True})
    pref, _, _, explicit = prepare_preferences(pref={'splitting': True}, keywords={'trig': True})
    assert explicit and not pref.splitting and not pref.enableFunqui


def test_explicit_default_domain_and_full_infinite_hscale():
    pref = ChebfunPref({'domain': [2., 8.]})
    assert prepare_preferences((-1., 1.), pref, operand_domain=(3., 7.))[1] == (-1., 1.)
    assert prepare_preferences(OMITTED, pref, operand_domain=(3., 7.))[1] == (3., 7.)
    assert prepare_preferences((-jnp.inf, 2., jnp.inf), pref)[2]['hscale'] == 1


class _Tech:
    coeffs = jnp.ones((1, 2))

    def __init__(self, value, seen):
        self.value, self.seen = value, seen

    def isempty(self):
        return False

    def __call__(self, x):
        self.seen.append(float(x))
        return jnp.asarray(self.value)


class _Fun:
    def __init__(self, value, seen):
        self.tech = _Tech(value, seen)

    def __call__(self, x):
        raise AssertionError('Source endpoint limits must bypass physical evaluation')


def test_endpoint_operator_first_inf_preserved_nan_components_only():
    seen = []
    funs = [_Fun([2., 4.], seen), _Fun([6., 8.], seen)]

    def op(x):
        assert seen == []
        seen.append('op')
        return jnp.asarray([[jnp.inf, 9.], [jnp.nan, 10.], [11., jnp.nan]])

    result = values_at_breakpoints(funs, (-1., 0., 1.), op)
    assert jnp.array_equal(result, jnp.asarray([[jnp.inf, 9.], [4., 10.], [11., 8.]]))
    assert seen == ['op', 1., -1., 1.]


def test_endpoint_operator_exception_not_swallowed():
    seen = []

    def op(x):
        raise RuntimeError('endpoint failure')

    with pytest.raises(RuntimeError, match='endpoint failure'):
        values_at_breakpoints([_Fun([2., 4.], seen)], (-1., 1.), op)
    assert seen == []


@pytest.mark.parametrize('dtype', [jnp.bool_, jnp.int32, jnp.float32, jnp.complex64])
@pytest.mark.parametrize('callable_op', [False, True])
def test_endpoint_assignment_uses_default_double_storage(dtype, callable_op):
    seen = []
    raw = jnp.asarray([[1, 0], [0, 1]], dtype=dtype)
    fun = _Fun(jnp.asarray([1, 0], dtype=dtype), seen)
    result = values_at_breakpoints([fun], (-1., 1.), (lambda x: raw) if callable_op else None)
    expected_dtype = jnp.complex128 if jnp.issubdtype(dtype, jnp.complexfloating) else jnp.float64
    assert result.dtype == expected_dtype
    expected = raw if callable_op else jnp.tile(jnp.asarray([1, 0]), (2, 1))
    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize('techname', ['chebtech1', 'chebtech2'])
def test_actual_unbounded_endpoint_limit_bypasses_inverse_map(techname):
    from chebfunjax.domain import Domain
    from chebfunjax.fun.unbndfun import Unbndfun
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

    cls = Chebtech1 if techname == 'chebtech1' else Chebtech2
    technology = cls.from_coeffs(jnp.asarray([1., 2., 3.]))
    fun = Unbndfun.from_chebtech(technology, Domain((-jnp.inf, jnp.inf)))
    assert endpoint_limit(fun) == 2.
    assert endpoint_limit(fun, True) == 6.


def test_actual_singular_endpoint_limit_uses_full_onefun():
    from chebfunjax.chebfun1d.chebfun import _Piece
    from chebfunjax.fun.singfun import Singfun
    from chebfunjax.tech.chebtech import Chebtech2

    smooth = Chebtech2.from_coeffs(jnp.asarray([2.]))
    singular = Singfun(smoothPart=smooth, exponents=(-.5, 0.))
    fun = _Piece(tech=singular, interval=(3., 7.))
    assert jnp.isposinf(endpoint_limit(fun))
    assert endpoint_limit(fun, True) == 2*jnp.asarray(2.)**(-.5)


def test_nonmatching_rank_one_constant_row_has_no_extra_probe():
    calls = []

    def op(x):
        calls.append(jnp.shape(x))
        return jnp.asarray([1., 2., 3.])

    wrapped = vector_check(op, (-1., 1.))
    assert calls == [(2,)]
    assert jnp.array_equal(wrapped(jnp.ones(4)), jnp.tile(jnp.asarray([1., 2., 3.]), (4, 1)))
    assert calls == [(2,), (4,)]


def test_matching_rank_one_ambiguous_two_stays_column_without_lazy_reprobe():
    calls = []

    def op(x):
        calls.append(jnp.shape(x))
        return jnp.asarray([1., 2.])

    wrapped = vector_check(op, (-1., 1.))
    assert calls == [(2,)]
    # This ambiguous spelling remains a column. Users intending a constant
    # row use shape(1,2), as the source row-expansion controls demonstrate.
    assert jnp.array_equal(wrapped(jnp.ones(4)), jnp.asarray([1., 2.]))
    assert calls == [(2,), (4,)]


def test_trig_endpoint_accessor_uses_coefficients_not_literal_cache():
    from chebfunjax.chebfun1d.chebfun import _Piece
    from chebfunjax.tech.trigtech import Trigtech

    technology = Trigtech(coeffs=jnp.asarray([2.+0j]), real_columns=(True,),
                          _values=jnp.asarray([7.]))
    fun = _Piece(tech=technology, interval=(3., 7.))
    assert technology.values[0] == 7.
    assert endpoint_limit(fun) == 2. and endpoint_limit(fun, True) == 2.

"""Public numeric source controls; no low-level transform API change."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils.quadrature import chebpts


@pytest.fixture(autouse=True)
def preferences():
    saved = ChebfunPref._defaults
    ChebfunPref.setDefaults('factory')
    yield
    ChebfunPref._defaults = saved


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('shape', [(), (3,), (1, 3), (3, 2)])
def test_session_tech_shape_and_full_domain(cls, shape):
    ChebfunPref.setDefaults('tech', cls)
    count = 1
    for size in shape:
        count *= size
    data = (jnp.arange(count).reshape(shape)+1)*(1+2j)
    f = chebfun(data, domain=(-1., 0., 1.))
    expected = cls.from_values(jnp.atleast_1d(data)).coeffs
    assert f.domain.breakpoints == (-1., 0., 1.)
    for piece in f.funs:
        assert isinstance(piece.tech, cls)
        assert bool(jnp.array_equal(piece.tech.coeffs, expected))


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
def test_numeric_horzcat_reuses_public_constructor(cls):
    ChebfunPref.setDefaults('tech', cls)
    f = chebfun(jnp.array([1., 2.]), domain=(-1., 0., 1.))
    q = Chebfun.horzcat([jnp.array([2j, 3.]), f])
    expected = chebfun(jnp.array([[2j, 3.]]), domain=f.domain.breakpoints)
    for result, reference in zip(q.funs, expected.funs):
        assert isinstance(result.tech, cls)
        assert bool(jnp.array_equal(result.tech.coeffs[0, :2], reference.tech.coeffs[0]))
        assert bool(jnp.all(result.tech.coeffs[1:, :2] == 0))
    other = Chebtech2 if cls is Chebtech1 else Chebtech1
    assert isinstance(chebfun(2., tech=other).funs[0].tech, other)


@pytest.mark.parametrize('cls,kind', [(Chebtech1, 1), (Chebtech2, 2)])
@pytest.mark.parametrize('bad', ['endpoint_nan', 'interior_inf', 'complex_multicolumn'])
def test_nonfinite_source_own_grid(cls, kind, bad):
    x = chebpts(9, kind=kind)
    values = 1+x+x*x
    if bad == 'endpoint_nan':
        corrupted = values.at[0].set(jnp.nan)
    elif bad == 'interior_inf':
        corrupted = values.at[4].set(jnp.inf)
    else:
        values = jnp.stack([values*(1+1j), 2-x], axis=1)
        corrupted = values.at[4, 0].set(jnp.nan+0j)
    f = chebfun(corrupted, tech=cls)
    expected = cls.from_values(values)
    error = jnp.max(jnp.abs(f.funs[0].tech.coeffs-expected.coeffs))
    assert float(error) < 30*jnp.finfo(jnp.float64).eps*jnp.max(jnp.abs(values))
    if bad == 'complex_multicolumn':
        assert jnp.iscomplexobj(f.funs[0].tech.coeffs)


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
def test_nan_bypass_and_no_good_row_error(cls):
    f = chebfun(jnp.full((3, 2), jnp.nan), tech=cls)
    assert bool(jnp.all(jnp.isnan(f.funs[0].tech.coeffs)))
    for values in (jnp.array([jnp.inf, jnp.inf]), jnp.array([jnp.inf, jnp.nan])):
        with pytest.raises(ValueError, match='extrapolate:nansInfs'):
            chebfun(values, tech=cls)


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('domain', [(0., float('inf')), (float('-inf'), 0.),
                                    (float('-inf'), float('inf')),
                                    (float('-inf'), 0., 1., float('inf'))])
def test_unbounded_zero_wrapper(cls, domain):
    ChebfunPref.setDefaults('tech', cls)
    f = chebfun(jnp.zeros((1, 2)), domain=domain)
    for p in f.funs:
        assert isinstance(p.tech, cls)
        if any(jnp.isinf(jnp.asarray(p.interval))):
            assert isinstance(p, Unbndfun)
    assert bool(jnp.all(f(jnp.array([.25, .5])) == 0))
    assert bool(jnp.all(f.point_values == 0))


@pytest.mark.parametrize('value', [1., 2j, jnp.nan, jnp.inf])
def test_unbounded_nonzero_native_error(value):
    with pytest.raises(ValueError, match='UNBNDFUN:unbndfun:inputValues'):
        chebfun(value, domain=(0., float('inf')))


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
def test_fixed_length_after_numeric_transform(cls):
    values = jnp.array([1., 2., -1., 3.])
    expected = cls.from_values(values).prolong(7).coeffs
    assert bool(jnp.array_equal(chebfun(values, tech=cls, n=7).funs[0].tech.coeffs, expected))
    ChebfunPref.setDefaults('tech', cls, 'fixedLength', 7)
    assert bool(jnp.array_equal(chebfun(values).funs[0].tech.coeffs, expected))
    # Low-level finite transformation retains its traced API.
    assert bool(jnp.array_equal(jax.jit(lambda v: cls.from_values(v).coeffs)(values),
                               cls.from_values(values).coeffs))


def test_session_trig_breaks_and_explicit_flag():
    ChebfunPref.setDefaults('tech', 'trigtech')
    f = chebfun(jnp.array([[1., 2j]]), domain=(-1., 0., 1.))
    assert all(isinstance(p.tech, Trigtech) for p in f.funs)
    assert all(p.tech.n == 1 for p in f.funs)
    with pytest.raises(ValueError, match='periodic'):
        chebfun(jnp.array([[1., 2j]]), domain=(-1., 0., 1.), trig=True)


def test_existing_numeric_doublelength_route():
    f = chebfun(jnp.asarray([3., 2., 1.]), doubleLength=True)
    assert len(f) == 5  # Native test_doubleLength.m slot3.


def test_existing_numeric_trunc_route():
    values = jnp.asarray([3., 2., 1.])
    f = chebfun(values, trunc=2)
    expected = chebfun(values).truncate(2)
    assert bool(jnp.array_equal(f.funs[0].tech.coeffs, expected.funs[0].tech.coeffs))


@pytest.mark.parametrize('kwargs', [{'exps': (1., 0.)}, {'blowup': True}])
def test_inherited_numeric_singular_route(kwargs):
    with pytest.raises(ValueError, match='requires a callable'):
        chebfun(jnp.array([1., 2.]), **kwargs)


def test_inherited_numeric_adaptive_override_route():
    with pytest.raises(ValueError, match='do not apply'):
        chebfun(jnp.array([1., 2.]), sample_test=False)

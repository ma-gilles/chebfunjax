"""Source controls for @trigtech/{qr,horzcat}.m and @chebfun/qr.m 7574c77."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.trigtech import Trigtech, _trig_qr_collate

EPS = jnp.finfo(jnp.float64).eps


def fixture(n=9, complex_column=False):
    x = -1 + 2*jnp.arange(n)/n
    values = jnp.stack((jnp.ones_like(x), jnp.sin(jnp.pi*x)), axis=1)
    if complex_column:
        values = values.astype(jnp.complex128)
        values = values.at[:, 1].add(1j*jnp.cos(jnp.pi*x))
    result = Trigtech.from_values(values)
    if complex_column:
        assert result.values.dtype == jnp.complex128
        assert result.real_columns == (True, False)
        assert jnp.max(jnp.abs(jnp.imag(result.values[:, 1]))) > 0
    else:
        assert result.real_columns == (True, True)
    return result


@pytest.mark.parametrize('complex_column', [False, True])
def test_dense_source_sign_scaling_and_cache(complex_column):
    f = fixture(complex_column=complex_column)
    q, r = f.qr()
    raw_q, raw_r = jnp.linalg.qr(f.values, mode='reduced')
    phase = jnp.sign(jnp.diag(raw_r))
    phase = jnp.where(phase == 0, 1, phase)
    expected_q = raw_q*phase[None, :]/jnp.sqrt(2/f.n)
    expected_r = jnp.sqrt(2/f.n)*phase[:, None]*raw_r
    assert jnp.max(jnp.abs(q.values-expected_q)) < 20*EPS
    assert jnp.max(jnp.abs(r-expected_r)) < 20*EPS
    assert q._values is not None
    assert q.real_columns == (not complex_column,)*2
    assert jnp.max(jnp.abs(jnp.imag(jnp.diag(raw_r)))) == 0
    assert jnp.min(jnp.real(jnp.diag(r))) >= 0


def test_cached_values_are_qr_input():
    f = fixture()
    cached = f.values.at[:, 1].add(0.125)
    f = Trigtech(coeffs=f.coeffs, real_columns=f.real_columns, _values=cached)
    q, r = f.qr()
    assert jnp.max(jnp.abs(q.values@r-cached)) < 30*EPS


def test_n_less_than_columns_restores_source_length():
    f = Trigtech.from_values(jnp.asarray([[1., 2., 3.]]))
    q, r = f.qr()
    assert q.n == 1 and q.num_columns == 3 and r.shape == (3, 3)
    assert q._values is not None
    assert jnp.max(jnp.abs(q.values@r-f.values)) < 100*EPS


def test_empty_identity_and_empty_factors():
    f = Trigtech.from_values(jnp.empty((0, 2)))
    q, r = f.qr()
    assert q is f and r.shape == (0, 0)


@pytest.mark.parametrize('value', [2., 2+3j])
def test_single_column(value):
    f = Trigtech.from_values(jnp.asarray([value]))
    q, r = f.qr()
    assert jnp.abs(r[0, 0]-jnp.sqrt(2)*jnp.abs(value)) < 20*EPS
    assert jnp.max(jnp.abs(q.values*r[0, 0]-f.values)) < 20*EPS


def test_single_zero_tech_keeps_native_nan_division():
    f = Trigtech.from_values(jnp.zeros(1))
    q, r = f.qr()
    assert r[0, 0] == 0 and jnp.all(jnp.isnan(q.values))


def test_collation_keeps_first_happiness_and_caches():
    first = Trigtech(coeffs=jnp.array([1.+0j]), real_columns=(True,),
                     ishappy=False, _values=jnp.array([2.]))
    second = Trigtech.from_values(jnp.array([3.+1j]))
    combined = _trig_qr_collate((first, second))
    assert not combined.ishappy
    assert combined.real_columns == (True, False)
    assert jnp.array_equal(combined.values, jnp.array([[2., 3.+1j]]))


def test_compiled_multiple_column_qr():
    f = fixture()
    q, r = jax.jit(lambda v: v.qr())(f)
    assert jnp.max(jnp.abs(q.values@r-f.values)) < 100*EPS


def test_public_physical_domain_periodic_dispatch():
    f = fixture()
    cf = Chebfun(funs=[_Piece(tech=f, interval=(2., 7.))],
                 domain=Domain((2., 7.)))
    q, r = cf.qr()
    assert isinstance(q, Chebfun)
    assert isinstance(q.funs[0].tech, Trigtech)
    assert q.domain == cf.domain
    x = jnp.linspace(2., 7., 21)
    assert jnp.max(jnp.abs(q(x)@r-cf(x))) < 100*EPS
    assert jnp.max(jnp.abs(q.funs[0].tech.innerProduct(q.funs[0].tech)*2.5-jnp.eye(2))) < 30*EPS


def test_public_single_zero_normalizes_constant():
    f = Trigtech.from_values(jnp.zeros(1))
    cf = Chebfun(funs=[_Piece(tech=f, interval=(2., 7.))],
                 domain=Domain((2., 7.)))
    q, r = cf.qr()
    assert r[0, 0] == 0
    assert jnp.max(jnp.abs(q(jnp.array([2., 4., 7.]))-1/jnp.sqrt(5.))) < 10*EPS



def test_same_domain_restrict_converts_but_preserves_input():
    from chebfunjax.tech.chebtech import Chebtech2

    f = fixture()
    cf = Chebfun(funs=[_Piece(tech=f, interval=(2., 7.))],
                 domain=Domain((2., 7.)))
    result = cf.restrict((2., 7.))
    assert isinstance(result.funs[0].tech, Chebtech2)
    assert cf.funs[0].tech is f
    assert jnp.array_equal(cf.funs[0].tech.values, f.values)
    x = jnp.linspace(2., 7., 21)
    assert jnp.max(jnp.abs(result(x)-cf(x))) < 100*EPS


def test_true_periodic_quasimatrix_restricts_before_dispatch():
    from chebfunjax.tech.chebtech import Chebtech2

    cf = Chebfun(funs=[_Piece(tech=fixture(), interval=(2., 7.))],
                 domain=Domain((2., 7.)))
    left, right = cf.mat2cell()
    q, r = left.qr([right])
    assert isinstance(q.funs[0].tech, Chebtech2)
    x = jnp.linspace(2., 7., 21)
    assert jnp.max(jnp.abs(q(x)@r-cf(x))) < 100*EPS


def test_true_quasimatrix_breakpoint_union():
    from chebfunjax.tech.chebtech import Chebtech2

    def constant(a, b):
        return _Piece(tech=Chebtech2.from_coeffs(jnp.array([1.])), interval=(a, b))

    first = Chebfun(funs=[constant(2., 4.), constant(4., 7.)],
                    domain=Domain((2., 4., 7.)))
    second = Chebfun(funs=[_Piece(tech=Trigtech.from_function(
        lambda t: jnp.sin(jnp.pi*t), n=9), interval=(2., 7.))],
        domain=Domain((2., 7.)))
    # Source quasi2cheb restricts the union even with mixed initial Techs.
    q, r = first.qr([second])
    assert q.domain.breakpoints == (2., 4., 7.)
    x = jnp.linspace(2., 7., 21)
    expected = jnp.stack((first(x), second(x)), axis=-1)
    assert jnp.max(jnp.abs(q(x)@r-expected)) < 100*EPS


def test_public_array_qr_uses_original_cache():
    f = fixture()
    values = f.values.at[:, 1].add(0.125)
    f = Trigtech(coeffs=f.coeffs, real_columns=f.real_columns, _values=values)
    cf = Chebfun(funs=[_Piece(tech=f, interval=(2., 7.))],
                 domain=Domain((2., 7.)))
    q, r = cf.qr()
    assert isinstance(q.funs[0].tech, Trigtech)
    assert jnp.max(jnp.abs(q.funs[0].tech.values@r-values)) < 100*EPS

"""Direct native factor-operation controls, Chebfun7574c77; no sampled norms."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d._svd import (
    _action,
    _axis_panel,
    _axis_qr,
    _conjugate,
    _core_svd,
    source_svd,
)
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech

EPS = jnp.finfo(jnp.float64).eps
BOUND = 1000*EPS  # Original native tests/chebfun2/test_norm.m multiplier.


def factor(cls, k=0):
    if cls is Trigtech:
        return Trigtech.from_function(
            lambda x: jnp.ones_like(x)/jnp.sqrt(2.) if k == 0 else jnp.sin(jnp.pi*x), n=9)
    coefficients = (jnp.array([1/jnp.sqrt(2.)]) if k == 0
                    else jnp.array([0., jnp.sqrt(1.5)]))
    return cls.from_coeffs(coefficients)


def approx(cls=Chebtech2, row_cls=None, weights=None, domain=(-1., 1., -1., 1.)):
    row_cls = cls if row_cls is None else row_cls
    return SeparableApprox(cols=[factor(cls, i) for i in range(2)],
                           rows=[factor(row_cls, i) for i in range(2)],
                           pivots=jnp.array([3., 1.]) if weights is None else weights,
                           domain=domain)


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2, Trigtech])
def test_orthonormal_axes_singular_values(cls):
    f = Chebfun2(approx=approx(cls))
    assert jnp.max(jnp.abs(f.svd()-jnp.array([3., 1.]))) < BOUND
    assert jnp.abs(f.norm()-jnp.sqrt(10.)) < BOUND


@pytest.mark.parametrize('cls,row_cls', [(Chebtech1, Trigtech), (Trigtech, Chebtech2)])
def test_mixed_axes_physical_scaling(cls, row_cls):
    f = Chebfun2(approx=approx(cls, row_cls, domain=(2., 6., -3., 3.)))
    assert jnp.max(jnp.abs(f.svd()-jnp.sqrt(6.)*jnp.array([3., 1.]))) < BOUND
    u, _, v = f.svd(full=True)
    assert type(u[0].funs[0].tech) is cls
    assert type(v[0].funs[0].tech) is row_cls


@pytest.mark.parametrize('weight', [2., 2+3j])
def test_scalar_normalization_analytic(weight):
    one = Chebtech2.from_coeffs(jnp.array([1.]))
    f = Chebfun2(approx=SeparableApprox(cols=[one], rows=[one],
                     pivots=jnp.asarray([weight]), domain=(0., 3., 0., 5.)))
    assert jnp.abs(f.svd()[0]-jnp.abs(weight)*jnp.sqrt(15.)) < BOUND


@pytest.mark.parametrize('cls', [Chebtech2, Trigtech])
def test_zero_full_literal_nonsquare_scales(cls):
    a = approx(cls, weights=jnp.zeros(2), domain=(0., 4., 0., 9.))
    f = Chebfun2(approx=a)
    u, s, v = f.svd(full=True)
    assert s.shape == (1,) and s[0] == 0 and len(u) == len(v) == 1
    assert jnp.abs(u[0](3.)-0.5) < 10*EPS
    assert jnp.abs(v[0](2.)-1/3) < 10*EPS
    assert f.norm() == 0


def test_empty_adapters():
    a = SeparableApprox(cols=[], rows=[], pivots=jnp.empty(0), domain=(-1., 1., -1., 1.))
    f = Chebfun2(approx=a)
    assert f.svd().shape == (0,)
    assert f.norm().shape == (0,) and a.norm().shape == (0,)
    u, s, v = f.svd(full=True)
    assert u == v == [] and s.shape == (0,)


def test_literal_complex_full_action():
    a = approx(weights=jnp.array([2+1j, 1-2j]))
    a = SeparableApprox(cols=a.cols, rows=[a.rows[0]*(1+2j), a.rows[1]],
                        pivots=a.pivots, domain=a.domain)
    left = _axis_panel(tuple(a.cols))
    right = _axis_panel(tuple(a.rows))
    ql, rl = _axis_qr(left, (-1., 1.))
    qr, rr = _axis_qr(right, (-1., 1.))
    du, ds, dvh = jnp.linalg.svd((rl@jnp.diag(a.pivots))@rr.T, full_matrices=False)
    u, s, v = source_svd(a, full=True)
    expected_u = _action(ql.funs[0].tech, du)
    expected_v = _action(qr.funs[0].tech, jnp.conj(dvh.T))
    assert jnp.max(jnp.abs(s-ds)) < BOUND
    for k in range(2):
        assert jnp.max(jnp.abs(u[k].funs[0].tech.coeffs-expected_u.coeffs[:, k])) < BOUND
        assert jnp.max(jnp.abs(v[k].funs[0].tech.coeffs-expected_v.coeffs[:, k])) < BOUND


def test_operator_nuclear_literal_factor_transform():
    a = approx(weights=jnp.array([2+1j, 1-2j]))
    a = SeparableApprox(cols=[a.cols[0]+1j*a.cols[1], a.cols[1]],
                        rows=[a.rows[0]+2j*a.rows[1], a.rows[1]],
                        pivots=a.pivots, domain=a.domain)
    left = _action(_axis_panel(tuple(a.cols)), jnp.diag(a.pivots))
    right = _conjugate(_axis_panel(tuple(a.rows)))
    _, rl = _axis_qr(left, (-1., 1.))
    _, rr = _axis_qr(right, (-1., 1.))
    expected = jnp.linalg.svd(rl@rr.T, compute_uv=False)
    f = Chebfun2(approx=a)
    assert jnp.abs(f.norm(2)-expected[0]) < BOUND
    assert jnp.abs(f.norm('nuc')-jnp.sum(expected)) < BOUND


def test_core_uses_plain_transpose():
    l = jnp.array([[1+1j, 2], [0, 1-2j]])
    r = jnp.array([[2-1j, 1j], [0, 3+2j]])
    d = jnp.array([1+2j, 2-1j])
    u, s, v = _core_svd(l, d, r)
    assert jnp.max(jnp.abs((u*s)@jnp.conj(v.T)-(l@jnp.diag(d))@r.T)) < BOUND


def test_rank_deficient_and_repeated_values():
    for weights in (jnp.array([1., 1.]), jnp.array([1., 0.])):
        got = source_svd(approx(weights=weights))
        assert got.shape == (2,) and jnp.all(jnp.diff(got) <= 0)
        assert jnp.max(jnp.abs(got-weights)) < BOUND


def test_trig_axis_keeps_authoritative_cache():
    a = approx(Trigtech)
    first = a.cols[0]
    changed = Trigtech(coeffs=first.coeffs, real_columns=first.real_columns,
                      _values=2*first.values)
    panel = _axis_panel((changed, a.cols[1]))
    assert jnp.array_equal(panel.values[:, 0], 2*first.values)
    assert panel.real_columns == (True, True)


def test_heterogeneous_axis_is_explicit_unqualified_interface():
    with pytest.raises(NotImplementedError, match='heterogeneous'):
        _axis_panel((factor(Chebtech2), factor(Trigtech)))



def test_public_marker_empty_adapter():
    f = Chebfun2.empty()
    assert f.svd().shape == (0,)
    u, s, v = f.svd(full=True)
    assert u == v == [] and s.shape == (0,)
    # Existing empty-aware norm wrapper returns an empty Chebfun2; retain
    # this Python API adapter, not native numeric-empty identity.
    assert isinstance(f.norm(), Chebfun2) and f.norm().isempty()


@pytest.mark.parametrize('raw', [(float('inf'), 2.), (1+1j, 2-1j)])
def test_retained_raw_weights_not_reciprocated_twice(raw):
    from chebfunjax.chebfun2d._pivot_metadata import _cdr_weights

    values = jnp.asarray(raw)
    weights = _cdr_weights(values)
    a = approx(weights=weights)
    a = SeparableApprox(cols=a.cols, rows=a.rows, pivots=weights,
                        pivot_values=values, domain=a.domain)
    expected = jnp.sort(jnp.abs(weights))[::-1]
    assert jnp.max(jnp.abs(source_svd(a)-expected)) < BOUND
    assert jnp.array_equal(a.pivot_values, values)


def test_scalar_zero_axis_with_nonzero_weight():
    zero = Chebtech2.from_coeffs(jnp.zeros(1))
    one = Chebtech2.from_coeffs(jnp.ones(1))
    a = SeparableApprox(cols=[zero], rows=[one], pivots=jnp.ones(1),
                        domain=(0., 3., 0., 5.))
    u, s, v = source_svd(a, full=True)
    assert s[0] == 0
    assert jnp.abs(u[0].innerProduct(u[0])-1) < BOUND
    assert jnp.abs(v[0].innerProduct(v[0])-1) < BOUND



def test_scalar_trig_even_nyquist_source_quadrature():
    # Native innerProduct prolongs to2*n, splitting the Nyquist cosine.
    cosine = Trigtech.from_values(jnp.array([-1., 1.]))
    one = Chebtech2.from_coeffs(jnp.ones(1))
    a = SeparableApprox(cols=[cosine], rows=[one], pivots=jnp.ones(1),
                        domain=(-1., 1., -1., 1.))
    assert jnp.abs(source_svd(a)[0]-jnp.sqrt(2.)) < BOUND



def test_independent_complex_frobenius_operator_nuclear():
    # In orthonormal e0,e1 basis: F=[[0,i],[i,1]], whereas the native
    # operator transform gives L=[[2,i],[-i,1]]. No same-provider oracle.
    e0, e1 = factor(Chebtech2, 0), factor(Chebtech2, 1)
    factors = [e0, 1j*e0+e1]
    f = Chebfun2(approx=SeparableApprox(cols=factors, rows=factors,
                      pivots=jnp.ones(2), domain=(-1., 1., -1., 1.)))
    expected = jnp.array([(jnp.sqrt(5.)+1)/2, (jnp.sqrt(5.)-1)/2])
    assert jnp.max(jnp.abs(f.svd()-expected)) < BOUND
    assert jnp.abs(f.norm()-jnp.sqrt(3.)) < BOUND
    assert jnp.abs(f.norm(2)-(3+jnp.sqrt(5.))/2) < BOUND
    assert jnp.abs(f.norm('nuc')-3) < BOUND

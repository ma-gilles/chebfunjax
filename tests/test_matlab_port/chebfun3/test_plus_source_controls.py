"""Added controls for active native plus and constructor cancellation edges.

Provenance
----------
MATLAB source : @chebfun3/{plus,iszero,chebfun3f}.m
Chebfun commit: 7574c77
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import _plus as plus
from chebfunjax.chebfun3d.chebfun3 import Chebfun3, _aca, _eval_tensor
from chebfunjax.tech.chebtech import Chebtech2

DOMAIN = (-1., 1., -1., 1., -1., 1.)


def field(coeffs=(2., 1.), *, core=1., domain=DOMAIN):
    t = Chebtech2.from_coeffs(jnp.asarray(coeffs))
    one = Chebtech2.from_coeffs(jnp.ones(1))
    return Chebfun3([t], [one], [one], jnp.asarray(core).reshape(1, 1, 1), domain)


@pytest.mark.parametrize('other', [[], Chebfun3.empty(), jnp.empty((0,))])
def test_empty_before_domain_and_types(other):
    assert plus.source_plus(field(), other).isempty()


@pytest.mark.parametrize('other', [True, 'bad', jnp.int32(2), jnp.float32(2)])
def test_native_double_type_rejection(other):
    with pytest.raises(ValueError, match='plus:unknown'):
        plus.source_plus(field(), other)


def test_domain_before_zero():
    with pytest.raises(ValueError, match='plus:domain'):
        plus.source_plus(field(core=0.), field(domain=(-2., 2., -1., 1., -1., 1.)))


@pytest.mark.parametrize('side', [0, 1])
def test_zero_identity(side):
    f, zero = field(), field(core=0.)
    assert plus.source_plus(zero, f) is f if side == 0 else plus.source_plus(f, zero) is f


@pytest.mark.parametrize('coeffs,core,expected', [((0.,), 3., True),
                                                ((2., 1.), 0., True),
                                                ((2., 1.), 1., False),
                                                ((jnp.nan,), 1., False)])
def test_literal_iszero(coeffs, core, expected):
    assert plus._source_iszero(field(coeffs, core=core)) is expected


@pytest.mark.parametrize('sumscale,expected', [(2., 2.5e-12), (0., float('inf'))])
def test_current_pref_kappa_and_unclamped_infinity(monkeypatch, sumscale, expected):
    f, g = field(), field((3., 1.))
    monkeypatch.setattr(plus, '_sum_vscale', lambda *a: jnp.asarray(sumscale))
    monkeypatch.setattr(plus, '_source_vscale', lambda obj: 2. if obj is f else 3.)
    monkeypatch.setattr(plus, 'ChebfunPref', lambda: SimpleNamespace(
        cheb3Prefs=SimpleNamespace(chebfun3eps=1e-12)))
    seen = []
    def construct(cls, op, *, domain, tol):
        seen.append((domain, tol))
        return 'constructed'
    monkeypatch.setattr(Chebfun3, 'from_function', classmethod(construct))
    assert plus.source_plus(f, g) == 'constructed'
    assert seen == [(DOMAIN, expected)]


@pytest.mark.parametrize('mixed', [False, True])
def test_51_point_sampling_branch(monkeypatch, mixed):
    calls = []
    def sample(*counts):
        calls.append(('sample', counts))
        return jnp.asarray([2.])
    f = SimpleNamespace(domain=DOMAIN, sample=sample)
    g = SimpleNamespace(domain=DOMAIN, sample=sample)
    monkeypatch.setattr(plus, '_is_periodic', lambda obj: mixed and obj is g)
    def evaluate(obj, *points):
        calls.append(('evaluate', points[0].shape))
        return jnp.asarray([3.])
    monkeypatch.setattr(plus, '_evaluate', evaluate)
    assert plus._sum_vscale(f, g) == (6. if mixed else 4.)
    expected = ('evaluate', (51, 51, 51)) if mixed else ('sample', (51, 51, 51))
    assert calls == [expected, expected]


@pytest.mark.parametrize('shape,tol', [((3, 2), 1e-12), ((3, 2), float('inf')),
                                     ((3, 0), 1e-12), ((0, 3), float('inf'))])
def test_native_aca_empty_pivot_sets(shape, tol):
    ac, ar, at, rows, cols = _aca(np.zeros(shape), tol, 3)
    assert rows.size == cols.size == 0
    assert ac.shape == (shape[0], 0) and ar.shape == (shape[1], 0) and at.shape == (0, 0)


@pytest.mark.parametrize('imaginary', [0., 2.])
def test_exact_component_assembly_preserves_complex_storage(imaginary):
    f, g = field(), field((1.,), core=1j*imaginary)
    result = plus._assemble_components(f, g)
    x = jnp.linspace(-1., 1., 11)
    assert jnp.iscomplexobj(result.core)
    assert result.core.shape == (2, 2, 2)
    assert jnp.max(jnp.abs(result(x, x, x)-(2+x+1j*imaginary))) < 1e-14


@pytest.mark.parametrize('case', ['real', 'scalar_left', 'scalar_right', 'exact_cancel',
    'near_cancel', 'near_cancel_resolvable', 'complex_add', 'complex_constructor', 'complex_power', 'numeric_values'])
def test_actual_plus(case):
    f = field()
    x = jnp.linspace(-1., 1., 17)
    if case == 'real':
        result, expected = f+field((3., -1.)), jnp.full_like(x, 5.)
    elif case == 'scalar_left':
        result, expected = 3+f, 5+x
    elif case == 'scalar_right':
        result, expected = f+3, 5+x
    elif case == 'exact_cancel':
        result, expected = f-f, jnp.zeros_like(x)
    elif case == 'near_cancel':
        g = field((-2., -1.), core=1-1e-8)
        # Literal native plus.m kappa, then chebfun3f.m getTol floor:
        # max|h|~3e-8 < eps*kappa~4.44e-8, hence zero ACA rank.
        maxh = plus._sum_vscale(f, g)
        scale = plus._source_vscale(f)+plus._source_vscale(g)
        tol = plus.ChebfunPref().cheb3Prefs.chebfun3eps*scale/maxh
        assert bool(jnp.isfinite(tol)) and tol > maxh
        result = f+g
        assert jnp.all(result.core == 0)
        assert jnp.all(result(x, x, x) == 0)
        return
    elif case == 'near_cancel_resolvable':
        result, expected = f+field((-2., -1.), core=1-1e-6), (2+x)*1e-6
    elif case == 'complex_add':
        result, expected = field(core=1j)+f, (1+1j)*(2+x)
    elif case == 'complex_constructor':
        result = Chebfun3.from_function(lambda x, y, z: 2+x+1j*(1+y))
        expected = 2+x+1j*(1+x)
    elif case == 'complex_power':
        result, expected = (1+1j)**field((1., 1.)), jnp.exp((1+x)*jnp.log(1+1j))
    else:
        result, expected = f+jnp.full((2, 2, 2), 1+2j), 3+x+2j
    assert jnp.max(jnp.abs(result(x, x, x)-expected)) < 2e-12
    if case == 'exact_cancel':
        assert jnp.all(result.core == 0)


@pytest.mark.parametrize('tol', [1e-13, float('inf')])
def test_actual_zero_constructor(tol):
    f = Chebfun3.from_function(lambda x, y, z: jnp.zeros_like(x), tol=tol)
    assert jnp.all(f.core == 0)


def test_actual_mixed_periodic_polynomial_sum():
    from chebfunjax.tech.trigtech import Trigtech

    wave = Trigtech.from_coeffs(jnp.asarray([.5, 0., .5]), is_real=True)
    one = Trigtech.from_coeffs(jnp.ones(1), is_real=True)
    periodic = Chebfun3([wave], [one], [one], jnp.ones((1, 1, 1)), DOMAIN)
    assert plus._is_periodic(periodic)
    assert not plus._is_periodic(field())
    result = periodic+field()
    x = jnp.linspace(-1., 1., 23)
    assert jnp.max(jnp.abs(result(x, x, x)-(jnp.cos(jnp.pi*x)+2+x))) < 2e-12


def test_empty_tensor_skips_callback():
    def forbidden(*args):
        pytest.fail('empty evaluation must not invoke the callback')
    grid = np.linspace(-1., 1., 3)
    out = _eval_tensor(forbidden, grid, grid, grid,
                       np.asarray([], dtype=int), np.arange(3), np.arange(2))
    assert out.shape == (0, 3, 2)


def test_actual_zero_constructor_with_empty_unsafe_callback():
    def zero(x, y, z):
        assert x.size > 0 and y.size > 0 and z.size > 0
        return jnp.zeros_like(x)
    result = Chebfun3.from_function(zero, tol=float('inf'))
    assert jnp.all(result.core == 0)


def test_nonzero_aca_reconstruction():
    matrix = np.asarray([[2., 1.], [1., 3.]])
    ac, ar, at, rows, cols = _aca(matrix, 1e-14, 3)
    assert rows.size == cols.size == 2
    reconstructed = jnp.asarray(ac) @ jnp.linalg.solve(jnp.asarray(at), jnp.asarray(ar.T))
    assert jnp.max(jnp.abs(reconstructed-jnp.asarray(matrix))) < 2e-15


@pytest.mark.parametrize('entry,rank', [(0.5e-9, 0), (1e-9, 1)])
def test_aca_strict_threshold(entry, rank):
    _, _, _, rows, cols = _aca(np.asarray([[entry]]), 1e-9, 2)
    assert rows.size == cols.size == rank


def test_actual_nonzero_constant_infinite_tolerance():
    result = Chebfun3.from_function(lambda x, y, z: jnp.ones_like(x), tol=float('inf'))
    assert jnp.all(result.core == 0)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_actual_nonfinite_function_rejected(value):
    with pytest.raises(ValueError, match='function returned Inf or NaN'):
        Chebfun3.from_function(lambda x, y, z: jnp.full_like(x, value))

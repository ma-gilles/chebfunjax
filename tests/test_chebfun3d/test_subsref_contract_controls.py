"""Source indexing controls; native predicates remain in their own files."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv
from chebfunjax.tech.chebtech import Chebtech2

COLON = slice(None)


def field(ranks=(2, 3, 4), complex_core=False):
    factors = [[Chebtech2.from_coeffs(jnp.eye(n)[i]) for i in range(n)] for n in ranks]
    core = jnp.arange(1, 1+ranks[0]*ranks[1]*ranks[2], dtype=jnp.float64).reshape(ranks)/100
    if complex_core:
        core = core+1j*core[::-1, ::-1, ::-1]
    return Chebfun3(*factors, core, (-1., 1.)*3)


@pytest.mark.parametrize('args', [(.5,), (.5, .5)])
def test_numeric_wrong_arity(args):
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:subsref:inputs:'):
        field()(*args)


def test_zero_arity_source_index_error():
    with pytest.raises(IndexError):
        field()()


@pytest.mark.parametrize('args', [(0., 'bad', 0.), (0., object(), 0.)])
def test_invalid_triple(args):
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:subsref:inputs3:'):
        field()(*args)


@pytest.mark.parametrize('n', [5, 7])
def test_brace_count(n):
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:subsref:dimensions:'):
        field().subsref({'type': '{}', 'subs': [0.]*n})


def test_case_sensitive_property():
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:get:propName:'):
        field().subsref({'type': '.', 'subs': 'Cols'})


@pytest.mark.parametrize('name', ['cols', 'rows', 'tubes', 'core', 'domain'])
def test_source_properties(name):
    f = field()
    value = f.subsref({'type': '.', 'subs': name})
    if name == 'core':
        assert jnp.array_equal(value, f.core)
    elif name == 'domain':
        assert jnp.array_equal(value, jnp.asarray(f.domain).reshape(1, 6))
    else:
        assert isinstance(value, Chebfun)
        assert value.n_columns == len(getattr(f, name))


@pytest.mark.parametrize('subs,expected', [((2,), .13), ((2, 3), .21)])
def test_numeric_dot_recursion_column_major(subs, expected):
    # core ranks2,3,4: linear2 is(2,1,1); (2,3) folds dimensions2/3.
    f = field()
    value = f.subsref([{'type': '.', 'subs': 'core'}, {'type': '()', 'subs': subs}])
    assert jnp.max(jnp.abs(value-expected)) < 1e-15


def test_factor_recursion():
    value = field().subsref([{'type': '.', 'subs': 'cols'},
                             {'type': '()', 'subs': (COLON, 2)},
                             {'type': '()', 'subs': (.25,)}])
    assert jnp.max(jnp.abs(value-.25)) < 1e-15


def test_unsupported_cell_recursion_visible():
    with pytest.raises(NotImplementedError, match='Recursive factor indexing'):
        field().subsref([{'type': '.', 'subs': 'cols'}, {'type': '{}', 'subs': (1,)}])


@pytest.mark.parametrize('kind,subs', [('()', (.1, .2, .3)), ('{}', (.1, .1, .2, .2, .3, .3))])
def test_non_dot_ignores_remaining_records(kind, subs):
    f = field()
    actual = f.subsref([{'type': kind, 'subs': subs}, {'type': 'invalid', 'subs': ()}])
    assert jnp.abs(actual-f.feval(.1, .2, .3)) < 1e-15


def test_empty_feval_precedes_colon_dispatch():
    assert Chebfun3.empty().feval(COLON, COLON, COLON).size == 0


def test_all_colon_identity():
    f = field()
    assert f(COLON, COLON, COLON) is f


@pytest.mark.parametrize('ranks,axis', [((2, 3, 4), 0), ((2, 3, 4), 1), ((2, 3, 4), 2),
                                       ((2, 1, 3), 0), ((2, 3, 1), 0), ((1, 3, 2), 1)])
def test_plane_contractions_and_singletons(monkeypatch, ranks, axis):
    f = field(ranks, True)
    monkeypatch.setattr(Chebfun2, 'from_function', lambda *a, **kw: pytest.fail('resampling'))
    args = [COLON]*3
    args[axis] = .2
    g = f(*args)
    xs = jnp.asarray([-.7, .1, .6])
    xyz = [xs, xs, xs]
    xyz[axis] = .2+0*xs
    assert g(xs, xs).shape == xs.shape
    assert jnp.max(jnp.abs(g(xs, xs)-f.feval(*xyz))) < 3e-13


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_complex_line_source_transposes(axis):
    f = field(complex_core=True)
    args = [.2, .2, .2]
    args[axis] = COLON
    g = f(*args)
    xs = jnp.asarray([-.7, .1, .6])
    xyz = [.2+0*xs]*3
    xyz[axis] = xs
    expected = f.feval(*xyz)
    if axis == 1:
        expected = jnp.conj(expected)  # Literal native feval conjugate transpose.
    assert g(xs).shape == xs.shape
    assert jnp.max(jnp.abs(g(xs)-expected)) < 3e-13


def test_vector_fixed_colon_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match='scalar fixed coordinates'):
        field()(jnp.asarray([.1, .2]), COLON, COLON)


def test_numeric_jit_and_ad_dispatch():
    f = field((2, 1, 1))
    evaluate = jax.jit(lambda x: f(x, .2, .3))
    assert jnp.abs(evaluate(.4)-.018) < 1e-15
    assert jnp.abs(jax.grad(lambda x: f(x, .2, .3))(.4)-.02) < 1e-15


def test_three_component_dispatch_reaches_compose(monkeypatch):
    marker = object()
    one = Chebtech2.from_coeffs(jnp.asarray([1.]))
    from chebfunjax.chebfun2d.separable_approx import SeparableApprox
    a = SeparableApprox(cols=[one], rows=[one], pivots=jnp.ones(1), domain=(-1., 1.)*2)
    vector = Chebfun2v([a, a, a])
    monkeypatch.setattr(Chebfun2v, 'compose', lambda self, outer: marker)
    assert field()(vector) is marker
    assert field()(vector, 999) is marker  # Literal first-input non-triple dispatch.
    assert field()(*(Chebfun2(approx=a) for _ in range(3))) is marker


def test_chebfun2v_three_domain_guard():
    vector = Chebfun2v.from_functions(lambda x, y: 2+0*x, lambda x, y: 0*x, lambda x, y: 0*x)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN2V:COMPOSE:DomainMismatch3'):
        field()(vector)


def test_chebfun2v_three_complex_guard():
    vector = Chebfun2v.from_functions(lambda x, y: 1j+0*x, lambda x, y: 0*x, lambda x, y: 0*x)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN2V:COMPOSE:Complex'):
        field()(vector)


def test_sphere_domain_guard():
    vector = Spherefunv.from_functions(lambda l, t: 2+0*l, lambda l, t: 0*l, lambda l, t: 0*l)
    with pytest.raises(ValueError, match='CHEBFUN:SPHEREFUNV:COMPOSE:DomainMismatch3'):
        field()(vector)


def test_sphere_type_guard():
    one = Spherefun.from_function(lambda l, t: 1+0*l)
    with pytest.raises(ValueError, match='CHEBFUN:SPHEREFUNV:COMPOSE:OP'):
        Spherefunv(one, one, one).compose(lambda x, y, z: x+y+z)


@pytest.mark.parametrize('ranks,axis', [((1, 3, 2), 2), ((1, 1, 1), 0),
                                       ((1, 1, 1), 1), ((1, 1, 1), 2)])
def test_restrict_scalar_plane_storage(monkeypatch, ranks, axis):
    f = field(ranks, True)
    monkeypatch.setattr(Chebfun2, 'from_function', lambda *a, **kw: pytest.fail('resampling'))
    domain = [-1., 1.]*3
    domain[2*axis:2*axis+2] = [.2, .2]
    plane = f.restrict(domain)
    xs = jnp.asarray([-.7, .1, .6])
    xyz = [xs, xs, xs]
    xyz[axis] = .2+0*xs
    actual = plane(xs, xs)
    assert actual.shape == xs.shape
    assert jnp.max(jnp.abs(actual-f.feval(*xyz))) < 3e-13


def test_plane_adapter_preserves_coefficient_entries_and_tech():
    from chebfunjax.chebfun1d.mtimes import _columns
    from chebfunjax.chebfun3d._mtimes import _panel
    from chebfunjax.chebfun3d._plane import scalar_plane_product
    f = field((1, 3, 2), True)
    rows = _panel(f.rows, (-1., 1.))
    cols = _panel(f.cols, (-1., 1.))
    matrix = f.core[:, :, 0].T
    product = rows @ matrix
    plane = scalar_plane_product(rows, matrix, cols)
    before = _columns(product)[0].funs[0].tech
    after = plane.approx.cols[0]
    assert type(after) is type(before)
    assert jnp.array_equal(after.coeffs.reshape(-1), before.coeffs.reshape(-1))

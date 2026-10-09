"""Source-derived vander/private column-cat controls, not public horzcat parity.

Provenance: @chebfun/vander.m, horzcat.m, fliplr.m; @chebtech/horzcat.m,
Chebfun7574c77680d7e82b79626300bf255498271a72df.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d._vander import source_column_horzcat
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech, _trig_prolong_coeffs

EPS = float(jnp.finfo(jnp.float64).eps)
TECHS = [Chebtech1, Chebtech2]


def function(Tech=Chebtech2, coeffs=(.25, .5), happy=True, domain=(-1., 1.)):
    return Chebfun(funs=[_Piece(tech=Tech(coeffs=jnp.asarray(coeffs),
                                         ishappy=happy), interval=domain)],
                   domain=Domain(domain))


def constant_pieces(domain):
    return Chebfun(funs=[_Piece(tech=Chebtech2(coeffs=jnp.asarray([1.])),
                                interval=(a, b))
                         for a, b in zip(domain[:-1], domain[1:])],
                   domain=Domain(domain))


@pytest.mark.parametrize('Tech', TECHS)
@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('n', [0, 1, 2, 4])
def test_vander_small_source_values_and_constant_zero_count(Tech, complex_values, n):
    phase = 1+1j if complex_values else 1.
    f = function(Tech, phase*jnp.asarray([.25, .5]))
    out = f.vander(n)
    assert isinstance(out, Chebfun)
    assert out.n_columns == max(n, 1)
    x = jnp.asarray([-.75, -.25, .125, .5])
    expected = jnp.stack([(phase*(.25+.5*x))**k
                          for k in reversed(range(max(n, 1)))], axis=1)
    actual = out(x)
    actual = actual[:, None] if actual.ndim == 1 else actual
    assert float(jnp.max(jnp.abs(actual-expected))) < 10*EPS*float(jnp.max(jnp.abs(expected)))
    # Native horzcat inherits ordinary constant constructor/default Tech.
    assert isinstance(out.funs[0].tech, Chebtech2)


@pytest.mark.parametrize('Tech', TECHS)
def test_literal_times_sequence_and_explicit_point_reversal(Tech, monkeypatch):
    f = function(Tech).set_point_values(jnp.asarray([2+1j, -1+2j]))
    original = Chebfun.__mul__
    records = []

    def times(left, right):
        result = original(left, right)
        records.append((left, right, result))
        return result

    monkeypatch.setattr(Chebfun, '__mul__', times)
    out = f.vander(4)
    assert len(records) == 3
    assert all(left is f for left, _, _ in records)
    assert records[1][1] is records[0][2]
    assert records[2][1] is records[1][2]
    # Copy oracle follows recorded sequential products, not independent pow.
    ascending = [records[0][1]] + [r[2] for r in records]
    expected = jnp.stack([g.point_values for g in ascending], axis=1)[:, ::-1]
    assert jax.device_get(out.point_values).tobytes() == jax.device_get(expected).tobytes()


@pytest.mark.parametrize('Tech', TECHS)
def test_full_break_domain_preserved_for_constant_and_products(Tech):
    domain = (-1., 0., 1.)
    pieces = [_Piece(tech=Tech(coeffs=jnp.asarray([(a+b)/2, (b-a)/2])),
                     interval=(a, b)) for a, b in zip(domain[:-1], domain[1:])]
    f = Chebfun(funs=pieces, domain=Domain(domain))
    for n in (0, 1, 3):
        out = f.vander(n)
        assert out.domain.breakpoints == domain
        assert out.point_values.shape[0] == 3
        assert len(out.funs) == 2


@pytest.mark.parametrize('kind', ['row', 'array'])
def test_vander_rejects_noncolumn_scalar_input(kind):
    f = function().T if kind == 'row' else function(coeffs=[[1., 2.], [0., 1.]])
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:vander:row'):
        f.vander(3)


@pytest.mark.parametrize('Tech', TECHS)
@pytest.mark.parametrize('happy', [False, True])
def test_private_cat_first_tech_metadata_and_exact_prolongation(Tech, happy):
    first = function(Tech, [1., 2.], happy)
    second = function(Chebtech2, [3., 4., 5.], not happy)
    out = source_column_horzcat([first, second])
    assert type(out.funs[0].tech) is Tech
    assert out.funs[0].tech.ishappy is happy
    assert bool(jnp.array_equal(out.coeffs, jnp.asarray([[1., 3.], [2., 4.], [0., 5.]])))
    assert bool(jnp.array_equal(out.point_values,
                               jnp.stack([first.point_values, second.point_values], axis=1)))


@pytest.mark.parametrize('kind', ['shifted', 'count', 'near', 'large_scale', 'unbounded'])
def test_private_cat_literal_domain_branch(kind):
    d1 = (-1., 0., 1.)
    d2 = (-1., .25, 1.)
    if kind == 'count':
        d2 = (-1., 1.)
    elif kind == 'near':
        d2 = (-1., EPS/4, 1.)
    elif kind == 'large_scale':
        d1, d2 = (1e16, 1.5e16, 2e16), (1e16, 1.5e16+2, 2e16)
    elif kind == 'unbounded':
        d1 = d2 = (0., float('inf'))
    first, second = constant_pieces(d1), constant_pieces(d2)
    out = source_column_horzcat([first, second])
    if kind == 'large_scale':
        # Native compares Boolean any(diff) against hscale*eps >1 here.
        assert isinstance(out, Chebfun)
        assert out.domain.breakpoints == d1
    else:
        assert isinstance(out, Quasimatrix)
        assert out.cols[0] is first and out.cols[1] is second


@pytest.mark.parametrize('kind', ['singular', 'delta', 'mixed_periodic', 'all_periodic'])
def test_private_cat_noncollatable_and_periodic_cases(kind):
    first = function()
    if kind == 'singular':
        second = Chebfun(funs=[_Piece(tech=Singfun(Chebtech2(coeffs=jnp.asarray([1.])),
                                                   (.5, 0.)), interval=(-1., 1.))],
                         domain=first.domain)
    elif kind == 'delta':
        second = Chebfun(funs=first.funs, domain=first.domain, deltas=((0., 1., 0),))
    else:
        second = Chebfun(funs=[_Piece(tech=Trigtech(coeffs=jnp.asarray([0., 2., 0.]),
                                                    is_real=True, ishappy=False),
                                      interval=(-1., 1.))], domain=first.domain)
        if kind == 'all_periodic':
            first = Chebfun(funs=[_Piece(tech=Trigtech(coeffs=jnp.asarray([1.]),
                                                       is_real=True, ishappy=True),
                                         interval=(-1., 1.))], domain=first.domain)
    out = source_column_horzcat([first, second])
    if kind == 'all_periodic':
        assert isinstance(out, Chebfun)
        assert out.funs[0].tech.ishappy
        expected = jnp.stack([_trig_prolong_coeffs(g.funs[0].tech.coeffs, 3)
                              for g in (first, second)], axis=1)
        assert bool(jnp.array_equal(out.coeffs, expected))
    else:
        assert isinstance(out, Quasimatrix)
        assert out.cols[0] is first and out.cols[1] is second


def test_singular_vander_keeps_source_column_representation():
    sf = Singfun(Chebtech2(coeffs=jnp.asarray([1.])), (.5, 0.))
    f = Chebfun(funs=[_Piece(tech=sf, interval=(-1., 1.))], domain=Domain((-1., 1.)))
    out = f.vander(3)
    assert isinstance(out, Quasimatrix)
    assert out.n_cols == 3
    assert isinstance(out.cols[1].funs[0].tech, Singfun)
    assert out.cols[1].funs[0].tech.exponents == (.5, 0.)
    x = jnp.asarray([-.75, -.25, .25, .75])
    expected = jnp.stack([1+x, jnp.sqrt(1+x), jnp.ones_like(x)], axis=1)
    assert float(jnp.max(jnp.abs(out(x)-expected))) < 10*EPS*float(jnp.max(expected))


def test_column_fliplr_preserves_unbounded_wrapper_and_points():
    domain = Domain((0., float('inf')))
    tech = Chebtech2(coeffs=jnp.asarray([[1., 2.], [3., 4.]]), ishappy=False)
    piece = Unbndfun(onefun=tech, domain=domain, mapping_type='right')
    values = jnp.asarray([[1+2j, 3+4j], [5+6j, 7+8j]])
    f = Chebfun(funs=[piece], domain=domain).set_point_values(values)
    out = f.fliplr()
    assert isinstance(out.funs[0], Unbndfun)
    assert out.funs[0].mapping_type == piece.mapping_type
    assert out.funs[0].domain == domain
    assert not out.funs[0].tech.ishappy
    assert bool(jnp.array_equal(out.coeffs, tech.coeffs[:, ::-1]))
    assert bool(jnp.array_equal(out.point_values, values[:, ::-1]))


def test_private_cat_empty_singleton_and_exact_endpoint_check():
    f = function()
    assert source_column_horzcat([Chebfun.empty(), f]) is f
    empty = Chebfun.empty()
    assert source_column_horzcat([empty, Chebfun.empty()]) is empty
    with pytest.raises(ValueError, match='horzcat:domains'):
        source_column_horzcat([f, function(domain=(-1., 1+EPS))])


def test_negative_integer_count_follows_empty_cell_expansion():
    assert function().vander(-1).n_columns == 1

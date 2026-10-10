"""Pinned native split-return and one-sided primitive contracts.

Source: @singfun/cumsum.m, @bndfun/cumsum.m, @chebfun/cumsum.m,
Chebfun 7574c77680d7e82b79626300bf255498271a72df.
Tagged tests isolate routing/scaling; they do not qualify approximation.
Analytic targets are predeclared in the accompanying review packet.
"""
import importlib
from fractions import Fraction

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(jnp.finfo(jnp.float64).eps)


def tech(*coeffs):
    return Chebtech2.from_coeffs(jnp.asarray(coeffs))


def exact(actual, wanted):
    assert bool(jnp.array_equal(jnp.asarray(actual), jnp.asarray(wanted)))


def test_tagged_singfun_split_order_and_half_scaling(monkeypatch):
    module = importlib.import_module('chebfunjax.fun.singfun')
    f = Singfun(tech(1.), (.5, .5))
    left, right = Singfun(tech(2.), (.5, 0.)), Singfun(tech(3.), (0., .5))
    calls = []

    def restrict(self, points):
        assert self is f and points == [-1., 0., 1.]
        calls.append('restrict')
        return [left, right]

    def integrate(piece):
        calls.append('left' if piece is left else 'right')
        return tech(4., 2.) if piece is left else tech(8., 4.)

    monkeypatch.setattr(Singfun, 'restrict', restrict)
    monkeypatch.setattr(module, '_sing_cumsum', integrate)
    result = f.cumsum()
    assert calls == ['restrict', 'left', 'right']
    assert isinstance(result, list) and len(result) == 2
    exact(result[0].coeffs, [2., 1.])
    exact(result[1].coeffs, [7., 2.])


@pytest.mark.parametrize('wrapper', ['piece', 'bndfun'])
def test_tagged_physical_child_width_scaling(monkeypatch, wrapper):
    f = Singfun(tech(1.), (.5, .5))
    monkeypatch.setattr(Singfun, 'cumsum', lambda self: [tech(2., 1.), tech(7., 2.)])
    obj = (_Piece(f, (2., 6.)) if wrapper == 'piece'
           else Bndfun(f, Domain((2., 6.))))
    result = obj.cumsum()
    assert len(result) == 2
    for child, interval, coeffs in zip(result, [(2., 4.), (4., 6.)],
                                        [[4., 2.], [14., 4.]]):
        if wrapper == 'piece':
            assert child.interval == interval
            exact(child.tech.coeffs, coeffs)
        else:
            assert (child.domain.a, child.domain.b) == interval
            exact(child.onefun.coeffs, coeffs)


@pytest.mark.parametrize('with_delta', [False, True])
def test_tagged_parent_carry_once_and_domains(monkeypatch, with_delta):
    def integrate(piece):
        if piece.interval == (-2., 2.):
            return _Piece(tech(7.), piece.interval)
        if piece.interval == (2., 6.):
            return [_Piece(tech(3.), (2., 4.)), _Piece(tech(5.), (4., 6.))]
        return _Piece(tech(2.), piece.interval)

    monkeypatch.setattr(_Piece, 'cumsum', integrate)
    f = Chebfun([_Piece(tech(1.), interval) for interval in
                 [(-2., 2.), (2., 6.), (6., 8.)]], Domain((-2., 2., 6., 8.)),
                deltas=((6., 11.),) if with_delta else ())
    g = f.T.cumsum()
    assert g.is_transposed
    exact(g.domain.breakpoints, [-2., 2., 4., 6., 8.])
    exact([p.tech.coeffs[0] for p in g.funs], [7., 10., 12., 25. if with_delta else 14.])


def test_tagged_single_parent_split_preserves_transpose(monkeypatch):
    monkeypatch.setattr(_Piece, 'cumsum', lambda self: [
        _Piece(tech(1.), (2., 4.)), _Piece(tech(2.), (4., 6.))])
    g = Chebfun([_Piece(tech(1.), (2., 6.))], Domain((2., 6.))).T.cumsum()
    assert len(g.funs) == 2 and g.is_transposed
    exact(g.domain.breakpoints, [2., 4., 6.])


def test_tagged_carry_uses_singular_value_not_smooth_factor(monkeypatch):
    def integrate(piece):
        return (_Piece(Singfun(tech(3.), (1., 0.)), piece.interval)
                if piece.interval == (0., 2.) else _Piece(tech(0.), piece.interval))
    monkeypatch.setattr(_Piece, 'cumsum', integrate)
    f = Chebfun([_Piece(tech(1.), (0., 2.)), _Piece(tech(1.), (2., 4.))], Domain((0., 2., 4.)))
    g = f.cumsum()
    exact(g.funs[1].tech.coeffs, [6.])


@pytest.mark.parametrize('kind', [Chebtech1, Chebtech2])
def test_smooth_and_zero_demote_to_input_tech(kind):
    smooth = Singfun(kind.from_coeffs(jnp.asarray([1.])), (0., 0.)).cumsum()
    zero = Singfun(kind.from_coeffs(jnp.asarray([0.])), (-1., -1.)).cumsum()
    assert isinstance(smooth, kind) and isinstance(zero, kind)
    exact(smooth.coeffs, [1., 1.])
    exact(zero.coeffs, [0.])


@pytest.mark.parametrize('exponents', [(-1., 0.), (0., -1.), (-1., -1.)])
def test_native_no_log_error(exponents):
    f = Singfun(tech(1.), exponents)
    with pytest.raises(ValueError, match='CHEBFUN:SINGFUN:cumsum:noLog'):
        f.cumsum()


def test_no_log_error_propagates_through_public_chebfun():
    f = Chebfun([_Piece(Singfun(tech(1.), (-1., 0.)), (2., 6.))], Domain((2., 6.)))
    with pytest.raises(ValueError, match='CHEBFUN:SINGFUN:cumsum:noLog'):
        f.cumsum()


@pytest.mark.parametrize('kind', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('amplitude', [1., 1.+2.j])
def test_integer_pole_without_log_retains_unbounded_left_constant(kind, amplitude):
    f = Singfun(kind.from_coeffs(jnp.asarray([amplitude])), (-2., 0.))
    g = f.cumsum()
    x = jnp.asarray([-.5, 0., .5, 1.])
    expected = -amplitude/(1+x)
    assert bool(jnp.all(jnp.abs(g(x)-expected) < 100*EPS*jnp.max(jnp.abs(expected))))
    assert isinstance(g.smoothPart, kind)


@pytest.mark.parametrize('amplitude', [1., 1.+2.j])
def test_two_integer_roots_independent_polynomial(amplitude):
    f = Singfun(tech(amplitude), (1., 1.))
    children = f.cumsum()
    assert isinstance(children, list) and len(children) == 2
    u = jnp.asarray([-1., -.5, 0., .5, 1.])
    for child, t in zip(children, [(u-1)/2, (u+1)/2]):
        expected = amplitude*(t-t**3/3+float(Fraction(2, 3)))
        assert bool(jnp.all(jnp.abs(child(u)-expected) < 100*EPS*max(1., abs(amplitude))))
    assert abs(children[0](1.)-children[1](-1.)) < 100*EPS*max(1., abs(amplitude))


def test_public_mapped_polynomial_split_and_constant():
    f = Chebfun([_Piece(Singfun(tech(1.), (1., 1.)), (2., 6.))], Domain((2., 6.)))
    g = f.cumsum()
    exact(g.domain.breakpoints, [2., 4., 6.])
    x = jnp.asarray([2., 3., 4., 5., 6.])
    t = (x-4.)/2.
    expected = 2*(t-t**3/3+float(Fraction(2, 3)))
    assert bool(jnp.all(jnp.abs(g(x)-expected) < 100*EPS*2))


@pytest.mark.parametrize('exponent,expected_calls', [(.5, 1), (-2., 0)])
def test_native_left_subtraction_is_not_thresholded(monkeypatch, exponent, expected_calls):
    original = Singfun.__sub__
    calls = []

    def subtract(self, value):
        calls.append(value)
        return original(self, value)

    monkeypatch.setattr(Singfun, '__sub__', subtract)
    Singfun(tech(1.), (exponent, 0.)).cumsum()
    assert len(calls) == expected_calls
    if calls:
        assert abs(calls[0]) < 1000*EPS  # source still subtracts this value


def test_two_fractional_poles_independent_asin():
    children = Singfun(tech(1.), (-.5, -.5)).cumsum()
    assert isinstance(children, list) and len(children) == 2
    u = jnp.asarray([-.5, 0., .5])
    for child, t in zip(children, [(u-1)/2, (u+1)/2]):
        expected = jnp.arcsin(t)+jnp.pi/2
        # Predeclared source-family accuracy target from native Chebfun
        # cumsum pass12. This is a NEW analytic control, not that predicate.
        assert jnp.max(jnp.abs(child(u)-expected)) < 5e7*EPS*jnp.max(jnp.abs(expected))


def test_bndfun_empty_native_return():
    f = Bndfun.empty()
    assert f.cumsum() is f


def test_nonconstant_demoted_chebtech1_carry_uses_rval():
    # Native rval is2; the final first-kind sample would be1+1/sqrt(2).
    first = Singfun(Chebtech1.from_coeffs(jnp.asarray([1.])), (0., 0.))
    second = Chebtech1.from_coeffs(jnp.asarray([1.]))
    f = Chebfun([_Piece(first, (0., 2.)), _Piece(second, (2., 4.))],
                Domain((0., 2., 4.)))
    g = f.cumsum()
    assert isinstance(g.funs[0].tech, Chebtech1)
    exact(g.funs[0].tech.coeffs, [1., 1.])
    exact(g.funs[1].tech(-1.), 2.)
    exact(g.funs[1].tech.coeffs, [3., 1.])

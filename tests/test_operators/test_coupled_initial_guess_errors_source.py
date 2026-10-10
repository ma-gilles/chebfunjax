"""Native linearize.m142–164 catch scope and exact identifiers,7574c77."""
import pytest

from chebfunjax.operators import _coupled_newton as coupled
from chebfunjax.operators.chebop import Chebop

IDS = [
    'CHEBFUN:CHEBTECH:extrapolate:nansInfs',
    'CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZeroChebfun',
]
TARGET = 'CHEBFUN:CHEBOP:linearize:invalidInitialGuess:'


def problem():
    return Chebop(lambda x, u, v: [u.diff(2)+v, v.diff(2)+u], (0., 1.))


def raise_error(error):
    def callback(*args, **kwargs):
        raise error
    return callback


@pytest.mark.parametrize('identifier', IDS)
def test_exact_operator_ids_translate(monkeypatch, identifier):
    op = problem()
    original = ValueError(identifier + ': source sentinel')
    monkeypatch.setattr(op, '_call_op', raise_error(original))
    with pytest.raises(ValueError) as caught:
        coupled.linearize(op)
    assert str(caught.value).startswith(TARGET)
    assert caught.value.__cause__ is original


@pytest.mark.parametrize('error', [
    ValueError(IDS[0] + 'Extra: distinct identifier'),
    ValueError('USER:NaN: unrelated error'),
    TypeError('user operator type error'),
    RuntimeError('user operator runtime error'),
])
def test_unrelated_operator_errors_rethrow_identity(monkeypatch, error):
    op = problem()
    monkeypatch.setattr(op, '_call_op', raise_error(error))
    with pytest.raises(type(error)) as caught:
        coupled.linearize(op)
    assert caught.value is error


@pytest.mark.parametrize('identifier', IDS)
def test_boundary_errors_outside_source_catch(identifier):
    op = problem()
    original = ValueError(identifier + ': boundary sentinel')
    # Explicit signature avoids callback arity inference as a separate issue.
    def boundary(u, v):
        raise original
    op.bc = boundary
    with pytest.raises(ValueError) as caught:
        coupled.linearize(op)
    assert caught.value is original


def test_initialization_errors_outside_source_catch(monkeypatch):
    original = ValueError(IDS[0] + ': initialization sentinel')
    monkeypatch.setattr(coupled, '_initial', raise_error(original))
    with pytest.raises(ValueError) as caught:
        coupled.linearize(problem())
    assert caught.value is original


@pytest.mark.parametrize('identifier', IDS)
def test_bare_operator_identifiers_translate(monkeypatch, identifier):
    op = problem()
    original = ValueError(identifier)
    monkeypatch.setattr(op, '_call_op', raise_error(original))
    with pytest.raises(ValueError) as caught:
        coupled.linearize(op)
    assert str(caught.value).startswith(TARGET)
    assert caught.value.__cause__ is original

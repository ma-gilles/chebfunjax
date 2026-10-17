"""Native solvebvp selects linear/Newton routes after AD classification.

Source: @chebop/solvebvp.m109,198-241 at7574c77. These are authored
routing controls, separate from the fixed48 supplemental residual check and
from the literal adaptive original parameter problem.
"""
import pytest

import chebfunjax as cj
from chebfunjax.operators import _mixed_linear, parameter_newton
from chebfunjax.operators.chebop import Chebop, _LinopVar


def problem(nonlinear=True):
    operator = Chebop(
        (lambda x, u, a: (1-x**2)*u + .1*u.diff(2) + a*u.exp())
        if nonlinear else (lambda x, u, a: u.diff()-a)
    )
    operator.lbc = lambda u, a: [u+a+1, u.diff()]
    operator.rbc = lambda u, a: u-1
    operator.init = [cj.chebfun(lambda x: x), -1.]
    return operator


def test_nonlinear_exp_reaches_parameter_newton_without_linear_prepare(monkeypatch):
    operator = problem()
    token = object()
    def forbidden(*args, **kwargs):
        pytest.fail('Nonlinear parameter problem entered linear preparation')
    monkeypatch.setattr(_mixed_linear, 'prepare', forbidden)
    monkeypatch.setattr(parameter_newton, 'solve_parameter', lambda *a, **k: token)
    assert operator.solve(0., n=48) is token


def test_linear_parameter_classification_runs_once_before_mixed_prepare(monkeypatch):
    operator = problem(False)
    events = []
    token, prepared = object(), object()
    def classify(self):
        events.append('classify')
        return True
    def prepare(op):
        assert op is operator
        events.append('prepare')
        return prepared
    def solve(op, rhs, data, **kwargs):
        assert op is operator and data is prepared and kwargs['n'] == 48
        events.append('solve')
        return token
    monkeypatch.setattr(Chebop, '_system_is_linear', classify)
    monkeypatch.setattr(_mixed_linear, 'prepare', prepare)
    monkeypatch.setattr(_mixed_linear, 'solve', solve)
    assert operator.solve(0., n=48) is token
    assert events == ['classify', 'prepare', 'solve']


@pytest.mark.parametrize('where', ['operator', 'left', 'right'])
def test_callback_value_errors_remain_visible(where, monkeypatch):
    operator = problem(False)
    message = 'intentional callback failure '+where
    if where == 'operator':
        original = operator.op
        def fail(x, u, a):
            if isinstance(u, _LinopVar):
                raise ValueError(message)
            return original(x, u, a)
        operator.op = fail
    else:
        def fail(u, a):
            raise ValueError(message)
        if where == 'left':
            operator.lbc = fail
        else:
            operator.rbc = fail
    def forbidden(*args, **kwargs):
        pytest.fail('A callback failure was converted to a solver fallback')
    monkeypatch.setattr(parameter_newton, 'solve_parameter', forbidden)
    with pytest.raises(ValueError, match=message):
        operator.solve(0., n=48)

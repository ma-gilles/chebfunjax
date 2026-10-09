"""Permanent source parameter-routing regression controls."""
import pytest

from chebfunjax.operators._coupled_newton import (
    CoupledParameterVariables,
    linearize,
    supported,
)
from chebfunjax.operators.chebop import Chebop


def test_fewer_equation_parameter_uses_existing_solver(monkeypatch):
    N = Chebop(lambda x, u, p: u.diff()-p)
    N.bc = lambda x, u, p: [u(-1), u(1)-1]
    assert not supported(N)
    token = object()
    monkeypatch.setattr(Chebop, '_system_is_linear', lambda self: True)
    monkeypatch.setattr(Chebop, '_solve_linear_system', lambda self, *a, **k: token)
    assert N.solve(0) is token


def test_exact_parameter_columns_delegate_public_and_solver(monkeypatch):
    N = Chebop(lambda x, u, p: [u.diff()+p, u-p])
    N.bc = lambda x, u, p: [u(-1), u(1)-1]
    assert supported(N)  # count heuristic alone cannot recognize this column
    with pytest.raises(CoupledParameterVariables):
        linearize(N)
    token = object()
    monkeypatch.setattr(Chebop, 'linop', lambda self: token)
    monkeypatch.setattr(Chebop, '_system_is_linear', lambda self: True)
    monkeypatch.setattr(Chebop, '_solve_linear_system', lambda self, *a, **k: token)
    assert N.linearize()[0] is token
    assert N.solve(0) is token

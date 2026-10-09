"""Six literal source predicates for public cell-style Brusselator IVPs.

Provenance: tests/chebop/test_ivp_chebmatrix_syntax.m and @chebmatrix/norm.m,
Chebfun7574c77680d7e82b79626300bf255498271a72df. Python SystemSolution is
converted only at the container boundary to a column ChebMatrix; all block
subtraction and continuous norms use the public library. The native marcher
provider is R2025b, not a new MATLAB2017 runtime capture.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref, ChebopPref
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop

DOM = (0., 5.)
TOL = 1e-14


@pytest.fixture(autouse=True)
def factory_preferences(monkeypatch):
    monkeypatch.setattr(ChebfunPref, '_defaults', None)
    monkeypatch.setattr(ChebopPref, '_defaults', None)
    assert ChebopPref().ivpSolver == 'ode113'
    yield


def _multiple():
    return Chebop(lambda t, u, v, w: [
        u.diff()-1-u**2*v+(w+1)*u,
        v.diff()-u*w+u**2*v,
        w.diff()-1.5+w*u], DOM)


def _cell():
    return Chebop(lambda t, u: [
        u[0].diff()-1-u[0]**2*u[1]+(u[2]+1)*u[0],
        u[1].diff()-u[0]*u[2]+u[0]**2*u[1],
        u[2].diff()-1.5+u[2]*u[0]], DOM)


def _column(solution):
    return ChebMatrix([[component] for component in solution], domain=DOM)


def _endpoint(solution, x):
    # Source FEVAL of a column ChebMatrix stacks component evaluations.
    return jnp.stack([component(x) for component in solution])


def _solve(n):
    solution = n.solve(0)
    assert n._ivp_backend_used == 'native_ode113'
    return solution


def test_all_matlab_assertions():
    n = _multiple()
    n.lbc = [1, 3, 4]
    u_left = _solve(n)
    n.lbc = None
    assert float(jnp.linalg.norm(_endpoint(u_left, DOM[0])-jnp.asarray([1, 3, 4]))) < TOL
    print("native predicate 1 passed", flush=True)
    n.lbc = None
    n.rbc = [.7, .9, 1.2]
    u_right = _solve(n)
    assert float(jnp.linalg.norm(_endpoint(u_right, DOM[-1])-jnp.asarray([.7, .9, 1.2]))) < TOL
    print("native predicate 2 passed", flush=True)

    m = _cell()
    m.lbc = lambda u: [u[0]-1, u[1]-3, u[2]-4]
    v_left1 = _solve(m)
    assert (_column(v_left1)-_column(u_left)).norm() == 0
    print("native predicate 3 passed", flush=True)
    m.lbc = [1, 3, 4]
    v_left2 = _solve(m)
    assert (_column(v_left2)-_column(u_left)).norm() == 0
    print("native predicate 4 passed", flush=True)
    m.lbc = None
    m.rbc = lambda u: [u[0]-.7, u[1]-.9, u[2]-1.2]
    v_right1 = _solve(m)
    assert (_column(v_right1)-_column(u_right)).norm() == 0
    print("native predicate 5 passed", flush=True)
    m.rbc = [.7, .9, 1.2]
    v_right2 = _solve(m)
    assert (_column(v_right2)-_column(u_right)).norm() == 0
    print("native predicate 6 passed", flush=True)

"""Independent backend selection controls for MATLAB @chebop/solvebvp.m,7574c77."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_linear_system_uses_requested_backend_without_default_seed(monkeypatch, backend):
    # The polynomial solution u=x,v=1 is independently known. Record all
    # public solve calls to detect a hidden default-discretization seed.
    N = Chebop(lambda x, u, v: [u.diff()-v, v.diff()], (-1.0, 1.0))
    N.lbc = lambda u, v: [u+1, v-1]
    original = N.solve
    calls = []
    def recorded(*args, **kwargs):
        calls.append(kwargs.get('discretization'))
        return original(*args, **kwargs)
    monkeypatch.setattr(N, 'solve', recorded)
    solution = N.solve(0, n=9, discretization=backend)
    assert calls == [backend], 'Requested backend must not silently use a default solve'
    x = chebfun(lambda t: t)
    assert (solution[0]-x).norm(jnp.inf) < 1e-10
    assert (solution[1]-1).norm(jnp.inf) < 1e-10


@pytest.mark.parametrize('initial, expected', [(None, False), ([1, 1], False)])
def test_native_product_linearity_depends_on_initial_state(initial, expected):
    """Source operatorBlock.mult retains structural dependence at zero values.

    In MATLAB 7574c77 operatorBlock.m397-424 a Chebfun multiplier never sets
    iszero, even when its function value is zero. Thus AD times must flag
    u*v nonlinear at both initial states; a zero numerical derivative does
    not make the operator structurally constant.
    """
    from chebfunjax.operators.chebop_altdisc import _linear_system_by_ad

    N = Chebop(lambda x, u, v: [u*v, v.diff()], (-1, 1))
    N.init = initial
    assert _linear_system_by_ad(N, (-1., 1.), 2) is expected


def test_nonlinear_operator_keeps_default_seed(monkeypatch):
    """A nondegenerate nonlinear operator retains the established seed route."""
    N = Chebop(lambda x, u, v: [u.sin(), v], (-1, 1))
    calls = []
    original = N.solve

    def recorded(*args, **kwargs):
        calls.append(kwargs.get('discretization'))
        if kwargs.get('discretization') is None:
            return [chebfun(0.), chebfun(0.)]
        return original(*args, **kwargs)

    monkeypatch.setattr(N, 'solve', recorded)
    N.solve(0, n=9, discretization='ultraS')
    assert calls == ['ultraS', None]


def test_supplied_initial_state_used_for_linearization():
    from chebfunjax.operators.chebop_altdisc import _linear_system_by_ad

    seen = []

    def operation(x, u, v):
        seen.append((float(u.func(0.)), float(v.func(0.))))
        return [u+v, v.diff()]

    N = Chebop(operation, (-1, 1))
    N.init = [2., 3.]
    assert _linear_system_by_ad(N, (-1., 1.), 2)
    assert seen == [(2., 3.)]

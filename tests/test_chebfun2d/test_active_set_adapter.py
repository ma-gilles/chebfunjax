"""Fixed native caller adapter; transaction failure controls live beside this."""
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d import _active_set_source as adapter
from chebfunjax.chebfun2d._extrema_fallback import ActiveOptions


def test_adapter_keeps_options_and_returns_nonpositive_status(monkeypatch):
    seen=[]
    point=jnp.asarray([.25, -.5])
    value=jnp.asarray(2.)
    def solver(fun, initial, lower, upper, **kwargs):
        seen.append((fun, initial, lower, upper, kwargs))
        return SimpleNamespace(x=point, value=value, exitflag=0)
    monkeypatch.setattr(adapter, 'active_set_box', solver)
    def fun(x):
        return jnp.sum(x)
    lo=jnp.asarray([-1., -1.])
    hi=-lo
    result=adapter.native_active_set(fun, point, lo, hi, ActiveOptions())
    assert result.x is point and result.fun is value
    assert all(a is b for a,b in zip(seen[0][:4], (fun, point, lo, hi)))
    assert seen[0][4] == {'finite_difference': adapter.finite_difference}


@pytest.mark.parametrize('field,value', [('display','iter'), ('algorithm','sqp'), ('tol_fun',1e-6), ('tol_x',1e-6)])
def test_adapter_rejects_other_preferences(field,value):
    options=dict(display='none', algorithm='active-set',tol_fun=2.**-52,tol_x=2.**-52)
    options[field]=value
    with pytest.raises(ValueError,match='Unsupported native extrema'):
        adapter.native_active_set(None,None,None,None,SimpleNamespace(**options))


def test_unsupported_branch_error_reaches_transaction_observer(monkeypatch):
    def solver(*args,**kwargs):
        raise RuntimeError('Native singular-set RNG stream is required by qpsub640')
    monkeypatch.setattr(adapter,'active_set_box',solver)
    with pytest.raises(RuntimeError,match='singular-set RNG'):
        adapter.native_active_set(None,None,None,None,ActiveOptions())

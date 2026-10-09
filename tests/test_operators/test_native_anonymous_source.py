"""Independent native anonymous assignment controls, source pin7574c77.

Source: @chebop/chebop.m setters, vectorizeOp.m and parseBC.m.
Small AD actions only; source6's public negative solve is tested separately.
"""
import inspect

import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun, ADMatrixDimensionError
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


@pytest.fixture
def u():
    return ADChebfun(chebfun(lambda x: x+2, domain=(0., 1.), n=2))


def test_default_and_signature(u):
    n = Chebop(Chebop.nativeAnonymous('@(u) u*diff(u)'))
    assert n.vectorize is True
    assert list(inspect.signature(n.op).parameters) == ['u']
    assert float(n.op(u).func(.5)) == 2.5


def test_assignment_flag_snapshot(u):
    tag = Chebop.nativeAnonymous('@(u) u*diff(u)')
    n = Chebop(tag)
    n.vectorize = False
    assert float(n.op(u).func(.5)) == 2.5
    n.op = tag
    n.vectorize = True
    with pytest.raises(ADMatrixDimensionError):
        n.op(u)


@pytest.mark.parametrize('flag', [False, True])
def test_explicit_dotted(flag, u):
    n = Chebop()
    n.vectorize = flag
    n.op = Chebop.nativeAnonymous('@(u) u.*diff(u)')
    assert float(n.op(u).func(.5)) == 2.5


def test_captured_bindings(u):
    workspace = {'a': 3., 'f': chebfun(lambda x: x+1, domain=(0., 1.), n=2)}
    tag = Chebop.nativeAnonymous('@(u) a*u+f.*u', workspace)
    workspace['a'] = 99.
    workspace['f'] = 0.
    n = Chebop(tag)
    assert abs(float(n.op(u).func(.5))-11.25) < 2e-13


@pytest.mark.parametrize('expression', ['@(u) 3*u', '@(u) u*3'])
def test_numeric_scaling(expression, u):
    n = Chebop()
    n.vectorize = False
    n.op = Chebop.nativeAnonymous(expression)
    assert float(n.op(u).func(.5)) == 7.5


@pytest.mark.parametrize('zero', [False, True])
def test_real_kernel_failure(zero, monkeypatch):
    u = ADChebfun(chebfun(lambda x: 0*x if zero else x+2, n=2))
    calls = []
    old = ADChebfun.mtimes
    def record(self, other):
        calls.append((self, other))
        return old(self, other)
    monkeypatch.setattr(ADChebfun, 'mtimes', record)
    n = Chebop()
    n.vectorize = False
    n.op = Chebop.nativeAnonymous('@(u) u*diff(u)')
    with pytest.raises(ValueError) as err:
        n.op(u)
    assert err.value.identifier == 'CHEBFUN:ADCHEBFUN:mtimes:dims'
    assert len(calls) == 1


@pytest.mark.parametrize('which', ['bc', 'lbc', 'rbc'])
def test_bc_assignment_snapshot(which, u):
    n = Chebop(lambda u: u)
    tag = Chebop.nativeAnonymous('@(u) u*diff(u)')
    setattr(n, which, tag)
    callback = getattr(n, which)
    n.vectorize = False
    assert float(callback(u).func(.5)) == 2.5
    setattr(n, which, tag)
    with pytest.raises(ADMatrixDimensionError):
        getattr(n, which)(u)


def test_vectorize_method_and_plain_callable(u):
    def fun(u):
        return u*u.diff()
    assert Chebop.vectorizeOp(fun) is fun
    n = Chebop(fun)
    n.vectorize = False
    n.op = fun
    assert n.op is fun
    assert float(n.op(u).func(.5)) == 2.5
    tagged = Chebop.nativeAnonymous('@(u) u*diff(u)')
    assert float(Chebop.vectorizeOp(tagged)(u).func(.5)) == 2.5


def test_legacy_string_unchanged(u):
    n = Chebop('u.*diff(u)')
    assert float(n.op(u).func(.5)) == 2.5


@pytest.mark.parametrize('expr', ['@(u) u/2', '@(u) u^2', '@(u) u**2',
                                  '@(u) u.foo', '@(u) u[0]', '@(u) missing*u',
                                  '@(u) u@u', '@(u) u~u'])
def test_unsupported_explicit(expr):
    with pytest.raises(ValueError, match='unsupported'):
        Chebop.nativeAnonymous(expr)


def test_bc_vector(u):
    n = Chebop(domain=(0., 1.))
    n.bc = Chebop.nativeAnonymous('@(x,u) [u(0)-2;u(1)-3]')
    assert all(float(v.func) == 0 for v in n.bc(jnp.asarray(.5), u))


@pytest.fixture
def restored_preferences():
    from chebfunjax.chebpref import ChebopPref
    saved = ChebopPref._defaults
    try:
        yield ChebopPref
    finally:
        ChebopPref._defaults = saved


@pytest.mark.parametrize('flag', [False, True])
def test_global_constructor_preference(flag, restored_preferences, u):
    prefs = restored_preferences
    prefs.setDefaults('vectorize', flag)
    n = Chebop(Chebop.nativeAnonymous('@(u) u*diff(u)'))
    assert n.vectorize is flag
    if flag:
        assert float(n.op(u).func(.5)) == 2.5
    else:
        with pytest.raises(ADMatrixDimensionError):
            n.op(u)


def test_global_change_keeps_existing_instance(restored_preferences, u):
    prefs = restored_preferences
    prefs.setDefaults('vectorize', False)
    n = Chebop()
    prefs.setDefaults('vectorize', True)
    n.op = Chebop.nativeAnonymous('@(u) u*diff(u)')
    assert n.vectorize is False
    with pytest.raises(ADMatrixDimensionError):
        n.op(u)
    fresh = Chebop(Chebop.nativeAnonymous('@(u) u*diff(u)'))
    assert float(fresh.op(u).func(.5)) == 2.5


def test_explicit_instance_override_and_factory(restored_preferences, u):
    prefs = restored_preferences
    prefs.setDefaults('vectorize', False)
    n = Chebop()
    n.vectorize = True
    n.op = Chebop.nativeAnonymous('@(u) u*diff(u)')
    assert float(n.op(u).func(.5)) == 2.5
    prefs.setDefaults('vectorize', 'factory')
    assert prefs().vectorize is True
    assert Chebop().vectorize is True

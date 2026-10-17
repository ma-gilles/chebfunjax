"""Supported scalar source AD routing; legacy assembly remains available."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('affine,initial', [(False, False), (True, False), (True, True)])
def test_public_local_source_route_and_manufactured_solution(monkeypatch, affine, initial):
    domain = (-.5, 1.5)
    x = chebfun(lambda t: t, domain=domain)
    if affine:
        op = Chebop(lambda x, u: u.diff(2)+u+x.exp(), domain)
        exact = x**2
        rhs = 2+x**2+x.exp()
    else:
        op = Chebop(lambda x, u: -((2+x*x)*u.diff()).diff()+(1+x)*u, domain)
        exact = 1+2*x-x*x+3*x**3
        rhs = -(2*x*(2-2*x+9*x*x)+(2+x*x)*(-2+18*x))+(1+x)*exact
    op.lbc, op.rbc = float(exact(domain[0])), float(exact(domain[-1]))
    if initial:
        op.init = 2*x+3

    def legacy(*args, **kwargs):
        raise AssertionError('Supported scalar linear AD expression entered legacy assembly')
    monkeypatch.setattr(op, '_linearize_op', legacy)
    result = op.solve(rhs)
    assert (result-exact).norm(jnp.inf) < 1e-10
    assert (op(result)-rhs).norm(jnp.inf) < 1e-10
    assert jnp.abs(result(domain[0])-op.lbc) < 1e-10
    assert jnp.abs(result(domain[-1])-op.rbc) < 1e-10


@pytest.mark.parametrize('capability', ['_coordinate_fn', '_values_capability'])
def test_missing_local_capability_preserves_legacy_route(monkeypatch, capability):
    import chebfunjax.operators.chebop_altdisc as module
    op = Chebop(lambda u: u.diff(2)+u, (-1., 1.), 0., 0.)
    original = module._linearize_scalar_ad
    calls = []

    def unsupported(*args, **kwargs):
        data = original(*args, **kwargs)
        setattr(data[0].A.blocks[0][0], capability, None)
        return data

    class Legacy:
        def solve(self, rhs, **kwargs):
            calls.append(kwargs)
            return 'legacy sentinel'

    monkeypatch.setattr(module, '_linearize_scalar_ad', unsupported)
    monkeypatch.setattr(op, '_build_linop', lambda: Legacy())
    assert op._solve_linear(0., n=12, n_min=8, n_max=64, tol=5e-13,
                            _known_local=True) == 'legacy sentinel'
    assert calls == [dict(n=12, n_min=8, n_max=64, tol=5e-13)]


@pytest.mark.parametrize('failure', ['unsupported', 'nonlinear'])
def test_ad_extraction_failure_preserves_legacy_route(monkeypatch, failure):
    import chebfunjax.operators.chebop_altdisc as module
    op = Chebop(lambda u: u.diff(2), (-1., 1.), 0., 0.)

    def unavailable(*args, **kwargs):
        if failure == 'unsupported':
            raise NotImplementedError('Unsupported AD expression')
        return None

    class Legacy:
        def solve(self, rhs, **kwargs):
            return 'legacy sentinel'

    monkeypatch.setattr(module, '_linearize_scalar_ad', unavailable)
    monkeypatch.setattr(op, '_build_linop', lambda: Legacy())
    assert op._solve_linear(0., n=None, n_min=32, n_max=4096, tol=5e-13,
                            _known_local=True) == 'legacy sentinel'


def test_existing_nonlocal_source_ad_route_preserved(monkeypatch):
    op = Chebop(lambda u: u.diff(2)+u.sum(), (-1., 1.), 0., 0.)
    def legacy(*args, **kwargs):
        raise AssertionError('Existing nonlocal source AD route regressed')
    monkeypatch.setattr(op, '_linearize_op', legacy)
    result = op.solve(1.)
    exact = chebfun(lambda x: 1.5*(x*x-1))
    assert (result-exact).norm(jnp.inf) < 1e-10
    assert (op(result)-1).norm(jnp.inf) < 1e-10

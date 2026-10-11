"""Public discretization preference selection and tiny resolved solves.

Provenance
----------
MATLAB source: @chebop/determineDiscretization.m; @chebop/solvebvp.m173-178;
    @cheboppref/cheboppref.m parseDiscretization.
Chebfun commit: 7574c77
Route expectations are literal native selector branches, not candidate output.
"""
import math
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop


@pytest.fixture(autouse=True)
def restore_session():
    previous = ChebopPref._defaults
    ChebopPref.setDefaults('factory')
    try:
        yield
    finally:
        ChebopPref._defaults = previous


@pytest.mark.parametrize('session,override,periodic,breaks,expected', [
    ('coeffs', None, False, None, 'ultraS'),
    ('values', None, False, None, 'default'),
    ('values', None, True, None, 'default'),
    ('coeffs', None, True, None, 'trigspec'),
    ('coeffs', 'values', True, None, 'default'),
    ('values', 'coeffs', False, None, 'ultraS'),
    ('values', 'chebcolloc2', True, None, 'periodic_c2'),
    ('chebcolloc1', None, False, None, 'chebcolloc1'),
    ('ULTRASPHERICAL', None, False, None, 'ultraS'),
    ('values', 'COLLOC1', False, None, 'chebcolloc1'),
    ('values', 'collocation', True, None, 'periodic_c2'),
    ('values', 'periodic', True, None, 'default'),
    ('values', None, True, 'domain', 'default'),
    ('coeffs', None, True, 'domain', 'ultraS'),
    ('values', None, True, 'rhs', 'default'),
    ('coeffs', None, True, 'rhs', 'ultraS'),
    ('values', None, True, 'init', 'default'),
    ('coeffs', None, True, 'init', 'ultraS'),
])
def test_native_public_selector_routes(monkeypatch, session, override,
                                       periodic, breaks, expected):
    from chebfunjax.operators import chebop_altdisc
    from chebfunjax.operators import periodic as periodic_module
    ChebopPref.setDefaults('discretization', session)
    op = Chebop(lambda x, u: u, domain=(-1., 0., 1.) if breaks == 'domain' else (-1., 1.))
    if periodic:
        op.bc = 'periodic'
    if breaks == 'init':
        op.init = SimpleNamespace(domain=(-1., 0., 1.))
    rhs = SimpleNamespace(domain=(-1., 0., 1.)) if breaks == 'rhs' else 0.
    monkeypatch.setattr(Chebop, '_is_linear', lambda self: True)
    monkeypatch.setattr(Chebop, '_simplify_solution', lambda self, u: u)
    monkeypatch.setattr(Chebop, '_solve_impl', lambda self, *a, **kw: 'default')
    monkeypatch.setattr(Chebop, '_solve_periodic_colloc', lambda self, *a, **kw: 'periodic_c2')
    monkeypatch.setattr(chebop_altdisc, 'solve_bvp_altdisc', lambda op, f, backend, **kw: [backend])
    monkeypatch.setattr(periodic_module, 'solve_coefficients', lambda *a, **kw: 'trigspec')
    assert op.solve(rhs, discretization=override) == expected


@pytest.mark.parametrize('backend', ['trigcolloc', 'trigspec', 'periodic'])
@pytest.mark.parametrize('breaks', ['domain', 'rhs', 'init'])
def test_native_explicit_trig_rejects_breaks(backend, breaks):
    op = Chebop(lambda x, u: u, domain=(-1., 0., 1.) if breaks == 'domain' else (-1., 1.))
    op.bc = 'periodic'
    if breaks == 'init':
        op.init = SimpleNamespace(domain=(-1., 0., 1.))
    rhs = SimpleNamespace(domain=(-1., 0., 1.)) if breaks == 'rhs' else 0.
    with pytest.raises(ValueError, match='breakpointsInDomain'):
        op.solve(rhs, discretization=backend)


@pytest.mark.parametrize('selector', ['coeffs', 'chebcolloc1', 'values'])
def test_actual_dirichlet_session_backend(selector):
    ChebopPref.setDefaults('discretization', selector)
    op = Chebop(lambda x, u: u.diff(2))
    op.lbc = 1.
    op.rbc = 1.
    u = op.solvebvp(2., n=24)[0]
    x = jnp.linspace(-1., 1., 31)
    assert float(jnp.max(jnp.abs(u(x)-x*x))) < 1e-10


@pytest.mark.parametrize('selector,tech', [('values', 'Trigtech'),
                                         ('coeffs', 'Trigtech'),
                                         ('chebcolloc2', 'Chebtech2')])
def test_actual_periodic_session_backend(selector, tech):
    ChebopPref.setDefaults('discretization', selector)
    op = Chebop(lambda x, u: u.diff(2)-u, domain=(0., 2*math.pi))
    op.bc = 'periodic'
    u = op.solve(lambda x: -2*jnp.cos(x), n=32)
    x = jnp.linspace(0., 2*math.pi, 31)
    assert float(jnp.max(jnp.abs(u(x)-jnp.cos(x)))) < 1e-10
    assert type(u.funs[0].tech).__name__ == tech

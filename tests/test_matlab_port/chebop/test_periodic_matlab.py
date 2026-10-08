"""All 41 literal predicates in MATLAB tests/chebop/test_periodic.m.

Provenance
----------
MATLAB source : tests/chebop/test_periodic.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Source clause 4 is deliberately a self-subtraction; it proves no endpoint
agreement. A separate independent control checks actual endpoint agreement.
"""
from functools import lru_cache

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop

TOL = 1e4 * ChebopPref().bvpTol
PI = float(jnp.pi)


def _cos(x):
    """MATLAB cosine dispatch for both Chebfun and numeric operator coordinates."""
    return x.cos() if hasattr(x, "cos") else jnp.cos(x)


def _fun(op, dom, periodic=False):
    return cj.chebfun(op, domain=dom, trig=periodic)


@lru_cache(None)
def _case(case):
    if case == 1:
        op = Chebop(lambda u: u.diff(2) - u)
        rhs = cj.chebfun(lambda x: jnp.sin(PI*x))
        op.bc = lambda u: [u(-1)-u(1), u.diff()(-1)-u.diff()(1)]
        u = op.solve(rhs)
        op.bc = 'periodic'
        return {'error': (u-op.solve(rhs)).norm()}
    if case in (2, 3):
        dom = [-PI, 0., PI]
        op = Chebop(lambda x, u, v: [u-v.diff(), u.diff(2)+v], dom)
        op.bc = 'periodic'
        if case == 2:
            rhs = [_fun(0., dom), _fun(jnp.cos, dom)]
            uv = op.solve(rhs)
            exact = [_fun(lambda x: jnp.cos(x+3*PI/4)/jnp.sqrt(2.), dom),
                     _fun(lambda x: jnp.cos(x+PI/4)/jnp.sqrt(2.), dom)]
            error = ChebMatrix([[uv[0]-exact[0]], [uv[1]-exact[1]]]).norm()
            return {'error': error, 'u': uv}
        mass = Chebop(lambda x, u, v: [v+u, v.diff()], dom)
        vectors, values = op.eigs_generalized(mass, k=5, sort='SM')
        eigenvalues = values
        values = values[jnp.argsort(jnp.real(values))]
        first, last = values[:2], values[2:]
        values = jnp.concatenate((first[jnp.argsort(jnp.imag(first))],
                                  last[jnp.argsort(jnp.imag(last))]))
        return {'vectors': [lambda x, j=j: jnp.stack([v[j](x) for v in vectors])
                            for j in range(2)], 'values': values, 'eigenpairs': vectors, 'eigenvalues': eigenvalues}
    if case == 5:
        dom = [0., 2*PI]
        op = Chebop(lambda u: u.diff()+u, dom)
        rhs = _fun(jnp.cos, dom)
        exact = _fun(lambda x: .5*jnp.cos(x)+.5*jnp.sin(x), dom, True)
    elif case in (7, 33):
        dom = [-2*PI, 2*PI] if case == 7 else [0., 2*PI]
        op = Chebop(lambda x, u: u.diff()+(1+_cos(x))*u, dom)
        rhs = _fun(lambda x: jnp.cos(2*x), dom)
    elif case == 10:
        dom = [-2*PI, 2*PI]
        op = Chebop(lambda u: u.diff(2)+10*u.diff()+5*u, dom)
        rhs = _fun(jnp.cos, dom)
        exact = _fun(lambda x: jnp.cos(x)/29+5*jnp.sin(x)/58, dom, True)
    elif case in (12, 16, 21, 37):
        dom = [-PI, PI]
        a = _fun(lambda x: 2+jnp.cos((4 if case in (12, 37) else 1)*x), dom)
        b = _fun(lambda x: jnp.sin(jnp.cos(2*x)), dom, case in (12, 37))
        c = _fun(lambda x: jnp.exp(jnp.cos(x)), dom)
        d = _fun(jnp.sin, dom)
        e = _fun(lambda x: jnp.sin(2*x), dom)
        if case in (12, 37):
            op = Chebop(lambda u: a*u.diff(2)+b*u.diff()+c*u, dom)
        elif case == 16:
            op = Chebop(lambda u: a*u.diff(3)+b*u.diff(2)+c*u.diff()+d*u, dom)
        else:
            op = Chebop(lambda u: a*u.diff(4)+b*u.diff(3)+c*u.diff(2)+d*u.diff()+e*u, dom)
        rhs = _fun(lambda x: jnp.cos((10 if case == 21 else 1)*x), dom)
    elif case == 27:
        dom = [0., PI, 2*PI]
        op = Chebop(lambda u: u.diff()+u, dom)
        rhs = _fun(jnp.cos, dom)
    elif case == 30:
        dom = [-1., 1.]
        op = Chebop(lambda x, u: u.diff(2)+abs(x)*u, dom)
        rhs = 1.
    else:
        raise AssertionError(case)
    op.bc = 'periodic'
    if case in (33, 37):
        u, _ = op.solvebvp(rhs, discretization='trigspec' if case == 33 else 'coeffs')
    else:
        u = op.solve(rhs)
    out = {'u': u, 'op': op, 'rhs': rhs, 'dom': dom}
    if case in (5, 10):
        out['exact'] = exact
    return out


CASES = [(1, 1, 'error'), (2, 2, 'error'), (3, 3, 'eigenvalues'),
         (4, 3, 'source_self_subtraction')]
for start, order, tech in ((5, 1, 'Trigtech'), (7, 1, 'Trigtech'),
                            (10, 2, 'Trigtech'), (12, 2, 'Trigtech'),
                            (16, 3, 'Trigtech'), (21, 4, 'Trigtech'),
                            (27, 1, 'Chebtech2'), (30, 1, 'Chebtech2'),
                            (33, 1, 'Trigtech'), (37, 2, 'Trigtech')):
    if start in (5, 10):
        CASES.extend(((start, start, 'exact'), (start+1, start, tech)))
    else:
        CASES.append((start, start, 'residual'))
        CASES.extend((start+1+d, start, d) for d in range(order if start != 30 else 1))
        last = start+1+(order if start != 30 else 1)
        CASES.append((last, start, tech))
        if start in (33, 37):
            CASES.append((last+1, start, 'real'))


@pytest.mark.parametrize('slot,case,check', CASES, ids=[f'source_{s:02d}' for s, _, _ in CASES])
def test_source_predicate(slot, case, check):
    data = _case(case)
    if check == 'error':
        assert data['error'] < TOL
    elif check == 'eigenvalues':
        e = data['values']
        assert (jnp.linalg.norm(jnp.real(e)-jnp.array([0, 0, 1, 1, 1]), ord=jnp.inf)
                + jnp.linalg.norm(jnp.imag(e)-jnp.array([-1, 1, -1, 0, 1]), ord=jnp.inf)) < TOL
    elif check == 'source_self_subtraction':
        v = data['vectors']
        assert (jnp.linalg.norm(jnp.atleast_1d(v[0](PI)-v[0](PI)), ord=jnp.inf)
                + jnp.linalg.norm(jnp.atleast_1d(v[1](PI)-v[1](PI)), ord=jnp.inf)) < TOL
    elif check == 'exact':
        assert (data['u']-data['exact']).norm(jnp.inf) < TOL
    elif check == 'residual':
        assert (data['op']*data['u']-data['rhs']).norm() < TOL
    elif check in ('Trigtech', 'Chebtech2'):
        assert type(data['u'].funs[0].tech).__name__ == check
    elif check == 'real':
        assert data['u'].isreal()
    else:
        derivative = data['u'].diff(check) if check else data['u']
        assert abs(derivative(data['dom'][0])-derivative(data['dom'][-1])) < TOL


def test_independent_eigenfunction_endpoints():
    """Actual endpoint agreement, independent of source clause 4's typo."""
    vectors = _case(3)['vectors']
    assert sum(jnp.linalg.norm(v(PI)-v(-PI), ord=jnp.inf) for v in vectors) < TOL


def test_independent_piecewise_result_domains():
    for u in _case(2)['u']:
        assert tuple(u.domain.breakpoints) == (-PI, 0., PI)
    for pair in _case(3)['eigenpairs']:
        for u in pair:
            assert tuple(u.domain.breakpoints) == (-PI, 0., PI)


def test_independent_generalized_eigenfunction_residual():
    data = _case(3)
    for (u, v), value in zip(data['eigenpairs'], data['eigenvalues']):
        residual = ChebMatrix([[u-v.diff()-value*(v+u)],
                               [u.diff(2)+v-value*v.diff()]])
        assert residual.norm() < TOL


def test_independent_coefficient_dispatch(monkeypatch):
    def no_collocation(*args, **kwargs):
        raise AssertionError('Explicit trigspec must select coefficient assembly')
    monkeypatch.setattr(Chebop, '_solve_periodic', no_collocation)
    dom = [0., 2*PI]
    op = Chebop(lambda u: u.diff()+u, dom)
    op.bc = 'periodic'
    u = op.solve(_fun(jnp.cos, dom), discretization='trigspec')
    exact = _fun(lambda x: .5*jnp.cos(x)+.5*jnp.sin(x), dom, True)
    assert (u-exact).norm(jnp.inf) < TOL

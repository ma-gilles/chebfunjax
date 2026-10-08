# uses-numpy: source numeric vector infinity and matrix spectral/infinity norms
"""All 62 literal predicates from MATLAB tests/misc/test_diffmat.m.

All source inputs are deterministic. No RNG fixture is required.

Provenance
----------
MATLAB source : tests/misc/test_diffmat.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebpref import ChebopPref
from chebfunjax.discretization.chebcolloc import ChebColloc1, ChebColloc2
from chebfunjax.utils.diffmat import diffmat
from chebfunjax.utils.quadrature import chebpts_ab, legpts

TOL = 1e-10


def infnorm(a):
    return float(np.linalg.norm(np.asarray(a), ord=np.inf))


def nodes(n, kind, domain=(-1, 1)):
    if kind == 'leg':
        return legpts(n, interval=domain)[0]
    return chebpts_ab(n, *domain, kind=1 if kind == 'chebkind1' else 2)


@pytest.mark.parametrize('clause', range(1, 7))
def test_source_square(clause):
    c1, c2 = ChebColloc1(5), ChebColloc2(5)
    if clause == 1:
        a, b = diffmat(5), c2.diffmat()
    elif clause == 2:
        a, b = diffmat(5, [-2, 2]), c2.diffmat()/2
    elif clause == 3:
        a, b = diffmat(5, 2, [-2, 2]), c2.diffmat(2)/4
    elif clause == 4:
        a, b = diffmat(5, 'chebkind1'), c1.diffmat(1)
    elif clause == 5:
        a, b = diffmat(5, 3, [-.5, .5], 'chebkind1'), c1.diffmat(3)*8
    else:
        saved = ChebopPref()
        ChebopPref.setDefaults('discretization', ChebColloc1)
        try:
            a, b = diffmat(5, 3), c2.diffmat(3)
        finally:
            ChebopPref.setDefaults(saved)
    assert np.linalg.norm(np.asarray(a-b), ord=2) < TOL


@pytest.mark.parametrize('clause', [7, 8])
def test_source_legendre_periodic(clause):
    if clause == 7:
        x = legpts(6)[0]
        got = diffmat(6, 2, 'leg') @ (x**3+1)
        expected = 6*x
    else:
        # Literal source x = diff(dom)/2*(trigpts(N)-dom(1)).
        x = jnp.pi*(-1+2*jnp.arange(6)/6)
        got = diffmat(6, 3, 'periodic', [0, 2*jnp.pi]) @ jnp.sin(x)
        expected = -jnp.cos(x)
    assert infnorm(got-expected) < TOL


@pytest.mark.parametrize('clause', range(9, 33))
def test_source_rectangular(clause):
    if clause <= 26:
        group, offset = divmod(clause-9, 6)
        source, target, n = [('chebkind1', 'chebkind1', 5),
                             ('chebkind2', 'chebkind1', 5),
                             ('leg', 'leg', 6)][group]
        domain = (-1, 1) if offset == 0 else (-2, 7)
        m = n-3 if offset == 5 else n-2
        grids = [source] if source == target else [source, target]
        # Clauses 21/22 explicitly name both Legendre grids.
        if group == 2 and offset in (0, 1):
            grids = ['leg', 'leg']
        d = diffmat([m, n], 2, domain, *grids)
        if offset in (2, 3, 4):
            if offset == 2:
                dd = diffmat([n, -2], 2, domain, *grids)
            elif offset == 3:
                dd = diffmat(n, 2, 'rect', domain, *grids)
            else:
                dd = diffmat(n, 2, domain, 'rect', *grids)
            assert infnorm(d-dd) < TOL
            return
    else:
        source, target = [('leg', 'chebkind1'), ('chebkind1', 'leg'),
                          ('chebkind2', 'leg'), ('chebkind1', 'chebkind2'),
                          ('chebkind2', 'chebkind2'), ('leg', 'chebkind2')][clause-27]
        n, m, domain = 6, 4, (-2, 7)
        d = diffmat([m, n], 2, domain, source, target)
    x, y = nodes(n, source, domain), nodes(m, target, domain)
    assert infnorm(6*y-d@(x**3+1)) < TOL


@pytest.mark.parametrize('clause', range(33, 57))
def test_source_boundary_conditions(clause):
    group, offset = divmod(clause-33, 8)
    source, target = [('chebkind1', 'chebkind1'),
                      ('chebkind2', 'chebkind1'), ('leg', 'leg')][group]
    p, n, domain = (1 if offset < 2 else 2), 30, (-2, 7)
    m = n-p
    x, y = nodes(n, source, domain), nodes(m, target, domain)
    lo, hi = jnp.exp(-2.), jnp.exp(7.)
    left, right = [(['dirichlet'], []), ([], ['dirichlet']),
                   (['dirichlet'], ['dirichlet']),
                   (['dirichlet', 'neumann'], []),
                   ([], ['neumann', 'dirichlet']),
                   (['dirichlet'], ['neumann']),
                   (['neumann'], ['dirichlet']),
                   (['neumann'], ['sum']) if group == 0 else ([], ['sum', 'dirichlet'])][offset]
    left_values = [hi-lo if bc == 'sum' else lo for bc in left]
    right_values = [hi-lo if bc == 'sum' else hi for bc in right]
    rhs = jnp.concatenate((jnp.asarray(left_values), jnp.exp(y), jnp.asarray(right_values)))
    # Preserve source scalar/cell/empty spellings through strings/lists.
    if clause in (33, 41, 49):
        boundary = [left if clause == 41 else 'dirichlet']
    elif clause in (34, 50):
        boundary = [[], 'dirichlet']
    elif clause == 43:
        boundary = ['dirichlet', ['dirichlet']]
    elif clause == 51:
        boundary = ['dirichlet', 'dirichlet']
    elif clause == 54:
        boundary = ['dirichlet', 'neumann']
    else:
        boundary = [left, right]
    size = [n, -p] if clause in (41, 42, 49, 50) else [m, n]
    grids = [source] if source == target else [source, target]
    d = diffmat(size, p, domain, *grids, *boundary)
    solution = jnp.linalg.solve(d, rhs)
    factors = ([1, 1, 10, 100, 50, 20, 100, 50],
               [1, 1, 20, 100, 100, 100, 1000, 10],
               [1, 1, 10, 100, 100, 100, 100, 10])
    assert infnorm(solution-jnp.exp(x)) < factors[group][offset]*TOL


@pytest.mark.parametrize('clause', range(57, 63))
def test_source_exact_symmetry(clause):
    n = 3 if clause < 60 else 4
    grid = [[], ['chebkind1'], ['leg']][(clause-57) % 3]
    d = diffmat(n, *grid)
    assert infnorm(d+d[::-1, ::-1]) == 0

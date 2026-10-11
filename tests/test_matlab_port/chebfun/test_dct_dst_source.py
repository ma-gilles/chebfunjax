"""All 64 literal source DCT/DST slots, with their original norms and bounds.

Provenance
----------
MATLAB source : tests/chebfun/test_dct.m, tests/chebfun/test_dst.m
Chebfun commit: 7574c77
The source seed42 MT19937 stream is adapted using NumPy RandomState and Fortran
matrix fill order. A matched R2025b capture qualifies all six exact R/x/c
arrays bitwise; no constructor or transform consumes random draws here. See
docs/dct_dst_native_source_cpu_20261011.json for bound capture evidence.
"""
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax import Chebfun

EPS = np.finfo(float).eps


@lru_cache(None)
def inputs(family):
    rng = np.random.RandomState(42)
    n = 5 if family == 'dct' else 15
    size = 13 if family == 'dct' else 15
    r = rng.rand(n*n).reshape((n,n), order='F')
    x = rng.rand(size*2).reshape((size,2), order='F')
    c = rng.rand(10,1) + 1j*rng.rand(10,1)
    return jnp.asarray(r), jnp.asarray(x), jnp.asarray(c)


@lru_cache(None)
def matrix(family, n, kind):
    i = jnp.arange(n, dtype=jnp.float64)[:, None]
    j = jnp.arange(n, dtype=jnp.float64)[None, :]
    if family == 'dct':
        if kind == 1:
            a = jnp.cos((jnp.pi/(n-1)*i)*j)
            return a.at[:, jnp.asarray([0,n-1])].multiply(.5)
        if kind == 2:
            return jnp.cos((jnp.pi/n*i)*(j+.5))
        if kind == 3:
            return jnp.cos((jnp.pi/n*(i+.5))*j).at[:,0].multiply(.5)
        if kind == 4:
            return jnp.cos((jnp.pi/n*(i+.5))*(j+.5))
        if kind == 5:
            return jnp.cos(((jnp.pi*i)*j)/(n-.5))
        if kind == 6:
            return jnp.cos(((jnp.pi*i)*(j+.5))/(n-.5))
        if kind == 7:
            return jnp.cos(((jnp.pi*(i+.5))*j)/(n-.5))
        return jnp.cos(((jnp.pi*(i+.5))*(j+.5))/(n+.5))
    if kind == 1:
        return jnp.sin((jnp.pi/(n+1)*(i+1))*(j+1))
    if kind == 2:
        return jnp.sin((jnp.pi/n*(i+1))*(j+.5))
    if kind == 3:
        return jnp.sin((jnp.pi/n*(i+.5))*(j+1)).at[:,-1].multiply(.5)
    if kind == 4:
        return jnp.sin((jnp.pi/n*(i+.5))*(j+.5))
    if kind == 5:
        return jnp.sin((jnp.pi/(n+.5)*(i+1))*(j+1))
    if kind == 6:
        return jnp.sin((jnp.pi/(n+.5)*(i+1))*(j+.5))
    if kind == 7:
        return jnp.sin((jnp.pi/(n+.5)*(i+.5))*(j+1))
    return jnp.sin((jnp.pi/(n+.5)*(i+.5))*(j+.5))


@pytest.mark.parametrize('family', ['dct', 'dst'])
@pytest.mark.parametrize('clause', range(1,33))
def test_native_dct_dst_slot(family, clause):
    r, x, c = inputs(family)
    forward = getattr(Chebfun, family)
    inverse = getattr(Chebfun, 'i'+family)
    tol = (50 if family == 'dct' else 100)*EPS
    if clause <= 8:
        # Native deliberately repeats kind1 eight times, not kinds1:8.
        difference = inverse(forward(r,1),1)-r
        bound = (5 if family == 'dct' else 50)*EPS
    elif clause <= 16:
        kind = clause-8
        difference = matrix(family,x.shape[0],kind)@x-forward(x,kind)
        bound = tol if kind <= 3 else 10*tol
    elif clause <= 24:
        if family == 'dst' and clause == 24:
            # Native pass(24)=true; inverse8 assertion is commented out.
            assert True
            return
        kind = clause-16
        difference = jnp.linalg.solve(matrix(family,x.shape[0],kind),x)-inverse(x,kind)
        bound = tol
    else:
        kind = clause-24
        difference = forward(c,kind)-(forward(jnp.real(c),kind)+1j*forward(jnp.imag(c),kind))
        bound = tol
    error = float(jnp.linalg.norm(difference,ord=2))
    print(f'{family} source clause {clause}: error={error:.17g} bound={bound:.17g}',flush=True)
    assert error < bound

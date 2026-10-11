"""Original large symmetric Jacobi quadrature clauses.

Provenance
----------
MATLAB source: tests/misc/test_jacpts.m, clauses 23, 26, 27.
Chebfun commit: 7574c77.
Clause 24 is disabled upstream and receives no execution credit.
"""

import jax.numpy as jnp

from chebfunjax.utils.quadrature import jacpts


def test_native_clause_23():
    x, w = jacpts(200000, .2, .2)
    assert abs(jnp.sum(w @ x)) < 10*jnp.finfo(jnp.float64).eps


def test_native_clauses_26_27():
    x, w = jacpts(100000, 1., 1.)
    eps = jnp.finfo(jnp.float64).eps
    assert jnp.sum(w)-4/3 < 100*eps
    assert w @ (x*x)-4/15 < 10*eps

"""Port of MATLAB Chebfun tests/ballfun/test_diff.m (Fable 5).

Adversarial Cartesian-derivative sweep (the check that exposed the
diskfun calculus bugs). Original native predicates are below.

Provenance
----------
MATLAB source : tests/ballfun/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import X0, Y0, Z0, val

CASES = [
    (lambda x, y, z: x, (1.0, 0.0, 0.0)),
    (lambda x, y, z: y, (0.0, 1.0, 0.0)),
    (lambda x, y, z: z, (0.0, 0.0, 1.0)),
    (lambda x, y, z: x * x, (2 * X0, 0.0, 0.0)),
    (lambda x, y, z: x * y, (Y0, X0, 0.0)),
    (lambda x, y, z: z * z, (0.0, 0.0, 2 * Z0)),
    (lambda x, y, z: x * y * z, (Y0 * Z0, X0 * Z0, X0 * Y0)),
]


class TestBallfunDiff:
    @pytest.mark.parametrize("i", range(len(CASES)))
    def test_cartesian_partials(self, i):
        op, want = CASES[i]
        f = Ballfun.from_function(op)
        for dim in (1, 2, 3):
            got = val(f.diff(dim))
            assert abs(got - want[dim - 1]) < 1e-7, (i, dim)


@pytest.mark.parametrize("op,dim,order,exact,spherical", [
    (lambda r, lam, th: r*jnp.sin(th)*jnp.cos(lam), 1, 1,
     lambda r, lam, th: 1, True),
    (lambda r, lam, th: r**2*jnp.sin(th)**2*jnp.cos(lam)**2, 1, 1,
     lambda r, lam, th: 2*r*jnp.sin(th)*jnp.cos(lam), True),
    (lambda r, lam, th: r*jnp.sin(th)*jnp.sin(lam), 2, 1,
     lambda r, lam, th: 1, True),
    (lambda r, lam, th: r**2*jnp.sin(th)**2*jnp.sin(lam)**2, 2, 1,
     lambda r, lam, th: 2*r*jnp.sin(th)*jnp.sin(lam), True),
    (lambda r, lam, th: r**2*jnp.sin(th)**2*jnp.sin(lam)**2, 2, 2,
     lambda r, lam, th: 2, True),
    (lambda r, lam, th: r*jnp.cos(th), 3, 1,
     lambda r, lam, th: 1, True),
    (lambda r, lam, th: r**2*jnp.cos(th)**2, 3, 2,
     lambda r, lam, th: 2, True),
    (lambda x, y, z: x**2+y**2+z**2, 1, 1,
     lambda x, y, z: 2*x, False),
])
def test_original_diff(op, dim, order, exact, spherical):
    from chebfunjax.chebpref import ChebfunPref
    tol = 1e2 * ChebfunPref().techPrefs.chebfuneps
    f = Ballfun.from_function(op, spherical=spherical)
    expected = Ballfun.from_function(exact, spherical=spherical)
    assert (f.diff(dim, order) - expected).norm() < tol

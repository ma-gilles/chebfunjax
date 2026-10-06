"""Independent complex-data GMRES controls using an independent exact solution.

Provenance
----------
MATLAB source: @chebop/gmres.m, @cheboppref/cheboppref.m;
Chebfun commit 7574c77. The original source GMRES tests use real data, so
these are independent complex-arithmetic controls, not copied MATLAB asserts.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.krylov import gmres

_PREF = ChebopPref()
_TOL = _PREF.bvpTol
_BOUND = 100.0 * _TOL
_F = 1.0 + 2.0j
_C = 2.0 + 1.0j
_D = -1.0 + 0.5j


def _complex_problem(nonzero_complex_bc):
    n = Chebop(lambda x, u: -u.diff(2) + u, domain=(-1.0, 1.0))
    f = cj.chebfun(lambda x: _F * (1.0 - 3.0 * x**2), domain=(-1.0, 1.0))
    if nonzero_complex_bc:
        ch, sh = jnp.cosh(1.0), jnp.sinh(1.0)
        n.lbc = -8.0 * _F + _C * ch - _D * sh
        n.rbc = -8.0 * _F + _C * ch + _D * sh

        def exact(x):
            return _F * (-5.0 - 3.0 * x**2) + _C * jnp.cosh(x) + _D * jnp.sinh(x)
    else:
        n.lbc = 0.0
        n.rbc = 0.0

        def exact(x):
            return _F * (-5.0 - 3.0 * x**2 + 8.0 * jnp.cosh(x) / jnp.cosh(1.0))

    return n, f, cj.chebfun(exact, domain=(-1.0, 1.0))


@pytest.mark.parametrize(
    "nonzero_complex_bc", [False, True], ids=["real-zero-bcs", "complex-double-bcs"]
)
def test_gmres_complex_rhs_and_double_boundary_data(nonzero_complex_bc):
    n, f, exact = _complex_problem(nonzero_complex_bc)
    solution, flag, relres, iteration, resvec = gmres(
        n, f, restart=None, tol=_TOL, maxit=40, full_output=True
    )

    assert flag == 0
    assert len(iteration) == 2
    assert float((solution - exact).norm(2)) < _BOUND
    assert float((n.feval(solution) - f).norm(2)) < _BOUND
    assert bool(jnp.isfinite(relres))
    assert bool(jnp.all(jnp.isfinite(resvec)))

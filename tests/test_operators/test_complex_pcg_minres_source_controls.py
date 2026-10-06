"""Independent analytic controls for complex scalar PCG/MINRES data.

Provenance
----------
MATLAB sources: @chebop/pcg.m, @chebop/minres.m,
@cheboppref/cheboppref.m; Chebfun commit 7574c77.

These are independent closed-form controls, not original MATLAB assertions:
the pinned original PCG/MINRES tests use real data. They check the complex
``double`` BC type branch and the complex Chebfun Krylov scalar path.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.krylov import minres, pcg

_PREF = ChebopPref()
_TOL = _PREF.bvpTol
_BOUND = 100.0 * _TOL
_F = 1.0 + 2.0j
_C = 2.0 + 1.0j
_D = -1.0 + 0.5j


def _problem(complex_bc: bool):
    n = Chebop(lambda x, u: -u.diff(2) + u, domain=(-1.0, 1.0))
    f = cj.chebfun(lambda x: _F * (1.0 - 3.0 * x**2), domain=(-1.0, 1.0))
    if complex_bc:
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


@pytest.mark.parametrize("solver", [pcg, minres], ids=["pcg", "minres"])
def test_complex_forcing_with_real_zero_dirichlet_data(solver):
    n, f, exact = _problem(complex_bc=False)
    u, flag, relres, _, _ = solver(
        n, f, tol=_TOL, maxit=40, full_output=True
    )
    assert flag == 0
    assert float((u - exact).norm(2)) < _BOUND
    assert float((n.feval(u) - f).norm(2)) < _BOUND
    assert bool(jnp.isfinite(relres))


@pytest.mark.parametrize("solver", [pcg, minres], ids=["pcg", "minres"])
def test_complex_double_dirichlet_data_and_forcing(solver):
    n, f, exact = _problem(complex_bc=True)
    u, flag, relres, _, _ = solver(
        n, f, tol=_TOL, maxit=40, full_output=True
    )
    assert flag == 0
    assert float((u - exact).norm(2)) < _BOUND
    assert float((n.feval(u) - f).norm(2)) < _BOUND
    assert bool(jnp.isfinite(relres))


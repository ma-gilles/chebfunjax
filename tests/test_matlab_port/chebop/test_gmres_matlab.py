"""Port of all seven assertions in MATLAB tests/chebop/test_gmres.m.

Provenance
----------
MATLAB source : tests/chebop/test_gmres.m
Chebfun commit: 7574c77

Each assertion uses the continuous Chebfun L2 norm and unchanged
100*cheboppref.bvpTol bound. Source preference tolerance and iteration cap
are passed explicitly; numerical evidence is recorded by the CPU harness.
"""

from __future__ import annotations

import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop

GMRES_PREF = ChebopPref()
TOL = 1e2 * GMRES_PREF.bvpTol
GMRES_TOL = GMRES_PREF.bvpTol
GMRES_MAXIT = GMRES_PREF.maxIter


def _op(a, c, domain=(-1.0, 1.0), b=None):
    if b is None:
        op = lambda x, u: -(a(x) * u.diff()).diff() + c(x) * u
    else:
        op = lambda x, u: (-(a(x) * u.diff()).diff()
                           + b(x) * u.diff() + c(x) * u)
    return Chebop(op, domain=domain)


def _error(u, v):
    return float((u - v).norm(2))


def test_gmres_pass1_domain_zero_to_one():
    dom = (0.0, 1.0)
    a = lambda x: 1.0 + 0 * x
    c = lambda x: 1.0 + 0 * x
    n = _op(a, c, domain=dom)
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1 - 3 * x**2, domain=dom)
    assert _error(n.solve(f), n.gmres(f, tol=GMRES_TOL, maxit=GMRES_MAXIT)) < TOL


def test_gmres_pass2_variable_diffusion_zero_reaction():
    a = lambda x: 2 + (jnp.pi * x).cos()
    c = lambda x: 0 * x
    n = _op(a, c)
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1 - 3 * x**2)
    assert _error(n.solve(f), n.gmres(f, tol=GMRES_TOL, maxit=GMRES_MAXIT)) < TOL


def test_gmres_pass3_variable_reaction():
    a = lambda x: 2 + (jnp.pi * x).cos()
    c = lambda x: 1 + 10 * x**2
    n = _op(a, c)
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1 - 3 * x**2)
    assert _error(n.solve(f), n.gmres(f, tol=GMRES_TOL, maxit=GMRES_MAXIT)) < TOL


def test_gmres_pass4_piecewise_forcing_and_coefficient():
    a = lambda x: 2 + (jnp.pi * abs(x - 0.5)).cos()
    c = lambda x: 1 + 10 * x**2
    n = _op(a, c)
    n.bc = 0.0
    f = cj.chebfun(lambda x: jnp.abs(x) - 0.5, splitting=True)
    assert _error(n.solve(f), n.gmres(f, tol=GMRES_TOL, maxit=GMRES_MAXIT)) < TOL


def test_gmres_pass5_inhomogeneous_dirichlet():
    a = lambda x: 2 + (jnp.pi * x).cos()
    c = lambda x: 1 + 10 * x**2
    n = _op(a, c)
    n.lbc = 1.0
    n.rbc = -1.0
    f = cj.chebfun(lambda x: 1 - 2 * x**2)
    assert _error(n.solve(f), n.gmres(f, tol=GMRES_TOL, maxit=GMRES_MAXIT)) < TOL


def test_gmres_pass6_first_derivative_term():
    a = lambda x: 2 + (jnp.pi * x).cos()
    b = lambda x: 1.0 + 0 * x
    c = lambda x: -10 * x**2
    n = _op(a, c, b=b)
    n.lbc = 1.0
    n.rbc = -1.0
    f = cj.chebfun(lambda x: 1 - 2 * x**2)
    assert _error(n.solve(f), n.gmres(f, tol=GMRES_TOL, maxit=GMRES_MAXIT)) < TOL


def test_gmres_pass7_piecewise_variable_coefficients():
    a = lambda x: 2 + (jnp.pi * x).cos()
    b = lambda x: abs((2 * jnp.pi * x).cos())
    c = lambda x: -10 * x**2 + x + 1
    n = _op(a, c, b=b)
    n.lbc = 1.0
    n.rbc = -1.0
    f = cj.chebfun(lambda x: jnp.sin(jnp.pi * x))
    assert _error(n.solve(f), n.gmres(f, tol=GMRES_TOL, maxit=GMRES_MAXIT)) < TOL

"""Port of MATLAB tests/chebop/test_pcg.m.

Provenance
----------
MATLAB source : tests/chebop/test_pcg.m
Chebfun commit: 7574c77

The comparisons use the source continuous Chebfun L2 norm and its
unchanged `1e2 * cheboppref.bvpTol` bound. The source Krylov preference tolerance and iteration cap are passed explicitly
because the current Python wrapper defaults differ. Numerical validation is
recorded separately in the captured CPU gate.
"""

from __future__ import annotations

import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop

PREF = ChebopPref()
TOL = 1e2 * PREF.bvpTol


def _op(a, c, domain=(-1.0, 1.0)):
    return Chebop(lambda x, u: -(a(x) * u.diff()).diff() + c(x) * u,
                  domain=domain)


def _error(u, v):
    return float((u - v).norm(2))


def test_pcg_pass1_constant_coefficients():
    a = lambda x: 1.0 + 0 * x
    c = lambda x: 1.0 + 0 * x
    n = _op(a, c)
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1 - 3 * x**2)
    assert _error(n.solve(f), n.pcg(f, tol=PREF.bvpTol, maxit=PREF.maxIter)) < TOL


def test_pcg_pass2_variable_diffusion_zero_reaction():
    a = lambda x: 2 + (jnp.pi * x).cos()
    c = lambda x: 0 * x
    n = _op(a, c)
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1 - 3 * x**2)
    assert _error(n.solve(f), n.pcg(f, tol=PREF.bvpTol, maxit=PREF.maxIter)) < TOL


def test_pcg_pass3_variable_reaction():
    a = lambda x: 2 + (jnp.pi * x).cos()
    c = lambda x: 1 + 10 * x**2
    n = _op(a, c)
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1 - 3 * x**2)
    assert _error(n.solve(f), n.pcg(f, tol=PREF.bvpTol, maxit=PREF.maxIter)) < TOL


def test_pcg_pass4_piecewise_forcing_and_coefficient():
    a = lambda x: 2 + (jnp.pi * abs(x - 0.5)).cos()
    c = lambda x: 1 + 10 * x**2
    n = _op(a, c)
    n.bc = 0.0
    f = cj.chebfun(lambda x: jnp.abs(x) - 0.5, splitting=True)
    assert _error(n.solve(f), n.pcg(f, tol=PREF.bvpTol, maxit=PREF.maxIter)) < TOL


def test_pcg_pass5_inhomogeneous_dirichlet():
    a = lambda x: 2 + (jnp.pi * x).cos()
    c = lambda x: 1 + 10 * x**2
    n = _op(a, c)
    n.lbc = 1.0
    n.rbc = -1.0
    f = cj.chebfun(lambda x: 1 - 2 * x**2)
    assert _error(n.solve(f), n.pcg(f, tol=PREF.bvpTol, maxit=PREF.maxIter)) < TOL


def test_pcg_pass6_piecewise_reaction_inhomogeneous_bcs():
    a = lambda x: 2 + (jnp.pi * x).cos()
    c = lambda x: abs((2 * jnp.pi * x).cos())
    n = _op(a, c)
    n.lbc = 1.0
    n.rbc = -1.0
    f = cj.chebfun(lambda x: 1 - 2 * x**2)
    assert _error(n.solve(f), n.pcg(f, tol=PREF.bvpTol, maxit=PREF.maxIter)) < TOL


def test_pcg_pass7_nonstandard_domain():
    a = lambda x: 2 + (jnp.pi * x).cos()
    c = lambda x: abs((2 * jnp.pi * x).cos())
    dom = (-1.5, 2.0)
    n = _op(a, c, domain=dom)
    n.lbc = 3.0
    n.rbc = -1.0
    f = cj.chebfun(lambda x: 1 - 2 * x**2, domain=dom)
    assert _error(n.solve(f), n.pcg(f, tol=PREF.bvpTol, maxit=PREF.maxIter)) < TOL

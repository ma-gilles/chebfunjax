"""MATLAB parity tests for chebop pcg/minres (ode-linear/Krylov).

Provenance
----------
MATLAB source : @chebop/pcg.m, @chebop/minres.m
Chebfun commit: 7574c77
"""
import sys

import jax.numpy as jnp

sys.path.insert(0, "src")

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.krylov import minres, pcg


def test_pcg_poisson():
    # -u'' = 1, u(+-1) = 0  ->  u = (1 - x^2)/2
    L = Chebop(lambda u: -u.diff(2), domain=(-1, 1))
    L.bc = 0
    f = cj.chebfun(lambda t: jnp.ones_like(t), domain=(-1, 1))
    u = pcg(L, f)
    x = cj.chebfun(lambda t: t, domain=(-1, 1))
    assert float((u - (1 - x**2) * 0.5).norm()) < 1e-10


def test_pcg_inhomogeneous_bcs():
    # -u'' + u = 0 with u(-1) = 3, u(1) = -5 (shift path)
    L = Chebop(lambda u: -u.diff(2) + u, domain=(-1, 1))
    L.lbc = 3
    L.rbc = -5
    f = cj.chebfun(lambda t: 0.0 * t, domain=(-1, 1))
    u = pcg(L, f)
    assert abs(float(u(jnp.array(-1.0))) - 3) < 1e-8
    assert abs(float(u(jnp.array(1.0))) + 5) < 1e-8
    r = -u.diff(2) + u
    assert float(r.norm()) < 1e-6


def test_minres_indefinite():
    # -u'' - 20 u = f is indefinite; compare with backslash
    L = Chebop(lambda u: -u.diff(2) - 20 * u, domain=(-1, 1))
    L.bc = 0
    x = cj.chebfun(lambda t: t, domain=(-1, 1))
    f = (3 * x).sin()
    u = minres(L, f, tol=1e-11, maxit=200)
    Lb = Chebop(lambda u: -u.diff(2) - 20 * u, domain=(-1, 1))
    Lb.bc = 0
    u_ref = Lb.solve(f)
    assert float((u - u_ref).norm()) < 1e-7

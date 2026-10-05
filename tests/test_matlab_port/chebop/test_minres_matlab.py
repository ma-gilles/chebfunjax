"""Exact five-assertion translation of MATLAB test_minres.m.

Provenance
----------
MATLAB source: tests/chebop/test_minres.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

# @cheboppref/cheboppref.m factory defaults; the MATLAB test sets tol to
# 100*pref.bvpTol and MINRES uses pref.maxIter unless maxit is supplied.
_SOURCE_BVP_TOL = 5e-13
_SOURCE_DEFAULT_MAXIT = 25
_SOURCE_ASSERTION_TOL = 100 * _SOURCE_BVP_TOL


def _operator(a, c):
    """Build -(a(x)u')' + c(x)u on MATLAB's default domain [-1, 1]."""
    return Chebop(
        lambda x, u: -((a(x) * u.diff()).diff()) + c(x) * u,
        domain=(-1.0, 1.0),
    )


def _source_pair(a, c, rhs, left=0.0, right=0.0, maxit=_SOURCE_DEFAULT_MAXIT):
    op = _operator(a, c)
    op.lbc = left
    op.rbc = right
    f = cj.chebfun(rhs, domain=(-1.0, 1.0))
    reference = op.solve(f)
    # Python's current defaults are tol=1e-10/maxit=100; pass the MATLAB
    # factory preferences explicitly so the source call semantics are tested.
    actual = op.minres(f, tol=_SOURCE_BVP_TOL, maxit=maxit)
    return reference, actual


def _assert_source_l2(reference, actual):
    # MATLAB norm(chebfun,2) is the continuous L2 norm, not a sampled max norm.
    assert float((reference - actual).norm(2)) < _SOURCE_ASSERTION_TOL


def test_minres_source_case_1_constant_diffusion_zero_potential():
    a = lambda x: 1.0 + 0.0 * x
    c = lambda x: 0.0 * x
    f = lambda x: 1.0 - 3.0 * x**2
    u, v = _source_pair(a, c, f)
    _assert_source_l2(u, v)


def test_minres_source_case_2_variable_diffusion_linear_potential():
    a = lambda x: 2.0 + (jnp.pi * x).cos()
    c = lambda x: -10.0 * x
    f = lambda x: 1.0 - 3.0 * x**2
    u, v = _source_pair(a, c, f)
    _assert_source_l2(u, v)


def test_minres_source_case_3_variable_diffusion_quadratic_potential():
    a = lambda x: 2.0 + (jnp.pi * x).cos()
    c = lambda x: 1.0 - 10.0 * x**2
    f = lambda x: 1.0 - 3.0 * x**2
    u, v = _source_pair(a, c, f)
    _assert_source_l2(u, v)


def test_minres_source_case_4_nonzero_mean_rhs():
    a = lambda x: 2.0 + (jnp.pi * x).cos()
    c = lambda x: 1.0 + 10.0 * x**2
    f = lambda x: 1.0 - 2.0 * x**2
    u, v = _source_pair(a, c, f)
    _assert_source_l2(u, v)


def test_minres_source_case_5_nonzero_dirichlet_data_maxit_40():
    a = lambda x: 2.0 + (jnp.pi * x).cos()
    c = lambda x: 1.0 - 100.0 * x**2
    f = lambda x: 1.0 - 2.0 * x**2
    u, v = _source_pair(a, c, f, left=1.0, right=-1.0, maxit=40)
    _assert_source_l2(u, v)

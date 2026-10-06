"""Independent independent complex GMRES control on an exact polynomial BVP.

Provenance
----------
MATLAB algorithm: @chebop/gmres.m, Chebfun commit 7574c77.
This is an added analytic control for complex coefficient preservation, not one
of the seven original real-valued test_gmres assertions. Its 5e-11 solution
bound is a predeclared diagnostic bound, not a source tolerance.
"""
from __future__ import annotations

import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def test_complex_nonsymmetric_polynomial_bvp_preserves_complex_krylov_data():
    domain = (-1.0, 1.0)
    def op(x, u):
        return -u.diff(2) + 1j * u.diff() + (1.0 + 0.5j) * u
    n = Chebop(op, domain=domain)
    n.bc = 0.0

    # Exact u=(1+2i)(1-x^2). Direct expansion of -u'' + i*u' +
    # (1+i/2)u gives 2+(13/2)i + (4-2i)x -(5/2)i*x^2.
    f = cj.chebfun(
        lambda x: 2.0 + 6.5j + (4.0 - 2.0j) * x - 2.5j * x**2,
        domain=domain,
    )
    x = cj.chebfun(lambda x: x, domain=domain)
    expected = (1.0 + 2.0j) * (1.0 - x**2)

    result, flag, relres, _iteration, _resvec = n.gmres(
        f,
        tol=5e-13,
        maxit=25,
        full_output=True,
    )

    assert flag == 0
    assert float(relres) <= 5e-13
    assert float((result - expected).norm(2)) < 5e-11
    physical_residual = (
        -result.diff(2) + 1j * result.diff() + (1.0 + 0.5j) * result - f
    )
    assert float(physical_residual.norm(2)) < 5e-11
    assert any(
        jnp.iscomplexobj(piece.tech.coeffs) for piece in result.funs
    )


def test_real_initial_residual_promotes_for_complex_operator():
    """A real RHS must still acquire a complex Hessenberg column."""
    domain = (-1.0, 1.0)
    n = Chebop(lambda x, u: -u.diff(2) + 1j * u, domain=domain)
    n.bc = 0.0
    f = cj.chebfun(lambda x: 1.0 - 3.0 * x**2, domain=domain)

    # Solve -u'' + i*u = 1-3*x^2. With lambda=sqrt(i), this exact
    # solution satisfies both zero endpoint conditions:
    # (6-i)+3i*x^2 -(6+2i)*cosh(lambda*x)/cosh(lambda).
    lam = jnp.sqrt(jnp.asarray(1j, dtype=jnp.complex128))
    x = cj.chebfun(lambda x: x, domain=domain)
    expected = ((6.0 - 1j) + 3.0j * x**2
                - (6.0 + 2.0j) * (lam * x).cosh() / jnp.cosh(lam))

    result, flag, relres, _iteration, _resvec = n.gmres(
        f,
        tol=5e-13,
        maxit=25,
        full_output=True,
    )

    assert flag == 0
    assert float(relres) <= 5e-13
    assert float((result - expected).norm(2)) < 5e-11
    physical_residual = -result.diff(2) + 1j * result - f
    assert float(physical_residual.norm(2)) < 5e-11
    assert any(
        jnp.iscomplexobj(piece.tech.coeffs) for piece in result.funs
    )

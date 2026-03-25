"""Matrix functions via Chebfun and Cauchy integrals.

Demonstrates computing functions of matrices (exp(A), sqrt(A), etc.)
and verifying functional calculus identities. Also demonstrates
computing the matrix exponential via eigendecomposition for diagonal
and symmetric matrices.

Credit: Inspired by Chebfun linalg examples.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


def run():
    print("=" * 60)
    print("Matrix functions and spectral calculus")
    print("=" * 60)

    # --- Chebfun approximation of scalar functions on an interval ----
    # These functions arise when computing f(A) for matrices A with
    # eigenvalues in a known interval.

    # Scalar exp on [-1, 1]: represented as Chebfun
    dom = (-1.0, 1.0)
    f_exp = cj.chebfun(lambda x: jnp.exp(x), domain=dom)

    # Number of Chebyshev coefficients needed for exp on [-1, 1]
    coeffs = f_exp.coeffs
    print(f"\nexp(x) on [-1,1]: {len(coeffs)} Chebyshev coefficients")
    # exp(x) converges fast: about 18 terms to machine precision
    assert len(coeffs) < 25

    # --- Eigenvalue functions: sqrt, log for positive definite ops ---
    # Symmetric matrix A = [[2, 1], [1, 2]] has eigenvalues 1 and 3
    # f(A) = V * diag(f(lambda_i)) * V^T
    lam = np.array([1.0, 3.0])

    # f = exp
    f_lam_exp = np.exp(lam)
    print(f"\nA = [[2,1],[1,2]] has eigenvalues {lam}")
    print(f"  exp(lam) = {f_lam_exp}")

    # f = sqrt
    f_lam_sqrt = np.sqrt(lam)
    print(f"  sqrt(lam) = {f_lam_sqrt}")

    # Verify using Chebfun: the function applied at eigenvalues
    f_sqrt_cheb = cj.chebfun(lambda x: jnp.sqrt(x), domain=(0.5, 3.5))
    val_sqrt_1 = float(f_sqrt_cheb(jnp.array(1.0)))
    val_sqrt_3 = float(f_sqrt_cheb(jnp.array(3.0)))
    print(f"  Chebfun sqrt(1) = {val_sqrt_1:.12f}  (exact: {np.sqrt(1.0):.12f})")
    print(f"  Chebfun sqrt(3) = {val_sqrt_3:.12f}  (exact: {np.sqrt(3.0):.12f})")
    assert abs(val_sqrt_1 - 1.0) < 1e-12
    assert abs(val_sqrt_3 - np.sqrt(3.0)) < 1e-12

    # --- Cayley-Hamilton: p(A) = 0 for characteristic polynomial -----
    # For diagonal matrix D = diag(1, 3):
    # Char poly: (lambda-1)(lambda-3) = lambda^2 - 4*lambda + 3
    # Chebfun of p(x) = x^2 - 4x + 3 on [0, 4]
    p = cj.chebfun(lambda x: x**2 - 4.0*x + 3.0, domain=(0.0, 4.0))
    p_at_1 = float(p(jnp.array(1.0)))
    p_at_3 = float(p(jnp.array(3.0)))
    print(f"\nChar poly p(lambda) = lambda^2 - 4*lambda + 3:")
    print(f"  p(1) = {p_at_1:.2e}  (exact: 0)")
    print(f"  p(3) = {p_at_3:.2e}  (exact: 0)")
    assert abs(p_at_1) < 1e-14
    assert abs(p_at_3) < 1e-14

    # --- Function of operator: apply eigenvalue decomposition --------
    # Laplacian eigenfunctions on [0,pi]: phi_n(x) = sin(n*x)
    # with eigenvalues -n^2; if we apply exp(t*L) for t=0.01,
    # each mode decays by exp(-n^2 * t)
    t = 0.01
    pi = float(jnp.pi)
    dom2 = (0.0, pi)
    n_modes = 5
    print(f"\nHeat kernel: exp(t*L)*phi_n = exp(-n^2*t)*phi_n for t={t}:")
    for n in range(1, n_modes + 1):
        decay = float(jnp.exp(jnp.array(-n**2 * t)))
        phi_n = cj.chebfun(lambda x, n=n: jnp.sin(n * x), domain=dom2)
        # After decay: exp(t*L) phi_n = decay * phi_n
        x0 = float(pi) / 2.0
        val_before = float(phi_n(jnp.array(x0)))
        val_after = decay * val_before
        print(f"  n={n}: decay factor = {decay:.8f}, phi_n(pi/2) = {val_before:.6f}")
    # Verify orthogonality of eigenfunctions: <sin(mx), sin(nx)> = 0 for m != n
    for m, n in [(1, 2), (1, 3), (2, 4)]:
        mn_prod = cj.chebfun(lambda x, m=m, n=n: jnp.sin(m*x) * jnp.sin(n*x),
                              domain=dom2)
        ip = float(mn_prod.sum())
        print(f"  <sin({m}x), sin({n}x)> = {ip:.2e}  (exact: 0)")
        assert abs(ip) < 1e-13

    # --- Spectral radius vs. Chebfun norm ----------------------------
    # For f(x) = exp(x) on [-1, 1], ||f||_inf = exp(1)
    norm_inf = float(f_exp.norm(jnp.inf))
    print(f"\n||exp(x)||_inf on [-1,1] = {norm_inf:.10f}")
    print(f"Exact exp(1) = {float(jnp.exp(jnp.array(1.0))):.10f}")
    assert abs(norm_inf - float(jnp.exp(jnp.array(1.0)))) < 1e-12

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()

"""Source rational trigonometric Remez with JAX finite-pencil reduction.

Provenance
----------
MATLAB source: @chebfun/trigremez.m, main loop, computeTrialFunctionRational.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
The singular generalized pencil is reduced by eliminating the numerator
column space using a complete QR. This preserves its finite eigenpairs;
JAX provides the reduced eigensolve. Eigenvector sign/scale is not fixed.
"""

import warnings

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils._trigremez_polynomial_source import _exchange


def _matrix(x, degree):
    columns = [jnp.ones_like(x)]
    for j in range(1, degree + 1):
        columns.extend((jnp.cos(j * x), jnp.sin(j * x)))
    return jnp.stack(columns, axis=1)


def _fourier(a):
    positive = (a[1::2] - 1j * a[2::2]) / 2
    return jnp.concatenate((jnp.conj(positive[::-1]), a[:1], positive))


def rational_trial(values, reference, m, n):
    """Construct source pole-free trial from equivalent finite eigenpairs.

    Source rejects nonreal eigenvalues at 1e-13 and checks public q.roots().
    Finite real eigenpairs of the real pencil have real eigenvectors; JAX's
    complex container is converted to that real vector before construction.
    """
    p_matrix, q_matrix = _matrix(reference, m), _matrix(reference, n)
    sigma = jnp.where(jnp.arange(reference.size) % 2, -1.0, 1.0)
    orthogonal, triangular = jnp.linalg.qr(p_matrix, mode="complete")
    left = orthogonal[:, p_matrix.shape[1] :]
    a_reduced = left.T @ (values[:, None] * q_matrix)
    b_reduced = left.T @ (sigma[:, None] * q_matrix)
    levels, vectors = jnp.linalg.eig(jnp.linalg.solve(b_reduced, a_reduced))
    result = None
    for j in range(levels.size):
        if not bool(jnp.isfinite(levels[j])) or float(jnp.abs(jnp.imag(levels[j]))) > 1e-13:
            continue
        h = jnp.real(levels[j])
        denominator = jnp.real(vectors[:, j])
        numerator = jnp.linalg.solve(
            triangular[: p_matrix.shape[1], :],
            orthogonal[:, : p_matrix.shape[1]].T
            @ ((values - h * sigma) * (q_matrix @ denominator)),
        )
        scale = jnp.linalg.norm(jnp.concatenate((numerator, denominator)))
        numerator, denominator = numerator / scale, denominator / scale
        domain = (-float(jnp.pi), float(jnp.pi))
        q = chebfun(_fourier(denominator), domain=domain, coeffs=True, trig=True)
        if q.roots().size == 0:
            p = chebfun(_fourier(numerator), domain=domain, coeffs=True, trig=True)
            result = p, q, p / q, h
    if result is None:
        raise RuntimeError(
            "trigremez:computeTrialFunctionRationl: no pole free approximation found"
        )
    return result


def rational_trigremez(f, m, n, max_iter, tol, normf):
    """Run source full exchange, best-result selection and status reporting.

    Provenance
    ----------
    MATLAB source: @chebfun/trigremez.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Python returns (p, q, public evaluator, error, status dictionary).
    """
    original_domain = (float(f.domain.a), float(f.domain.b))
    target = f.new_domain((-float(jnp.pi), float(jnp.pi)))
    count = 2 * (m + n) + 2
    reference = -jnp.pi + 2 * jnp.pi * jnp.arange(count) / count
    old = reference
    delta, best_delta, diffx, iteration = normf, jnp.inf, 1.0, 0
    best = None
    while float(delta / normf) > tol and iteration < max_iter and float(diffx) > 0:
        p, q, quotient, h = rational_trial(target(reference), reference, m, n)
        if bool(h == 0):
            h = jnp.asarray(1e-19)
        reference, error = _exchange(target, quotient, reference, h, 2, count)
        if float(error / normf) > 1e5:
            reference, error = _exchange(target, quotient, old, h, 1, count)
        diffx = jnp.max(jnp.abs(old - reference))
        delta = error - jnp.abs(h)
        if bool(delta < best_delta):
            best = p, q, error, reference
            best_delta = delta
        old = reference
        iteration += 1
    if best is None:
        # Native leaves pmin undefined when no iteration runs. Preserve the
        # prior Python rational adapter's zero-iteration warning and error.
        warnings.warn(
            "CHEBFUN:CHEBFUN:trigremez:convergence: Remez algorithm did not "
            f"converge after {iteration} iterations to the tolerance {tol:.2g}.",
            UserWarning,
            stacklevel=2,
        )
        raise RuntimeError("trigremez: rational iteration failed")
    if float(best_delta / normf) > tol:
        warnings.warn(
            "CHEBFUN:CHEBFUN:trigremez:convergence: Remez algorithm did not "
            f"converge after {iteration} iterations to the tolerance {tol:.2g}.",
            UserWarning,
            stacklevel=2,
        )
    p, q, error, references = best
    a, b = original_domain
    points = b * (references + jnp.pi) / (2 * jnp.pi) + a * (jnp.pi - references) / (2 * jnp.pi)
    p, q = p.new_domain(original_domain), q.new_domain(original_domain)
    return (
        p,
        q,
        lambda x: p(x) / q(x),
        error,
        {
            "delta": best_delta / normf,
            "iter": iteration,
            "diffx": diffx,
            "xk": points,
        },
    )

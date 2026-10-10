"""Watson's continuous L1 polynomial iteration.

Provenance
----------
MATLAB source : @chebfun/polyfitL1.m
Chebfun commit: 7574c77
Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

import warnings

import jax
import jax.numpy as jnp

from chebfunjax.utils.quadrature import chebpts, legpts
from chebfunjax.utils.transforms import ultra2ultra


@jax.jit
def _clenshaw_u(x, c):
    """Source paired-step Chebyshev-U evaluation, one column per series."""
    x = 2 * jnp.reshape(x, (-1, 1))
    bk1 = jnp.zeros((x.shape[0], c.shape[1]), dtype=jnp.result_type(x, c))
    bk2 = bk1
    degree = c.shape[0] - 1
    for k in range(degree, 1, -2):
        bk2 = c[k] + x * bk1 - bk2
        bk1 = c[k - 1] + x * bk2 - bk1
    if degree % 2:
        bk1, bk2 = c[1] + x * bk1 - bk2, bk1
    return c[0] + x * bk1 - bk2


def _intpsign(n, roots, domain):
    """Source alternating Legendre integrals, including its one-root branch."""
    a, b = domain
    basis = jnp.eye(n + 1)

    def integral(left, right):
        x, w = legpts(n, (left, right))
        return w @ _clenshaw_u(2 * (x - a) / (b - a) - 1, basis)

    sums = integral(a, roots[0])
    for j in range(1, len(roots)):
        sums = sums + (-1) ** j * integral(roots[j - 1], roots[j])
    if len(roots) > 1:
        sums = sums + (-1) ** len(roots) * integral(roots[-1], b)
    return sums


def polyfit_l1(f, n):
    """Literal eager adapter of the native Watson iteration and stopping rule."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun

    if len(f.funs) == 1 and f.funs[0].n <= n + 1:
        return f
    a, b = float(f.domain.a), float(f.domain.b)
    domain = (a, b)
    x = chebpts(n + 3)
    if (a, b) != (-1.0, 1.0):
        # chebpts.m scaleNodes uses this endpoint-weighted expression.
        x = b * (x + 1) / 2 + a * (1 - x) / 2
    x = x[1:-1]
    p = Chebfun.interp1(x, f(x), domain)
    roots = (f - p).roots()
    if len(roots) == len(x):
        return p

    roots = (f - p).roots()
    sums = jnp.sign((f - p)(a)) * _intpsign(n, roots, domain)
    history = jnp.linalg.norm(sums, ord=jnp.inf)
    for iteration in range(100):
        residual = f - p
        roots = residual.roots()
        g = jnp.sign(residual(a)) * _intpsign(n, roots, domain)
        basis = _clenshaw_u(2 * (roots - a) / (b - a) - 1, jnp.eye(n + 1))
        derivative = residual.diff()
        diagonal = jnp.diag(2 / jnp.abs(derivative(roots)))
        hessian = basis.T @ diagonal @ basis
        dc = jnp.linalg.solve(hessian, g)
        dp = chebfun(ultra2ultra(dc, 1, 0), domain=domain, coeffs=True)
        damping = 1.0
        previous = p
        while damping > 1e-5:
            p = previous + damping * dp
            roots = (f - p).roots()
            sums = _intpsign(n, roots, domain)
            if jnp.linalg.norm(sums, ord=jnp.inf) < history and len(roots) >= n + 1:
                break
            damping /= 2
        history = jnp.linalg.norm(sums, ord=jnp.inf)
        if history < 1e-10 / (b - a):
            break
    if iteration == 99:
        warnings.warn(
            "CHEBFUN:POLYFIT:MAXITER: The maximum number of iterations was reach. "
            "Answer may not be accurate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return p

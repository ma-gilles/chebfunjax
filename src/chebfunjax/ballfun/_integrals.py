"""Native CFF norm and triple integration using JAX arrays.

Provenance
----------
MATLAB source: @ballfun/norm.m, sum3.m, vals2coeffs.m, constructor.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original authors: Copyright 2019 University of Oxford and Chebfun Developers.
"""

from functools import partial

import jax
import jax.numpy as jnp

from chebfunjax._ball_plot_data import (
    prolong_plot_coefficients,
    source_coefficients_to_values,
)


def values_to_coefficients(values):
    """Literal radial-first forward CFF transform; retain complex roundoff."""
    x = jnp.asarray(values, dtype=jnp.complex128)
    m, n, p = x.shape
    if m > 1:
        x = jnp.fft.ifft(jnp.concatenate((x[m - 1:0:-1], x)),
                         n=2 * (m - 1), axis=0)
        x = 2 * x[:m]
        x = x.at[0].divide(2).at[m - 1].divide(2)
    x = jnp.fft.fftshift(jnp.fft.fftshift(
        jnp.fft.fft(jnp.fft.fft(x, axis=1), axis=2), axes=1), axes=2)
    kn, kp = jnp.arange(n) - n // 2, jnp.arange(p) - p // 2
    sn = jnp.where(kn % 2 == 0, 1., -1.)
    sp = (1. / n / p) * jnp.where(kp % 2 == 0, 1., -1.)
    return x * (sn[:, None] * sp[None, :])[None, :, :]


def coefficient_is_real(c):
    """Native numeric coefficient constructor, including even Nyquist rows."""
    _, n, p = c.shape
    check = (c[:, 1 - n % 2:n // 2 + 1, 1 - p % 2:p // 2 + 1]
             - jnp.conj(c[:, n // 2:, p // 2:][:, ::-1, ::-1]))
    tol = 1e7 * jnp.finfo(jnp.float64).eps
    result = jnp.max(jnp.abs(check)) < tol
    if n % 2 == 0:
        result = result & (jnp.max(jnp.abs(c[:, 0, :])) < tol)
    if p % 2 == 0:
        result = result & (jnp.max(jnp.abs(c[:, :, 0])) < tol)
    return result


def sum_coefficients(coefficients, is_real):
    """Literal sum3 prolongation, Jacobian matrices, and integration order."""
    c = jnp.asarray(coefficients, dtype=jnp.complex128)
    m0, n, p0 = c.shape
    m, p = m0 + 2, p0 + 2
    f = prolong_plot_coefficients(c, (m, n, p))[:, n // 2, :]
    f = jnp.pad(f, ((0, 2), (1, 1)))
    m, p = m + 2, p + 2
    # ultraS.multmat(m,[.5,0,.5],0): multiplication of Chebyshev T terms.
    mr = .5 * jnp.eye(m)
    for k in range(m):
        if k + 2 < m:
            mr = mr.at[k + 2, k].add(.25)
        mr = mr.at[abs(k - 2), k].add(.25)
    delta = jnp.arange(p)[:, None] - jnp.arange(p)[None, :]
    ms = jnp.where(delta == -1, .5j, jnp.where(delta == 1, -.5j, 0j))
    f = mr @ f @ ms.T
    k = jnp.arange(m, dtype=jnp.float64)
    # Safe denominators avoid creating nonfinite values in unselected branches.
    even = -1 / jnp.where(k * k == 1, 1, k * k - 1)
    odd3 = -1 / jnp.where(k == 1, 1, k - 1)
    wr = jnp.where(k % 2 == 0, even,
                   jnp.where(k % 4 == 1, 1 / (k + 1), odd3))
    q = jnp.arange(p) - p // 2
    wt = -1j * (jnp.where(q % 2 == 0, 1., -1.) - 1) / jnp.where(q == 0, 1, q)
    wt = wt.at[p // 2].set(jnp.pi)
    wt = 2 * jnp.pi * wt
    result = wr @ f @ wt
    return jnp.where(is_real, jnp.real(result), result)


def norm_coefficients(coefficients):
    """Native exact doubled dimensions and coefficient-constructor realness."""
    c = jnp.asarray(coefficients, dtype=jnp.complex128)
    doubled = prolong_plot_coefficients(c, tuple(2 * k for k in c.shape))
    values = source_coefficients_to_values(doubled)
    squared = values_to_coefficients(values * jnp.conj(values))
    return jnp.sqrt(jnp.abs(sum_coefficients(squared, coefficient_is_real(squared))))


@partial(jax.jit, static_argnames=("dims",))
def sum2_coefficients(coefficients, dims):
    """Native sum2 matrix order, physical weights and complex coefficients.

    Provenance: @ballfun/sum2.m, Chebfun 7574c77.
    """
    c = jnp.asarray(coefficients, dtype=jnp.complex128)
    m0, n, p0 = c.shape
    m, p = m0 + 2, p0 + 2
    f = prolong_plot_coefficients(c, (m, n, p))
    delta = jnp.arange(p)[:, None] - jnp.arange(p)[None, :]
    ms = jnp.where(delta == -1, .5j, jnp.where(delta == 1, -.5j, 0j))
    q = jnp.arange(p) - p // 2
    wt = -1j * (jnp.where(q % 2 == 0, 1., -1.) - 1) / jnp.where(q == 0, 1, q)
    wt = wt.at[p // 2].set(jnp.pi)
    if dims == (2, 3):
        return (f[:, n // 2, :] @ ms.T) @ (2*jnp.pi*wt)
    mr = .5*jnp.eye(m)
    for k in range(m):
        if k + 2 < m:
            mr = mr.at[k + 2, k].add(.25)
        mr = mr.at[abs(k - 2), k].add(.25)
    f = (mr @ f.reshape(m, -1)).reshape(m, n, p)
    k = jnp.arange(m, dtype=jnp.float64)
    even = -1 / jnp.where(k*k == 1, 1, k*k - 1)
    odd3 = -1 / jnp.where(k == 1, 1, k - 1)
    wr = jnp.where(k % 2 == 0, even,
                   jnp.where(k % 4 == 1, 1/(k+1), odd3))
    if dims == (1, 2):
        return wr @ (2*jnp.pi*f[:, n // 2, :])
    f = (f.reshape(-1, p) @ ms.T).reshape(m, n, p)
    return jax.vmap(lambda block: (wr @ block) @ wt, in_axes=1)(f)

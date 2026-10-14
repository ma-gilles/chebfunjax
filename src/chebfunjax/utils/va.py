"""Vandermonde with Arnoldi orthogonalization (VAorthog / VAeval).

Polynomial least-squares fitting on arbitrary points is ill-conditioned in
the monomial basis. Arnoldi orthogonalization builds a discretely orthogonal
polynomial or rational basis for stable fitting.

Provenance
----------
MATLAB source : VAorthog.m, VAeval.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original authors: Copyright 2017 by The University of Oxford and the Chebfun
Developers.
Algorithm: P. D. Brubeck, Y. Nakatsukasa, and L. N. Trefethen,
    "Vandermonde with Arnoldi", SIAM Review 63 (2021), 405-415.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["va_orthog", "va_eval"]


def _dtype(z):
    return jnp.result_type(z, jnp.float64)


def _polynomial_orthog(z, degree):
    m = z.shape[0]
    dtype = jnp.float64 if degree == 0 else _dtype(z)
    cols = [jnp.ones((m,), dtype=dtype)]
    h = jnp.zeros((degree + 1, degree), dtype=dtype)
    for k in range(degree):
        q = z * cols[k]
        for j in range(k + 1):
            h_jk = jnp.vdot(cols[j], q) / m
            h = h.at[j, k].set(h_jk)
            q = q - h_jk * cols[j]
        h_next = jnp.linalg.norm(q) / jnp.sqrt(jnp.asarray(m, dtype=dtype))
        h = h.at[k + 1, k].set(h_next)
        cols.append(q / h_next)
    return h, jnp.stack(cols, axis=1)


def _pole_orthog(z, poles):
    m, npoles = z.shape[0], poles.shape[0]
    dtype = jnp.float64 if npoles == 0 else jnp.result_type(z, poles, jnp.float64)
    cols = [jnp.ones((m,), dtype=dtype)]
    if npoles == 0:
        # Native zeros(0,-1) plus zero loop iterations leaves real 0-by-0 H.
        return jnp.zeros((0, 0), dtype=dtype), jnp.stack(cols, axis=1)
    h = jnp.zeros((npoles + 1, npoles), dtype=dtype)
    for k in range(npoles):
        q = cols[k] / (z - poles[k])
        for j in range(k + 1):
            h_jk = jnp.vdot(cols[j], q) / m
            h = h.at[j, k].set(h_jk)
            q = q - h_jk * cols[j]
        h_next = jnp.linalg.norm(q) / jnp.sqrt(jnp.asarray(m, dtype=dtype))
        h = h.at[k + 1, k].set(h_next)
        cols.append(q / h_next)
    return h, jnp.stack(cols, axis=1)


@partial(jax.jit, static_argnums=(1,))
def _orthog(z, degree, pole_groups):
    h, r = _polynomial_orthog(z, degree)
    hs, blocks = [h], [r]
    for poles in pole_groups:
        hp, rp = _pole_orthog(z, poles)
        hs.append(hp)
        blocks.append(rp[:, 1:])
    return tuple(hs), jnp.concatenate(blocks, axis=1)


def va_orthog(Z, n: int, Pol=None):
    """Build polynomial VA basis; explicit ``Pol`` adds rational blocks.

    ``Z`` is flattened as in the historical adapter. ``n`` is a nonnegative
    integer degree and is static while tracing with JIT. With ``Pol`` omitted,
    return ``(H, R)``. With ``Pol`` supplied, return ``(Hes, R)`` where
    ``Hes`` is a tuple containing the polynomial recurrence followed by one
    recurrence per pole group. Pole-group count and lengths are also static
    for JIT.

    Provenance
    ----------
    MATLAB source : VAorthog.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    z = jnp.asarray(Z).reshape((-1,))
    if Pol is None:
        hes, r = _orthog(z, n, ())
        return hes[0], r
    groups = tuple(jnp.asarray(p).reshape((-1,)) for p in Pol)
    return _orthog(z, n, groups)


def _blocks(Hes):
    if isinstance(Hes, (tuple, list)):
        return tuple(jnp.asarray(h) for h in Hes)
    return (jnp.asarray(Hes),)


def _eval_block(z, h, derivative):
    m, n = z.shape[0], h.shape[1]
    dtype = jnp.float64 if n == 0 else jnp.result_type(z, h, jnp.float64)
    qcols = [jnp.ones((m,), dtype=dtype)]
    dcols = [jnp.zeros((m,), dtype=dtype)]
    for k in range(n):
        hc = h[:k + 1, k]
        q = (z * qcols[k] - jnp.stack(qcols[:k + 1], 1) @ hc) / h[k + 1, k]
        qcols.append(q)
        if derivative:
            d = (z * dcols[k] - jnp.stack(dcols[:k + 1], 1) @ hc + qcols[k]) / h[k + 1, k]
            dcols.append(d)
    return jnp.stack(qcols, 1), jnp.stack(dcols, 1)


def _eval_pole_block(z, h, poles, derivative):
    m = z.shape[0]
    npoles = poles.shape[0]
    dtype = jnp.float64 if npoles == 0 else jnp.result_type(z, h, poles, jnp.float64)
    qcols = [jnp.ones((m,), dtype=dtype)]
    dcols = [jnp.zeros((m,), dtype=dtype)]
    for k in range(npoles):
        zpk = 1 / (z - poles[k])
        hc = h[:k + 1, k]
        q = (qcols[k] * zpk - jnp.stack(qcols[:k + 1], 1) @ hc) / h[k + 1, k]
        qcols.append(q)
        if derivative:
            d = (dcols[k] * zpk - jnp.stack(dcols[:k + 1], 1) @ hc - qcols[k] * zpk**2) / h[k + 1, k]
            dcols.append(d)
    return jnp.stack(qcols, 1), jnp.stack(dcols, 1)


@partial(jax.jit, static_argnames=("return_derivative",))
def _evaluate(z, hes, pole_groups, return_derivative=False):
    q, d = _eval_block(z, hes[0], return_derivative)
    r0s, r1s = [q], [d]
    for i, poles in enumerate(pole_groups):
        q, d = _eval_pole_block(z, hes[i + 1], poles, return_derivative)
        r0s.append(q[:, 1:])
        r1s.append(d[:, 1:])
    r0, r1 = jnp.concatenate(r0s, 1), jnp.concatenate(r1s, 1)
    return (r0, r1) if return_derivative else r0


def va_eval(Z, Hes, Pol=None, *, return_derivative=False):
    """Evaluate basis values; optional ``return_derivative`` yields ``(R0,R1)``.

    ``Hes`` accepts the legacy polynomial matrix or recurrence-block tuple.
    For rational blocks, pass their pole groups through ``Pol``. Inputs are
    flattened to preserve the historical Python point adapter.

    Provenance
    ----------
    MATLAB source : VAeval.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    z = jnp.asarray(Z).reshape((-1,))
    hs = _blocks(Hes)
    groups = () if Pol is None else tuple(jnp.asarray(p).reshape((-1,)) for p in Pol)
    if len(hs) != len(groups) + 1:
        raise ValueError("rational recurrence blocks require matching Pol groups")
    return _evaluate(z, hs, groups, return_derivative=return_derivative)

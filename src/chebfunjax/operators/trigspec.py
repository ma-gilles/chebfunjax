"""Fourier spectral-collocation matrices (MATLAB ``trigspec`` class,
Fable 5).

Provenance
----------
MATLAB source : @trigspec/multmat.m, @trigspec/sptoeplitz.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The
    Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp


def sptoeplitz(col, row):
    """Toeplitz matrix with first column ``col`` and first row ``row``
    (MATLAB ``trigspec.sptoeplitz``; dense here)."""
    col = jnp.asarray(col).reshape(-1)
    row = jnp.asarray(row).reshape(-1)
    n = col.shape[0]
    m = row.shape[0]
    i = jnp.arange(n)[:, None]
    j = jnp.arange(m)[None, :]
    d = i - j
    full = jnp.concatenate([row[1:][::-1], col])   # index d + (m-1)
    return full[d + (m - 1)]


def multmat(N: int, f):
    """``N x N`` multiplication matrix by ``f`` in the Fourier basis
    (MATLAB ``trigspec.multmat(N, f)``): ``f`` is a periodic Chebfun or
    a vector of Fourier coefficients.

    Provenance
    ----------
    MATLAB source : @trigspec/multmat.m
    Chebfun commit: 7574c77
    """
    if hasattr(f, "trigcoeffs"):
        a = jnp.asarray(f.trigcoeffs(), dtype=jnp.complex128).reshape(-1)
    elif hasattr(f, "coeffs"):
        a = jnp.asarray(f.coeffs, dtype=jnp.complex128).reshape(-1)
    else:
        a = jnp.asarray(f, dtype=jnp.complex128).reshape(-1)
    N = int(N)
    if a.shape[0] == 1:
        return a[0] * jnp.eye(N, dtype=jnp.complex128)
    if a.shape[0] % 2 == 0:
        a = jnp.concatenate([a[:1] / 2, a[1:], a[:1] / 2])
    Na = a.shape[0] // 2 + 1          # 1-based index of the zero mode
    if Na < N:
        col = jnp.concatenate([a[Na - 1:], jnp.zeros(N - Na, dtype=a.dtype)])
        row = jnp.concatenate([a[Na - 1::-1], jnp.zeros(N - Na, dtype=a.dtype)])
    else:
        col = a[Na - 1:a.shape[0] - Na + N]
        row = a[Na - 1:Na - N - 1:-1] if Na - N - 1 >= 0 else a[Na - 1::-1][:N]
    return sptoeplitz(col, row)

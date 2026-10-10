"""Polynomial coefficient and value transforms.

Chebyshev <-> Legendre, Chebyshev <-> Jacobi, Legendre values/coefficients,
ultra-spherical, Jacobi-to-Jacobi, and Chebyshev values <-> coefficients.

Translated from MATLAB Chebfun (commit 7574c77): cheb2leg.m, leg2cheb.m,
cheb2jac.m, jac2cheb.m, jac2jac.m, ultra2ultra.m, ultracoeffs.m,
chebvals2legcoeffs.m, chebcoeffs2legvals.m, legvals2chebcoeffs.m,
legvals2chebvals.m, legvals2legcoeffs.m, legcoeffs2chebvals.m,
legcoeffs2legvals.m, chebvals2legvals.m, chebvals2chebvals.m,
chebcoeffs2chebvals.m, chebvals2chebcoeffs.m, and related files.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

# ===========================================================================
# Chebyshev values <-> coefficients (DCT-I based)
# ===========================================================================

@jax.jit
def _vals2coeffs_jax(values: jnp.ndarray) -> jnp.ndarray:
    """Unconditional JAX Chebyshev values-to-coefficients transform.

    Provenance
    ----------
    MATLAB source : @chebtech2/vals2coeffs.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    Algorithm: existing JAX DCT-I traced body; no NumPy dispatch.
    """
    n = values.shape[0]
    if n <= 1:
        return values
    tmp = jnp.concatenate([values[n - 1:0:-1], values[:n - 1]])
    if jnp.iscomplexobj(values):
        # The cosine transform is real-linear. Two real FFTs preserve
        # conjugacy exactly; a general complex FFT can mix roundoff between
        # components and violate the source's strict cancellation predicate.
        # This is a numerical adaptation of @chebtech2/vals2coeffs.m's IFFT.
        coeffs = (jnp.real(jnp.fft.ifft(jnp.real(tmp), axis=0))
                  + 1j*jnp.real(jnp.fft.ifft(jnp.imag(tmp), axis=0)))
    else:
        coeffs = jnp.real(jnp.fft.ifft(tmp, axis=0))
    coeffs = coeffs[:n]
    coeffs = coeffs.at[1:n - 1].multiply(2.0)
    vflip = values[::-1]
    is_even = jnp.max(jnp.abs(values - vflip), axis=0) == 0
    is_odd = jnp.max(jnp.abs(values + vflip), axis=0) == 0
    k = jnp.arange(n).reshape((n,) + (1,) * (coeffs.ndim - 1))
    sym = jnp.where((k % 2 == 1) & is_even, 0.0, coeffs)
    sym = jnp.where((k % 2 == 0) & is_odd, 0.0, sym)
    delta = jnp.where(jnp.isfinite(coeffs), sym - coeffs, 0.0)
    coeffs = coeffs + jax.lax.stop_gradient(delta)
    return coeffs


@jax.jit
def _coeffs2vals_jax(coeffs: jnp.ndarray) -> jnp.ndarray:
    """Unconditional JAX Chebyshev coefficients-to-values transform.

    Provenance
    ----------
    MATLAB source : @chebtech2/coeffs2vals.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    Algorithm: existing JAX DCT-I traced body; no NumPy dispatch.
    """
    n = coeffs.shape[0]
    if n <= 1:
        return coeffs
    c = coeffs.at[1:n - 1].multiply(0.5)
    tmp = jnp.concatenate([c, c[n - 2:0:-1]])
    if jnp.iscomplexobj(coeffs):
        # Real-linear FFT evaluation preserves real/imaginary components and
        # conjugacy exactly, as in the inverse transform above.
        values = (jnp.real(jnp.fft.fft(jnp.real(tmp), axis=0))
                  + 1j*jnp.real(jnp.fft.fft(jnp.imag(tmp), axis=0)))
    else:
        values = jnp.real(jnp.fft.fft(tmp, axis=0))
    values = values[n - 1::-1]
    is_even = jnp.max(jnp.abs(coeffs[1::2]), axis=0, initial=0.0) == 0
    is_odd = jnp.max(jnp.abs(coeffs[0::2]), axis=0, initial=0.0) == 0
    vflip = values[::-1]
    sym = jnp.where(is_even, (values + vflip) / 2.0, values)
    sym = jnp.where(is_odd, (values - vflip) / 2.0, sym)
    delta = jnp.where(jnp.isfinite(values), sym - values, 0.0)
    values = values + jax.lax.stop_gradient(delta)
    return values


def vals2coeffs(values: jnp.ndarray) -> jnp.ndarray:
    """Convert values at 2nd-kind Chebyshev points to Chebyshev coefficients.

    Given values V_k = f(x_k) at Chebyshev points of the 2nd kind
    x_k = cos(k*pi/(n-1)), k=0,...,n-1, returns the Chebyshev coefficients c
    such that f(x) = c[0]*T_0(x) + c[1]*T_1(x) + ... + c[n-1]*T_{n-1}(x).

    This is equivalent to the inverse Discrete Cosine Transform of Type I.

    Parameters
    ----------
    values : jnp.ndarray, shape (n,)
        Function values at 2nd-kind Chebyshev points, ordered from x=1 to x=-1
        (descending order, MATLAB convention) or ascending — the transform is
        symmetric so ordering does not matter for even/odd structure.

    Returns
    -------
    coeffs : jnp.ndarray, shape (n,)
        Chebyshev series coefficients.

    Provenance
    ----------
    MATLAB source : @chebtech2/vals2coeffs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Inverse DCT-I via FFT, see Mason & Handscomb,
        "Chebyshev Polynomials", Section 4.7 (2003).

    See Also
    --------
    coeffs2vals
    """
    n = values.shape[0]
    if n <= 1:
        return values
    return _vals2coeffs_jax(values)

def coeffs2vals(coeffs: jnp.ndarray) -> jnp.ndarray:
    """Convert Chebyshev coefficients to values at 2nd-kind Chebyshev points.

    Given Chebyshev coefficients c, returns the values
    V_k = c[0]*T_0(x_k) + ... + c[n-1]*T_{n-1}(x_k)
    at Chebyshev points of the 2nd kind x_k = cos(k*pi/(n-1)).

    This is equivalent to the Discrete Cosine Transform of Type I.

    Parameters
    ----------
    coeffs : jnp.ndarray, shape (n,)
        Chebyshev series coefficients.

    Returns
    -------
    values : jnp.ndarray, shape (n,)
        Function values at 2nd-kind Chebyshev points (descending x order).

    Provenance
    ----------
    MATLAB source : @chebtech2/coeffs2vals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: DCT-I via FFT, see Mason & Handscomb,
        "Chebyshev Polynomials", Sections 4.7 and 6.3 (2003).

    See Also
    --------
    vals2coeffs
    """
    n = coeffs.shape[0]
    if n <= 1:
        return coeffs
    return _coeffs2vals_jax(coeffs)

# ===========================================================================
# Chebyshev <-> Legendre (direct O(n^2) method)
# ===========================================================================

def cheb2leg(c_cheb: jnp.ndarray, normalize: bool | str = False) -> jnp.ndarray:
    """Convert Chebyshev coefficients to Legendre coefficients.

    C_LEG = cheb2leg(C_CHEB) converts the vector C_CHEB of Chebyshev
    coefficients to a vector C_LEG of Legendre coefficients such that
        C_CHEB[0]*T_0 + ... + C_CHEB[N-1]*T_{N-1}
      = C_LEG[0]*P_0 + ... + C_LEG[N-1]*P_{N-1},
    where P_k is the degree-k Legendre polynomial normalized so that
    max(|P_k|) = 1 (the standard normalization P_k(1) = 1).

    Parameters
    ----------
    c_cheb : jnp.ndarray, shape (n,) or (n, m)
        Chebyshev coefficients. Matrix columns are converted independently.
    normalize : bool or str, default False
        If true, or if the string begins with the native four-character
        abbreviation ``"norm"``, use orthonormal Legendre normalization.

    Returns
    -------
    c_leg : jnp.ndarray, shape (n,) or (n, m)
        Legendre coefficients.

    Notes
    -----
    Uses the direct Clenshaw-Curtis projection below 513 coefficient rows and
    the native pivoted-Cholesky Toeplitz-Hankel method from [1] at 513 rows.

    References
    ----------
    .. [1] A. Townsend, M. Webb, and S. Olver, "Fast polynomial transforms
       based on Toeplitz and Hankel matrices", Math. Comp., 87, 2018.

    Provenance
    ----------
    MATLAB source : cheb2leg.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend, Nick Hale.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.

    See Also
    --------
    leg2cheb, cheb2jac, jac2cheb
    """
    if isinstance(normalize, str):
        normalize = normalize[:4].lower() == "norm"
    c_cheb = jnp.asarray(c_cheb)
    if c_cheb.ndim not in (1, 2):
        raise ValueError("cheb2leg expects a coefficient vector or matrix")
    n = c_cheb.shape[0]
    # Native cheb2leg returns before normalization for N < 2.
    if n < 2:
        return c_cheb
    if c_cheb.ndim == 2:
        return jax.vmap(lambda col: cheb2leg(col, normalize=normalize),
                        in_axes=1, out_axes=1)(c_cheb)
    if n < 513:
        return _cheb2leg_direct(c_cheb, normalize)
    return _cheb2leg_fast(c_cheb, normalize)


def _cheb2leg_fast(c_cheb: jnp.ndarray, normalize: bool) -> jnp.ndarray:
    """Native pivoted-Cholesky Toeplitz-Hankel transform (N >= 513)."""
    n = c_cheb.shape[0]
    # The pivot sequence depends only on static N. Build and cache the exact
    # source stopping-rank plan at trace time, so runtime work and buffers use
    # only the retained low-rank columns (no workspace cap or truncation).
    with jax.ensure_compile_time_eval():
        vals, chol = _cheb2leg_cholesky_plan(n)
    num = jnp.arange(1, n, dtype=jnp.float64)

    # First row of the conversion matrix, with the singular zero-mode
    # expressions assigned directly as in the MATLAB routine.
    rownum = jnp.arange(n, dtype=jnp.float64)
    l1top = jnp.concatenate((jnp.ones((2,)), vals[:n - 2]))
    l2top = jnp.concatenate((jnp.ones((1,)), vals[:n - 1]))
    l1 = jnp.where(rownum == 0, 1.0, l1top / jnp.where(rownum == 0, 1.0, rownum))
    l2 = l2top / (rownum + 1.0)
    first_row = -0.5 * rownum * l1 * l2
    first_row = jnp.where((rownum.astype(jnp.int32) % 2) == 1, 0.0, first_row)
    first_row = first_row.at[0].set(1.0)

    # Toeplitz row and its circulant embedding for the FFT product.
    toeplitz = jnp.concatenate((jnp.zeros((2,)), vals[:n - 3]))
    denom = jnp.arange(n - 1, dtype=jnp.float64)
    toeplitz = jnp.where(denom == 0, 0.0, toeplitz / jnp.where(denom == 0, 1.0, denom))
    toeplitz = jnp.where((jnp.arange(n - 1) % 2) == 1, 0.0, toeplitz)
    embed = jnp.concatenate((jnp.zeros((n,), dtype=jnp.float64), toeplitz[-1:0:-1]))
    a_fft = jnp.fft.fft(embed)

    diag_scale = 0.5 * jnp.sqrt(jnp.pi) / vals[jnp.arange(2, 2 * n - 1, 2)]
    scale = (num + 0.5) / num
    tail = c_cheb[1:]
    tmp = -chol * tail[:, None]
    f1 = jnp.fft.fft(tmp, n=2 * n - 2, axis=0)
    b = jnp.fft.ifft(f1 * a_fft[:, None], axis=0)[:n - 1]
    result = jnp.sum(chol * b, axis=1)
    out_tail = scale * result + diag_scale * tail
    out = jnp.concatenate((jnp.dot(first_row, c_cheb)[None], out_tail))
    if normalize:
        inverse_norms = 1.0 / jnp.sqrt(jnp.arange(n, dtype=jnp.float64) + 0.5)
        out = out * inverse_norms
    if not jnp.issubdtype(c_cheb.dtype, jnp.complexfloating):
        out = jnp.real(out)
    return out


@lru_cache(maxsize=8)
def _cheb2leg_cholesky_plan(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cache the native static-N Cholesky factors using JAX arithmetic only."""
    from jax import lax

    vals = jnp.zeros((2 * n,), dtype=jnp.float64)
    vals = vals.at[0].set(jnp.sqrt(jnp.pi)).at[1].set(2.0 / jnp.sqrt(jnp.pi))

    def _vals_step(k, values):
        i = 2 + 2 * k
        values = values.at[i].set(values[i - 2] * (1.0 - 1.0 / i))
        values = values.at[i + 1].set(values[i - 1] * (1.0 - 1.0 / (i + 1)))
        return values

    vals = lax.fori_loop(0, n - 1, _vals_step, vals)
    num = jnp.arange(1, n, dtype=jnp.float64)
    diag = vals[2 * jnp.arange(1, n, dtype=jnp.int32) - 1] * (
        num**2 / (2.0 * num + 1.0)
    )
    tol = 1e-14 * jnp.log(float(n))
    chol = jnp.empty((n - 1, 0), dtype=jnp.float64)
    pivots = jnp.empty((0,), dtype=jnp.float64)
    peak = float(jnp.max(diag))
    while peak > tol:
        idx = int(jnp.argmax(diag))
        mx = diag[idx]
        vals_idx = idx + 1 + jnp.arange(n - 1, dtype=jnp.int32)
        col = vals[vals_idx] * (num * (idx + 1.0) / (idx + num + 2.0))
        if chol.shape[1]:
            col = col - chol @ (chol[idx, :] * pivots)
        chol = jnp.concatenate((chol, col[:, None]), axis=1)
        pivots = jnp.concatenate((pivots, (1.0 / mx)[None]))
        diag = diag - col**2 / mx
        peak = float(jnp.max(diag))
    chol = chol * jnp.sqrt(pivots)[None, :]
    return vals, chol


def _cheb2leg_direct(c_cheb: jnp.ndarray, normalize: bool) -> jnp.ndarray:
    """Convert Chebyshev to Legendre coefficients using the 3-term recurrence.

    Uses Clenshaw-Curtis quadrature on a 2N+1 grid to compute the Legendre
    projection integrals.
    """
    N = c_cheb.shape[0] - 1  # polynomial degree

    # 2*N+1 Chebyshev grid (descending order, cos(pi*k/(2N)) for k=0..2N)
    k = jnp.arange(2 * N + 1, dtype=jnp.float64)
    x = jnp.cos(0.5 * jnp.pi * k / N)

    # Values of the Chebyshev expansion on the 2N+1 grid via DCT
    # Pad coefficients to length 2N+1
    c_padded = jnp.concatenate([c_cheb, jnp.zeros(N, dtype=jnp.float64)])
    f = _dct1(c_padded)

    # Clenshaw-Curtis quadrature weights for 2N+1 points
    w = _cc_weights(2 * N + 1)

    # Rolling three-term recurrence via lax.scan: the previous
    # implementation built the full Legendre-Vandermonde with N JAX
    # .at[].set copies (O(N^3) memory traffic -- minutes at N ~ 4000);
    # the scan is O(N^2) flops, O(N) memory, and stays jit/grad
    # traceable (seconds at N ~ 30000).
    from jax import lax

    wf = f * w
    p0 = jnp.ones_like(x)
    c0 = 0.5 * jnp.dot(wf, p0)
    p1 = x
    c1 = 1.5 * jnp.dot(wf, p1)

    def _step(carry, j):
        pjm1, pj = carry
        pjp1 = ((2 * j + 1) * x * pj - j * pjm1) / (j + 1)
        contrib = (2 * (j + 1) + 1) / 2.0 * jnp.dot(wf, pjp1)
        return (pj, pjp1), contrib

    if N >= 2:
        _, rest = lax.scan(_step, (p0, p1),
                           jnp.arange(1, N, dtype=jnp.float64))
        c_leg = jnp.concatenate([jnp.stack([c0, c1]), rest])
    else:
        c_leg = jnp.stack([c0, c1])[: N + 1]

    if normalize:
        inverse_norms = 1.0 / jnp.sqrt(jnp.arange(N + 1, dtype=jnp.float64) + 0.5)
        c_leg = c_leg * inverse_norms

    return c_leg


def _dct1(c: jnp.ndarray) -> jnp.ndarray:
    """Compute a (scaled) DCT of type I using FFT.

    Returns T(X)*C where X = cos(pi*k/N) for k=0..N, and
    T(X) = [T_0, T_1, ..., T_N](X).
    N = len(c) - 1.
    """
    n = c.shape[0]
    if n <= 1:
        return c

    # Scale endpoints
    c_scaled = c.at[0].multiply(2.0)
    c_scaled = c_scaled.at[-1].multiply(2.0)

    # Mirror: [c_0, c_1, ..., c_{N}, c_{N-1}, ..., c_1]
    tmp = jnp.concatenate([c_scaled, c_scaled[-2:0:-1]])

    # FFT and take first n entries, then scale
    v = jnp.fft.fft(tmp)
    if not jnp.issubdtype(c.dtype, jnp.complexfloating):
        v = jnp.real(v)
    v = v[:n] / 2.0

    return v


def _idct1(v: jnp.ndarray) -> jnp.ndarray:
    """Inverse DCT-I: convert values on a Chebyshev grid to coefficients.

    Returns T(X)\\V where X = cos(pi*k/N), T(X) = [T_0, ..., T_N](X).
    """
    n = v.shape[0]
    if n <= 1:
        return v

    # Mirror along the coefficient axis. MATLAB's chebfun.idct supports
    # column batches and complex inputs; preserve both contracts here.
    vector_input = v.ndim == 1
    values = v[:, None] if vector_input else v
    tmp = jnp.concatenate([values, values[-2:0:-1]], axis=0)

    # leg2cheb.m's local idct1 uses (2/(n-1))*chebfun.dct(v,1),
    # then halves the two output endpoints. The mirrored IFFT already
    # divides by 2*(n-1), so the source pre-endpoint factor is two.
    c = 2.0 * jnp.fft.ifft(tmp, axis=0)
    if not jnp.iscomplexobj(v):
        c = jnp.real(c)
    else:
        # Source chebfun.dct(v,1) delegates to Chebtech2.coeffs2vals,
        # preserving exact real/imaginary output for those global cases.
        real_c = 2.0 * jnp.real(jnp.fft.ifft(jnp.real(tmp), axis=0))
        imag_c = 2.0j * jnp.real(jnp.fft.ifft(jnp.imag(tmp), axis=0))
        c = jnp.where(jnp.all(jnp.imag(v) == 0), real_c,
                      jnp.where(jnp.all(jnp.real(v) == 0), imag_c, c))
    c = c[:n]

    # Scale endpoints by 1/2
    c = c.at[0, :].multiply(0.5)
    c = c.at[-1, :].multiply(0.5)

    return c[:, 0] if vector_input else c


def _cc_weights(n: int) -> jnp.ndarray:
    """Clenshaw-Curtis quadrature weights for n 2nd-kind Chebyshev points.

    Points ordered descending: x_k = cos(k*pi/(n-1)), k=0,...,n-1.
    """
    if n == 1:
        return jnp.array([2.0], dtype=jnp.float64)
    if n == 2:
        return jnp.array([1.0, 1.0], dtype=jnp.float64)

    N = n - 1

    # Chebyshev moments: integral of T_k(x) over [-1,1]
    # = 2/(1-k^2) for even k, 0 for odd k
    c = jnp.zeros(N + 1, dtype=jnp.float64)
    k_even = jnp.arange(0, N + 1, 2, dtype=jnp.float64)
    c = c.at[0::2].set(2.0 / (1.0 - k_even**2))

    # Mirror for IFFT
    v = jnp.concatenate([c, c[N - 1:0:-1]])

    w = 2.0 * jnp.real(jnp.fft.ifft(v))
    w = w[:N + 1]

    # Halve endpoints
    w = w.at[0].set(w[0] / 2.0)
    w = w.at[N].set(w[N] / 2.0)

    return w


def leg2cheb(
    c_leg: jnp.ndarray, *, normalize: bool = False, trans: bool = False,
    max_rank: int = 128,
) -> jnp.ndarray:
    """Convert Legendre coefficients to Chebyshev coefficients.

    C_CHEB = leg2cheb(C_LEG) converts the vector C_LEG of Legendre
    coefficients to a vector C_CHEB of Chebyshev coefficients such that
        C_LEG[0]*P_0 + ... + C_LEG[N-1]*P_{N-1}
      = C_CHEB[0]*T_0 + ... + C_CHEB[N-1]*T_{N-1}.

    Parameters
    ----------
    c_leg : jnp.ndarray, shape (n,)
        Legendre coefficients.
    normalize : bool, default False
        If True, the input uses Legendre polynomials normalized to be
        orthonormal (i.e., multiply by sqrt(k+1/2) to get standard).
    trans : bool, default False
        Apply the transpose of the Legendre-to-Chebyshev conversion operator.
    max_rank : int, default 128
        Static workspace bound for the source fast branch when ``n > 512``.

    Returns
    -------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.

    Notes
    -----
    Uses the direct source method through degree 511 and the source fast
    Hankel-Toeplitz method above that size. The direct transpose follows
    MATLAB's literal ``L.T @ idct1(c)`` branch.

    References
    ----------
    .. [1] A. Townsend, M. Webb, and S. Olver, "Fast polynomial transforms
       based on Toeplitz and Hankel matrices", Math. Comp., 87, 2018.

    Provenance
    ----------
    MATLAB source : leg2cheb.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend, Nick Hale.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.

    See Also
    --------
    cheb2leg, cheb2jac, jac2cheb
    """
    c_leg = jnp.asarray(c_leg)
    if c_leg.ndim not in (1, 2):
        raise ValueError("c_leg must be a vector or a 2-D column matrix")
    n = c_leg.shape[0]
    if normalize:
        norms = jnp.sqrt(jnp.arange(n, dtype=jnp.float64) + 0.5)
        c_leg = c_leg * (norms[:, None] if c_leg.ndim == 2 else norms)
    if n <= 1:
        return c_leg.copy()

    if n <= 512:
        if trans:
            x = jnp.cos(jnp.pi * jnp.arange(n, dtype=jnp.float64) / (n - 1))
            vandermonde = _legendre_vandermonde(n - 1, x)
            return vandermonde.T @ _idct1(c_leg)
        return _leg2cheb_direct(c_leg)

    import equinox as eqx

    from chebfunjax.utils.legendre_fast import _leg2cheb_fast

    result, complete = _leg2cheb_fast(
        c_leg, trans=trans, normalize=False, max_rank=max_rank
    )
    return eqx.error_if(
        result,
        ~complete,
        "leg2cheb fast Cholesky approximation did not reach source tolerance; "
        "increase max_rank",
    )


def _leg2cheb_direct(c_leg: jnp.ndarray) -> jnp.ndarray:
    """Convert Legendre to Chebyshev coefficients using Vandermonde + vals2coeffs."""
    N = c_leg.shape[0] - 1  # degree

    # Chebyshev grid of N+1 points (descending order: cos(0)=1, ..., cos(pi)=-1)
    k = jnp.arange(N + 1, dtype=jnp.float64)
    x = jnp.cos(jnp.pi * k / N) if N > 0 else jnp.array([1.0], dtype=jnp.float64)

    # Rolling three-term recurrence via lax.scan (see cheb2leg):
    # accumulate v(x) = sum_j c_j P_j(x) without materialising the
    # Vandermonde; jit/grad traceable.
    from jax import lax

    vector_input = c_leg.ndim == 1
    c = c_leg[:, None] if vector_input else c_leg
    p0 = jnp.ones_like(x)
    if N == 0:
        v_desc = p0[:, None] * c[0, :][None, :]
    else:
        p1 = x
        v_init = c[0, :][None, :] * p0[:, None] + c[1, :][None, :] * p1[:, None]

        def _step(carry, j):
            pjm1, pj, v = carry
            pjp1 = ((2 * j + 1) * x * pj - j * pjm1) / (j + 1)
            jd = j.astype(jnp.int32)
            v = v + c[jd + 1, :][None, :] * pjp1[:, None]
            return (pj, pjp1, v), None

        if N >= 2:
            (_, _, v_desc), _ = lax.scan(
                _step, (p0, p1, v_init.astype(
                    jnp.result_type(c, x))),
                jnp.arange(1, N, dtype=jnp.float64))
        else:
            v_desc = v_init

    # Literal source local idct1, using only JAX for eager and traced input.
    # Its grid is descending; the public vals2coeffs uses ascending order.
    c_cheb = _idct1(v_desc)

    return c_cheb[:, 0] if vector_input else c_cheb


def _legendre_vandermonde(N: int, x: jnp.ndarray) -> jnp.ndarray:
    """Compute the Legendre-Chebyshev Vandermonde matrix.

    L[i, j] = P_j(x[i]) for j = 0, ..., N using the 3-term recurrence.
    """
    from jax import lax

    p0 = jnp.ones_like(x)
    if N == 0:
        return p0[:, None]
    p1 = x

    def _step(carry, j):
        pjm1, pj = carry
        pjp1 = ((2 * j + 1) * x * pj - j * pjm1) / (j + 1)
        return (pj, pjp1), pjp1

    if N >= 2:
        _, cols = lax.scan(_step, (p0, p1),
                           jnp.arange(1, N, dtype=jnp.float64))
        return jnp.concatenate(
            [jnp.stack([p0, p1], axis=1), cols.T], axis=1)
    return jnp.stack([p0, p1], axis=1)


# ===========================================================================
# Chebyshev <-> Jacobi (direct method)
# ===========================================================================

def cheb2jac(c_cheb: jnp.ndarray, alpha: float, beta: float) -> jnp.ndarray:
    """Convert Chebyshev coefficients to Jacobi coefficients.

    Converts the Chebyshev expansion
        c_cheb[0]*T_0(x) + ... + c_cheb[N-1]*T_{N-1}(x)
    to a Jacobi expansion
        c_jac[0]*P_0^{(a,b)}(x) + ... + c_jac[N-1]*P_{N-1}^{(a,b)}(x),
    where P_k^{(a,b)} is the degree-k Jacobi polynomial for the weight
    function w(x) = (1-x)^a * (1+x)^b.

    Parameters
    ----------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.
    alpha : float
        Jacobi parameter alpha (exponent for 1-x).
    beta : float
        Jacobi parameter beta (exponent for 1+x).

    Returns
    -------
    c_jac : jnp.ndarray, shape (n,)
        Jacobi coefficients.

    Provenance
    ----------
    MATLAB source : cheb2jac.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend, Nick Hale.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.
    Algorithm:
        [1] A. Townsend, M. Webb, and S. Olver, "Fast polynomial transforms
            based on Toeplitz and Hankel matrices", Math. Comp., 87, 2018.

    See Also
    --------
    jac2cheb, cheb2leg, leg2cheb
    """
    n = c_cheb.shape[0]
    if n <= 1:
        return c_cheb

    # Special case: alpha=beta=0 is Legendre
    if alpha == 0.0 and beta == 0.0:
        return cheb2leg(c_cheb)

    # Chebyshev basis is a diagonally scaled Jacobi (-1/2,-1/2) basis.
    nn = jnp.arange(n, dtype=jnp.float64)
    scl = jnp.concatenate([
        jnp.array([1.0], dtype=jnp.float64),
        jnp.cumprod((0.5 + nn[:-1]) / (1.0 + nn[:-1]))
    ])
    if alpha == -0.5 and beta == -0.5:
        return c_cheb / (scl if c_cheb.ndim == 1 else scl[:, None])

    if n > 512:
        scaled = c_cheb / (scl if c_cheb.ndim == 1 else scl[:, None])
        return _jac2jac_source(scaled, -0.5, -0.5, alpha, beta)

    return _cheb2jac_direct(c_cheb, alpha, beta)


def _cheb2jac_direct(c_cheb: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Convert Chebyshev to Jacobi coefficients via Vandermonde solve.

    Evaluates the Chebyshev expansion at N+1 Chebyshev-1st-kind points,
    then solves the Jacobi Vandermonde system to get Jacobi coefficients.
    This mirrors the MATLAB jac2cheb_direct approach (reversed).
    """
    N = c_cheb.shape[0] - 1  # degree
    if N <= 0:
        return c_cheb

    # Evaluate Chebyshev expansion at 1st-kind Chebyshev points
    n_pts = N + 1
    k = jnp.arange(n_pts, 0, -1, dtype=jnp.float64)
    x = jnp.cos((2.0 * k - 1.0) * jnp.pi / (2.0 * n_pts))

    # Evaluate the Chebyshev expansion at these points
    # T_k(x_j) = cos(k * arccos(x_j))
    theta = jnp.arccos(x)
    kk = jnp.arange(N + 1, dtype=jnp.float64)
    T = jnp.cos(kk[None, :] * theta[:, None])  # T[j, k] = T_k(x_j)
    v = T @ c_cheb  # values at the points

    # Jacobi Vandermonde at the same points
    P = _jacobi_vandermonde(N, x, a, b)

    # Solve P @ c_jac = v
    c_jac = jnp.linalg.solve(P, v)

    return c_jac


def jac2cheb(c_jac: jnp.ndarray, alpha: float, beta: float) -> jnp.ndarray:
    """Convert Jacobi coefficients to Chebyshev coefficients.

    Converts the Jacobi expansion
        c_jac[0]*P_0^{(a,b)}(x) + ... + c_jac[N-1]*P_{N-1}^{(a,b)}(x)
    to a Chebyshev expansion
        c_cheb[0]*T_0(x) + ... + c_cheb[N-1]*T_{N-1}(x).

    Parameters
    ----------
    c_jac : jnp.ndarray, shape (n,)
        Jacobi coefficients.
    alpha : float
        Jacobi parameter alpha (exponent for 1-x).
    beta : float
        Jacobi parameter beta (exponent for 1+x).

    Returns
    -------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.

    Provenance
    ----------
    MATLAB source : jac2cheb.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend, Nick Hale.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.

    See Also
    --------
    cheb2jac, cheb2leg, leg2cheb
    """
    n = c_jac.shape[0]
    if n <= 1:
        return c_jac

    # Special case: alpha=beta=0 is Legendre. Preserve matrix columns.
    if alpha == 0.0 and beta == 0.0:
        if c_jac.ndim == 1:
            return leg2cheb(c_jac)
        return jax.vmap(leg2cheb, in_axes=1, out_axes=1)(c_jac)

    if n > 512:
        converted = _jac2jac_source(c_jac, alpha, beta, -0.5, -0.5)
        nn = jnp.arange(n, dtype=jnp.float64)
        scl = jnp.concatenate([
            jnp.array([1.0], dtype=jnp.float64),
            jnp.cumprod((0.5 + nn[:-1]) / (1.0 + nn[:-1]))
        ])
        return converted * (scl if converted.ndim == 1 else scl[:, None])

    return _jac2cheb_direct(c_jac, alpha, beta)


def _jac2cheb_direct(c_jac: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Convert Jacobi to Chebyshev coefficients using Vandermonde evaluation."""
    N = c_jac.shape[0] - 1  # degree
    if N <= 0:
        return c_jac

    # Evaluate Jacobi expansion at 1st-kind Chebyshev points (like MATLAB)
    n_pts = N + 1
    # 1st-kind Chebyshev points: cos((2k-1)*pi/(2n)), k=1..n
    k = jnp.arange(n_pts, 0, -1, dtype=jnp.float64)
    x = jnp.cos((2.0 * k - 1.0) * jnp.pi / (2.0 * n_pts))

    # Jacobi Vandermonde matrix at these points
    P = _jacobi_vandermonde(N, x, a, b)

    # Values on Chebyshev grid
    v_cheb = P @ c_jac

    # Convert values at 1st-kind Chebyshev points to Chebyshev coefficients
    c_cheb = _vals2coeffs_kind1(v_cheb)

    return c_cheb


def _vals2coeffs_kind1(values: jnp.ndarray) -> jnp.ndarray:
    """Convert values at 1st-kind Chebyshev points to Chebyshev coefficients.

    Uses the relation: c_k = (2/n) sum_{j=0}^{n-1} v_j T_k(x_j), with
    appropriate scaling for k=0.
    """
    if values.ndim == 2:
        return jax.vmap(_vals2coeffs_kind1, in_axes=1, out_axes=1)(values)
    n = values.shape[0]
    if n <= 1:
        return values

    # DCT-III: c_k = (2/n) * sum_j v_j * cos(k*(2j+1)*pi/(2n))
    # Use FFT-based approach
    # Rearrange for FFT: we need DCT-III of the values

    # Using the relation to FFT:
    # The 1st-kind Chebyshev points are cos((2j-1)*pi/(2n)), j=1..n
    # values are ordered from j=n down to j=1 (ascending x), so
    # values[i] corresponds to x_i = cos((2*(n-i)-1)*pi/(2n))

    # For DCT-III via FFT:
    # Embed in length 2n DFT
    v = jnp.zeros(2 * n, dtype=jnp.float64)
    v = v.at[:n].set(values[::-1])  # reverse to match cos((2j-1)pi/2n), j=0..n-1
    v = v.at[n:].set(-values)  # antisymmetric extension

    # Shift by half-sample: multiply by exp(-i*pi*k/(2n)) in freq domain
    # Alternative: direct DCT-III computation
    # c_k = (2/n) sum_{j=0}^{n-1} values[j] * T_k(x_j)
    # where x_j = cos((2*(n-1-j)+1)*pi/(2n)) (ascending order)

    # Direct approach using explicit DCT-III formula
    j = jnp.arange(n, dtype=jnp.float64)
    k = jnp.arange(n, dtype=jnp.float64)

    # values[j] at x = cos((2j+1)*pi/(2n)) for j=0..n-1 (after reversing)
    vals_rev = values[::-1]

    # T_k(cos(theta)) = cos(k*theta), theta_j = (2j+1)*pi/(2n)
    theta = (2.0 * j + 1.0) * jnp.pi / (2.0 * n)
    # Cosine matrix: M[k,j] = cos(k * theta_j)
    M = jnp.cos(k[:, None] * theta[None, :])

    coeffs = (2.0 / n) * M @ vals_rev
    coeffs = coeffs.at[0].multiply(0.5)

    return coeffs


def _jacobi_vandermonde(
    N: int, x: jnp.ndarray, a: float, b: float
) -> jnp.ndarray:
    """Jacobi polynomial Vandermonde matrix P[i,j] = P_j^{(a,b)}(x[i]).

    Uses the standard three-term recurrence for Jacobi polynomials.
    """
    m = x.shape[0]
    apb = a + b

    P = jnp.zeros((m, N + 1), dtype=jnp.float64)
    P = P.at[:, 0].set(1.0)

    if N >= 1:
        P = P.at[:, 1].set(0.5 * (2.0 * (a + 1.0) + (apb + 2.0) * (x - 1.0)))

    aa = a * a
    bb = b * b
    for k in range(2, N + 1):
        k2 = 2 * k
        k2apb = k2 + apb
        q1 = k2 * (k + apb) * (k2apb - 2)
        q2 = (k2apb - 1) * (aa - bb)
        q3 = (k2apb - 2) * (k2apb - 1) * k2apb
        q4 = 2 * (k + a - 1) * (k + b - 1) * k2apb
        P = P.at[:, k].set(
            ((q2 + q3 * x) * P[:, k - 1] - q4 * P[:, k - 2]) / q1
        )

    return P


# ===========================================================================
# Convenience wrappers (matching MATLAB naming)
# ===========================================================================

def chebcoeffs2legcoeffs(c_cheb: jnp.ndarray) -> jnp.ndarray:
    """Convert Chebyshev coefficients to Legendre coefficients.

    Wrapper for cheb2leg.

    Provenance
    ----------
    MATLAB source : chebcoeffs2legcoeffs.m
    Chebfun commit: 7574c77
    """
    return cheb2leg(c_cheb)


def chebcoeffs2legvals(c_cheb: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a Chebyshev series at the Gauss--Legendre points.

    The output has the same leading length as the coefficient vector. For a
    two-dimensional coefficient array, columns are transformed independently.

    Provenance
    ----------
    MATLAB source : chebcoeffs2legvals.m, via @chebfun/ndct.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and
        the Chebfun Developers.
    """
    return _legendre_ndct(c_cheb)


def legcoeffs2chebcoeffs(c_leg: jnp.ndarray) -> jnp.ndarray:
    """Convert Legendre coefficients to Chebyshev coefficients.

    Wrapper for leg2cheb.

    Provenance
    ----------
    MATLAB source : legcoeffs2chebcoeffs.m
    Chebfun commit: 7574c77
    """
    return leg2cheb(c_leg)


def chebvals2legcoeffs(
    v_cheb: jnp.ndarray, *, kind: int = 2, normalize: bool = False
) -> jnp.ndarray:
    """Convert Chebyshev values to Legendre coefficients.

    First converts values at Chebyshev points to Chebyshev coefficients,
    then converts to Legendre coefficients.

    Parameters
    ----------
    v_cheb : jnp.ndarray, shape (n,)
        Values at Chebyshev points of the specified kind.
    kind : {1, 2}, default 2
        Which Chebyshev points the values come from.
    normalize : bool, default False
        If True, use orthonormal Legendre polynomials.

    Returns
    -------
    c_leg : jnp.ndarray, shape (n,)
        Legendre coefficients.

    Provenance
    ----------
    MATLAB source : chebvals2legcoeffs.m
    Chebfun commit: 7574c77
    """
    if kind == 2:
        c_cheb = vals2coeffs(v_cheb)
    elif kind == 1:
        c_cheb = _vals2coeffs_kind1(v_cheb)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")
    return cheb2leg(c_cheb, normalize=normalize)


# ===========================================================================
# Legendre values <-> coefficients/values  (DLT/iDLT wrappers)
# ===========================================================================

def _legendre_dlt(c_leg: jnp.ndarray) -> jnp.ndarray:
    """Delegate to source DLT, including matrix and large-transform branches.

    Provenance: ``legcoeffs2legvals.m`` delegates to ``@chebfun/dlt.m``,
    Chebfun commit 7574c77. The local import avoids a module import cycle.
    """
    from chebfunjax.utils.fasttransforms import dlt

    return dlt(c_leg)


def _legendre_idlt(v_leg: jnp.ndarray, *, force_fast: bool = False) -> jnp.ndarray:
    """Inverse DLT: values at Gauss-Legendre points -> Legendre coefficients.

    Uses Gauss-Legendre quadrature:
        c_k = (2k+1)/2 * sum_j w_j * P_k(x_j) * v_j

    ``force_fast`` selects the source transpose-NDCT algorithm below its
    normal 5000-row crossover.  The large-QR path uses this when its own
    matrix-free branch begins at 4001 rows; the recurrence-based direct IDLT
    loses enough accuracy in the overlapping 4001..4999 range to violate the
    unchanged QR reconstruction regression bound.

    This is the analogue of ``chebfun.idlt`` (MATLAB).
    """
    v_leg = jnp.asarray(v_leg)
    if v_leg.ndim not in (1, 2):
        raise ValueError("v_leg must be a vector or a 2-D column matrix")
    n = v_leg.shape[0]
    if n == 0:
        return v_leg
    if n == 1:
        # Literal @chebfun/idlt.m idlt_direct special case: c = 1 + 0*c.
        return jnp.ones_like(v_leg) + 0 * v_leg

    from chebfunjax.utils.quadrature import legpts

    if n >= 5000 or force_fast:
        from chebfunjax.utils.legendre_fast import _idlt_ndct_transpose_source

        x, w, _v, theta = legpts(n, newtheta=True)
        weighted = w[:, None] * v_leg if v_leg.ndim == 2 else w * v_leg
        stage1 = _idlt_ndct_transpose_source(weighted, theta)
        stage2 = leg2cheb(stage1, trans=True)
        scale = jnp.arange(n, dtype=jnp.float64) + 0.5
        return stage2 * (scale[:, None] if stage2.ndim == 2 else scale)

    x, w = legpts(n)

    # rolling scan projection: c_k = (2k+1)/2 * sum_j w_j P_k(x_j) v_j
    from jax import lax

    x = jnp.asarray(x)
    wv = (jnp.asarray(w)[:, None] * v_leg
          if v_leg.ndim == 2 else jnp.asarray(w) * v_leg)
    p0 = jnp.ones_like(x)
    p1 = x
    axis = 0
    c0 = 0.5 * jnp.sum(wv * p0[:, None], axis=axis) if v_leg.ndim == 2 else 0.5 * jnp.dot(wv, p0)
    c1 = 1.5 * jnp.sum(wv * p1[:, None], axis=axis) if v_leg.ndim == 2 else 1.5 * jnp.dot(wv, p1)

    def _step(carry, j):
        pjm1, pj = carry
        pjp1 = ((2 * j + 1) * x * pj - j * pjm1) / (j + 1)
        if v_leg.ndim == 2:
            contrib = (2 * (j + 1) + 1) / 2.0 * jnp.sum(
                wv * pjp1[:, None], axis=0
            )
        else:
            contrib = (2 * (j + 1) + 1) / 2.0 * jnp.dot(wv, pjp1)
        return (pj, pjp1), contrib

    if n >= 3:
        _, rest = lax.scan(_step, (p0, p1),
                           jnp.arange(1, n - 1, dtype=jnp.float64))
        if v_leg.ndim == 2:
            return jnp.concatenate([jnp.stack([c0, c1]), rest], axis=0)
        return jnp.concatenate([jnp.stack([c0, c1]), rest])
    if v_leg.ndim == 2:
        return jnp.stack([c0, c1], axis=0)[:n, :]
    return jnp.stack([c0, c1])[:n]


def ndct(x: jnp.ndarray, coeffs: jnp.ndarray | None = None,
         theta: jnp.ndarray | None = None) -> jnp.ndarray:
    """Evaluate Chebyshev coefficient columns with MATLAB's K=16 fast NDCT.

    ``ndct(coeffs)`` uses Gauss-Legendre nodes. ``ndct(x, coeffs)`` uses
    supplied nodes; ``ndct(x, coeffs, theta)`` uses the supplied angles
    instead of computing acos(x). Node/angle matrices are flattened in
    column order. Coefficient vectors produce vectors and coefficient
    matrices produce matrices with one output row per evaluation point.

    The source returns the imaginary component as a real result when the
    entire coefficient array is purely imaginary. That unusual numerical
    behavior is preserved. Complex input retains a complex JAX output dtype,
    since JIT output dtypes cannot depend on coefficient values.

    Provenance
    ----------
    MATLAB source : @chebfun/ndct.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.utils.ndct_fast import _ndct_fast

    if coeffs is None:
        coeffs = jnp.asarray(x)
        from chebfunjax.utils.quadrature import legpts
        x, _w, _v, source_theta = legpts(coeffs.shape[0], newtheta=True)
        if theta is None:
            theta = source_theta
    else:
        coeffs = jnp.asarray(coeffs)
    if theta is None:
        # MATLAB takes real(acos(x)), including complex/outside-interval x.
        nodes = jnp.asarray(x).astype(jnp.complex128)
        theta = jnp.real(jnp.arccos(nodes))
    theta = jnp.real(jnp.asarray(theta)).reshape(-1, order="F")
    values = _ndct_fast(coeffs, theta)
    if jnp.issubdtype(coeffs.dtype, jnp.complexfloating):
        values = jnp.where(jnp.all(jnp.real(coeffs) == 0),
                           jnp.imag(values), values)
    return values


def _legendre_ndct(c_cheb: jnp.ndarray) -> jnp.ndarray:
    """Source backwards-compatible NDCT wrapper at Gauss-Legendre nodes."""
    return ndct(c_cheb)


def legvals2legcoeffs(v_leg: jnp.ndarray) -> jnp.ndarray:
    """Convert values at Gauss-Legendre points to Legendre coefficients.

    LEGCOEFFS = LEGVALS2LEGCOEFFS(V_LEG) converts the vector V_LEG of values
    at Gauss-Legendre points (LEGPTS(N)) to Legendre series coefficients
    C such that
        f(x) = C[0]*P_0(x) + C[1]*P_1(x) + ... + C[N-1]*P_{N-1}(x).

    Parameters
    ----------
    v_leg : jnp.ndarray, shape (n,)
        Values at n Gauss-Legendre points.

    Returns
    -------
    c_leg : jnp.ndarray, shape (n,)
        Legendre series coefficients.

    Provenance
    ----------
    MATLAB source : legvals2legcoeffs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    return _legendre_idlt(v_leg)


def legcoeffs2legvals(c_leg: jnp.ndarray) -> jnp.ndarray:
    """Convert Legendre coefficients to values at Gauss-Legendre points.

    LEGVALS = LEGCOEFFS2LEGVALS(C_LEG) evaluates the Legendre expansion
        f(x) = C_LEG[0]*P_0(x) + ... + C_LEG[N-1]*P_{N-1}(x)
    at LEGPTS(N), i.e., the N Gauss-Legendre nodes.

    Parameters
    ----------
    c_leg : jnp.ndarray, shape (n,)
        Legendre coefficients.

    Returns
    -------
    v_leg : jnp.ndarray, shape (n,)
        Values at Gauss-Legendre points.

    Provenance
    ----------
    MATLAB source : legcoeffs2legvals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    return _legendre_dlt(c_leg)


def legvals2chebcoeffs(v_leg: jnp.ndarray) -> jnp.ndarray:
    """Convert Legendre values to Chebyshev coefficients.

    C_CHEB = LEGVALS2CHEBCOEFFS(V_LEG) converts values at Gauss-Legendre
    points to Chebyshev coefficients of the interpolating polynomial.

    Parameters
    ----------
    v_leg : jnp.ndarray, shape (n,)
        Values at Gauss-Legendre points.

    Returns
    -------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev series coefficients.

    Provenance
    ----------
    MATLAB source : legvals2chebcoeffs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    c_leg = _legendre_idlt(v_leg)
    return leg2cheb(c_leg)


def legvals2chebvals(v_leg: jnp.ndarray, *, kind: int = 2) -> jnp.ndarray:
    """Convert values at Gauss-Legendre points to values at Chebyshev points.

    CHEBVALS = LEGVALS2CHEBVALS(LEGVALS) converts values of a polynomial at
    Gauss-Legendre points to values at 2nd-kind Chebyshev points.

    Parameters
    ----------
    v_leg : jnp.ndarray, shape (n,)
        Values at Gauss-Legendre points.
    kind : {1, 2}, default 2
        Target Chebyshev grid kind.

    Returns
    -------
    v_cheb : jnp.ndarray, shape (n,)
        Values at Chebyshev points of the given kind.

    Provenance
    ----------
    MATLAB source : legvals2chebvals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    c_leg = _legendre_idlt(v_leg)
    return legcoeffs2chebvals(c_leg, kind=kind)


def legcoeffs2chebvals(c_leg: jnp.ndarray, *, kind: int = 2) -> jnp.ndarray:
    """Convert Legendre coefficients to values at Chebyshev points.

    V_CHEB = LEGCOEFFS2CHEBVALS(C_LEG) converts Legendre coefficients to
    values at 2nd-kind Chebyshev points.

    Parameters
    ----------
    c_leg : jnp.ndarray, shape (n,)
        Legendre coefficients.
    kind : {1, 2}, default 2
        Target Chebyshev grid kind.  1 = first-kind Chebyshev points,
        2 = second-kind (Clenshaw-Curtis).

    Returns
    -------
    v_cheb : jnp.ndarray, shape (n,)
        Values at Chebyshev points.

    Provenance
    ----------
    MATLAB source : legcoeffs2chebvals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    c_cheb = leg2cheb(c_leg)
    if kind == 2:
        return coeffs2vals(c_cheb)
    elif kind == 1:
        n = c_cheb.shape[0]
        if n <= 1:
            return c_cheb
        # Evaluate the T-series at 1st-kind Chebyshev points: plain T @ c.
        # (leg2cheb returns standard T-series coefficients; the previous
        # halve-endpoints-then-double scaling belonged to a DCT
        # normalization and corrupted the result.)
        from chebfunjax.utils.quadrature import chebpts
        x = chebpts(n, kind=1)
        theta = jnp.arccos(jnp.clip(x, -1.0, 1.0))
        k = jnp.arange(n, dtype=jnp.float64)
        T = jnp.cos(k[None, :] * theta[:, None])
        return T @ c_cheb
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")


def chebvals2legvals(v_cheb: jnp.ndarray, *, kind: int = 2) -> jnp.ndarray:
    """Convert Chebyshev values to values at Gauss-Legendre points.

    LEGVALS = CHEBVALS2LEGVALS(CHEBVALS) converts values at 2nd-kind
    Chebyshev points to values at Gauss-Legendre points.

    Parameters
    ----------
    v_cheb : jnp.ndarray, shape (n,)
        Values at Chebyshev points.
    kind : {1, 2}, default 2
        Kind of source Chebyshev points.

    Returns
    -------
    v_leg : jnp.ndarray, shape (n,)
        Values at Gauss-Legendre points.

    Provenance
    ----------
    MATLAB source : chebvals2legvals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    if kind == 2:
        c_cheb = vals2coeffs(v_cheb)
    elif kind == 1:
        c_cheb = _vals2coeffs_kind1(v_cheb)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")
    return _legendre_ndct(c_cheb)


def chebvals2chebvals(
    v_in: jnp.ndarray, kind1: int, kind2: int
) -> jnp.ndarray:
    """Convert between first- and second-kind Chebyshev grids.

    V_OUT = CHEBVALS2CHEBVALS(V_IN, KIND1, KIND2) converts values of a
    polynomial on a Chebyshev grid of kind KIND1 to a grid of kind KIND2.

    Parameters
    ----------
    v_in : jnp.ndarray, shape (n,)
        Input values on a Chebyshev grid.
    kind1 : {1, 2}
        Source grid kind.
    kind2 : {1, 2}
        Target grid kind.

    Returns
    -------
    v_out : jnp.ndarray, shape (n,)
        Values on the target grid.

    Provenance
    ----------
    MATLAB source : chebvals2chebvals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    if kind1 == kind2:
        return v_in
    elif kind1 == 1 and kind2 == 2:
        c = _vals2coeffs_kind1(v_in)
        return coeffs2vals(c)
    elif kind1 == 2 and kind2 == 1:
        c = vals2coeffs(v_in)
        n = c.shape[0]
        if n <= 1:
            return c
        from chebfunjax.utils.quadrature import chebpts
        x = chebpts(n, kind=1)
        theta = jnp.arccos(jnp.clip(x, -1.0, 1.0))
        k = jnp.arange(n, dtype=jnp.float64)
        T = jnp.cos(k[None, :] * theta[:, None])
        # vals2coeffs returns standard T-series coefficients
        # (p(x) = sum c_k T_k), so evaluation is a plain T @ c — the
        # halve-interior-then-double scaling belonged to a DCT
        # normalization and corrupted the result.
        return T @ c
    else:
        raise ValueError(f"kind1 and kind2 must be 1 or 2, got {kind1}, {kind2}")


def chebcoeffs2chebvals(c_cheb: jnp.ndarray, *, kind: int = 2) -> jnp.ndarray:
    """Convert Chebyshev coefficients to values at Chebyshev points.

    Wrapper for coeffs2vals (kind=2) or evaluation at 1st-kind points.

    Parameters
    ----------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.
    kind : {1, 2}, default 2

    Returns
    -------
    v : jnp.ndarray, shape (n,)

    Provenance
    ----------
    MATLAB source : chebcoeffs2chebvals.m
    Chebfun commit: 7574c77
    """
    if kind == 2:
        return coeffs2vals(c_cheb)
    elif kind == 1:
        return chebvals2chebvals(coeffs2vals(c_cheb), kind1=2, kind2=1)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")


def chebvals2chebcoeffs(v: jnp.ndarray, *, kind: int = 2) -> jnp.ndarray:
    """Convert Chebyshev values to Chebyshev coefficients.

    Wrapper for vals2coeffs (kind=2) or 1st-kind variant.

    Parameters
    ----------
    v : jnp.ndarray, shape (n,)
        Values at Chebyshev points.
    kind : {1, 2}, default 2

    Returns
    -------
    c_cheb : jnp.ndarray, shape (n,)

    Provenance
    ----------
    MATLAB source : chebvals2chebcoeffs.m
    Chebfun commit: 7574c77
    """
    if kind == 2:
        return vals2coeffs(v)
    elif kind == 1:
        return _vals2coeffs_kind1(v)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")


# ===========================================================================
# Jacobi-to-Jacobi transform  (jac2jac)
# ===========================================================================

# uses-numpy: iterative pivoted Cholesky and FFT-based Toeplitz-Hankel multiply

def jac2jac(
    c_jac: jnp.ndarray,
    alpha: float,
    beta: float,
    gam: float,
    delta: float,
) -> jnp.ndarray:
    """Convert Jacobi (alpha, beta) coefficients to Jacobi (gam, delta) coefficients.

    C_OUT = JAC2JAC(C_IN, A, B, G, D) converts the vector C_IN of Jacobi
    P^{(A,B)} coefficients to P^{(G,D)} coefficients such that
        C_IN[0]*P_0^{(A,B)}(x) + ... + C_IN[N-1]*P_{N-1}^{(A,B)}(x)
      = C_OUT[0]*P_0^{(G,D)}(x) + ... + C_OUT[N-1]*P_{N-1}^{(G,D)}(x).

    Parameters
    ----------
    c_jac : jnp.ndarray, shape (n,)
        Jacobi (alpha, beta) coefficients.
    alpha, beta : float
        Source Jacobi parameters.
    gam, delta : float
        Target Jacobi parameters.

    Returns
    -------
    c_out : jnp.ndarray, shape (n,)
        Jacobi (gam, delta) coefficients.

    Notes
    -----
    Uses the algorithm from [1]: the conversion matrix decomposes as
    D1*(T.*H)*D2 where T is Toeplitz and H is a Hankel matrix approximated
    by pivoted Cholesky.  O(n log n) per rank-1 update.

    References
    ----------
    .. [1] A. Townsend, M. Webb, and S. Olver, "Fast polynomial transforms
       based on Toeplitz and Hankel matrices", Math. Comp., 87, 2018.

    Provenance
    ----------
    MATLAB source : jac2jac.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend, Marcus Webb, Sheehan Olver.
        Copyright 2017 by The University of Oxford and The Chebfun Developers.

    See Also
    --------
    cheb2jac, jac2cheb, ultra2ultra
    """
    # Work in numpy (iterative algorithm, data-dependent branching)
    v = np.array(c_jac, dtype=np.float64)
    v = _jac2jac_np(v, alpha, beta, gam, delta)
    return jnp.array(v)


def _jac2jac_source(
    c_jac: jnp.ndarray,
    alpha: float,
    beta: float,
    gam: float,
    delta: float,
) -> jnp.ndarray:
    """Jacobi conversion through the native integer and fractional stages."""
    values = jnp.asarray(c_jac)
    values, alpha, beta = _jacobi_integer_conversion(
        values, alpha, beta, gam, delta
    )
    if abs(alpha - gam) > 1e-15:
        values = _jacobi_fractional_conversion(values, alpha, beta, gam)
    if abs(beta - delta) > 1e-15:
        values = values.at[1::2, ...].multiply(-1)
        values = _jacobi_fractional_conversion(values, beta, gam, delta)
        values = values.at[1::2, ...].multiply(-1)
    return jnp.asarray(values)


def _jac2jac_np(
    v: np.ndarray,
    alpha: float,
    beta: float,
    gam: float,
    delta: float,
) -> np.ndarray:
    """Numpy O(n^2) implementation of jac2jac via Gauss-Jacobi quadrature.

    Evaluates the source Jacobi expansion at (gam,delta) Gauss-Jacobi nodes,
    then projects onto the target basis via the Gauss-Jacobi inner products.
    This is O(n^2) but correct and numerically stable for n up to a few hundred.
    """
    N = len(v)
    if N == 0:
        return v
    if N == 1:
        return v

    # If source == target, identity
    if abs(alpha - gam) < 1e-14 and abs(beta - delta) < 1e-14:
        return v.copy()

    # Gauss-Jacobi nodes and weights for (gam, delta) — used for quadrature
    from chebfunjax.utils.quadrature import jacpts
    x, w = jacpts(N, gam, delta)
    x_np = np.array(x, dtype=np.float64)
    w_np = np.array(w, dtype=np.float64)

    # Evaluate source expansion at x_np: f(x) = sum_k v[k] * P_k^{(alpha,beta)}(x)
    P_src = np.array(_jacobi_vandermonde(N - 1, jnp.array(x_np), alpha, beta))  # (N, N)
    f_vals = P_src @ v

    # Project f onto target Jacobi basis using Gauss-Jacobi quadrature:
    # c_k = (2k + gam + delta + 1)/(2^{gam+delta+1}) * B(k+gam+1,k+delta+1)/(k! * ...)
    # Actually: c_k = h_k^{-1} * sum_j w_j * P_k^{(gam,delta)}(x_j) * f(x_j)
    # where h_k = 2^{gam+delta+1} * Gamma(k+gam+1)*Gamma(k+delta+1) / ((2k+gam+delta+1)*Gamma(k+1)*Gamma(k+gam+delta+1))

    P_tgt = np.array(_jacobi_vandermonde(N - 1, jnp.array(x_np), gam, delta))  # (N, N)

    # Normalization constants h_k (squared norm of P_k^{(gam,delta)})
    k = np.arange(N, dtype=np.float64)
    s = gam + delta + 1
    # Denominator log((2k+s) * Gamma(k+s)): at k=0 this is log(s*Gamma(s))
    # = gammaln(s+1), which stays finite as s -> 0 (e.g. gam=delta=-0.5,
    # the Chebyshev weight) where the naive form gives inf - inf = NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_den = np.log(2 * k + s) + gammaln(k + s)
    log_den[0] = gammaln(s + 1)
    h_k = np.exp(
        s * np.log(2)
        + gammaln(k + gam + 1) + gammaln(k + delta + 1)
        - log_den
        - gammaln(k + 1)
    )

    c_out = (P_tgt.T @ (w_np * f_vals)) / h_k
    return c_out


def _jacobi_integer_conversion(
    v: jnp.ndarray,
    alpha: float,
    beta: float,
    gam: float,
    delta: float,
) -> tuple[jnp.ndarray, float, float]:
    """Move (alpha,beta) to (A,B) so that |A-gam|<1 and |B-delta|<1."""
    a, b = float(alpha), float(beta)

    while a <= gam - 1:
        v = _right_jacobi(v, a, b)
        a += 1
    while a >= gam + 1:
        v = _left_jacobi(v, a - 1, b)
        a -= 1
    while b <= delta - 1:
        v = _up_jacobi(v, a, b)
        b += 1
    while b >= delta + 1:
        v = _down_jacobi(v, a, b - 1)
        b -= 1

    return v, a, b


def _up_jacobi(v: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Convert Jacobi (a,b) -> (a,b+1) in O(n) operations."""
    vector_input = v.ndim == 1
    values = jnp.asarray(v)[:, None] if vector_input else jnp.asarray(v)
    N = values.shape[0]
    nn = jnp.arange(N, dtype=jnp.float64)
    apb = a + b
    d1 = jnp.concatenate((
        jnp.ones((1,), dtype=jnp.float64),
        jnp.asarray([(apb + 2) / (apb + 3)], dtype=jnp.float64) if N > 1 else jnp.empty((0,), dtype=jnp.float64),
        (apb + 3 + nn[:-2]) / (apb + 5 + 2 * nn[:-2]) if N > 2 else jnp.empty((0,), dtype=jnp.float64),
    ))
    d2 = (a + 1 + nn[:N - 1]) / (apb + 3 + 2 * nn[:N - 1])
    out = d1[:, None] * values
    out = out.at[:N - 1].add(d2[:, None] * values[1:])
    return out[:, 0] if vector_input else out


def _down_jacobi(v: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Convert Jacobi (a,b+1) -> (a,b) by inverting _up_jacobi."""
    vector_input = v.ndim == 1
    values = jnp.asarray(v)[:, None] if vector_input else jnp.asarray(v)
    N = values.shape[0]
    nn = jnp.arange(N, dtype=jnp.float64)
    apb = a + b
    ratio1 = (a + 1) / (apb + 2)
    factors = (a + jnp.arange(2, N, dtype=jnp.float64)) / (apb + jnp.arange(3, N + 1, dtype=jnp.float64))
    top_tail = ratio1 * jnp.cumprod(factors)
    topRow = jnp.concatenate((jnp.ones((1,), dtype=jnp.float64), jnp.asarray([ratio1]), top_tail))[:N]
    signs = (-1.0) ** nn
    topRow = topRow * signs
    tv = topRow[:, None] * values
    vecsum = jnp.cumsum(tv[::-1, :], axis=0)[::-1, :]
    ratios = jnp.concatenate((
        jnp.ones((1,), dtype=jnp.float64),
        jnp.asarray([-(apb + 3) / (a + 1)], dtype=jnp.float64) if N > 1 else jnp.empty((0,), dtype=jnp.float64),
        ((apb + 5 + 2 * nn[:-2]) / (apb + 3 + nn[:-2]) / topRow[2:]) if N > 2 else jnp.empty((0,), dtype=jnp.float64),
    ))

    out = ratios[:, None] * vecsum
    return out[:, 0] if vector_input else out


def _right_jacobi(v: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Convert Jacobi (a,b) -> (a+1,b) using reflection formula."""
    v = v.at[1::2].multiply(-1)
    v = _up_jacobi(v, b, a)
    v = v.at[1::2].multiply(-1)
    return v


def _left_jacobi(v: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Convert Jacobi (a+1,b) -> (a,b) using reflection formula."""
    v = v.at[1::2].multiply(-1)
    v = _down_jacobi(v, b, a)
    v = v.at[1::2].multiply(-1)
    return v


def _jacobi_fractional_conversion(
    v: jnp.ndarray,
    alpha: float,
    beta: float,
    gam: float,
) -> jnp.ndarray:
    """Convert Jacobi (alpha,beta) -> (gam,beta) with |alpha-gam|<1.

    Uses the Toeplitz-Hankel decomposition from Townsend-Webb-Olver [1].
    The conversion matrix A = D1*(T.*H)*D2, where T is Toeplitz and
    H is Hankel approximated by pivoted Cholesky.

    References
    ----------
    .. [1] A. Townsend, M. Webb, and S. Olver, 2018.
    """
    vector_input = v.ndim == 1
    values = jnp.asarray(v)[:, None] if vector_input else jnp.asarray(v)
    N = values.shape[0]
    if N <= 1:
        return v

    d1, d2, T_row, vals_first, C_arr, a_fft = _jacobi_fractional_plan(
        N, alpha, beta, gam
    )
    c_work = d2[:, None] * values

    def transform_column(column):
        tmp = C_arr * column[:, None]
        f1 = jnp.fft.fft(tmp, n=2 * N - 1, axis=0)
        b = jnp.fft.ifft(f1 * a_fft[:, None], axis=0)
        return d1 * jnp.sum(b[:N, :] * C_arr, axis=1)

    # MATLAB processes each coefficient column separately; lax.map keeps the
    # FFT workspace at O(N * numerical_rank) for each column.
    result = jax.lax.map(transform_column, c_work.T).T

    matrow1 = (
        jax.scipy.special.gamma(gam + beta + 2)
        / jax.scipy.special.gamma(beta + 1)
        * d2 * T_row * vals_first
    )
    result = result.at[0, :].set(matrow1 @ values + values[0, :])

    # The real source transform maps real coefficients to real coefficients;
    # FFT roundoff can leave a tiny imaginary component in the implementation.
    if not jnp.issubdtype(values.dtype, jnp.complexfloating):
        result = jnp.real(result)
    return result[:, 0] if vector_input else result


@lru_cache(maxsize=8)
def _jacobi_fractional_plan(
    N: int,
    alpha: float,
    beta: float,
    gam: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cache native Jacobi conversion factors at their source stopping rank."""
    with jax.ensure_compile_time_eval():
        def lambda1(z):
            return jnp.exp(
                jax.scipy.special.gammaln(z + alpha + beta + 1)
                - jax.scipy.special.gammaln(z + gam + beta + 2)
            )

        def lambda2(z):
            return jnp.exp(
                jax.scipy.special.gammaln(z + alpha - gam)
                - jax.scipy.special.gammaln(z + 1)
            )

        def lambda3(z):
            return jnp.exp(
                jax.scipy.special.gammaln(z + gam + beta + 1)
                - jax.scipy.special.gammaln(z + beta + 1)
            )

        def lambda4(z):
            return jnp.exp(
                jax.scipy.special.gammaln(z + beta + 1)
                - jax.scipy.special.gammaln(z + alpha + beta + 1)
            )

        nn = jnp.arange(N, dtype=jnp.float64)
        source_indices = jnp.concatenate((jnp.asarray([1.0]), nn[1:]))
        d1 = (2 * nn + gam + beta + 1) * lambda3(source_indices)
        d1 = d1.at[0].set(1.0)
        d2 = lambda4(source_indices) / jax.scipy.special.gamma(alpha - gam)
        d2 = d2.at[0].set(0.0)

        vals = lambda1(jnp.arange(1, 2 * N + 1, dtype=jnp.float64))
        vals_h = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), vals[:-1]))
        diagonal = vals_h[0::2]
        tol = 1e-14 * math.log(N)
        chol = jnp.empty((N, 0), dtype=jnp.float64)
        pivots = jnp.empty((0,), dtype=jnp.float64)
        peak = float(jnp.max(diagonal))
        while peak > tol:
            pivot = int(jnp.argmax(diagonal))
            mx = diagonal[pivot]
            col = vals_h[pivot:pivot + N]
            if chol.shape[1]:
                col = col - chol @ (chol[pivot, :] * pivots)
            chol = jnp.concatenate((chol, col[:, None]), axis=1)
            pivots = jnp.concatenate((pivots, (1.0 / mx)[None]))
            diagonal = jnp.maximum(diagonal - col**2 / mx, 0.0)
            peak = float(jnp.max(diagonal))
        chol = chol * jnp.sqrt(pivots)[None, :]

        T_row = lambda2(nn).at[0].set(
            jax.scipy.special.gamma(alpha - gam + 1) / (alpha - gam)
        )
        Z = jnp.concatenate((T_row[:1], jnp.zeros((N - 1,), dtype=jnp.float64)))
        a_fft = jnp.fft.fft(jnp.concatenate((Z, T_row[N - 1:0:-1])))
        return d1, d2, T_row, vals_h[:N], chol, a_fft


# ===========================================================================
# Ultra-spherical transforms
# ===========================================================================

def ultra2ultra(c: jnp.ndarray, lam_in: float, lam_out: float) -> jnp.ndarray:
    """Convert between ultraspherical (Gegenbauer) expansions.

    C_OUT = ULTRA2ULTRA(C_IN, LAM_IN, LAM_OUT) converts the vector C_IN of
    ultraspherical C^{(lam_in)} coefficients to C^{(lam_out)} coefficients.

    Ultraspherical polynomials C_n^{(lambda)} are a special case of Jacobi
    polynomials:  C_n^{(lam)} ∝ P_n^{(lam-1/2, lam-1/2)}.
    This function uses the jac2jac algorithm internally.

    Parameters
    ----------
    c : jnp.ndarray, shape (n,)
        Ultraspherical C^{(lam_in)} coefficients.
    lam_in : float
        Source ultraspherical parameter (must be >= 0).
    lam_out : float
        Target ultraspherical parameter (must be >= 0).

    Returns
    -------
    c_out : jnp.ndarray, shape (n,)
        Ultraspherical C^{(lam_out)} coefficients.

    Notes
    -----
    The scaling from ultraspherical to Jacobi and back follows DLMF Table 18.3.1.
    For lam=0 the polynomial reduces to Legendre / T_n (Chebyshev).

    Provenance
    ----------
    MATLAB source : ultra2ultra.m
    Chebfun commit: 7574c77
    Original authors: Alex Townsend. Copyright 2017 by The University of
        Oxford and The Chebfun Developers.

    See Also
    --------
    jac2jac, ultracoeffs
    """
    n = c.shape[0] - 1

    def _scl(lam: float) -> np.ndarray:
        """Scaling from Jacobi to ultraspherical (DLMF Table 18.3.1)."""
        if lam == 0.0:
            nn_scl = np.arange(n, dtype=np.float64)
            s = np.concatenate([[1.0], np.cumprod((nn_scl + 0.5) / (nn_scl + 1.0))])
        else:
            nn_scl = np.arange(n + 1, dtype=np.float64)
            s = (np.exp(gammaln(2 * lam) - gammaln(lam + 0.5))
                 * np.exp(gammaln(lam + 0.5 + nn_scl) - gammaln(2 * lam + nn_scl)))
        return s

    c_np = np.array(c, dtype=np.float64)

    # Scale from US to Jacobi
    c_np = c_np / _scl(lam_in)

    # Convert Jacobi (lam_in-0.5, lam_in-0.5) -> (lam_out-0.5, lam_out-0.5)
    c_np = _jac2jac_np(c_np, lam_in - 0.5, lam_in - 0.5, lam_out - 0.5, lam_out - 0.5)

    # Scale from Jacobi to US
    c_np = c_np * _scl(lam_out)

    return jnp.array(c_np)


def ultracoeffs(c_cheb: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Compute ultraspherical series coefficients from Chebyshev coefficients.

    A = ULTRACOEFFS(C_CHEB, LAM) converts the Chebyshev coefficients C_CHEB
    to ultraspherical C^{(lam)} coefficients A such that
        f(x) ≈ A[0]*C_0^{(lam)}(x) + A[1]*C_1^{(lam)}(x) + ...

    Parameters
    ----------
    c_cheb : jnp.ndarray, shape (n,)
        Chebyshev coefficients.
    lam : float
        Ultraspherical parameter.  Must be > 0.

    Returns
    -------
    c_ultra : jnp.ndarray, shape (n,)
        Ultraspherical coefficients.

    Notes
    -----
    Internally converts Chebyshev -> Jacobi (lam-0.5, lam-0.5) -> ultraspherical.

    Provenance
    ----------
    MATLAB source : ultracoeffs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    ultra2ultra, cheb2jac
    """
    if lam <= 0.0:
        raise ValueError("Ultraspherical polynomials require lam > 0.")
    if lam == 0.5:
        # US(0.5) == Legendre
        return cheb2leg(c_cheb)
    if lam == 1.0:
        # C^(1) = U (Chebyshev 2nd kind), but T-series coefficients must be
        # CONVERTED to U-series coefficients (returning them unchanged was
        # wrong): T_0 = U_0, T_1 = U_1/2, T_n = (U_n - U_{n-2})/2 for n >= 2,
        # so b_0 = a_0 - a_2/2 and b_k = (a_k - a_{k+2})/2 for k >= 1.
        n = c_cheb.shape[0]
        a = jnp.asarray(c_cheb, dtype=jnp.float64)
        if n <= 1:
            return a
        a_shift = jnp.concatenate(
            [a[2:], jnp.zeros(min(2, n), dtype=a.dtype)]
        )[:n]
        b = 0.5 * (a - a_shift)
        b0 = a[0] - 0.5 * a[2] if n > 2 else a[0]
        return b.at[0].set(b0)

    # Convert Chebyshev -> Jacobi (ab, ab) where ab = lam - 0.5
    ab = lam - 0.5
    c_jac = cheb2jac(c_cheb, ab, ab)

    # Scale from Jacobi to ultraspherical
    n = c_jac.shape[0]
    from scipy.special import gammaln as _gammaln
    scl = jnp.array(
        np.exp(_gammaln(2 * lam) - _gammaln(lam + 0.5))
        * np.exp(
            np.array([float(_gammaln(lam + 0.5 + k)) for k in range(n)])
            - np.array([float(_gammaln(2 * lam + k)) for k in range(n)])
        )
    )
    return c_jac * scl


# ===========================================================================
# Discrete Sine Transform (DST) and inverse (IDST)
# ===========================================================================


# uses-numpy: DST/IDST use scipy.fft which is not JAX-JIT-safe
def dst(u: jnp.ndarray, kind: int = 1) -> jnp.ndarray:
    r"""Discrete Sine Transform (DST) of type *kind* on a vector or matrix.

    Implements the Wikipedia / MATLAB Chebfun convention.  Types 1–4 are
    supported.  If *u* is a 2-D array the transform is applied column-wise.

    The implementation delegates to ``scipy.fft.dst`` for numerical accuracy
    and is therefore NOT JIT-safe.

    Parameters
    ----------
    u : jnp.ndarray, shape (n,) or (n, m)
        Input values (real).
    kind : {1, 2, 3, 4}
        DST type.  Default 1 (consistent with MATLAB PDE toolbox / Chebfun).

    Returns
    -------
    y : jnp.ndarray, same shape as *u*

    Provenance
    ----------
    MATLAB source : @chebfun/dst.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    idst

    Examples
    --------
    Round-trip test (DST-1 is its own inverse up to a scale):

    >>> import jax.numpy as jnp
    >>> from chebfunjax.utils.transforms import dst, idst
    >>> u = jnp.array([1.0, 2.0, 3.0])
    >>> v = dst(u, 1)
    >>> u2 = idst(v, 1)
    >>> float(jnp.max(jnp.abs(u - u2))) < 1e-12
    True
    """
    import scipy.fft as _sfft
    u_np = np.array(u, dtype=np.float64)
    y_np = _sfft.dst(u_np, type=kind, axis=0, norm="backward")
    return jnp.array(y_np, dtype=jnp.float64)


def idst(c: jnp.ndarray, kind: int = 1) -> jnp.ndarray:
    r"""Inverse Discrete Sine Transform of type *kind*.

    Computes the exact inverse of :func:`dst` under the same
    Wikipedia / MATLAB Chebfun scaling convention.  Delegates to
    ``scipy.fft.idst`` — not JIT-safe.

    Parameters
    ----------
    c : jnp.ndarray, shape (n,) or (n, m)
        DST coefficients produced by :func:`dst`.
    kind : {1, 2, 3, 4}
        DST type.

    Returns
    -------
    u : jnp.ndarray, same shape as *c*

    Provenance
    ----------
    MATLAB source : @chebfun/idst.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    dst

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from chebfunjax.utils.transforms import dst, idst
    >>> u = jnp.array([1.0, 2.0, 3.0, 4.0])
    >>> v = dst(u, 2)
    >>> u2 = idst(v, 2)
    >>> float(jnp.max(jnp.abs(u - u2))) < 1e-12
    True
    """
    import scipy.fft as _sfft
    c_np = np.array(c, dtype=np.float64)
    u_np = _sfft.idst(c_np, type=kind, axis=0, norm="backward")
    return jnp.array(u_np, dtype=jnp.float64)

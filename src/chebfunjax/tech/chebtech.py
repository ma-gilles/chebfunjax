"""Chebyshev technology — smooth function approximation on [-1, 1].

Translated from MATLAB Chebfun class @chebtech2 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import warnings
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.utils.misc import standard_chop
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Clenshaw evaluation (JIT-safe, grad-safe, vmap-safe)
# ============================================================================


def _as_fun_dtype(v: jax.Array) -> jax.Array:
    """Coerce sampled data to float64, or complex128 for complex input.

    MATLAB Chebfun represents complex-valued functions natively; forcing
    float64 silently discarded imaginary parts. The dtype check is a
    trace-time constant, so callers stay JIT-safe.
    """
    v = jnp.asarray(v)
    if jnp.iscomplexobj(v):
        return v.astype(jnp.complex128)
    return v.astype(jnp.float64)


def _as_scalar(v):
    """Coerce a Python/JAX scalar preserving complexness."""
    if isinstance(v, complex) or jnp.iscomplexobj(v):
        return jnp.complex128(v)
    return jnp.float64(v)


def _clenshaw(coeffs: jax.Array, x: jax.Array) -> jax.Array:
    """Evaluate a Chebyshev series at point(s) x via Clenshaw's algorithm.

    Computes  f(x) = c[0]*T_0(x) + c[1]*T_1(x) + ... + c[n-1]*T_{n-1}(x)
    using the three-term recurrence for Chebyshev polynomials of the first kind.

    Parameters
    ----------
    coeffs : jax.Array, shape (n,)
        Chebyshev series coefficients c[0], c[1], ..., c[n-1].
    x : jax.Array, shape ()  or (m,)
        Evaluation point(s) in [-1, 1].

    Returns
    -------
    y : jax.Array, same shape as x
        Evaluated values.

    Notes
    -----
    This function is JIT-safe, grad-safe, and vmap-safe. It uses
    ``jax.lax.fori_loop`` so the number of iterations is determined only by the
    static shape of ``coeffs``, which makes it trace-friendly.

    The algorithm is the standard Clenshaw recurrence:
        b_{n+1} = b_n = 0
        b_k = c[k] + 2*x*b_{k+1} - b_{k+2}    for k = n-1, ..., 1
        f(x) = c[0] + x*b_1 - b_2

    Provenance
    ----------
    MATLAB source : @chebtech/clenshaw.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebtech2.__call__
    """
    n = coeffs.shape[0]

    # Array-valued series: coeffs (n, m) evaluated at x (...,) gives
    # values of shape x.shape + (m,) -- the recurrence broadcasts a
    # trailing column axis (Fable 5 array-valued support).
    multi = coeffs.ndim == 2
    if multi:
        x = jnp.asarray(x)[..., None]      # (..., 1) vs (n, m) rows

    # Edge cases
    if n == 0:
        return jnp.zeros_like(x, dtype=jnp.float64)
    if n == 1:
        if multi:
            return jnp.broadcast_to(
                coeffs[0], x.shape[:-1] + coeffs.shape[1:])
        return jnp.broadcast_to(coeffs[0], x.shape)

    x2 = 2.0 * x

    # Clenshaw recurrence from the top.
    # We use lax.fori_loop for JIT friendliness.
    # State: (bk1, bk2)  — one step behind and two steps behind.
    def body(i, state):
        bk1, bk2 = state
        # k = n - 1 - i  (we iterate i = 0..n-2)
        k = n - 1 - i
        bk = coeffs[k] + x2 * bk1 - bk2
        return (bk, bk1)

    # Carry dtype must match the series dtype (complex chebfuns give a
    # complex recurrence; a float64 carry breaks the lax scan/loop typing).
    out_dtype = jnp.result_type(coeffs.dtype, x.dtype)
    carry_shape = jnp.broadcast_shapes(
        x.shape, coeffs.shape[1:] if multi else ())
    init = (
        jnp.zeros(carry_shape, dtype=out_dtype),
        jnp.zeros(carry_shape, dtype=out_dtype),
    )
    bk1, bk2 = jax.lax.fori_loop(0, n - 1, body, init)

    # Final step: f(x) = c[0] + x*bk1 - bk2
    return coeffs[0] + x * bk1 - bk2


# ============================================================================
# Helper: values / coefficients conversion (private aliases)
# ============================================================================


def _coeffs_to_values(c: jax.Array) -> jax.Array:
    """Convert Chebyshev coefficients to values at 2nd-kind Chebyshev points."""
    return coeffs2vals(c)


def _values_to_coeffs(v: jax.Array) -> jax.Array:
    """Convert values at 2nd-kind Chebyshev points to Chebyshev coefficients."""
    return vals2coeffs(v)


# ============================================================================
# Helper: zero-pad / truncate coefficient array
# ============================================================================


def _prolong_coeffs(coeffs: jax.Array, n: int) -> jax.Array:
    """Zero-pad or truncate Chebyshev coefficients to length *n*."""
    m = coeffs.shape[0]
    if m >= n:
        return coeffs[:n]
    pad = jnp.zeros((n - m,) + coeffs.shape[1:], dtype=coeffs.dtype)
    return jnp.concatenate([coeffs, pad])


def _alias_chebtech2(coeffs: jax.Array, m: int) -> jax.Array:
    """Alias 2nd-kind Chebyshev coefficients to length ``m``.

    Direct port of ``@chebtech2/alias.m``.  If ``m`` exceeds the current
    length the coefficients are zero-padded; otherwise the higher modes are
    folded down onto the retained ones per eq. (4.4) of Trefethen, ATAP.

    Not JIT-safe (Python-int branching + fancy-index accumulation); uses
    numpy for the folding, mirroring the other coefficient-surgery helpers.
    """
    import numpy as np

    orig = jnp.asarray(coeffs)
    twod = orig.ndim == 2
    c = np.asarray(orig)
    if not twod:
        c = c.reshape(-1, 1)
    else:
        c = c.copy()
    n = c.shape[0]
    if m > n:
        c = np.concatenate([c, np.zeros((m - n,) + c.shape[1:], dtype=c.dtype)], axis=0)
    elif m == 1:
        e = np.ones(int(np.ceil(n / 2)), dtype=c.dtype)
        e[1::2] = -1
        c = (e @ c[0::2, :]).reshape((1,) + c.shape[1:])
    else:
        c = c.copy()
        if m > n / 2:
            # Only single coefficients are aliased (k is unique), so the
            # fancy-indexed accumulation matches MATLAB's vectorised assign.
            j = np.arange(m + 1, n + 1)
            k = np.abs(np.mod(j + m - 3, 2 * m - 2) - m + 2) + 1
            c[k - 1, :] = c[k - 1, :] + c[j - 1, :]
        else:
            for j in range(m + 1, n + 1):
                k = abs((j + m - 3) % (2 * m - 2) - m + 2) + 1
                c[k - 1, :] = c[k - 1, :] + c[j - 1, :]
        c = c[:m, :]
    out = jnp.asarray(c, dtype=orig.dtype)
    return out if twod else out.reshape(-1)


def _alias_chebtech1(coeffs: jax.Array, m: int) -> jax.Array:
    """Alias 1st-kind Chebyshev coefficients to length ``m``.

    Direct port of ``@chebtech1/alias.m``.  The folding formula differs from
    the 2nd-kind grid even though the coefficients are for 1st-kind
    Chebyshev polynomials in both cases.  Not JIT-safe (see
    :func:`_alias_chebtech2`).
    """
    import numpy as np

    orig = jnp.asarray(coeffs)
    twod = orig.ndim == 2
    c = np.asarray(orig)
    if not twod:
        c = c.reshape(-1, 1)
    else:
        c = c.copy()
    n = c.shape[0]
    if m > n:
        c = np.concatenate([c, np.zeros((m - n,) + c.shape[1:], dtype=c.dtype)], axis=0)
    elif m == 1:
        e = np.ones(int(np.ceil(n / 2)), dtype=c.dtype)
        e[1::2] = -1
        c = (e @ c[0::2, :]).reshape((1,) + c.shape[1:])
    else:
        c = c.copy()
        if m > n / 2:
            j = np.arange(m + 1, n + 1)
            k = np.abs(np.mod(j + m - 2, 2 * m) - m + 1) + 1
            p = np.floor((j - 1 + m) / (2 * m))
            t = ((-1.0) ** p).astype(c.dtype)
            c[k - 1, :] = c[k - 1, :] + t[:, None] * c[j - 1, :]
        else:
            for j in range(m + 1, n + 1):
                k = abs((j + m - 2) % (2 * m) - m + 1) + 1
                sgn = 1 - 2 * (int(np.floor((j - 1 + m) / (2 * m))) % 2)
                c[k - 1, :] = c[k - 1, :] + sgn * c[j - 1, :]
        c = c[:m, :]
    out = jnp.asarray(c, dtype=orig.dtype)
    return out if twod else out.reshape(-1)


def _cheb_coeffs_turbo(op: Callable, rho: float, n: int) -> jax.Array:
    """Compute the first ``n`` Chebyshev coefficients of an analytic ``op``
    via Cauchy (contour) integrals over the Bernstein ellipse of parameter
    ``rho``.

    Port of the ``chebCoeffsTurbo`` subfunction of
    ``@chebtech/constructorTurbo.m``.  ``op`` must be vectorised and accept
    complex inputs.
    """
    K = 4 * n
    z = jnp.exp(2j * jnp.pi * jnp.arange(K, dtype=jnp.float64) / K)
    g = jnp.asarray(op((rho * z + 1.0 / (rho * z)) / 2.0), dtype=jnp.complex128)
    c = jnp.fft.fft(g) / K / (rho ** jnp.arange(K, dtype=jnp.float64))
    return jnp.concatenate([c[:1], 2.0 * c[1:n]])


def _turbo_coeffs(op: Callable, plain_coeffs: jax.Array, num: int) -> jax.Array:
    """Recompute ``num`` coefficients of ``op`` to high accuracy from a plain
    construction (``plain_coeffs``) using the turbo contour integral.

    Port of ``@chebtech/constructorTurbo.m``: picks the Bernstein ellipse
    from the plain length, computes the coefficients, then respects the
    real/pure-imaginary structure of the plain representation.
    """
    length = plain_coeffs.shape[0]
    rho_cheb = jnp.exp(abs(jnp.log(_EPS)) / length)
    rho = rho_cheb ** (2.0 / 3.0)
    c = _cheb_coeffs_turbo(op, float(rho), num)

    # Respect the real / pure-imaginary structure of the plain series
    # (MATLAB @chebtech/constructorTurbo.m: real(c) / imag(c) / c).
    if not bool(jnp.iscomplexobj(plain_coeffs)):
        return jnp.real(c)
    if float(jnp.max(jnp.abs(jnp.real(plain_coeffs)))) == 0.0:
        return jnp.asarray(jnp.imag(c), dtype=plain_coeffs.dtype)
    return c


def _trigcoeffs_from_tech(tech, N: int | None) -> jax.Array:
    """Trigonometric (complex-exponential) coefficients of a CHEBTECH.

    Port of ``@chebtech/trigcoeffs.m``: the ``k``-th Fourier mode is built as
    a tech of the same kind and its (unconjugated) integral against ``f``
    gives the coefficient ``0.5 * sum(exp(-i pi k x) * f)``.
    """
    if N is None:
        N = len(tech)
    if N is None or N <= 0:
        return jnp.array([], dtype=jnp.complex128)

    half = (N - 1) // 2 if N % 2 == 1 else N // 2
    if N % 2 == 1:
        modes = range(-half, half + 1)
    else:
        modes = range(-half, half)

    cls = type(tech)
    out = []
    for k in modes:
        mode = cls.from_function(lambda x, k=k: jnp.exp(-1j * jnp.pi * k * x))
        out.append(0.5 * (mode * tech).sum())
    return jnp.asarray(out)


def _chop_columns(coeffs: jax.Array, tol: float | None) -> int:
    """standard_chop applied column-wise; the cutoff is the max across
    columns (MATLAB @chebtech/simplify.m and standardCheck.m loop over
    the columns of an array-valued chebtech and keep the largest)."""
    if coeffs.ndim == 1:
        return standard_chop(coeffs, tol)
    return max(standard_chop(coeffs[:, j], tol)
               for j in range(coeffs.shape[1]))


# ============================================================================
# Coefficient-level differentiation (JIT-safe)
# ============================================================================


def _diff_coeffs_once(c: jax.Array) -> jax.Array:
    """Single differentiation via the Chebyshev coefficient recurrence.

    Given Chebyshev coefficients c_0, ..., c_{n-1} of a polynomial p,
    returns coefficients d_0, ..., d_{n-2} of p'.

    The recurrence (Mason & Handscomb, p. 34):
        d_{n-1} = d_n = 0
        d_r     = d_{r+2} + 2*(r+1)*c_{r+1}   for r = n-2, n-3, ..., 1
        d_0     = d_2 / 2 + c_1

    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @chebtech/diff.m  (computeDerCoeffs)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Page 34 of Mason & Handscomb, "Chebyshev Polynomials",
        Chapman & Hall/CRC, 2003.
    """
    n = c.shape[0]
    if n <= 1:
        return jnp.zeros((1,) + c.shape[1:], dtype=jnp.float64)

    # w[k] = 2*(k+1) for k = 0 .. n-2
    # (trailing singleton axes broadcast over array-valued columns)
    w = 2.0 * jnp.arange(1, n, dtype=jnp.float64)
    w = w.reshape((n - 1,) + (1,) * (c.ndim - 1))
    v = w * c[1:]  # v[k] = 2*(k+1)*c_{k+1}

    # Accumulate from the tail, even and odd indices separately.
    # (buffer dtype follows the series: complex chebfuns stay complex)
    out = jnp.zeros((n - 1,) + c.shape[1:], dtype=c.dtype)

    # Slice1: indices n-2, n-4, ..., i.e. v[-1], v[-3], ...
    s1 = v[::-1][::2]  # reversed, take every other
    cs1 = jnp.cumsum(s1, axis=0)
    # Slice2: indices n-3, n-5, ..., i.e. v[-2], v[-4], ...
    s2 = v[::-1][1::2]
    cs2 = jnp.cumsum(s2, axis=0)

    # Place back
    out = out.at[::-1].set(0.0)  # reset
    out = out.at[-1::-2].set(cs1)
    if cs2.shape[0] > 0:
        out = out.at[-2::-2].set(cs2)

    # Fix the c_0 coefficient: d_0 = d_2/2 + c_1 => already in out but halved
    out = out.at[0].multiply(0.5)

    return out


def _diff_coeffs(coeffs: jax.Array, k: int) -> jax.Array:
    """Differentiate Chebyshev coefficients *k* times.

    JIT-safe: yes (k must be a static integer).

    Provenance
    ----------
    MATLAB source : @chebtech/diff.m
    Chebfun commit: 7574c77
    """
    c = coeffs
    for _ in range(k):
        c = _diff_coeffs_once(c)
    return c


# ============================================================================
# Coefficient-level antiderivative (JIT-safe)
# ============================================================================


def _cumsum_coeffs(c: jax.Array) -> jax.Array:
    """Antiderivative via the Chebyshev coefficient recurrence, with F(-1)=0.

    Given c_0, ..., c_{n-1}, returns b_0, ..., b_n where
        b_1 = c_0 - c_2/2,
        b_r = (c_{r-1} - c_{r+1}) / (2*r)  for r >= 2,
        b_0 = sum_{r=1}^{n} (-1)^{r+1} b_r   (ensures F(-1)=0).

    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @chebtech/cumsum.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Pages 32-33 of Mason & Handscomb, "Chebyshev Polynomials",
        Chapman & Hall/CRC, 2003.
    """
    n = c.shape[0]
    if n == 0:
        return jnp.zeros(1, dtype=jnp.float64)

    # Pad with two zeros so that c_{n} = c_{n+1} = 0
    # (dtype follows the series: complex chebfuns stay complex;
    # trailing singleton axes broadcast over array-valued columns)
    cp = jnp.concatenate(
        [c, jnp.zeros((2,) + c.shape[1:], dtype=c.dtype)])

    b = jnp.zeros((n + 1,) + c.shape[1:], dtype=c.dtype)

    # b[r] = (c[r-1] - c[r+1]) / (2*r) for r = 2, ..., n
    rk = jnp.arange(2, n + 1, dtype=jnp.float64)
    rk = rk.reshape((n - 1,) + (1,) * (c.ndim - 1))
    b = b.at[2 : n + 1].set((cp[1:n] - cp[3 : n + 2]) / (2.0 * rk))

    # b[1] = c[0] - c[2]/2
    b = b.at[1].set(cp[0] - cp[2] / 2.0)

    # b[0]: choose so that F(-1) = 0
    # F(-1) = sum_r b_r * T_r(-1) = sum_r b_r * (-1)^r = 0
    # => b_0 = - sum_{r=1}^{n} (-1)^r * b_r = sum_{r=1}^{n} (-1)^{r+1} * b_r
    vv = jnp.ones(n, dtype=jnp.float64)
    vv = vv.at[1::2].set(-1.0)
    b = b.at[0].set(jnp.tensordot(vv, b[1 : n + 1], axes=(0, 0)))

    return b


# ============================================================================
# Coefficient-level definite integral (JIT-safe)
# ============================================================================


def _definite_integral(coeffs: jax.Array) -> jax.Array:
    r"""Definite integral of a Chebyshev expansion over [-1, 1].

    Uses the fact that \int_{-1}^{1} T_k(x) dx = 2/(1-k^2) for even k,
    0 for odd k.  (Trefethen, ATAP, Thm 19.2.)

    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @chebtech/sum.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    n = coeffs.shape[0]
    if n == 0:
        return jnp.array(0.0, dtype=jnp.float64)
    if n == 1:
        return 2.0 * coeffs[0]

    # Chebyshev moments: m_k = 2/(1-k^2) for even k, 0 for odd k
    k = jnp.arange(n, dtype=jnp.float64)
    moments = jnp.where(
        k % 2 == 0,
        2.0 / (1.0 - k**2),
        0.0,
    )
    # k=0: 2/(1-0)=2 is already correct.
    # (tensordot over axis 0 handles array-valued (n, m) coefficients,
    # returning one integral per column)
    return jnp.tensordot(moments, coeffs, axes=(0, 0))


# ============================================================================
# Coefficient-level L2 inner product (JIT-safe)
# ============================================================================


def _inner_product(f_coeffs: jax.Array, g_coeffs: jax.Array) -> jax.Array:
    r"""L^2 inner product <f, g> = \int_{-1}^{1} f(x) g(x) dx.

    Computed by prolonging both to length n_f + n_g (so quadrature is exact),
    converting to values, and applying Clenshaw-Curtis quadrature weights.

    JIT-safe: yes (shapes fixed once called).

    Provenance
    ----------
    MATLAB source : @chebtech/innerProduct.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    from chebfunjax.utils.quadrature import chebweights

    nf = f_coeffs.shape[0]
    ng = g_coeffs.shape[0]
    n = nf + ng

    # Prolong both to length n (dtype follows the operands)
    dt = jnp.result_type(f_coeffs.dtype, g_coeffs.dtype)
    fc = jnp.zeros((n,) + f_coeffs.shape[1:], dtype=dt).at[:nf].set(f_coeffs)
    gc = jnp.zeros((n,) + g_coeffs.shape[1:], dtype=dt).at[:ng].set(g_coeffs)

    # Convert to values
    fv = _coeffs_to_values(fc)
    gv = _coeffs_to_values(gc)

    # Clenshaw-Curtis weights
    w = chebweights(n, kind=2)

    # MATLAB @chebtech/innerProduct.m is conjugate-linear in F.
    if fv.ndim == 1 and gv.ndim == 1:
        return jnp.dot(w * jnp.conj(fv), gv)
    # Array-valued: pairwise column inner products (MATLAB returns the
    # m_f x m_g matrix F' * W * G)
    fv2 = fv if fv.ndim == 2 else fv[:, None]
    gv2 = gv if gv.ndim == 2 else gv[:, None]
    return (w[:, None] * jnp.conj(fv2)).T @ gv2


# ============================================================================
# Coefficient-level polynomial multiplication via FFT (JIT-safe)
# ============================================================================


def _coeff_multiply(fc: jax.Array, gc: jax.Array) -> jax.Array:
    """Multiply two Chebyshev series in coefficient space via FFT.

    Given coefficients f_0, ..., f_{m-1} and g_0, ..., g_{p-1},
    returns the coefficients of f*g (length m+p-1).

    Uses the Toeplitz-plus-Hankel-plus-rank-one embedding into a circulant
    matrix and applied using the FFT (Olver & Townsend, SIAM Review, 2013).

    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @chebtech/times.m  (coeff_times)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    nf = fc.shape[0]
    ng = gc.shape[0]
    mn = nf + ng - 1

    # Array-valued case: promote a scalar-column operand so both sides
    # share the trailing column shape (MATLAB @chebtech/times.m allows
    # scalar-valued * array-valued)
    if fc.ndim != gc.ndim:
        if fc.ndim == 1:
            fc = fc[:, None]
        if gc.ndim == 1:
            gc = gc[:, None]
    cols = jnp.broadcast_shapes(fc.shape[1:], gc.shape[1:])

    # Pad both to length mn (dtype follows the operands so complex
    # chebfun products keep their imaginary parts)
    out_dtype = jnp.result_type(fc.dtype, gc.dtype)
    f = jnp.zeros((mn,) + cols, dtype=out_dtype).at[:nf].set(
        jnp.broadcast_to(fc, (nf,) + cols))
    g = jnp.zeros((mn,) + cols, dtype=out_dtype).at[:ng].set(
        jnp.broadcast_to(gc, (ng,) + cols))

    # Embed into circulant: double the first coefficient
    t = jnp.concatenate([2.0 * f[:1], f[1:]])
    x = jnp.concatenate([2.0 * g[:1], g[1:]])

    # Circulant multiply via FFT (axis=0 keeps columns independent)
    t_ext = jnp.concatenate([t, t[-1:0:-1]])
    x_ext = jnp.concatenate([x, x[-1:0:-1]])
    product = jnp.fft.ifft(
        jnp.fft.fft(t_ext, axis=0) * jnp.fft.fft(x_ext, axis=0),
        axis=0)
    if not jnp.iscomplexobj(f):
        product = jnp.real(product)

    # Extract result
    hc = 0.25 * jnp.concatenate([product[:1], product[1:mn] + product[-1 : mn - 1 : -1]])

    return hc


# ============================================================================
# Root-finding helpers (numpy, NOT JIT-safe)
# ============================================================================


def _roots_colleague(coeffs: jax.Array) -> jax.Array:
    import numpy as np
    """Find all real roots of a Chebyshev expansion in [-1, 1].

    Uses recursive subdivision for degree > 50 and colleague matrix
    eigenvalue computation for degree <= 50.

    NOT JIT-safe (variable output size, recursive subdivision).

    Provenance
    ----------
    MATLAB source : @chebtech/roots.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] I. J. Good, "The colleague matrix, a Chebyshev analogue of the
            companion matrix", QJM 12, 1961.
        [2] J. A. Boyd, "Computing zeros on a real interval through Chebyshev
            expansion and polynomial rootfinding", SIAM J. Numer. Anal. 40, 2002.
        [3] L. N. Trefethen, ATAP, SIAM, 2013, Chapter 18.
    """
    # Complex coefficients stay complex: the float64 cast silently
    # found roots of the REAL PART only (e.g. exp(2i pi x) "roots" at
    # +-1/4, +-3/4 -- Fable 5, flip-roots audit).  The colleague
    # matrix and imag-part filters below are complex-safe: a real root
    # of a genuinely complex series requires both parts to vanish.
    c = np.asarray(coeffs)
    if not np.iscomplexobj(c):
        c = c.astype(np.float64)
    htol = 100.0 * np.finfo(np.float64).eps

    # Normalize
    vscl = np.max(np.abs(c))
    if vscl == 0.0:
        return jnp.array([0.0], dtype=jnp.float64)
    c_scaled = c / vscl

    r = _roots_main(c_scaled, htol)
    r = np.sort(r)
    return jnp.asarray(r, dtype=jnp.float64)


def _roots_main(c, htol: float):
    import numpy as np
    """Recursive root-finding engine (numpy, NOT JIT-safe).

    Follows MATLAB Chebfun's roots.m strategy:
    - Trim trailing small coefficients.
    - If degree > 50, subdivide at a slightly off-center point and recurse.
    - If degree <= 50, form the colleague matrix and compute eigenvalues.
    """
    SPLIT_POINT = -0.004849834917525
    MAX_EIG_SIZE = 50

    # Trim small trailing coefficients
    tail_max = 5.0 * np.finfo(np.float64).eps * np.linalg.norm(c, 1)
    idx = np.where(np.abs(c) > tail_max)[0]
    if idx.size == 0:
        return np.array([0.0])
    n = int(idx[-1]) + 1
    c = c[:n]

    # Trivial cases
    if n == 1:
        if c[0] == 0.0:
            return np.array([0.0])
        return np.array([], dtype=np.float64)

    if n == 2:
        r = np.array([-c[0] / c[1]])
        mask_im = np.abs(np.imag(r)) < htol
        r = np.real(r[mask_im])
        r = r[(r >= -(1.0 + htol)) & (r <= (1.0 + htol))]
        r = np.clip(r, -1.0, 1.0)
        return r

    if n - 1 <= MAX_EIG_SIZE:
        # Form the colleague matrix
        c_adj = -0.5 * c[:-1] / c[-1]
        c_adj[-2] += 0.5

        nn = n - 1
        oh = 0.5 * np.ones(nn - 1)
        A = np.diag(oh, 1) + np.diag(oh, -1)
        if np.iscomplexobj(c_adj):
            A = A.astype(np.complex128)
        A[-2, -1] = 1.0
        A[:, 0] = c_adj[::-1]

        rts = np.linalg.eigvals(A)

        # Filter: keep roots with small imaginary part and inside [-1, 1]
        mask = np.abs(np.imag(rts)) < htol
        rts = np.real(rts[mask])
        rts = rts[np.abs(rts) <= 1.0 + htol]
        rts = np.sort(rts)
        if rts.size > 0:
            rts[0] = max(rts[0], -1.0)
            rts[-1] = min(rts[-1], 1.0)
        return rts

    # Subdivide and recurse
    pts = np.asarray(chebpts(n, kind=2))

    # Map Chebyshev points to left and right subintervals
    a_left, b_left = -1.0, SPLIT_POINT
    a_right, b_right = SPLIT_POINT, 1.0

    x_left = 0.5 * ((b_left - a_left) * pts + (b_left + a_left))
    x_right = 0.5 * ((b_right - a_right) * pts + (b_right + a_right))

    # Evaluate using numpy Clenshaw
    def _eval_cheb(x_arr, cc):
        """Evaluate Chebyshev series at numpy points."""
        nn = cc.shape[0]
        bk1 = np.zeros_like(x_arr)
        bk2 = np.zeros_like(x_arr)
        for k in range(nn - 1, 0, -1):
            bk1_new = 2.0 * x_arr * bk1 - bk2 + cc[k]
            bk2 = bk1
            bk1 = bk1_new
        return x_arr * bk1 - bk2 + cc[0]

    v_left = _eval_cheb(x_left, c)
    v_right = _eval_cheb(x_right, c)

    # Convert values to coefficients
    c_left = np.asarray(vals2coeffs(jnp.asarray(v_left)))
    c_right = np.asarray(vals2coeffs(jnp.asarray(v_right)))

    # Recurse
    r_left = _roots_main(c_left, 2.0 * htol)
    r_right = _roots_main(c_right, 2.0 * htol)

    # Map back to original interval
    r_left_mapped = 0.5 * (SPLIT_POINT - 1.0) + 0.5 * (SPLIT_POINT + 1.0) * r_left
    r_right_mapped = 0.5 * (SPLIT_POINT + 1.0) + 0.5 * (1.0 - SPLIT_POINT) * r_right

    return np.concatenate([r_left_mapped, r_right_mapped])


# ============================================================================
# Chebtech2 — the core class
# ============================================================================


def _is_empty_tech(obj) -> bool:
    """True if ``obj`` is a marker-empty tech (see ``Chebtech2.empty``)."""
    return getattr(obj, "_is_empty_object", False)


class Chebtech2(eqx.Module):
    """Chebyshev interpolant on 2nd-kind points.

    Represents a smooth function on [-1, 1] via coefficients of the
    corresponding 1st-kind Chebyshev series expansion.

    Attributes
    ----------
    coeffs : jax.Array, shape (n,)
        Chebyshev series coefficients (T_0, T_1, ..., T_{n-1}).
    ishappy : bool
        True if the representation is resolved to the requested tolerance.

    Provenance
    ----------
    MATLAB source : @chebtech2/chebtech2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebtech1, Trigtech, Bndfun
    """

    coeffs: jax.Array
    ishappy: bool = eqx.field(static=True, default=True)

    # ------------------------------------------------------------------
    # Empty representation (MATLAB chebtech2() with no arguments)
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> "Chebtech2":
        """The empty Chebtech2 (MATLAB ``chebtech2()``).

        ``isempty()`` is True; arithmetic with it propagates empties.  Built
        without ``__init__`` (no coefficient data), so its fields must not be
        accessed — guard with ``isempty()`` first.

        Provenance
        ----------
        MATLAB source : @chebtech/isempty.m
        Chebfun commit: 7574c77
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty Chebtech2 (MATLAB ``isempty``).

        Provenance
        ----------
        MATLAB source : @chebtech/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

    # ------------------------------------------------------------------
    # Construction (class methods — NOT __init__)
    # ------------------------------------------------------------------

    @classmethod
    def from_coeffs(cls, coeffs: jax.Array,
                    ishappy: bool = True) -> "Chebtech2":
        """Construct a Chebtech2 from Chebyshev coefficients.

        Parameters
        ----------
        coeffs : array_like, shape (n,)
            Chebyshev series coefficients c[0], ..., c[n-1].

        Returns
        -------
        Chebtech2
            A new Chebtech2 instance.

        Examples
        --------
        >>> c = jnp.array([1.0, 0.0, -0.5])
        >>> f = Chebtech2.from_coeffs(c)
        >>> f.n
        3
        """
        coeffs = jnp.atleast_1d(_as_fun_dtype(coeffs))
        return cls(coeffs=coeffs, ishappy=bool(ishappy))

    @classmethod
    def from_values(cls, values: jax.Array) -> "Chebtech2":
        """Construct a Chebtech2 from values at 2nd-kind Chebyshev points.

        Parameters
        ----------
        values : array_like, shape (n,)
            Function values at n Chebyshev points of the 2nd kind on [-1, 1],
            ordered from x = -1 to x = 1 (ascending, matching ``chebpts``).

        Returns
        -------
        Chebtech2
            A new Chebtech2 instance.

        Examples
        --------
        >>> x = chebpts(5)
        >>> f = Chebtech2.from_values(jnp.sin(x))
        """
        values = jnp.atleast_1d(_as_fun_dtype(values))
        c = vals2coeffs(values)
        return cls(coeffs=c)

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        *,
        n: int | None = None,
        maxpow2: int = 16,
        tol: float | None = None,
        turbo: bool = False,
    ) -> "Chebtech2":
        """Construct a Chebtech2 from a callable.

        If ``n`` is given, evaluates the function on an ``n``-point 2nd-kind
        Chebyshev grid and forms the interpolant directly (non-adaptive).

        If ``n`` is ``None`` (the default), uses an adaptive algorithm that
        doubles the number of points until the Chebyshev coefficients decay
        below the tolerance set by ``standard_chop``.

        Parameters
        ----------
        f : callable
            Function mapping an array of points to an array of values.
            Must be vectorised (accept and return arrays of the same shape).
        n : int or None, optional
            Fixed number of points. If ``None``, adaptive construction is used.
        maxpow2 : int, default 16
            Maximum power of 2 for adaptive grid size (grid will be
            ``2**maxpow2 + 1`` at most). Only used when ``n is None``.

        Returns
        -------
        Chebtech2
            A new Chebtech2 instance.

        Notes
        -----
        Adaptive construction is NOT JIT-safe (Python while loop with
        data-dependent termination). Fixed-length construction IS JIT-safe
        in principle, but typically called outside JIT.

        The adaptive algorithm mirrors MATLAB Chebfun's refine/happinessCheck
        cycle: it evaluates on grids of size 2^k + 1 for k = 4, 5, ...,
        maxpow2, converts to coefficients, and calls ``standard_chop`` to
        check for convergence.

        Examples
        --------
        >>> f = Chebtech2.from_function(jnp.sin)
        >>> f.n  # typically ~14 for sin(x) on [-1, 1]
        14
        >>> f(0.5)  # close to sin(0.5)
        Array(0.47942554, dtype=float64)

        Provenance
        ----------
        MATLAB source : @chebtech2/chebtech2.m, @chebtech/populate.m,
            @chebtech2/refine.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if turbo:
            # "Turbo" construction: build the plain (adaptive) representation,
            # then recompute the coefficients to high accuracy via contour
            # integrals.  Port of @chebtech/constructorTurbo.m: the plain
            # construction stays adaptive even when a fixed output length is
            # requested (the constructor only prolongs for numeric data), so
            # the Bernstein ellipse is fixed from the adaptive length while the
            # number of computed coefficients is ``fixedLength`` (here ``n``)
            # or ``2*length`` otherwise.
            plain = cls._adaptive_construct(f, maxpow2, tol=tol)
            num = n if n is not None else 2 * len(plain)
            c = _turbo_coeffs(f, plain.coeffs, num)
            return cls(coeffs=c, ishappy=plain.ishappy)
        if n is not None:
            return cls._fixed_construct(f, n)
        return cls._adaptive_construct(f, maxpow2, tol=tol)

    @classmethod
    def _fixed_construct(
        cls, f: Callable[[jax.Array], jax.Array], n: int
    ) -> "Chebtech2":
        """Fixed-length construction on an n-point Chebyshev-2 grid."""
        if n <= 0:
            return cls(coeffs=jnp.array([], dtype=jnp.float64))
        x = chebpts(n, kind=2)
        values = _as_fun_dtype(f(x))
        c = vals2coeffs(values)
        return cls(coeffs=c)

    @classmethod
    def _adaptive_construct(
        cls,
        f: Callable[[jax.Array], jax.Array],
        maxpow2: int = 16,
        start_pow2: int = 4,
        tol: float | None = None,
    ) -> "Chebtech2":
        """Adaptive construction — Python-level loop, NOT JIT-safe.

        Evaluates f on grids of size 2^k + 1 for k = start_pow2, ..., maxpow2
        and uses ``happiness_check`` (standard_chop + sample test) to detect
        convergence. Returns a happy Chebtech2 if convergence is detected, or
        an unhappy one at the maximum grid size otherwise.

        Parameters
        ----------
        f : callable
            Function mapping an array of points to an array of values.
        maxpow2 : int, default 16
            Maximum power of 2 for adaptive grid size.
        start_pow2 : int, default 4
            Starting power of 2 (minimum grid size is ``2**start_pow2 + 1``).
            Used by ``compose`` to start from a larger grid.
        tol : float, optional
            Construction tolerance (``eps``).  If None, ``happiness_check``
            uses machine epsilon.  Threaded from ``chebfun(..., eps=...)``.
        """
        vscale = 0.0
        c = None
        for k in range(start_pow2, maxpow2 + 1):
            n = 2**k + 1
            x = chebpts(n, kind=2)
            values = _as_fun_dtype(f(x))
            c = vals2coeffs(values)
            vscale = max(vscale, float(jnp.max(jnp.abs(values))))
            ishappy, cutoff = cls.happiness_check(
                c,
                values,
                op=f,
                tol=tol,
                vscale=vscale,
            )
            if ishappy:
                return cls(coeffs=c[:cutoff], ishappy=True)

        # Did not converge — return unhappy at max length
        warnings.warn(
            f"Chebtech2.from_function: function did not converge with "
            f"{2**maxpow2 + 1} points. Returning unhappy representation.",
            stacklevel=2,
        )
        return cls(coeffs=c, ishappy=False)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the Chebyshev interpolant at point(s) x in [-1, 1].

        Uses Clenshaw's algorithm.

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            Evaluation point(s).

        Returns
        -------
        y : jax.Array, same shape as x
            Evaluated values.

        Notes
        -----
        This method is JIT-safe, grad-safe, and vmap-safe.

        Provenance
        ----------
        MATLAB source : @chebtech/clenshaw.m, @chebtech/feval.m
        Chebfun commit: 7574c77
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        return _clenshaw(self.coeffs, x)

    # ------------------------------------------------------------------
    # Static methods: vals2coeffs / coeffs2vals
    # ------------------------------------------------------------------

    @staticmethod
    def vals2coeffs(values: jax.Array) -> jax.Array:
        """Convert values at 2nd-kind Chebyshev points to coefficients.

        Delegates to ``chebfunjax.utils.transforms.vals2coeffs``.

        Parameters
        ----------
        values : jax.Array, shape (n,)

        Returns
        -------
        coeffs : jax.Array, shape (n,)

        Provenance
        ----------
        MATLAB source : @chebtech2/vals2coeffs.m
        Chebfun commit: 7574c77
        """
        return vals2coeffs(values)

    @staticmethod
    def coeffs2vals(coeffs: jax.Array) -> jax.Array:
        """Convert Chebyshev coefficients to values at 2nd-kind Chebyshev points.

        Delegates to ``chebfunjax.utils.transforms.coeffs2vals``.

        Parameters
        ----------
        coeffs : jax.Array, shape (n,)

        Returns
        -------
        values : jax.Array, shape (n,)

        Provenance
        ----------
        MATLAB source : @chebtech2/coeffs2vals.m
        Chebfun commit: 7574c77
        """
        return coeffs2vals(coeffs)

    @staticmethod
    def alias(coeffs: jax.Array, m: int) -> jax.Array:
        """Alias 2nd-kind Chebyshev coefficients to length ``m``.

        ``ALIAS(C, M)`` folds the coefficients ``C`` down to length ``M``
        (or zero-pads if ``M`` exceeds ``len(C)``).  Aliasing to length
        ``M`` gives exactly the coefficients of the interpolant through the
        underlying function on the ``M``-point 2nd-kind grid.

        Provenance
        ----------
        MATLAB source : @chebtech2/alias.m
        Chebfun commit: 7574c77
        """
        return _alias_chebtech2(coeffs, m)

    @staticmethod
    def barywts(n: int) -> jax.Array:
        """Barycentric weights for the ``n`` 2nd-kind Chebyshev points.

        Provenance
        ----------
        MATLAB source : @chebtech2/barywts.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.diffmat import _cheb2_barywts

        return _cheb2_barywts(n)

    @staticmethod
    def bary(x: jax.Array, gvals: jax.Array) -> jax.Array:
        """Barycentric interpolation of values on the 2nd-kind grid.

        Evaluates at ``x`` the polynomial interpolant through the data
        ``gvals`` given on the ``len(gvals)``-point 2nd-kind Chebyshev
        grid, using the closed-form barycentric weights.

        Provenance
        ----------
        MATLAB source : @chebtech2/bary.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.diffmat import _cheb2_barywts
        from chebfunjax.utils.interpolation import bary as _bary

        n = gvals.shape[0]
        return _bary(jnp.asarray(x, dtype=gvals.dtype), gvals,
                     chebpts(n, kind=2), _cheb2_barywts(n))

    @staticmethod
    def angles(n: int) -> jax.Array:
        """Angles ``acos(x)`` of the ``n`` 2nd-kind Chebyshev points.

        Provenance
        ----------
        MATLAB source : @chebtech2/angles.m
        Chebfun commit: 7574c77
        """
        if n == 0:
            return jnp.array([], dtype=jnp.float64)
        if n == 1:
            return jnp.array([jnp.pi / 2], dtype=jnp.float64)
        m = n - 1
        return jnp.arange(m, -1, -1, dtype=jnp.float64) * jnp.pi / m

    def sample(self, n: int | None = None):
        """Sample the tech at ``n`` 2nd-kind Chebyshev points.

        Returns ``(values, points)`` where ``values`` are the function
        values on the ``n``-point 2nd-kind grid (``n = len(self)`` if
        omitted) and ``points`` is that grid.

        Provenance
        ----------
        MATLAB source : @chebtech/sample.m
        Chebfun commit: 7574c77
        """
        if n is None:
            n = len(self)
        values = coeffs2vals(_alias_chebtech2(self.coeffs, n))
        points = chebpts(n, kind=2)
        return values, points

    def trigcoeffs(self, N: int | None = None) -> jax.Array:
        """Trigonometric (complex-exponential) coefficients of the tech.

        Provenance
        ----------
        MATLAB source : @chebtech/trigcoeffs.m
        Chebfun commit: 7574c77
        """
        return _trigcoeffs_from_tech(self, N)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of Chebyshev coefficients (= polynomial degree + 1)."""
        return self.coeffs.shape[0]

    @property
    def values(self) -> jax.Array:
        """Function values at 2nd-kind Chebyshev points (ascending order).

        Computed from coefficients via coeffs2vals. Not cached — equinox
        modules are frozen pytrees, so we recompute on access.
        """
        return coeffs2vals(self.coeffs)

    @property
    def vscale(self) -> float:
        """Vertical scale: max absolute function value."""
        return float(jnp.max(jnp.abs(self.values)))

    def __len__(self) -> int:
        """Number of Chebyshev coefficients, same as ``self.n``."""
        return self.n

    def __repr__(self) -> str:
        """Compact display like Chebfun.

        Examples
        --------
        >>> f = Chebtech2.from_function(jnp.sin)
        >>> repr(f)
        'Chebtech2(n=14, vscale=8.415e-01)'
        """
        vs = self.vscale
        return f"Chebtech2(n={self.n}, vscale={vs:.4g})"

    # ------------------------------------------------------------------
    # Core operations (return new Chebtech2 objects — immutability)
    # ------------------------------------------------------------------

    def prolong(self, n: int) -> "Chebtech2":
        """Return a new Chebtech2 with n coefficients.

        If ``n > self.n``, zero-pads the coefficient array.
        If ``n < self.n``, truncates (which may lose accuracy).
        If ``n == self.n``, returns a copy.

        Parameters
        ----------
        n : int
            Desired number of coefficients.

        Returns
        -------
        Chebtech2
            New instance with ``n`` coefficients.

        Provenance
        ----------
        MATLAB source : @chebtech/prolong.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        m = self.n
        if n == m:
            return self
        if n > m:
            padded = jnp.concatenate(
                [
                    self.coeffs,
                    jnp.zeros((n - m,) + self.coeffs.shape[1:],
                              dtype=self.coeffs.dtype),
                ]
            )
            return Chebtech2(coeffs=padded, ishappy=self.ishappy)
        # n < m: truncate
        n = max(n, 0)
        return Chebtech2(coeffs=self.coeffs[:n], ishappy=self.ishappy)

    def simplify(self, tol: float | None = None) -> "Chebtech2":
        """Return a new Chebtech2 with trailing coefficients chopped.

        Uses ``standard_chop`` to determine a suitable cutoff for the
        coefficient series. If the Chebtech2 is not happy, returns ``self``
        unchanged.

        Parameters
        ----------
        tol : float or None, optional
            Tolerance for ``standard_chop``. Default is machine epsilon.

        Returns
        -------
        Chebtech2
            Simplified instance (possibly shorter).

        Notes
        -----
        Following the MATLAB Chebfun convention, the coefficient array is
        first prolonged (zero-padded) to at least ``max(17, round(n * 1.25 + 5))``
        so that ``standard_chop`` has enough room for its plateau-detection
        logic. The result is then capped at the original length so that
        simplification never increases the number of coefficients.

        Provenance
        ----------
        MATLAB source : @chebtech/simplify.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        standard_chop
        """
        if not self.ishappy:
            return self

        nold = self.n
        # Prolong to give standard_chop room for plateau detection
        N = max(17, round(nold * 1.25 + 5))
        prolonged = self.prolong(N)

        # Round-trip through vals/coeffs to create a slightly noisy plateau
        # (standard_chop uses logarithms and needs non-zero plateau values)
        c = vals2coeffs(coeffs2vals(prolonged.coeffs))

        cutoff = _chop_columns(c, tol)
        cutoff = min(cutoff, nold)

        return Chebtech2(coeffs=self.coeffs[:cutoff], ishappy=self.ishappy)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(
        self,
        op: Callable,
        g: "Chebtech2 | None" = None,
        *,
        maxpow2: int = 16,
    ) -> "Chebtech2":
        """Compose an operator with this Chebtech2.

        ``self.compose(op)`` returns a new ``Chebtech2`` representing
        ``op(self(x))``.  When a second Chebtech2 ``g`` is supplied,
        returns ``op(self(x), g(x))``.

        ``self.compose(g)`` where ``g`` is a ``Chebtech2`` returns
        ``g(self(x))`` (function composition). The range of ``self``
        must lie inside ``[-1, 1]``.

        Parameters
        ----------
        op : callable or Chebtech2
            If callable: a function handle ``op(y)`` or ``op(y, z)``.
            If Chebtech2: computes ``op(self(x))``.
        g : Chebtech2 or None, optional
            Second argument for binary operators ``op(self(x), g(x))``.
        maxpow2 : int, default 16
            Maximum power of 2 for the adaptive grid.

        Returns
        -------
        Chebtech2
            The composed function.

        Notes
        -----
        This is NOT JIT-safe because it uses adaptive construction internally.

        The method mirrors MATLAB's ``@chebtech/compose.m`` and
        ``@chebtech2/compose.m``. The adaptive construction starts from
        ``max(self.n, g.n if g else 0)`` points (matching MATLAB's
        ``pref.minSamples``).

        Provenance
        ----------
        MATLAB source : @chebtech/compose.m, @chebtech2/compose.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        Examples
        --------
        >>> sin_cheb = Chebtech2.from_function(jnp.sin)
        >>> exp_sin = sin_cheb.compose(jnp.exp)
        >>> float(exp_sin(jnp.float64(0.5)))  # ~ exp(sin(0.5))
        1.632...

        See Also
        --------
        from_function, restrict
        """
        if isinstance(op, Chebtech2):
            # Compose two Chebtech2 objects: op(self(x))
            op_cheb = op
            composed_func = lambda x: op_cheb(self(x))  # noqa: E731
            min_n = max(self.n, op_cheb.n)
        elif g is not None:
            # Binary operator: op(self(x), g(x)).  A scalar-valued
            # operand broadcasts against an array-valued one via a
            # trailing column axis (MATLAB @chebtech/compose.m repmats
            # the scalar operand to matching columns).
            f_cols = self.coeffs.ndim == 2
            g_cols = g.coeffs.ndim == 2
            if f_cols and not g_cols:
                composed_func = lambda x: op(self(x), g(x)[..., None])  # noqa: E731
            elif g_cols and not f_cols:
                composed_func = lambda x: op(self(x)[..., None], g(x))  # noqa: E731
            else:
                composed_func = lambda x: op(self(x), g(x))  # noqa: E731
            min_n = max(self.n, g.n)
        else:
            # Unary operator: op(self(x))
            composed_func = lambda x: op(self(x))  # noqa: E731
            min_n = self.n

        # Match MATLAB: minSamples = max(pref.minSamples, length(f))
        # Start from a power of 2 grid large enough to hold min_n points.
        import math

        start_pow2 = max(4, math.ceil(math.log2(max(min_n - 1, 1))))
        return Chebtech2._adaptive_construct(
            composed_func,
            maxpow2=maxpow2,
            start_pow2=start_pow2,
        )

    # ------------------------------------------------------------------
    # Restriction
    # ------------------------------------------------------------------

    def restrict(self, a: float, b: float) -> "Chebtech2":
        """Restrict this Chebtech2 to a sub-interval [a, b] of [-1, 1].

        Returns a new ``Chebtech2`` representing the same function on [a, b],
        re-parameterized so the new object still lives on the standard
        interval [-1, 1].

        Parameters
        ----------
        a : float
            Left endpoint of the sub-interval (must satisfy ``-1 <= a < b``).
        b : float
            Right endpoint of the sub-interval (must satisfy ``a < b <= 1``).

        Returns
        -------
        Chebtech2
            A new Chebtech2 on [-1, 1] representing ``self`` restricted to
            ``[a, b]``.

        Raises
        ------
        ValueError
            If ``[a, b]`` is not a valid sub-interval of ``[-1, 1]``.

        Notes
        -----
        The restriction is computed by evaluating ``self`` at the n
        Chebyshev-2 points mapped from [-1, 1] into [a, b] via the affine
        map ``y = (b - a)/2 * x + (b + a)/2``, then converting the resulting
        values to Chebyshev coefficients.  This matches the MATLAB
        ``@chebtech/restrict.m`` implementation.

        The result is NOT simplified (following MATLAB convention). Call
        ``.simplify()`` explicitly if desired.

        Provenance
        ----------
        MATLAB source : @chebtech/restrict.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        Examples
        --------
        >>> f = Chebtech2.from_function(jnp.sin)
        >>> g = f.restrict(0.0, 0.5)
        >>> # g(x) on [-1, 1] now represents sin on [0, 0.5]
        >>> float(g(jnp.float64(0.0)))  # maps to midpoint (0+0.5)/2=0.25
        0.247...

        See Also
        --------
        compose, prolong
        """
        a = float(a)
        b = float(b)
        if a < -1.0 - 10 * _EPS or b > 1.0 + 10 * _EPS or a >= b:
            raise ValueError(
                f"[a, b] = [{a}, {b}] is not a valid sub-interval of [-1, 1]. "
                f"Require -1 <= a < b <= 1."
            )
        # Trivial case: full interval
        if abs(a - (-1.0)) < 10 * _EPS and abs(b - 1.0) < 10 * _EPS:
            return Chebtech2(coeffs=self.coeffs.copy(), ishappy=self.ishappy)

        n = self.n
        # Chebyshev points of the 2nd kind on [-1, 1]
        x = chebpts(n, kind=2)
        # Map x from [-1, 1] into [a, b]:  y = (b-a)/2 * x + (a+b)/2
        y = 0.5 * (b - a) * x + 0.5 * (a + b)
        # Evaluate self at the mapped points
        new_values = self(y)
        # Convert to coefficients
        new_coeffs = vals2coeffs(new_values)
        return Chebtech2(coeffs=new_coeffs, ishappy=self.ishappy)

    # ------------------------------------------------------------------
    # Happiness check
    # ------------------------------------------------------------------

    @staticmethod
    def happiness_check(
        coeffs: jax.Array,
        values: jax.Array,
        op: Callable | None = None,
        tol: float | None = None,
        vscale: float = 0.0,
        hscale: float = 1.0,
    ) -> tuple[bool, int]:
        """Standard happiness check for adaptive construction.

        Tests whether a Chebyshev coefficient sequence has converged by
        calling ``standard_chop``, with the tolerance scaled by
        ``max(hscale, vscale / vscale_local)`` (matching MATLAB's
        ``@chebtech/standardCheck.m``).

        Optionally performs a sample test: evaluates the operator ``op``
        and the Chebyshev interpolant at two off-grid points and checks
        that they agree to within ``sqrt(tol) * vscale``.

        Parameters
        ----------
        coeffs : jax.Array, shape (n,)
            Chebyshev coefficients.
        values : jax.Array, shape (n,)
            Function values at 2nd-kind Chebyshev points.
        op : callable or None, optional
            Original function handle for sample testing.
        tol : float or None, optional
            Target relative tolerance. Default: machine epsilon.
        vscale : float, default 0.0
            Global vertical scale (possibly from a larger approximation
            interval). Updated to ``max(vscale, max(|values|))``.
        hscale : float, default 1.0
            Horizontal scale factor.

        Returns
        -------
        ishappy : bool
            True if the representation has converged.
        cutoff : int
            Number of coefficients to retain (1-based length).

        Notes
        -----
        The tolerance scaling ``max(hscale, vscale / vscale_local)``
        matches MATLAB's ``standardCheck.m``. For single-domain
        approximation with hscale = 1, the scaling has no effect.

        When the sample test fails, ``cutoff`` is set to ``len(coeffs)``
        and ``ishappy`` is False.

        Provenance
        ----------
        MATLAB source : @chebtech/happinessCheck.m, @chebtech/standardCheck.m,
            @chebtech/sampleTest.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        standard_chop
        """
        import numpy as _np

        if tol is None:
            tol = _EPS

        n = coeffs.shape[0]
        vscale_local = float(jnp.max(jnp.abs(values)))
        vscale = max(vscale, vscale_local)

        # Scale tolerance by max(hscale, vscale / vscale_local)
        # (see MATLAB standardCheck.m lines 60-62)
        if vscale_local > 0:
            scaled_tol = tol * max(hscale, vscale / vscale_local)
        else:
            scaled_tol = tol * hscale

        cutoff = _chop_columns(coeffs, scaled_tol)
        ishappy = cutoff < n

        # Sample test: verify the interpolant matches the operator at
        # two off-grid points (MATLAB sampleTest.m)
        if ishappy and op is not None:
            # Fixed test points from MATLAB (not on any Chebyshev grid)
            xeval = jnp.array(
                [-0.357998918959666, 0.036785641195074], dtype=jnp.float64
            )
            # Build a temporary Chebtech2 with the truncated coefficients
            f_test = Chebtech2(coeffs=coeffs[:cutoff])
            v_fun = f_test(xeval)
            v_op = _as_fun_dtype(op(xeval))
            err = float(jnp.max(jnp.abs(v_op - v_fun)))
            sample_tol = _np.sqrt(max(_EPS, tol)) * max(hscale * vscale_local, vscale)
            if err > sample_tol:
                ishappy = False
                cutoff = n

        return ishappy, cutoff

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other) -> "Chebtech2":
        """Add a Chebtech2 or scalar.

        Provenance
        ----------
        MATLAB source : @chebtech/plus.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech2.empty()
        if isinstance(other, Chebtech2):
            # Prolong to the same length (zero-pad shorter one)
            nf = self.n
            ng = other.n
            n = max(nf, ng)
            fc = _prolong_coeffs(self.coeffs, n)
            gc = _prolong_coeffs(other.coeffs, n)
            return Chebtech2.from_coeffs(fc + gc, ishappy=self.ishappy and other.ishappy)
        else:
            # Scalar addition: only the c_0 coefficient changes. Promote the
            # coefficient dtype first — scattering a complex scalar into a
            # float64 buffer silently drops the imaginary part.
            s = _as_scalar(other)
            c = self.coeffs.astype(jnp.result_type(self.coeffs.dtype, s.dtype))
            c = c.at[0].add(s)
            return Chebtech2.from_coeffs(c, ishappy=self.ishappy)

    def __radd__(self, other) -> "Chebtech2":
        return self.__add__(other)

    def __sub__(self, other) -> "Chebtech2":
        """Subtract a Chebtech2 or scalar.

        Provenance
        ----------
        MATLAB source : @chebtech/minus.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech2.empty()
        return self + (-other)

    def __rsub__(self, other) -> "Chebtech2":
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech2.empty()
        return -(self - other)

    def __neg__(self) -> "Chebtech2":
        """Unary minus.

        Provenance
        ----------
        MATLAB source : @chebtech/uminus.m
        Chebfun commit: 7574c77
        """
        return Chebtech2.from_coeffs(-self.coeffs, ishappy=self.ishappy)

    def __pos__(self) -> "Chebtech2":
        """Unary plus (identity)."""
        return self

    def __mul__(self, other) -> "Chebtech2":
        """Pointwise multiplication.

        Chebtech2 * Chebtech2 uses coefficient-space FFT multiplication.
        Chebtech2 * scalar scales all coefficients.

        Provenance
        ----------
        MATLAB source : @chebtech/times.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech2.empty()
        if isinstance(other, Chebtech2):
            hc = _coeff_multiply(self.coeffs, other.coeffs)
            return Chebtech2.from_coeffs(hc, ishappy=self.ishappy and other.ishappy)
        else:
            return Chebtech2.from_coeffs(self.coeffs * _as_scalar(other), ishappy=self.ishappy)

    def __rmul__(self, other) -> "Chebtech2":
        return self.__mul__(other)

    def __matmul__(self, other) -> "Chebtech2":
        """MATLAB mtimes ``f * A``: right-multiply an array-valued tech
        by a matrix, mixing its columns (coeffs @ A).

        Provenance
        ----------
        MATLAB source : @chebtech/mtimes.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self):
            return Chebtech2.empty()
        A = jnp.asarray(other)
        c = self.coeffs if self.coeffs.ndim == 2 else self.coeffs[:, None]
        return Chebtech2(coeffs=c @ A, ishappy=self.ishappy)

    def fliplr(self) -> "Chebtech2":
        """Reverse the column order of an array-valued tech (a no-op
        for scalar-valued input).

        Provenance
        ----------
        MATLAB source : @chebtech/fliplr.m
        Chebfun commit: 7574c77
        """
        if self.coeffs.ndim == 1:
            return self
        return Chebtech2(coeffs=self.coeffs[:, ::-1],
                         ishappy=self.ishappy)

    def flipud(self) -> "Chebtech2":
        """Return g with g(x) = f(-x): negate the odd coefficients.

        Provenance
        ----------
        MATLAB source : @chebtech/flipud.m
        Chebfun commit: 7574c77
        """
        return type(self)(coeffs=self.coeffs.at[1::2].multiply(-1.0),
                          ishappy=self.ishappy)

    def real(self) -> "Chebtech2":
        """Real part (a zero tech if the input was purely imaginary).

        Provenance
        ----------
        MATLAB source : @chebtech/real.m
        Chebfun commit: 7574c77
        """
        c = jnp.real(self.coeffs)
        if not bool(jnp.any(c)):
            c = jnp.zeros((1,) + self.coeffs.shape[1:],
                          dtype=jnp.float64)
            return type(self)(coeffs=c, ishappy=True)
        return type(self)(coeffs=c, ishappy=self.ishappy)

    def imag(self) -> "Chebtech2":
        """Imaginary part (a zero tech if the input was real).

        Provenance
        ----------
        MATLAB source : @chebtech/imag.m
        Chebfun commit: 7574c77
        """
        c = jnp.imag(self.coeffs)
        if not bool(jnp.any(c)):
            c = jnp.zeros((1,) + self.coeffs.shape[1:],
                          dtype=jnp.float64)
            return type(self)(coeffs=c, ishappy=True)
        return type(self)(coeffs=c, ishappy=self.ishappy)

    def conj(self) -> "Chebtech2":
        """Complex conjugate.

        Provenance
        ----------
        MATLAB source : @chebtech/conj.m
        Chebfun commit: 7574c77
        """
        return type(self)(coeffs=jnp.conj(self.coeffs),
                          ishappy=self.ishappy)

    def mat2cell(self, sizes) -> list:
        """Split an array-valued tech into a list of techs with the
        given column counts (MATLAB ``mat2cell(f, 1, sizes)``); a
        size-1 block becomes a scalar-valued tech.

        Provenance
        ----------
        MATLAB source : @chebtech/mat2cell.m
        Chebfun commit: 7574c77
        """
        c = self.coeffs if self.coeffs.ndim == 2 \
            else self.coeffs[:, None]
        out = []
        j = 0
        for s in sizes:
            block = c[:, j:j + s]
            j += s
            out.append(type(self)(
                coeffs=block[:, 0] if s == 1 else block,
                ishappy=self.ishappy))
        return out

    @classmethod
    def cell2mat(cls, techs) -> "Chebtech2":
        """Horizontally concatenate techs into one array-valued tech
        (MATLAB ``cell2mat([g h])``).

        Provenance
        ----------
        MATLAB source : @chebtech/cell2mat.m
        Chebfun commit: 7574c77
        """
        n = max(t.n for t in techs)
        cols = []
        for t in techs:
            c = t.prolong(n).coeffs
            cols.append(c if c.ndim == 2 else c[:, None])
        dt = jnp.result_type(*(c.dtype for c in cols))
        return cls(coeffs=jnp.concatenate(
                       [c.astype(dt) for c in cols], axis=1),
                   ishappy=all(t.ishappy for t in techs))

    def assign_columns(self, cols, g) -> "Chebtech2":
        """Overwrite the columns ``cols`` (0-based) of an array-valued
        tech with the columns of ``g`` (MATLAB assignColumns);
        ``g=None`` deletes the columns instead.

        Provenance
        ----------
        MATLAB source : @chebtech/assignColumns.m
        Chebfun commit: 7574c77
        """
        fc = self.coeffs if self.coeffs.ndim == 2 \
            else self.coeffs[:, None]
        cols = [cols] if isinstance(cols, int) else list(cols)
        if g is None:
            keep = [j for j in range(fc.shape[1]) if j not in cols]
            return type(self)(coeffs=fc[:, keep], ishappy=self.ishappy)
        gc = g.coeffs if g.coeffs.ndim == 2 else g.coeffs[:, None]
        n = max(fc.shape[0], gc.shape[0])
        fc = _prolong_coeffs(fc, n)
        gc = _prolong_coeffs(gc, n).astype(
            jnp.result_type(fc.dtype, gc.dtype))
        out = fc.astype(gc.dtype).at[:, jnp.asarray(cols)].set(gc)
        return type(self)(coeffs=out,
                          ishappy=self.ishappy and g.ishappy)

    def __truediv__(self, other) -> "Chebtech2":
        """Division: Chebtech2 / scalar or Chebtech2 / Chebtech2.

        Division by a scalar simply scales the coefficients.
        Division by another Chebtech2 evaluates on a fine grid and
        re-interpolates (NOT JIT-safe when dividing by a Chebtech2).

        Provenance
        ----------
        MATLAB source : @chebtech/rdivide.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebtech2):
            # MATLAB: compose(f, @rdivide, g) — adaptive re-construction so
            # the quotient is resolved to machine precision (a fixed grid
            # silently under-resolves, e.g. 1/(1+25x^2) needs ~185 coeffs).
            return self.compose(lambda a, b: a / b, other)
        else:
            return Chebtech2.from_coeffs(self.coeffs / _as_scalar(other), ishappy=self.ishappy)

    def __rtruediv__(self, other) -> "Chebtech2":
        """Scalar / Chebtech2 (adaptive, like MATLAB compose)."""
        return self.compose(lambda y: _as_scalar(other) / y)

    def __pow__(self, exponent) -> "Chebtech2":
        """Raise to a power.

        Integer powers via repeated multiplication.
        Non-integer powers via evaluation on a grid and re-interpolation.

        Provenance
        ----------
        MATLAB source : @chebtech/power.m
        Chebfun commit: 7574c77
        """
        if isinstance(exponent, int) and exponent >= 0:
            if exponent == 0:
                # ones with the same column count (array-valued f**0
                # keeps m columns, MATLAB power.m)
                return Chebtech2.from_coeffs(
                    jnp.ones((1,) + self.coeffs.shape[1:],
                             dtype=jnp.float64))
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        elif isinstance(exponent, Chebtech2):
            # f^g via adaptive composition (MATLAB: compose(f, @power, g))
            return self.compose(lambda a, b: a ** b, exponent)
        else:
            # Fractional power: adaptive composition (MATLAB compose)
            return self.compose(lambda y: y ** _as_scalar(exponent))

    def __abs__(self) -> "Chebtech2":
        """Absolute value (evaluated on a grid, re-interpolated).

        NOT JIT-safe (may introduce kinks).
        """
        n = max(2 * self.n, 17)
        x = chebpts(n, kind=2)
        fv = jnp.abs(_clenshaw(self.coeffs, x))
        return Chebtech2.from_values(fv)

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, k: int = 1, dim: int = 1) -> "Chebtech2":
        """Differentiate *k* times.

        Uses the Chebyshev coefficient recurrence (Mason & Handscomb, p. 34).

        JIT-safe: yes (k must be a static integer).

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.
        dim : int, default 1
            ``dim=2`` takes k-th finite differences ACROSS the columns
            of an array-valued tech (MATLAB ``diff(f, k, 2)``); returns
            an empty-coefficient tech for scalar-valued input.

        Returns
        -------
        Chebtech2
            The k-th derivative.

        Provenance
        ----------
        MATLAB source : @chebtech/diff.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Page 34 of Mason & Handscomb, "Chebyshev Polynomials",
            Chapman & Hall/CRC, 2003.

        See Also
        --------
        cumsum, sum
        """
        if dim == 2:
            if self.coeffs.ndim == 1:
                return Chebtech2(
                    coeffs=jnp.zeros((0,), dtype=self.coeffs.dtype),
                    ishappy=self.ishappy)
            return Chebtech2(coeffs=jnp.diff(self.coeffs, n=k, axis=1),
                             ishappy=self.ishappy)
        if k == 0:
            return self
        new_coeffs = _diff_coeffs(self.coeffs, k)
        return Chebtech2.from_coeffs(new_coeffs, ishappy=self.ishappy)

    def cumsum(self) -> "Chebtech2":
        """Indefinite integral (antiderivative with F(-1) = 0).

        Uses the Chebyshev coefficient recurrence.

        JIT-safe: yes.

        Returns
        -------
        Chebtech2
            The antiderivative satisfying F(-1) = 0.

        Provenance
        ----------
        MATLAB source : @chebtech/cumsum.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Pages 32-33 of Mason & Handscomb, "Chebyshev Polynomials",
            Chapman & Hall/CRC, 2003.

        See Also
        --------
        diff, sum
        """
        new_coeffs = _cumsum_coeffs(self.coeffs)
        return Chebtech2.from_coeffs(new_coeffs, ishappy=self.ishappy)

    def sum(self, dim: int = 1) -> "jax.Array | Chebtech2":
        r"""Definite integral over [-1, 1].

        Uses the Chebyshev moments: integral of T_k = 2/(1-k^2) for even k.

        JIT-safe: yes.

        Parameters
        ----------
        dim : int, default 1
            ``dim=1`` integrates each column (MATLAB ``sum(f)``);
            ``dim=2`` sums across the columns of an array-valued tech
            and returns a scalar-column Chebtech2 (MATLAB ``sum(f, 2)``,
            a no-op for scalar-valued input).

        Returns
        -------
        jax.Array (scalar or (m,)) or Chebtech2
            The definite integral(s), or the column-sum tech if dim=2.

        Provenance
        ----------
        MATLAB source : @chebtech/sum.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Trefethen, ATAP, Thm 19.2.

        See Also
        --------
        diff, cumsum, inner
        """
        if dim == 2:
            if self.coeffs.ndim == 1:
                return self
            return Chebtech2(coeffs=jnp.sum(self.coeffs, axis=1),
                             ishappy=self.ishappy)
        return _definite_integral(self.coeffs)

    def inner(self, other: "Chebtech2") -> jax.Array:
        r"""L^2 inner product <self, other> = \int_{-1}^{1} f(x) g(x) dx.

        Computed by prolonging to sum of degrees and applying Clenshaw-Curtis
        quadrature (exact for polynomials of this combined degree).

        JIT-safe: yes (shapes fixed once called).

        Parameters
        ----------
        other : Chebtech2
            The other function.

        Returns
        -------
        jax.Array (scalar)
            The inner product.

        Provenance
        ----------
        MATLAB source : @chebtech/innerProduct.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        sum, norm
        """
        out = _inner_product(self.coeffs, other.coeffs)
        # MATLAB @chebtech/innerProduct.m forces a nonnegative real result
        # when f == g (isequal branch).  The identity check is JIT-safe;
        # the value check runs only on concrete (non-traced) arrays.
        same = other is self
        if not same and self.coeffs.shape == other.coeffs.shape:
            if not isinstance(self.coeffs, jax.core.Tracer) and \
                    not isinstance(other.coeffs, jax.core.Tracer):
                same = bool(jnp.all(self.coeffs == other.coeffs))
        if same and out.ndim == 0:
            return jnp.abs(out)
        return out

    def norm(self, p: float = 2.0) -> jax.Array:
        """Lp norm of the Chebtech2.

        Parameters
        ----------
        p : float, default 2.0
            The exponent for the Lp norm.
            - ``p=2``: L2 norm via inner product (= sqrt(<f, f>)).
            - ``p=jnp.inf``: L-infinity norm via max of |values| on a fine grid.
            - Other p: computed via quadrature of |f|^p.

        Returns
        -------
        jax.Array (scalar)

        Provenance
        ----------
        MATLAB source : @chebtech/normest.m (and norm.m at the chebfun level)
        Chebfun commit: 7574c77
        """
        if p == 2:
            return jnp.sqrt(jnp.abs(self.inner(self)))
        elif p == jnp.inf or p == float("inf"):
            # Sample on a fine grid
            n = max(2 * self.n + 1, 65)
            x = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)
            return jnp.max(jnp.abs(_clenshaw(self.coeffs, x)))
        else:
            # General Lp: integrate |f|^p via (|f|^p).sum()
            fp = self.__abs__().__pow__(p)
            return fp.sum() ** (1.0 / p)

    # ------------------------------------------------------------------
    # Rootfinding
    # ------------------------------------------------------------------

    def roots(self) -> jax.Array:
        """Real roots in [-1, 1] via colleague matrix eigenvalues.

        NOT JIT-safe (variable output size, recursive subdivision).

        Returns
        -------
        jax.Array, shape (n_roots,)
            Sorted roots in [-1, 1].

        Provenance
        ----------
        MATLAB source : @chebtech/roots.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm:
            [1] I. J. Good, "The colleague matrix, a Chebyshev analogue of the
                companion matrix", QJM 12, 1961.
            [2] L. N. Trefethen, ATAP, SIAM, 2013, Chapter 18.

        See Also
        --------
        diff, sum
        """
        if self.coeffs.ndim == 2:
            # Array-valued: roots per column, NaN-padded to equal length
            # (MATLAB @chebtech/roots.m does exactly this)
            import numpy as _np
            cols = [_np.asarray(_roots_colleague(self.coeffs[:, j]))
                    for j in range(self.coeffs.shape[1])]
            nmax = max((len(c) for c in cols), default=0)
            out = _np.full((nmax, len(cols)), _np.nan)
            for j, c in enumerate(cols):
                out[: len(c), j] = c
            return jnp.asarray(out)
        return _roots_colleague(self.coeffs)

    def minandmax(self) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        """Global minimum and maximum of the function on [-1, 1].

        Returns the global minimum and maximum values together with the
        positions at which they are achieved.  Computed by finding the roots
        of the derivative and evaluating at those interior critical points as
        well as at the endpoints.

        NOT JIT-safe (depends on rootfinding which has variable output size).

        Returns
        -------
        (min_val, min_pos) : tuple[jax.Array, jax.Array]
            Global minimum value and the x-position where it is achieved.
        (max_val, max_pos) : tuple[jax.Array, jax.Array]
            Global maximum value and the x-position where it is achieved.

        Provenance
        ----------
        MATLAB source : @chebtech/minandmax.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        roots, diff
        """
        import numpy as _np

        if jnp.iscomplexobj(self.coeffs):
            # Complex-valued: extrema of |f| located via |f|^2 (avoids
            # the abs singularity), values are f at those positions
            # (MATLAB @chebtech/minandmax.m lines 23-36).
            realf = self.real()
            imagf = self.imag()
            h = (realf * realf + imagf * imagf).simplify()
            (_, min_pos), (_, max_pos) = h.minandmax()
            if self.coeffs.ndim == 2:
                # f(pos) is (m, m); the diagonal pairs column k with
                # its own extremum position (MATLAB's stride trick).
                min_val = jnp.diagonal(self(jnp.atleast_1d(min_pos)))
                max_val = jnp.diagonal(self(jnp.atleast_1d(max_pos)))
            else:
                min_val = self(min_pos)
                max_val = self(max_pos)
            return (min_val, min_pos), (max_val, max_pos)

        if self.coeffs.ndim == 2:
            # Array-valued: extremum per column (MATLAB
            # @chebtech/minandmax.m returns 2 x m values/positions)
            per_col = [
                Chebtech2(coeffs=self.coeffs[:, j],
                          ishappy=self.ishappy).minandmax()
                for j in range(self.coeffs.shape[1])
            ]
            min_val = jnp.stack([p[0][0] for p in per_col])
            min_pos = jnp.stack([p[0][1] for p in per_col])
            max_val = jnp.stack([p[1][0] for p in per_col])
            max_pos = jnp.stack([p[1][1] for p in per_col])
            return (min_val, min_pos), (max_val, max_pos)

        # Compute turning points (roots of derivative)
        fp = self.diff()
        r = fp.roots()

        # Include endpoints
        endpoints = jnp.array([-1.0, 1.0], dtype=jnp.float64)
        if r.shape[0] > 0:
            candidates = jnp.concatenate([endpoints, r])
        else:
            candidates = endpoints

        # Evaluate at all candidate points
        v = self(candidates)
        v_np = _np.array(v)
        cand_np = _np.array(candidates)

        min_idx = int(_np.argmin(v_np))
        max_idx = int(_np.argmax(v_np))

        min_val = jnp.array(v_np[min_idx], dtype=jnp.float64)
        max_val = jnp.array(v_np[max_idx], dtype=jnp.float64)
        min_pos = jnp.array(cand_np[min_idx], dtype=jnp.float64)
        max_pos = jnp.array(cand_np[max_idx], dtype=jnp.float64)

        return (min_val, min_pos), (max_val, max_pos)

    def min(self) -> tuple[jax.Array, jax.Array]:
        """Global minimum of the function on [-1, 1].

        Returns
        -------
        (val, pos) : tuple[jax.Array, jax.Array]
            Global minimum value and the x-position where it is achieved.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebtech/min.m
        Chebfun commit: 7574c77
        """
        (min_val, min_pos), _ = self.minandmax()
        return min_val, min_pos

    def max(self) -> tuple[jax.Array, jax.Array]:
        """Global maximum of the function on [-1, 1].

        Returns
        -------
        (val, pos) : tuple[jax.Array, jax.Array]
            Global maximum value and the x-position where it is achieved.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebtech/max.m
        Chebfun commit: 7574c77
        """
        _, (max_val, max_pos) = self.minandmax()
        return max_val, max_pos


# ============================================================================
# Chebtech1 vals <-> coeffs (DCT-II / DCT-III based)
# ============================================================================


def _chebtech1_vals2coeffs(values: jax.Array) -> jax.Array:
    r"""Convert values at 1st-kind Chebyshev points to Chebyshev coefficients.

    Given values v[k] = f(x_k) at Chebyshev points of the 1st kind
    x_k = cos((2*(N-k) - 1)*pi / (2*N)), k = 0,...,N-1  (ascending order),
    returns the Chebyshev coefficients c such that
        f(x) = c[0]*T_0(x) + c[1]*T_1(x) + ... + c[N-1]*T_{N-1}(x).

    Equivalent to the inverse Discrete Cosine Transform of Type II (IDCT-II),
    which is also called DCT-III.

    Parameters
    ----------
    values : jax.Array, shape (n,)
        Function values at n Chebyshev points of the 1st kind (ascending).

    Returns
    -------
    coeffs : jax.Array, shape (n,)
        Chebyshev series coefficients c[0], ..., c[n-1].

    Notes
    -----
    JIT-safe: yes.

    The transform mirrors MATLAB's ``@chebtech1/vals2coeffs.m`` (commit 7574c77)
    which uses the weight vector ``w = 2*exp(i*k*pi/(2*n))`` applied after a
    mirrored IFFT.

    The input values are expected in ascending order (as returned by
    ``chebpts(n, kind=1)``).  The MATLAB implementation works with descending
    order; we flip internally and flip back.

    Provenance
    ----------
    MATLAB source : @chebtech1/vals2coeffs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: DCT-II via FFT, Section 4.7, Mason & Handscomb,
        "Chebyshev Polynomials", Chapman & Hall/CRC, 2003.

    See Also
    --------
    _chebtech1_coeffs2vals, vals2coeffs
    """
    n = values.shape[0]
    if n <= 1:
        return values.astype(jnp.float64)

    # Weight vector: w = 2 * exp(i * k * pi / (2*n)), k = 0..n-1
    # (trailing singleton axes broadcast over array-valued columns)
    k = jnp.arange(n, dtype=jnp.float64)
    w = 2.0 * jnp.exp(1j * k * jnp.pi / (2.0 * n))
    w = w.reshape((n,) + (1,) * (values.ndim - 1))

    # Complex data: the FFT/DCT trick below assumes REAL values (the
    # final jnp.real would silently DROP the imaginary part -- Fable 5
    # audit, bug #6).  Split like MATLAB @chebtech1/vals2coeffs.m.
    if jnp.iscomplexobj(values):
        return (_chebtech1_vals2coeffs(jnp.real(values))
                + 1j * _chebtech1_vals2coeffs(jnp.imag(values)))
    # MATLAB vals2coeffs: tmp = [values(n:-1:1); values]
    # values is ascending (left-to-right); values(n:-1:1) is descending.
    # In Python (values ascending): tmp = [values[::-1], values]
    # = [descending, ascending]
    tmp = jnp.concatenate([values[::-1], values])
    tmp = tmp.astype(jnp.complex128)
    coeffs_complex = jnp.fft.ifft(tmp, axis=0)[:n] * w

    # Scale the constant term (c_0 halved)
    coeffs_complex = coeffs_complex.at[0].multiply(0.5)

    coeffs = jnp.real(coeffs_complex)

    # Enforce symmetries exactly (MATLAB @chebtech1/vals2coeffs.m
    # lines 74-76): even values -> odd coeffs zero, odd values -> even
    # coeffs zero.  Branch-free, JIT-safe; correction through
    # stop_gradient so autodiff is not projected onto the symmetry
    # manifold.
    vflip = values[::-1]
    is_even = jnp.max(jnp.abs(values - vflip), axis=0) == 0
    is_odd = jnp.max(jnp.abs(values + vflip), axis=0) == 0
    kk = jnp.arange(n).reshape((n,) + (1,) * (coeffs.ndim - 1))
    sym = jnp.where((kk % 2 == 1) & is_even, 0.0, coeffs)
    sym = jnp.where((kk % 2 == 0) & is_odd, 0.0, sym)
    # Guard non-finite entries: inf - inf would turn them into NaN.
    delta = jnp.where(jnp.isfinite(coeffs), sym - coeffs, 0.0)
    return coeffs + jax.lax.stop_gradient(delta)


def _chebtech1_coeffs2vals(coeffs: jax.Array) -> jax.Array:
    r"""Convert Chebyshev coefficients to values at 1st-kind Chebyshev points.

    Given Chebyshev coefficients c, returns the values
    v[k] = c[0]*T_0(x_k) + ... + c[n-1]*T_{n-1}(x_k)
    at Chebyshev points of the 1st kind.

    Equivalent to the Discrete Cosine Transform of Type III (DCT-III).

    Parameters
    ----------
    coeffs : jax.Array, shape (n,)
        Chebyshev series coefficients c[0], ..., c[n-1].

    Returns
    -------
    values : jax.Array, shape (n,)
        Function values at n 1st-kind Chebyshev points (ascending x order).

    Notes
    -----
    JIT-safe: yes.

    The transform mirrors MATLAB's ``@chebtech1/coeffs2vals.m`` (commit 7574c77)
    which uses weight vector ``w = (exp(-i*k*pi/(2*n))/2)``.
    The output is in ascending order to match ``chebpts(n, kind=1)``.

    Provenance
    ----------
    MATLAB source : @chebtech1/coeffs2vals.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: DCT-III via FFT, Section 4.7, Mason & Handscomb,
        "Chebyshev Polynomials", Chapman & Hall/CRC, 2003.

    See Also
    --------
    _chebtech1_vals2coeffs, coeffs2vals
    """
    n = coeffs.shape[0]
    # Complex coefficients: split into real/imag (the mirror-FFT below
    # assumes real data; jnp.real dropped the imaginary part -- Fable 5
    # audit, bug #6).
    if jnp.iscomplexobj(coeffs):
        return (_chebtech1_coeffs2vals(jnp.real(coeffs))
                + 1j * _chebtech1_coeffs2vals(jnp.imag(coeffs)))
    if n <= 1:
        return coeffs.astype(jnp.float64)

    # Weight vector (length 2n): w_k = exp(-i*k*pi/(2*n))/2
    # (trailing singleton axes broadcast over array-valued columns)
    k = jnp.arange(2 * n, dtype=jnp.float64)
    w = jnp.exp(-1j * k * jnp.pi / (2.0 * n)) / 2.0
    # Special entries: w[0] = 2*w[0] = 1, w[n] = 0, w[n+1:] flipped sign
    w = w.at[0].set(1.0)
    w = w.at[n].set(0.0)
    w = w.at[n + 1:].multiply(-1.0)
    w = w.reshape((2 * n,) + (1,) * (coeffs.ndim - 1))

    # Mirror: [c; 1; c_{n-1}, ..., c_1]   (MATLAB convention, descending coeffs)
    c_mirror = jnp.concatenate([
        coeffs,
        jnp.ones((1,) + coeffs.shape[1:], dtype=jnp.float64),
        coeffs[-1:0:-1],
    ]).astype(jnp.complex128)

    c_weighted = c_mirror * w
    values_complex = jnp.fft.fft(c_weighted, axis=0)

    # Truncate to n entries; MATLAB returns in descending order (flip to ascending)
    values = jnp.real(values_complex[n - 1::-1])

    # Enforce symmetries exactly (MATLAB @chebtech1/coeffs2vals.m
    # lines 75-77): odd coeffs zero -> even values, even coeffs zero ->
    # odd values.  Branch-free, JIT-safe; correction through
    # stop_gradient (see _chebtech1_vals2coeffs).
    is_even = jnp.max(jnp.abs(coeffs[1::2]), axis=0, initial=0.0) == 0
    is_odd = jnp.max(jnp.abs(coeffs[0::2]), axis=0, initial=0.0) == 0
    vflip = values[::-1]
    sym = jnp.where(is_even, (values + vflip) / 2.0, values)
    sym = jnp.where(is_odd, (values - vflip) / 2.0, sym)
    # Guard non-finite entries: inf - inf would turn them into NaN.
    delta = jnp.where(jnp.isfinite(values), sym - values, 0.0)
    return values + jax.lax.stop_gradient(delta)


# ============================================================================
# Chebtech1 — Chebyshev interpolant on 1st-kind points
# ============================================================================


class Chebtech1(eqx.Module):
    """Chebyshev interpolant on 1st-kind points (Gauss-Chebyshev nodes).

    Represents a smooth function on [-1, 1] via coefficients of the
    corresponding 1st-kind Chebyshev series expansion.  The coefficient
    basis is *identical* to that of ``Chebtech2`` (the series
    ``c[0]*T_0 + c[1]*T_1 + ...``); only the grid used for sampling and
    the associated transforms differ.

    ``Chebtech1`` uses the **interior** Gauss-Chebyshev nodes
    ``x_k = cos((2k-1)*pi/(2n))``, k = 1,...,n (no endpoints).
    This makes it suitable for functions that are smooth up to the boundary
    but where endpoint evaluation should be avoided.

    The Chebyshev coefficient stored is the same first-kind Chebyshev
    expansion, so all calculus operations (``diff``, ``cumsum``, ``sum``,
    ``roots``) and evaluation via Clenshaw's algorithm are inherited from
    the shared private helpers.

    Attributes
    ----------
    coeffs : jax.Array, shape (n,)
        Chebyshev series coefficients (T_0, T_1, ..., T_{n-1}).
    ishappy : bool
        True if the representation is resolved to the requested tolerance.

    Provenance
    ----------
    MATLAB source : @chebtech1/chebtech1.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebtech2, Trigtech
    """

    coeffs: jax.Array
    ishappy: bool = eqx.field(static=True, default=True)

    # ------------------------------------------------------------------
    # Empty representation (MATLAB chebtech1() with no arguments)
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> "Chebtech1":
        """The empty Chebtech1 (MATLAB ``chebtech1()``).

        Provenance
        ----------
        MATLAB source : @chebtech/isempty.m
        Chebfun commit: 7574c77
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty Chebtech1 (MATLAB ``isempty``).

        Provenance
        ----------
        MATLAB source : @chebtech/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

    # ------------------------------------------------------------------
    # Construction (class methods)
    # ------------------------------------------------------------------

    @classmethod
    def from_coeffs(cls, coeffs: jax.Array,
                    ishappy: bool = True) -> "Chebtech1":
        """Construct a Chebtech1 from Chebyshev coefficients.

        Parameters
        ----------
        coeffs : array_like, shape (n,)
            Chebyshev series coefficients c[0], ..., c[n-1].

        Returns
        -------
        Chebtech1
        """
        coeffs = jnp.atleast_1d(_as_fun_dtype(coeffs))
        return cls(coeffs=coeffs, ishappy=bool(ishappy))

    @classmethod
    def from_values(cls, values: jax.Array) -> "Chebtech1":
        """Construct a Chebtech1 from values at 1st-kind Chebyshev points.

        Parameters
        ----------
        values : array_like, shape (n,)
            Function values at n Chebyshev points of the 1st kind on [-1, 1],
            ordered from x = -1 to x = 1 (ascending, matching
            ``chebpts(n, kind=1)``).

        Returns
        -------
        Chebtech1
        """
        values = jnp.atleast_1d(_as_fun_dtype(values))
        c = _chebtech1_vals2coeffs(values)
        return cls(coeffs=c)

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        *,
        n: int | None = None,
        maxpow2: int = 16,
        turbo: bool = False,
    ) -> "Chebtech1":
        """Construct a Chebtech1 from a callable.

        Samples the function on 1st-kind Chebyshev grids.  Adaptive if
        ``n`` is ``None``; fixed-length otherwise.

        Parameters
        ----------
        f : callable
            Vectorised function.
        n : int or None, optional
            Fixed number of points.  If ``None``, adaptive.
        maxpow2 : int, default 16
            Maximum power of 2 for adaptive grid.

        Returns
        -------
        Chebtech1

        Provenance
        ----------
        MATLAB source : @chebtech1/chebtech1.m, @chebtech/populate.m,
            @chebtech1/refine.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if turbo:
            # "Turbo" construction (see Chebtech2.from_function): the plain
            # construction is adaptive; only the number of computed
            # coefficients is fixed by ``n`` (fixedLength).
            plain = cls._adaptive_construct(f, maxpow2)
            num = n if n is not None else 2 * len(plain)
            c = _turbo_coeffs(f, plain.coeffs, num)
            return cls(coeffs=c, ishappy=plain.ishappy)
        if n is not None:
            return cls._fixed_construct(f, n)
        return cls._adaptive_construct(f, maxpow2)

    @classmethod
    def _fixed_construct(
        cls, f: Callable[[jax.Array], jax.Array], n: int
    ) -> "Chebtech1":
        """Fixed-length construction on an n-point Chebyshev-1 grid."""
        if n <= 0:
            return cls(coeffs=jnp.array([], dtype=jnp.float64))
        x = chebpts(n, kind=1)
        values = _as_fun_dtype(f(x))
        c = _chebtech1_vals2coeffs(values)
        return cls(coeffs=c)

    @classmethod
    def _adaptive_construct(
        cls,
        f: Callable[[jax.Array], jax.Array],
        maxpow2: int = 16,
        start_pow2: int = 4,
    ) -> "Chebtech1":
        """Adaptive construction — Python-level loop, NOT JIT-safe.

        Evaluates f on grids of size 2^k for k = start_pow2, ..., maxpow2
        (note: 1st-kind grids have exactly 2^k points, not 2^k+1).
        """
        vscale = 0.0
        c = None
        for k in range(start_pow2, maxpow2 + 1):
            n = 2**k
            x = chebpts(n, kind=1)
            values = _as_fun_dtype(f(x))
            c = _chebtech1_vals2coeffs(values)
            vscale = max(vscale, float(jnp.max(jnp.abs(values))))
            ishappy, cutoff = cls.happiness_check(
                c,
                values,
                op=f,
                vscale=vscale,
            )
            if ishappy:
                return cls(coeffs=c[:cutoff], ishappy=True)

        warnings.warn(
            f"Chebtech1.from_function: function did not converge with "
            f"{2**maxpow2} points. Returning unhappy representation.",
            stacklevel=2,
        )
        return cls(coeffs=c, ishappy=False)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the Chebyshev interpolant at point(s) x in [-1, 1].

        Uses Clenshaw's algorithm — same as Chebtech2 because both store
        the same Chebyshev coefficient basis.

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            Evaluation point(s).

        Returns
        -------
        y : jax.Array, same shape as x

        Notes
        -----
        JIT-safe, grad-safe, vmap-safe.

        Provenance
        ----------
        MATLAB source : @chebtech/clenshaw.m, @chebtech/feval.m
        Chebfun commit: 7574c77
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        return _clenshaw(self.coeffs, x)

    # ------------------------------------------------------------------
    # Static methods: vals2coeffs / coeffs2vals (1st-kind specific)
    # ------------------------------------------------------------------

    @staticmethod
    def vals2coeffs(values: jax.Array) -> jax.Array:
        """Convert values at 1st-kind Chebyshev points to Chebyshev coefficients.

        Parameters
        ----------
        values : jax.Array, shape (n,)

        Returns
        -------
        coeffs : jax.Array, shape (n,)

        Provenance
        ----------
        MATLAB source : @chebtech1/vals2coeffs.m
        Chebfun commit: 7574c77
        """
        return _chebtech1_vals2coeffs(values)

    @staticmethod
    def coeffs2vals(coeffs: jax.Array) -> jax.Array:
        """Convert Chebyshev coefficients to values at 1st-kind Chebyshev points.

        Parameters
        ----------
        coeffs : jax.Array, shape (n,)

        Returns
        -------
        values : jax.Array, shape (n,)

        Provenance
        ----------
        MATLAB source : @chebtech1/coeffs2vals.m
        Chebfun commit: 7574c77
        """
        return _chebtech1_coeffs2vals(coeffs)

    @staticmethod
    def alias(coeffs: jax.Array, m: int) -> jax.Array:
        """Alias 1st-kind Chebyshev coefficients to length ``m``.

        Note the 1st-kind folding formula differs from the 2nd-kind grid
        even though the coefficients are for 1st-kind Chebyshev polynomials
        in both cases.

        Provenance
        ----------
        MATLAB source : @chebtech1/alias.m
        Chebfun commit: 7574c77
        """
        return _alias_chebtech1(coeffs, m)

    @staticmethod
    def barywts(n: int) -> jax.Array:
        """Barycentric weights for the ``n`` 1st-kind Chebyshev points.

        Provenance
        ----------
        MATLAB source : @chebtech1/barywts.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.diffmat import _cheb1_barywts

        return _cheb1_barywts(n)

    @staticmethod
    def bary(x: jax.Array, gvals: jax.Array) -> jax.Array:
        """Barycentric interpolation of values on the 1st-kind grid.

        Evaluates at ``x`` the polynomial interpolant through the data
        ``gvals`` given on the ``len(gvals)``-point 1st-kind Chebyshev
        grid, using the closed-form barycentric weights.

        Provenance
        ----------
        MATLAB source : @chebtech1/bary.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.diffmat import _cheb1_barywts
        from chebfunjax.utils.interpolation import bary as _bary

        n = gvals.shape[0]
        return _bary(jnp.asarray(x, dtype=gvals.dtype), gvals,
                     chebpts(n, kind=1), _cheb1_barywts(n))

    @staticmethod
    def angles(n: int) -> jax.Array:
        """Angles ``acos(x)`` of the ``n`` 1st-kind Chebyshev points.

        Provenance
        ----------
        MATLAB source : @chebtech1/angles.m
        Chebfun commit: 7574c77
        """
        if n == 0:
            return jnp.array([], dtype=jnp.float64)
        return jnp.arange(n - 0.5, 0.0, -1.0, dtype=jnp.float64) * jnp.pi / n

    def sample(self, n: int | None = None):
        """Sample the tech at ``n`` 1st-kind Chebyshev points.

        Returns ``(values, points)``; ``n = len(self)`` if omitted.

        Provenance
        ----------
        MATLAB source : @chebtech/sample.m
        Chebfun commit: 7574c77
        """
        if n is None:
            n = len(self)
        values = _chebtech1_coeffs2vals(_alias_chebtech1(self.coeffs, n))
        points = chebpts(n, kind=1)
        return values, points

    def trigcoeffs(self, N: int | None = None) -> jax.Array:
        """Trigonometric (complex-exponential) coefficients of the tech.

        Provenance
        ----------
        MATLAB source : @chebtech/trigcoeffs.m
        Chebfun commit: 7574c77
        """
        return _trigcoeffs_from_tech(self, N)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of Chebyshev coefficients (= polynomial degree + 1)."""
        return self.coeffs.shape[0]

    @property
    def values(self) -> jax.Array:
        """Function values at 1st-kind Chebyshev points (ascending order)."""
        return _chebtech1_coeffs2vals(self.coeffs)

    @property
    def vscale(self) -> float:
        """Vertical scale: max absolute function value."""
        return float(jnp.max(jnp.abs(self.values)))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        vs = self.vscale
        return f"Chebtech1(n={self.n}, vscale={vs:.4g})"

    # ------------------------------------------------------------------
    # Core operations (delegate to the shared helpers)
    # ------------------------------------------------------------------

    def prolong(self, n: int) -> "Chebtech1":
        """Return a new Chebtech1 with n coefficients (zero-pad or truncate).

        Provenance
        ----------
        MATLAB source : @chebtech/prolong.m
        Chebfun commit: 7574c77
        """
        m = self.n
        if n == m:
            return self
        if n > m:
            padded = jnp.concatenate(
                [self.coeffs,
                 jnp.zeros((n - m,) + self.coeffs.shape[1:],
                           dtype=self.coeffs.dtype)]
            )
            return Chebtech1(coeffs=padded, ishappy=self.ishappy)
        return Chebtech1(coeffs=self.coeffs[:max(n, 0)], ishappy=self.ishappy)

    def simplify(self, tol: float | None = None) -> "Chebtech1":
        """Return a new Chebtech1 with trailing coefficients chopped.

        Provenance
        ----------
        MATLAB source : @chebtech/simplify.m
        Chebfun commit: 7574c77
        """
        if not self.ishappy:
            return self
        nold = self.n
        N = max(17, round(nold * 1.25 + 5))
        prolonged_c = jnp.concatenate(
            [self.coeffs,
             jnp.zeros((N - nold,) + self.coeffs.shape[1:],
                       dtype=self.coeffs.dtype)]
        )
        # Round-trip through values to create a plateau
        c = _chebtech1_vals2coeffs(_chebtech1_coeffs2vals(prolonged_c))
        cutoff = _chop_columns(c, tol)
        cutoff = min(cutoff, nold)
        return Chebtech1(coeffs=self.coeffs[:cutoff], ishappy=self.ishappy)

    # ------------------------------------------------------------------
    # Arithmetic (returns Chebtech1)
    # ------------------------------------------------------------------

    def __add__(self, other) -> "Chebtech1":
        """Add a Chebtech1 or scalar.

        Provenance
        ----------
        MATLAB source : @chebtech/plus.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech1.empty()
        if isinstance(other, Chebtech1):
            n = max(self.n, other.n)
            fc = _prolong_coeffs(self.coeffs, n)
            gc = _prolong_coeffs(other.coeffs, n)
            return Chebtech1.from_coeffs(fc + gc, ishappy=self.ishappy and other.ishappy)
        else:
            c = self.coeffs.at[0].add(_as_scalar(other))
            return Chebtech1.from_coeffs(c, ishappy=self.ishappy)

    def __radd__(self, other) -> "Chebtech1":
        return self.__add__(other)

    def __sub__(self, other) -> "Chebtech1":
        """Subtract a Chebtech1 or scalar.

        Provenance
        ----------
        MATLAB source : @chebtech/minus.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech1.empty()
        return self + (-other)

    def __rsub__(self, other) -> "Chebtech1":
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech1.empty()
        return -(self - other)

    def __neg__(self) -> "Chebtech1":
        """Unary minus.

        Provenance
        ----------
        MATLAB source : @chebtech/uminus.m
        Chebfun commit: 7574c77
        """
        return Chebtech1.from_coeffs(-self.coeffs, ishappy=self.ishappy)

    def __pos__(self) -> "Chebtech1":
        return self

    def __mul__(self, other) -> "Chebtech1":
        """Pointwise multiplication.

        Provenance
        ----------
        MATLAB source : @chebtech/times.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech1.empty()
        if isinstance(other, Chebtech1):
            hc = _coeff_multiply(self.coeffs, other.coeffs)
            return Chebtech1.from_coeffs(hc, ishappy=self.ishappy and other.ishappy)
        else:
            return Chebtech1.from_coeffs(self.coeffs * _as_scalar(other), ishappy=self.ishappy)

    def __rmul__(self, other) -> "Chebtech1":
        return self.__mul__(other)

    def __matmul__(self, other) -> "Chebtech1":
        """MATLAB mtimes ``f * A``: right-multiply an array-valued tech
        by a matrix, mixing its columns (coeffs @ A).

        Provenance
        ----------
        MATLAB source : @chebtech/mtimes.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self):
            return Chebtech1.empty()
        A = jnp.asarray(other)
        c = self.coeffs if self.coeffs.ndim == 2 else self.coeffs[:, None]
        return Chebtech1(coeffs=c @ A, ishappy=self.ishappy)

    def fliplr(self) -> "Chebtech1":
        """Reverse the column order of an array-valued tech (a no-op
        for scalar-valued input).

        Provenance
        ----------
        MATLAB source : @chebtech/fliplr.m
        Chebfun commit: 7574c77
        """
        if self.coeffs.ndim == 1:
            return self
        return Chebtech1(coeffs=self.coeffs[:, ::-1],
                         ishappy=self.ishappy)

    def flipud(self) -> "Chebtech1":
        """Return g with g(x) = f(-x): negate the odd coefficients.

        Provenance
        ----------
        MATLAB source : @chebtech/flipud.m
        Chebfun commit: 7574c77
        """
        return Chebtech2.flipud(self)

    def real(self) -> "Chebtech1":
        """Real part (MATLAB @chebtech/real.m)."""
        return Chebtech2.real(self)

    def imag(self) -> "Chebtech1":
        """Imaginary part (MATLAB @chebtech/imag.m)."""
        return Chebtech2.imag(self)

    def conj(self) -> "Chebtech1":
        """Complex conjugate (MATLAB @chebtech/conj.m)."""
        return Chebtech2.conj(self)

    def assign_columns(self, cols, g) -> "Chebtech1":
        """Overwrite the columns ``cols`` (0-based) of an array-valued
        tech with the columns of ``g`` (MATLAB assignColumns);
        ``g=None`` deletes the columns instead.

        Provenance
        ----------
        MATLAB source : @chebtech/assignColumns.m
        Chebfun commit: 7574c77
        """
        return Chebtech2.assign_columns(self, cols, g)

    def mat2cell(self, sizes) -> list:
        """Split an array-valued tech by column counts (MATLAB
        ``mat2cell(f, 1, sizes)``).

        Provenance
        ----------
        MATLAB source : @chebtech/mat2cell.m
        Chebfun commit: 7574c77
        """
        return Chebtech2.mat2cell(self, sizes)

    @classmethod
    def cell2mat(cls, techs) -> "Chebtech1":
        """Horizontally concatenate techs into one array-valued tech
        (MATLAB ``cell2mat([g h])``).

        Provenance
        ----------
        MATLAB source : @chebtech/cell2mat.m
        Chebfun commit: 7574c77
        """
        n = max(t.n for t in techs)
        cols = []
        for t in techs:
            c = t.prolong(n).coeffs
            cols.append(c if c.ndim == 2 else c[:, None])
        dt = jnp.result_type(*(c.dtype for c in cols))
        return cls(coeffs=jnp.concatenate(
                       [c.astype(dt) for c in cols], axis=1),
                   ishappy=all(t.ishappy for t in techs))

    def __truediv__(self, other) -> "Chebtech1":
        """Division.

        Provenance
        ----------
        MATLAB source : @chebtech/rdivide.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebtech1):
            # Adaptive re-construction so the quotient is fully resolved
            # (MATLAB: compose(f, @rdivide, g)).
            return Chebtech1.from_function(
                lambda x: _clenshaw(self.coeffs, x) / _clenshaw(other.coeffs, x)
            )
        else:
            return Chebtech1.from_coeffs(self.coeffs / _as_scalar(other), ishappy=self.ishappy)

    def __rtruediv__(self, other) -> "Chebtech1":
        return Chebtech1.from_function(
            lambda x: _as_scalar(other) / _clenshaw(self.coeffs, x)
        )

    def __pow__(self, exponent) -> "Chebtech1":
        """Raise to a power.

        Provenance
        ----------
        MATLAB source : @chebtech/power.m
        Chebfun commit: 7574c77
        """
        if isinstance(exponent, int) and exponent >= 0:
            if exponent == 0:
                return Chebtech1.from_coeffs(jnp.array([1.0], dtype=jnp.float64))
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        else:
            # Fractional power: adaptive re-construction (MATLAB compose)
            return Chebtech1.from_function(
                lambda x: _clenshaw(self.coeffs, x) ** _as_scalar(exponent)
            )

    def __abs__(self) -> "Chebtech1":
        n = max(2 * self.n, 17)
        x = chebpts(n, kind=1)
        fv = jnp.abs(_clenshaw(self.coeffs, x))
        return Chebtech1.from_values(fv)

    # ------------------------------------------------------------------
    # Calculus (same coefficient-level helpers as Chebtech2)
    # ------------------------------------------------------------------

    def diff(self, k: int = 1, dim: int = 1) -> "Chebtech1":
        """Differentiate *k* times (dim=2 takes finite differences
        across the columns of an array-valued tech, MATLAB
        ``diff(f, k, 2)``).

        Provenance
        ----------
        MATLAB source : @chebtech/diff.m
        Chebfun commit: 7574c77
        Algorithm: Page 34 of Mason & Handscomb, "Chebyshev Polynomials", 2003.
        """
        if dim == 2:
            if self.coeffs.ndim == 1:
                return Chebtech1(
                    coeffs=jnp.zeros((0,), dtype=self.coeffs.dtype),
                    ishappy=self.ishappy)
            return Chebtech1(coeffs=jnp.diff(self.coeffs, n=k, axis=1),
                             ishappy=self.ishappy)
        if k == 0:
            return self
        new_coeffs = _diff_coeffs(self.coeffs, k)
        return Chebtech1.from_coeffs(new_coeffs, ishappy=self.ishappy)

    def cumsum(self) -> "Chebtech1":
        """Indefinite integral with F(-1) = 0.

        Provenance
        ----------
        MATLAB source : @chebtech/cumsum.m
        Chebfun commit: 7574c77
        Algorithm: Pages 32-33 of Mason & Handscomb, "Chebyshev Polynomials".
        """
        new_coeffs = _cumsum_coeffs(self.coeffs)
        return Chebtech1.from_coeffs(new_coeffs, ishappy=self.ishappy)

    def sum(self, dim: int = 1) -> "jax.Array | Chebtech1":
        r"""Definite integral over [-1, 1] (dim=2 sums the columns of an
        array-valued tech, MATLAB ``sum(f, 2)``).

        Provenance
        ----------
        MATLAB source : @chebtech/sum.m
        Chebfun commit: 7574c77
        Algorithm: Trefethen, ATAP, Thm 19.2.
        """
        if dim == 2:
            if self.coeffs.ndim == 1:
                return self
            return Chebtech1(coeffs=jnp.sum(self.coeffs, axis=1),
                             ishappy=self.ishappy)
        return _definite_integral(self.coeffs)

    def inner(self, other: "Chebtech1") -> jax.Array:
        r"""L^2 inner product <self, other> = \int_{-1}^{1} f(x) g(x) dx.

        Provenance
        ----------
        MATLAB source : @chebtech/innerProduct.m
        Chebfun commit: 7574c77
        """
        out = _inner_product(self.coeffs, other.coeffs)
        # MATLAB @chebtech/innerProduct.m forces a nonnegative real result
        # when f == g (isequal branch).  The identity check is JIT-safe;
        # the value check runs only on concrete (non-traced) arrays.
        same = other is self
        if not same and self.coeffs.shape == other.coeffs.shape:
            if not isinstance(self.coeffs, jax.core.Tracer) and \
                    not isinstance(other.coeffs, jax.core.Tracer):
                same = bool(jnp.all(self.coeffs == other.coeffs))
        if same and out.ndim == 0:
            return jnp.abs(out)
        return out

    def norm(self, p: float = 2.0) -> jax.Array:
        """Lp norm on [-1, 1].

        Provenance
        ----------
        MATLAB source : @chebtech/normest.m
        Chebfun commit: 7574c77
        """
        if p == 2:
            return jnp.sqrt(jnp.abs(self.inner(self)))
        elif p == jnp.inf or p == float("inf"):
            n = max(2 * self.n + 1, 65)
            x = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)
            return jnp.max(jnp.abs(_clenshaw(self.coeffs, x)))
        else:
            fp = self.__abs__().__pow__(p)
            return fp.sum() ** (1.0 / p)

    # ------------------------------------------------------------------
    # Rootfinding
    # ------------------------------------------------------------------

    def roots(self) -> jax.Array:
        """Real roots in [-1, 1] via colleague matrix eigenvalues.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebtech/roots.m
        Chebfun commit: 7574c77
        """
        if self.coeffs.ndim == 2:
            # Array-valued: roots per column, NaN-padded to equal length
            # (MATLAB @chebtech/roots.m), same as Chebtech2.roots.
            import numpy as _np
            cols = [_np.asarray(_roots_colleague(self.coeffs[:, j]))
                    for j in range(self.coeffs.shape[1])]
            nmax = max((len(c) for c in cols), default=0)
            out = _np.full((nmax, len(cols)), _np.nan)
            for j, c in enumerate(cols):
                out[: len(c), j] = c
            return jnp.asarray(out)
        return _roots_colleague(self.coeffs)

    # ------------------------------------------------------------------
    # Happiness check (mirrors Chebtech2 but uses 1st-kind sampling)
    # ------------------------------------------------------------------

    @staticmethod
    def happiness_check(
        coeffs: jax.Array,
        values: jax.Array,
        op: Callable | None = None,
        tol: float | None = None,
        vscale: float = 0.0,
        hscale: float = 1.0,
    ) -> tuple[bool, int]:
        """Standard happiness check for adaptive construction.

        Same logic as Chebtech2.happiness_check but sample-tests at
        1st-kind off-grid points.

        Provenance
        ----------
        MATLAB source : @chebtech/happinessCheck.m, @chebtech/standardCheck.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        if tol is None:
            tol = _EPS

        n = coeffs.shape[0]
        vscale_local = float(jnp.max(jnp.abs(values)))
        vscale = max(vscale, vscale_local)

        if vscale_local > 0:
            scaled_tol = tol * max(hscale, vscale / vscale_local)
        else:
            scaled_tol = tol * hscale

        cutoff = _chop_columns(coeffs, scaled_tol)
        ishappy = cutoff < n

        if ishappy and op is not None:
            xeval = jnp.array(
                [-0.357998918959666, 0.036785641195074], dtype=jnp.float64
            )
            f_test = Chebtech1(coeffs=coeffs[:cutoff])
            v_fun = f_test(xeval)
            v_op = _as_fun_dtype(op(xeval))
            err = float(jnp.max(jnp.abs(v_op - v_fun)))
            sample_tol = _np.sqrt(max(_EPS, tol)) * max(hscale * vscale_local, vscale)
            if err > sample_tol:
                ishappy = False
                cutoff = n

        return ishappy, cutoff

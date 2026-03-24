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
import numpy as np

from chebfunjax.utils.misc import standard_chop
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# Clenshaw evaluation (JIT-safe, grad-safe, vmap-safe)
# ============================================================================


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

    # Edge cases
    if n == 0:
        return jnp.zeros_like(x, dtype=jnp.float64)
    if n == 1:
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

    init = (
        jnp.zeros_like(x, dtype=jnp.float64),
        jnp.zeros_like(x, dtype=jnp.float64),
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
    return jnp.concatenate([coeffs, jnp.zeros(n - m, dtype=jnp.float64)])


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
        return jnp.zeros(1, dtype=jnp.float64)

    # w[k] = 2*(k+1) for k = 0 .. n-2
    w = 2.0 * jnp.arange(1, n, dtype=jnp.float64)
    v = w * c[1:]  # v[k] = 2*(k+1)*c_{k+1}

    # Accumulate from the tail, even and odd indices separately.
    out = jnp.zeros(n - 1, dtype=jnp.float64)

    # Slice1: indices n-2, n-4, ..., i.e. v[-1], v[-3], ...
    s1 = v[::-1][::2]  # reversed, take every other
    cs1 = jnp.cumsum(s1)
    # Slice2: indices n-3, n-5, ..., i.e. v[-2], v[-4], ...
    s2 = v[::-1][1::2]
    cs2 = jnp.cumsum(s2)

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
    cp = jnp.concatenate([c, jnp.zeros(2, dtype=jnp.float64)])

    b = jnp.zeros(n + 1, dtype=jnp.float64)

    # b[r] = (c[r-1] - c[r+1]) / (2*r) for r = 2, ..., n
    rk = jnp.arange(2, n + 1, dtype=jnp.float64)
    b = b.at[2 : n + 1].set((cp[1:n] - cp[3 : n + 2]) / (2.0 * rk))

    # b[1] = c[0] - c[2]/2
    b = b.at[1].set(cp[0] - cp[2] / 2.0)

    # b[0]: choose so that F(-1) = 0
    # F(-1) = sum_r b_r * T_r(-1) = sum_r b_r * (-1)^r = 0
    # => b_0 = - sum_{r=1}^{n} (-1)^r * b_r = sum_{r=1}^{n} (-1)^{r+1} * b_r
    vv = jnp.ones(n, dtype=jnp.float64)
    vv = vv.at[1::2].set(-1.0)
    b = b.at[0].set(jnp.dot(vv, b[1 : n + 1]))

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
    return jnp.dot(coeffs, moments)


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

    # Prolong both to length n
    fc = jnp.zeros(n, dtype=jnp.float64).at[:nf].set(f_coeffs)
    gc = jnp.zeros(n, dtype=jnp.float64).at[:ng].set(g_coeffs)

    # Convert to values
    fv = _coeffs_to_values(fc)
    gv = _coeffs_to_values(gc)

    # Clenshaw-Curtis weights
    w = chebweights(n, kind=2)

    return jnp.dot(w * fv, gv)


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

    # Pad both to length mn
    f = jnp.zeros(mn, dtype=jnp.float64).at[:nf].set(fc)
    g = jnp.zeros(mn, dtype=jnp.float64).at[:ng].set(gc)

    # Embed into circulant: double the first coefficient
    t = jnp.concatenate([2.0 * f[:1], f[1:]])
    x = jnp.concatenate([2.0 * g[:1], g[1:]])

    # Circulant multiply via FFT
    t_ext = jnp.concatenate([t, t[-1:0:-1]])
    x_ext = jnp.concatenate([x, x[-1:0:-1]])
    product = jnp.real(jnp.fft.ifft(jnp.fft.fft(t_ext) * jnp.fft.fft(x_ext)))

    # Extract result
    hc = 0.25 * jnp.concatenate([product[:1], product[1:mn] + product[-1 : mn - 1 : -1]])

    return hc


# ============================================================================
# Root-finding helpers (numpy, NOT JIT-safe)
# ============================================================================


def _roots_colleague(coeffs: jax.Array) -> jax.Array:
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
    c = np.asarray(coeffs, dtype=np.float64)
    htol = 100.0 * np.finfo(np.float64).eps

    # Normalize
    vscl = np.max(np.abs(c))
    if vscl == 0.0:
        return jnp.array([0.0], dtype=jnp.float64)
    c_scaled = c / vscl

    r = _roots_main(c_scaled, htol)
    r = np.sort(r)
    return jnp.asarray(r, dtype=jnp.float64)


def _roots_main(c: np.ndarray, htol: float) -> np.ndarray:
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
    # Construction (class methods — NOT __init__)
    # ------------------------------------------------------------------

    @classmethod
    def from_coeffs(cls, coeffs: jax.Array) -> "Chebtech2":
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
        coeffs = jnp.atleast_1d(jnp.asarray(coeffs, dtype=jnp.float64))
        return cls(coeffs=coeffs)

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
        values = jnp.atleast_1d(jnp.asarray(values, dtype=jnp.float64))
        c = vals2coeffs(values)
        return cls(coeffs=c)

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        *,
        n: int | None = None,
        maxpow2: int = 16,
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
        if n is not None:
            return cls._fixed_construct(f, n)
        return cls._adaptive_construct(f, maxpow2)

    @classmethod
    def _fixed_construct(
        cls, f: Callable[[jax.Array], jax.Array], n: int
    ) -> "Chebtech2":
        """Fixed-length construction on an n-point Chebyshev-2 grid."""
        if n <= 0:
            return cls(coeffs=jnp.array([], dtype=jnp.float64))
        x = chebpts(n, kind=2)
        values = jnp.asarray(f(x), dtype=jnp.float64)
        c = vals2coeffs(values)
        return cls(coeffs=c)

    @classmethod
    def _adaptive_construct(
        cls,
        f: Callable[[jax.Array], jax.Array],
        maxpow2: int = 16,
        start_pow2: int = 4,
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
        """
        vscale = 0.0
        c = None
        for k in range(start_pow2, maxpow2 + 1):
            n = 2**k + 1
            x = chebpts(n, kind=2)
            values = jnp.asarray(f(x), dtype=jnp.float64)
            c = vals2coeffs(values)
            vscale = max(vscale, float(jnp.max(jnp.abs(values))))
            ishappy, cutoff = cls.happiness_check(
                c,
                values,
                op=f,
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
                    jnp.zeros(n - m, dtype=jnp.float64),
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

        cutoff = standard_chop(c, tol)
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
            # Binary operator: op(self(x), g(x))
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

        cutoff = standard_chop(coeffs, scaled_tol)
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
            v_op = jnp.asarray(op(xeval), dtype=jnp.float64)
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
        if isinstance(other, Chebtech2):
            # Prolong to the same length (zero-pad shorter one)
            nf = self.n
            ng = other.n
            n = max(nf, ng)
            fc = _prolong_coeffs(self.coeffs, n)
            gc = _prolong_coeffs(other.coeffs, n)
            return Chebtech2.from_coeffs(fc + gc)
        else:
            # Scalar addition: only the c_0 coefficient changes
            c = self.coeffs.at[0].add(jnp.float64(other))
            return Chebtech2.from_coeffs(c)

    def __radd__(self, other) -> "Chebtech2":
        return self.__add__(other)

    def __sub__(self, other) -> "Chebtech2":
        """Subtract a Chebtech2 or scalar.

        Provenance
        ----------
        MATLAB source : @chebtech/minus.m
        Chebfun commit: 7574c77
        """
        return self + (-other)

    def __rsub__(self, other) -> "Chebtech2":
        return -(self - other)

    def __neg__(self) -> "Chebtech2":
        """Unary minus.

        Provenance
        ----------
        MATLAB source : @chebtech/uminus.m
        Chebfun commit: 7574c77
        """
        return Chebtech2.from_coeffs(-self.coeffs)

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
        if isinstance(other, Chebtech2):
            hc = _coeff_multiply(self.coeffs, other.coeffs)
            return Chebtech2.from_coeffs(hc)
        else:
            return Chebtech2.from_coeffs(self.coeffs * jnp.float64(other))

    def __rmul__(self, other) -> "Chebtech2":
        return self.__mul__(other)

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
            # Evaluate both on a fine grid and divide
            n = self.n + other.n
            x = chebpts(n, kind=2)
            fv = _clenshaw(self.coeffs, x)
            gv = _clenshaw(other.coeffs, x)
            return Chebtech2.from_values(fv / gv)
        else:
            return Chebtech2.from_coeffs(self.coeffs / jnp.float64(other))

    def __rtruediv__(self, other) -> "Chebtech2":
        """Scalar / Chebtech2."""
        n = max(self.n, 17)
        x = chebpts(n, kind=2)
        fv = jnp.float64(other) / _clenshaw(self.coeffs, x)
        return Chebtech2.from_values(fv)

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
                return Chebtech2.from_coeffs(jnp.array([1.0], dtype=jnp.float64))
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        elif isinstance(exponent, Chebtech2):
            # f^g via evaluation
            n = self.n + exponent.n
            x = chebpts(n, kind=2)
            fv = _clenshaw(self.coeffs, x)
            gv = _clenshaw(exponent.coeffs, x)
            return Chebtech2.from_values(fv**gv)
        else:
            # Fractional power: evaluate on a grid
            n = max(2 * self.n, 17)
            x = chebpts(n, kind=2)
            fv = _clenshaw(self.coeffs, x) ** jnp.float64(exponent)
            return Chebtech2.from_values(fv)

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

    def diff(self, k: int = 1) -> "Chebtech2":
        """Differentiate *k* times.

        Uses the Chebyshev coefficient recurrence (Mason & Handscomb, p. 34).

        JIT-safe: yes (k must be a static integer).

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

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
        if k == 0:
            return self
        new_coeffs = _diff_coeffs(self.coeffs, k)
        return Chebtech2.from_coeffs(new_coeffs)

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
        return Chebtech2.from_coeffs(new_coeffs)

    def sum(self) -> jax.Array:
        r"""Definite integral over [-1, 1].

        Uses the Chebyshev moments: integral of T_k = 2/(1-k^2) for even k.

        JIT-safe: yes.

        Returns
        -------
        jax.Array (scalar)
            The definite integral.

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
        return _inner_product(self.coeffs, other.coeffs)

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
        return _roots_colleague(self.coeffs)

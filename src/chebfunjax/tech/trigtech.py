"""Trigonometric technology — smooth periodic function approximation on [-1, 1].

Translated from MATLAB Chebfun class @trigtech (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Coefficient convention
----------------------
Fourier series: f(x) = sum_k c_k * exp(i*pi*k*x), x in [-1, 1].

Coefficients are stored in *descending wavenumber* order:
  - Odd N=2M+1:  c_{-M}, c_{-M+1}, ..., c_0, ..., c_M
    (c_0 at index M = N//2)
  - Even N=2M:   c_{-M}, c_{-M+1}, ..., c_0, ..., c_{M-1}
    (c_0 at index M = N//2)

The coefficients are always stored as complex128 arrays.  For real-valued
functions the Hermitian symmetry c_{-k} = conj(c_k) holds approximately up
to floating-point precision; the ``is_real`` flag records whether the original
function was sampled from real values.
"""

from __future__ import annotations

import warnings
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.utils.misc import standard_chop

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# FFT-based transforms (JIT-safe)
# ============================================================================


def trig_vals2coeffs(values: jax.Array) -> jax.Array:
    r"""Convert values at N equally spaced points on [-1,1) to Fourier coefficients.

    Given values ``v[k] = f(x_k)`` at ``x_k = -1 + 2k/N``, k = 0,...,N-1,
    returns complex Fourier coefficients ``c[j]`` such that the trigonometric
    interpolant is

    .. math::

        f(x) = \sum_k c_k \exp(i \pi k x)

    Odd N: sum over k = -(N-1)/2, ..., (N-1)/2.
    Even N: sum over k = -N/2, ..., N/2-1.

    Coefficients are stored in descending wavenumber order (lowest k first).

    Parameters
    ----------
    values : jax.Array, shape (N,) real or complex
        Function values at N equispaced trigonometric points on [-1, 1).

    Returns
    -------
    coeffs : jax.Array, shape (N,) complex128
        Fourier coefficients in descending-wavenumber order.

    Notes
    -----
    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @trigtech/vals2coeffs.m
    Chebfun commit: 7574c77
    """
    values = jnp.asarray(values, dtype=jnp.complex128)
    n = values.shape[0]

    if n <= 1:
        return values

    # coeffs = (1/n) * fftshift(fft(values))
    coeffs = jnp.fft.fftshift(jnp.fft.fft(values)) / n

    # The FFT is for [0, 2) but we want [-1, 1).
    # Fix: multiply c_k by (-1)^k.
    if n % 2 == 1:
        half = (n - 1) // 2
        ks = jnp.arange(-half, half + 1, dtype=jnp.float64)
    else:
        half = n // 2
        ks = jnp.arange(-half, half, dtype=jnp.float64)

    even_odd_fix = (-1.0 + 0j) ** ks
    return coeffs * even_odd_fix


def trig_coeffs2vals(coeffs: jax.Array) -> jax.Array:
    r"""Convert Fourier coefficients to values at N equally spaced points on [-1,1).

    Inverse of ``trig_vals2coeffs``.

    Parameters
    ----------
    coeffs : jax.Array, shape (N,) complex
        Fourier coefficients in descending-wavenumber order.

    Returns
    -------
    values : jax.Array, shape (N,) complex128
        Function values at the N equispaced points x_k = -1 + 2k/N.

    Notes
    -----
    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @trigtech/coeffs2vals.m
    Chebfun commit: 7574c77
    """
    coeffs = jnp.asarray(coeffs, dtype=jnp.complex128)
    n = coeffs.shape[0]

    if n <= 1:
        return coeffs

    if n % 2 == 1:
        half = (n - 1) // 2
        ks = jnp.arange(-half, half + 1, dtype=jnp.float64)
    else:
        half = n // 2
        ks = jnp.arange(-half, half, dtype=jnp.float64)

    # Undo the even/odd fix applied in vals2coeffs
    even_odd_fix = (-1.0 + 0j) ** ks
    c = coeffs * even_odd_fix

    return jnp.fft.ifft(jnp.fft.ifftshift(n * c))


# ============================================================================
# Trigonometric grid points
# ============================================================================


def trigpts(n: int) -> jax.Array:
    """Return N equally spaced points on [-1, 1).

    The points are x_k = -1 + 2k/N for k = 0, 1, ..., N-1.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    jax.Array, shape (n,) float64
        Equispaced points on [-1, 1).
    """
    return jnp.linspace(-1.0, 1.0, n, endpoint=False, dtype=jnp.float64)


# ============================================================================
# Evaluation (JIT-safe, grad-safe, vmap-safe)
# ============================================================================


def _trig_eval(coeffs: jax.Array, x: jax.Array, is_real: bool = True) -> jax.Array:
    r"""Evaluate a trigonometric series at points x.

    For real-valued functions (``is_real=True``), uses real arithmetic via
    the cosine/sine decomposition (Horner scheme from MATLAB @trigtech/horner.m).
    For complex-valued functions, uses the complex Horner scheme.

    Parameters
    ----------
    coeffs : jax.Array, shape (N,) complex
        Fourier coefficients in descending wavenumber order.
    x : jax.Array, scalar or shape (m,)
        Evaluation points.
    is_real : bool, default True
        Whether to use real arithmetic and return a real result.

    Returns
    -------
    y : jax.Array
        Evaluated values. float64 if is_real, complex128 otherwise.

    Notes
    -----
    JIT-safe: yes. vmap-safe: yes. grad-safe: yes.

    Provenance
    ----------
    MATLAB source : @trigtech/horner.m
    Chebfun commit: 7574c77
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    scalar_input = x.ndim == 0
    x_1d = jnp.atleast_1d(x)

    n = coeffs.shape[0]
    coeffs_cx = jnp.asarray(coeffs, dtype=jnp.complex128)

    if n == 0:
        if is_real:
            result = jnp.zeros_like(x_1d, dtype=jnp.float64)
        else:
            result = jnp.zeros_like(x_1d, dtype=jnp.complex128)
        return result[0] if scalar_input else result

    if n == 1:
        c0 = coeffs_cx[0]
        if is_real:
            val = jnp.real(c0).astype(jnp.float64)
        else:
            val = c0.astype(jnp.complex128)
        result = jnp.broadcast_to(val, x_1d.shape)
        return result[0] if scalar_input else result

    if is_real:
        result = _trig_eval_real(coeffs_cx, x_1d)
    else:
        result = _trig_eval_complex(coeffs_cx, x_1d)

    return result[0] if scalar_input else result


def _trig_eval_real(coeffs_cx: jax.Array, x: jax.Array) -> jax.Array:
    """Real Horner evaluation for real-valued trig series.

    Translates the real-arithmetic path from @trigtech/horner.m.

    For N odd (N = 2M+1):
      c_{-M}, ..., c_0, ..., c_M  (c_0 at index M)
      f(x) = a_0 + 2 * sum_{k=1}^{M} [a_k*cos(k*pi*x) - b_k*sin(k*pi*x)]
    where a_k = Re(c_{-k}), b_k = Im(c_{-k})  (negative-indexed coeffs, per MATLAB).

    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @trigtech/horner.m (horner_scl_real, horner_vec_real)
    """
    n = coeffs_cx.shape[0]
    c0_idx = n // 2  # index of constant mode c_0

    # MATLAB: c = c(n_half:-1:1,:) picks from c_0 down to c_{-(n_half-1)}
    # n_half = ceil((N+1)/2)
    # For odd N=5: n_half=3, 1-based indices 3,2,1 -> 0-based 2,1,0
    #   = c_0, c_{-1}, c_{-2}
    # For even N=4: n_half=3 (ceil(5/2)=3), indices 3,2,1 -> 0-based 2,1,0
    #   wavenumbers: -2,-1,0,1; c_0 at index 2
    #   picks: c[2]=c_0, c[1]=c_{-1}, c[0]=c_{-2}
    (n + 2) // 2  # = ceil((n+1)/2) but using integer arithmetic

    # Slice from c0_idx down to 0 (inclusive): c_0, c_{-1}, ..., c_{-c0_idx}
    c_slice = coeffs_cx[c0_idx::-1]  # shape (c0_idx+1,) = (n_half,) for odd; same for even
    a = jnp.real(c_slice)  # cosine amplitudes
    b = jnp.imag(c_slice)  # sine amplitudes

    # For even N: the highest negative mode is c_{-N/2} which pairs with itself
    # (it's a pure cosine mode). MATLAB halves it: a(n_half) /= 2, b(n_half) = 0.
    if n % 2 == 0:
        a = a.at[-1].set(a[-1] / 2.0)
        b = b.at[-1].set(0.0)

    n_h = a.shape[0]
    u = jnp.cos(jnp.pi * x)   # shape (m,)
    v = jnp.sin(jnp.pi * x)   # shape (m,)

    if n_h == 1:
        return jnp.broadcast_to(a[0], x.shape)

    # Horner recurrence: start from the highest-frequency pair and work down
    # Initialize with the highest-k term (index n_h-1)
    co = jnp.broadcast_to(a[n_h - 1], x.shape)
    si = jnp.broadcast_to(b[n_h - 1], x.shape)

    def body(j, state):
        co_, si_ = state
        # j = 0, ..., n_h-3; inner index k = n_h-2-j goes from n_h-2 down to 1
        k = n_h - 2 - j
        temp = a[k] + u * co_ + v * si_
        si_new = b[k] + u * si_ - v * co_
        return (temp, si_new)

    co, si = jax.lax.fori_loop(0, n_h - 2, body, (co, si))

    # Final: f(x) = a_0 + 2*(u*co + v*si)
    return a[0] + 2.0 * (u * co + v * si)


def _trig_eval_complex(coeffs_cx: jax.Array, x: jax.Array) -> jax.Array:
    """Complex Horner evaluation for general trig series.

    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @trigtech/horner.m (horner_scl_cmplx)
    """
    n = coeffs_cx.shape[0]
    z = jnp.exp(1j * jnp.pi * x.astype(jnp.float64))  # shape (m,)

    # Horner from highest wavenumber (index N-1) down
    q = jnp.broadcast_to(coeffs_cx[n - 1].astype(jnp.complex128), z.shape)

    def body(i, q_):
        j = n - 2 - i  # goes from n-2 down to 1
        return coeffs_cx[j] + z * q_

    q = jax.lax.fori_loop(0, n - 2, body, q)

    # Apply lowest-mode prefactor
    if n % 2 == 1:
        # Odd N: q = exp(-i*pi*(N-1)/2 * x) * (c[0] + z*q)
        prefactor = jnp.exp(-1j * jnp.pi * ((n - 1) / 2) * x)
        return prefactor * (coeffs_cx[0] + z * q)
    else:
        # Even N: q = exp(-i*pi*(N/2-1)*x)*q + cos(N*pi*x/2)*c[0]
        prefactor = jnp.exp(-1j * jnp.pi * (n / 2 - 1) * x)
        return prefactor * q + jnp.cos(n / 2 * jnp.pi * x) * coeffs_cx[0]


# ============================================================================
# Spectral differentiation (JIT-safe)
# ============================================================================


def _trig_diff_coeffs(coeffs: jax.Array, k: int) -> jax.Array:
    r"""Differentiate Fourier coefficients k times.

    Multiplies c_j by (i*pi*j)^k (spectral differentiation in Fourier space).

    Parameters
    ----------
    coeffs : jax.Array, shape (N,) complex
        Fourier coefficients in descending wavenumber order.
    k : int
        Differentiation order (must be static for JIT).

    Returns
    -------
    jax.Array, shape (N,) complex128

    Notes
    -----
    JIT-safe: yes (k static).

    Provenance
    ----------
    MATLAB source : @trigtech/diff.m (diffContinuousDim)
    Chebfun commit: 7574c77
    """
    if k == 0:
        return jnp.asarray(coeffs, dtype=jnp.complex128)

    coeffs_cx = jnp.asarray(coeffs, dtype=jnp.complex128)
    n = coeffs_cx.shape[0]

    if n % 2 == 1:
        half = (n - 1) // 2
        wavenumbers = jnp.arange(-half, half + 1, dtype=jnp.float64)
    else:
        half = n // 2
        wavenumbers = jnp.arange(-half, half, dtype=jnp.float64)

    factor = (1j * jnp.pi * wavenumbers) ** k
    return coeffs_cx * factor


# ============================================================================
# Spectral antiderivative (JIT-safe)
# ============================================================================


def _trig_cumsum_coeffs(coeffs: jax.Array) -> jax.Array:
    r"""Antiderivative of a trigonometric series (F(-1) = 0).

    Given c_k, returns b_k = c_k / (i*pi*k) for k != 0.
    b_0 is determined by the condition F(-1) = 0.

    The function must have zero mean (c_0 = 0) for the antiderivative to be
    periodic.

    Parameters
    ----------
    coeffs : jax.Array, shape (N,) complex
        Fourier coefficients in descending wavenumber order.

    Returns
    -------
    jax.Array, shape (N,) complex128

    Notes
    -----
    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @trigtech/cumsum.m (cumsumContinuousDim)
    Chebfun commit: 7574c77
    """
    coeffs_cx = jnp.asarray(coeffs, dtype=jnp.complex128)
    n = coeffs_cx.shape[0]
    is_even = (n % 2 == 0)

    if is_even:
        # Expand even N to odd by splitting the c_{-N/2} mode
        c0_half = 0.5 * coeffs_cx[0]
        c_expanded = jnp.concatenate([c0_half[None], coeffs_cx[1:], c0_half[None]])
        n_exp = n + 1
        half_exp = (n_exp - 1) // 2
        wavenumbers = jnp.arange(-half_exp, half_exp + 1, dtype=jnp.float64)
        c0_idx = half_exp
    else:
        c_expanded = coeffs_cx
        n_exp = n
        half_exp = (n - 1) // 2
        wavenumbers = jnp.arange(-half_exp, half_exp + 1, dtype=jnp.float64)
        c0_idx = half_exp

    # Zero out the constant mode
    c_work = c_expanded.at[c0_idx].set(0.0 + 0j)

    # Integration factor: 1/(i*pi*k) for k != 0
    safe_wn = jnp.where(wavenumbers == 0, 1.0, wavenumbers)
    int_factor = jnp.where(
        wavenumbers == 0,
        0.0 + 0j,
        1.0 / (1j * jnp.pi * safe_wn + 0j),
    )
    b = c_work * int_factor

    # For even original N: zero out the ±N/2 modes (they're pure cos, don't integrate)
    if is_even:
        b = b.at[0].set(0.0 + 0j)
        b = b.at[-1].set(0.0 + 0j)

    # Determine b_0 from F(-1) = 0:
    # F(-1) = sum_k b_k * exp(-i*pi*k) = sum_k b_k * (-1)^k = 0
    # => b_0 = -sum_{k != 0} b_k * (-1)^k
    signs = (-1.0 + 0j) ** wavenumbers
    b_no_const = b.at[c0_idx].set(0.0 + 0j)
    b = b.at[c0_idx].set(-jnp.dot(signs, b_no_const))

    # Shrink back to original N if we expanded
    if is_even:
        b = b[:n]

    return b


# ============================================================================
# Definite integral (JIT-safe)
# ============================================================================


def _trig_definite_integral(coeffs: jax.Array) -> jax.Array:
    r"""Definite integral of a trigonometric series over [-1, 1].

    By orthogonality:
    .. math::
        \int_{-1}^{1} f(x) dx = 2 c_0

    where c_0 is the zero-wavenumber Fourier coefficient.

    Parameters
    ----------
    coeffs : jax.Array, shape (N,) complex

    Returns
    -------
    jax.Array scalar (complex128)

    Notes
    -----
    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @trigtech/sum.m
    Chebfun commit: 7574c77
    """
    n = coeffs.shape[0]
    if n == 0:
        return jnp.array(0.0, dtype=jnp.float64)
    # c_0 is at index floor((n+2)/2) - 1 = (n+2)//2 - 1 (0-based)
    # = n//2 for both odd and even N
    c0_idx = n // 2
    return 2.0 * coeffs[c0_idx].astype(jnp.complex128)


# ============================================================================
# Coefficient prolong/truncate
# ============================================================================


def _trig_prolong_coeffs(coeffs: jax.Array, n_out: int) -> jax.Array:
    """Zero-pad or truncate Fourier coefficients to length n_out.

    Padding adds zeros symmetrically at high frequencies.
    Truncation removes high-frequency coefficients symmetrically.

    Parameters
    ----------
    coeffs : jax.Array, shape (n,) complex
        Fourier coefficients in descending wavenumber order.
    n_out : int
        Target number of coefficients.

    Returns
    -------
    jax.Array, shape (n_out,) complex128

    Provenance
    ----------
    MATLAB source : @trigtech/prolong.m
    Chebfun commit: 7574c77
    """
    n = coeffs.shape[0]
    if n_out == n:
        return jnp.asarray(coeffs, dtype=jnp.complex128)

    coeffs_cx = jnp.asarray(coeffs, dtype=jnp.complex128)

    # If n is even, expand to n+1 by splitting the first (lowest) coefficient
    if n % 2 == 0:
        c_low = 0.5 * coeffs_cx[0]
        coeffs_cx = jnp.concatenate([c_low[None], coeffs_cx[1:], c_low[None]])
        n = n + 1

    if n_out == n:
        return coeffs_cx

    if n_out > n:
        k_up = (n_out - n + 1) // 2   # ceil((n_out-n)/2)
        k_down = (n_out - n) // 2      # floor((n_out-n)/2)
        coeffs_cx = jnp.concatenate([
            jnp.zeros(k_up, dtype=jnp.complex128),
            coeffs_cx,
            jnp.zeros(k_down, dtype=jnp.complex128),
        ])
    else:
        # Truncate: remove k_up from top (lowest wavenumbers) and k_down from bottom
        k_up = (n - n_out) // 2       # floor
        k_down = (n - n_out + 1) // 2 # ceil
        if k_down > 0:
            coeffs_cx = coeffs_cx[k_up: n - k_down]
        else:
            coeffs_cx = coeffs_cx[k_up:]
        # If more was removed from bottom than top, scale first coeff
        if k_up < k_down:
            coeffs_cx = coeffs_cx.at[0].set(2.0 * coeffs_cx[0])

    return coeffs_cx


# ============================================================================
# Happiness check helpers
# ============================================================================


def _trig_abs_coeffs_for_chop(coeffs: jax.Array) -> jax.Array:
    """Prepare Fourier coefficient magnitudes for standard_chop.

    Follows the MATLAB @trigtech/simplify.m strategy: pair symmetric modes
    (k and -k) by summing their absolute values, producing a 1D non-negative
    sequence ordered from lowest to highest frequency.

    The result is in the form expected by ``standard_chop`` (monotone envelope
    from low to high frequency, high-to-low decay expected).

    Parameters
    ----------
    coeffs : jax.Array, shape (N,) complex

    Returns
    -------
    jax.Array, 1D non-negative float64 array suitable for standard_chop.
    """
    n = len(coeffs)
    abs_c = jnp.abs(coeffs)
    c0_idx = n // 2  # index of constant mode

    if n % 2 == 1:
        # Odd N: c0_idx = (N-1)/2
        # MATLAB ordering: [pair_M; ...; pair_1; c_0] then flipud -> [c_0; pair_1; ...; pair_M]
        # pair_k = |c_{-k}| + |c_k|
        # In our array: c_{-k} is at index c0_idx - k, c_k is at index c0_idx + k
        neg = abs_c[:c0_idx][::-1]    # |c_{-1}|, |c_{-2}|, ..., |c_{-M}|  (k=1..M)
        pos = abs_c[c0_idx + 1:]      # |c_1|, |c_2|, ..., |c_M|            (k=1..M)
        paired = neg + pos             # pair_k for k=1..M
        c0_val = abs_c[c0_idx:c0_idx + 1]
        # Assemble in MATLAB order (after flipud): [c_0, pair_1, pair_2, ..., pair_M]
        chop_in = jnp.concatenate([c0_val, paired])
    else:
        # Even N: c0_idx = N/2
        # c_{-N/2} is the unpaired highest mode (index 0 in our array)
        # MATLAB: [highest; pair_{N/2-1}; ...; pair_1; c_0] then flipud
        # -> [c_0; pair_1; ...; pair_{N/2-1}; highest]
        highest = abs_c[:1]            # |c_{-N/2}|
        neg = abs_c[1:c0_idx][::-1]   # |c_{-1}|,...,|c_{-(N/2-1)}|  (k=1..N/2-1)
        c0_val = abs_c[c0_idx:c0_idx + 1]
        pos = abs_c[c0_idx + 1:]      # |c_1|,...,|c_{N/2-1}|          (k=1..N/2-1)
        paired = neg + pos
        # Assemble: [c_0, pair_1, ..., pair_{N/2-1}, highest]
        chop_in = jnp.concatenate([c0_val, paired, highest])

    # Expand each entry (except the first = c_0) into a duplicate pair [x, x]
    # This matches MATLAB: [coeffs(1,:) ; kron(coeffs(2:end,:), [1;1])]
    if chop_in.shape[0] > 1:
        tail = jnp.repeat(chop_in[1:], 2)
        chop_final = jnp.concatenate([chop_in[:1], tail])
    else:
        chop_final = chop_in

    return chop_final


def _chop_cutoff_to_ncoeffs(chop_cutoff: int, n_full: int) -> int:
    """Map a standard_chop cutoff (in expanded space) to full coefficient count.

    Parameters
    ----------
    chop_cutoff : int
        Output of standard_chop on the _trig_abs_coeffs_for_chop array.
    n_full : int
        Original number of Fourier coefficients.

    Returns
    -------
    int
        Number of Fourier coefficients to retain (odd preferred).
    """
    if chop_cutoff <= 1:
        return 1
    # Reverse the kron expansion: (cutoff - 1) / 2 pairs after the first element
    paired_idx = (chop_cutoff + 1) // 2  # = ceil(chop_cutoff / 2)
    # paired_idx modes (including constant) -> n_keep = 2*paired_idx - 1 (odd, centered)
    n_keep = max(1, 2 * paired_idx - 1)
    return min(n_keep, n_full)


# ============================================================================
# Root-finding (NOT JIT-safe)
# ============================================================================


def _trig_roots(coeffs: jax.Array) -> jax.Array:
    """Find real roots of a trigonometric series in [-1, 1].

    Converts the trigonometric interpolant to a Chebyshev representation
    by sampling on Chebyshev points, then calls Chebyshev rootfinding.
    This mirrors MATLAB's default @trigtech/roots.m strategy.

    NOT JIT-safe (variable output size).

    Parameters
    ----------
    coeffs : jax.Array, shape (N,) complex

    Returns
    -------
    jax.Array, shape (r,) float64
        Real roots in [-1, 1], sorted.

    Provenance
    ----------
    MATLAB source : @trigtech/roots.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.tech.chebtech import Chebtech2
    from chebfunjax.utils.quadrature import chebpts

    n = coeffs.shape[0]
    if n == 0:
        return jnp.array([], dtype=jnp.float64)

    # Sample on Chebyshev-2 points and call Chebtech2.roots()
    n_sample = max(2 * n + 1, 33)
    x_cheb = chebpts(n_sample, kind=2)
    vals = _trig_eval(coeffs, x_cheb, is_real=jnp.isrealobj(coeffs))
    if jnp.iscomplexobj(vals):
        vals = jnp.real(vals)

    g = Chebtech2.from_values(vals.astype(jnp.float64))
    return g.roots()


# ============================================================================
# Trigtech class
# ============================================================================


class Trigtech(eqx.Module):
    """Trigonometric interpolant for smooth periodic functions on [-1, 1].

    Represents a smooth periodic function via complex Fourier coefficients
    on an equispaced trigonometric grid.

    Attributes
    ----------
    coeffs : jax.Array, shape (N,) complex128
        Fourier coefficients in descending-wavenumber order.
        Constant mode c_0 is at index ``N // 2``.
    is_real : bool
        True if the underlying function is real-valued. Controls whether
        evaluation returns real (float64) or complex (complex128) values.
    ishappy : bool
        True if the representation is resolved to tolerance.

    Notes
    -----
    The function is represented as

    .. math::

        f(x) = \\sum_k c_k \\exp(i \\pi k x), \\quad x \\in [-1, 1]

    Provenance
    ----------
    MATLAB source : @trigtech/trigtech.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebtech2, Bndfun
    """

    coeffs: jax.Array  # complex128, shape (N,)
    is_real: bool = eqx.field(static=True, default=True)
    ishappy: bool = eqx.field(static=True, default=True)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_coeffs(
        cls,
        coeffs: jax.Array,
        *,
        is_real: bool | None = None,
        ishappy: bool = True,
    ) -> "Trigtech":
        """Construct a Trigtech from Fourier coefficients.

        Parameters
        ----------
        coeffs : array_like, shape (N,) real or complex
            Fourier coefficients in descending-wavenumber order.
        is_real : bool or None
            If None, inferred: True if coeffs is real-dtype.
        ishappy : bool, default True

        Returns
        -------
        Trigtech
        """
        coeffs = jnp.atleast_1d(jnp.asarray(coeffs, dtype=jnp.complex128))
        if is_real is None:
            is_real = True  # default: treat as real unless caller says otherwise
        return cls(coeffs=coeffs, is_real=bool(is_real), ishappy=ishappy)

    @classmethod
    def from_values(
        cls,
        values: jax.Array,
        *,
        ishappy: bool = True,
    ) -> "Trigtech":
        """Construct a Trigtech from values at equispaced trigonometric points.

        Parameters
        ----------
        values : array_like, shape (N,) real or complex
            Function values at N equispaced points x_k = -1 + 2k/N.
        ishappy : bool, default True

        Returns
        -------
        Trigtech
        """
        values = jnp.atleast_1d(jnp.asarray(values))
        is_real = jnp.isrealobj(values)
        c = trig_vals2coeffs(values.astype(jnp.complex128))
        return cls(coeffs=c, is_real=bool(is_real), ishappy=ishappy)

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        *,
        n: int | None = None,
        maxpow2: int = 16,
    ) -> "Trigtech":
        """Construct a Trigtech from a callable.

        If ``n`` is given, evaluates the function on an ``n``-point equispaced
        trigonometric grid (non-adaptive). If ``n`` is None, uses an adaptive
        algorithm.

        Parameters
        ----------
        f : callable
            Vectorised function on [-1, 1]. Should be periodic.
        n : int or None
            Fixed number of points, or None for adaptive.
        maxpow2 : int, default 16
            Maximum grid size = 2^maxpow2 for adaptive construction.

        Returns
        -------
        Trigtech

        Notes
        -----
        Adaptive construction is NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @trigtech/trigtech.m, @trigtech/populate.m
        Chebfun commit: 7574c77
        """
        if n is not None:
            return cls._fixed_construct(f, n)
        return cls._adaptive_construct(f, maxpow2)

    @classmethod
    def _fixed_construct(cls, f: Callable, n: int) -> "Trigtech":
        """Fixed-size construction."""
        if n <= 0:
            return cls(coeffs=jnp.array([], dtype=jnp.complex128), is_real=True)
        x = trigpts(n)
        values = jnp.asarray(f(x), dtype=jnp.float64)
        c = trig_vals2coeffs(values.astype(jnp.complex128))
        return cls(coeffs=c, is_real=True, ishappy=True)

    @classmethod
    def _adaptive_construct(
        cls,
        f: Callable,
        maxpow2: int = 16,
        start_pow2: int = 4,
    ) -> "Trigtech":
        """Adaptive construction — Python loop, NOT JIT-safe.

        Evaluates f on grids of 2^k points for k = start_pow2, ..., maxpow2.
        Note: start_pow2=4 gives n=16, producing a chop array of length 17,
        which is the minimum required by standard_chop.
        """
        vscale = 0.0
        c = None
        for k in range(start_pow2, maxpow2 + 1):
            n = 2**k
            x = trigpts(n)
            values = jnp.asarray(f(x), dtype=jnp.float64)
            c = trig_vals2coeffs(values.astype(jnp.complex128))
            vscale = max(vscale, float(jnp.max(jnp.abs(values))))

            # Check happiness using paired coefficient magnitudes
            chop_in = _trig_abs_coeffs_for_chop(c)
            cutoff = standard_chop(chop_in)
            ishappy = cutoff < len(chop_in)

            if ishappy:
                # Map cutoff back to number of Fourier modes
                n_keep = _chop_cutoff_to_ncoeffs(cutoff, n)
                # Ensure odd (symmetric spectrum)
                if n_keep % 2 == 0:
                    n_keep = max(1, n_keep - 1)
                c_keep = _trig_prolong_coeffs(c, n_keep)
                return cls(coeffs=c_keep, is_real=True, ishappy=True)

        # Did not converge
        warnings.warn(
            f"Trigtech.from_function: function did not converge with "
            f"{2**maxpow2} points. Returning unhappy representation.",
            stacklevel=2,
        )
        c_final = trig_vals2coeffs(
            jnp.asarray(f(trigpts(2**maxpow2)), dtype=jnp.float64).astype(jnp.complex128)
        )
        return cls(coeffs=c_final, is_real=True, ishappy=False)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate at point(s) x in [-1, 1].

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)

        Returns
        -------
        y : jax.Array, float64 if is_real else complex128

        Notes
        -----
        JIT-safe: yes. vmap-safe: yes. grad-safe: yes.

        Provenance
        ----------
        MATLAB source : @trigtech/feval.m, @trigtech/horner.m
        Chebfun commit: 7574c77
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        return _trig_eval(self.coeffs, x, is_real=self.is_real)

    # ------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------

    @staticmethod
    def vals2coeffs(values: jax.Array) -> jax.Array:
        """Equispaced values → Fourier coefficients.

        See ``trig_vals2coeffs`` for details.

        Provenance
        ----------
        MATLAB source : @trigtech/vals2coeffs.m
        Chebfun commit: 7574c77
        """
        return trig_vals2coeffs(values)

    @staticmethod
    def coeffs2vals(coeffs: jax.Array) -> jax.Array:
        """Fourier coefficients → equispaced values.

        See ``trig_coeffs2vals`` for details.

        Provenance
        ----------
        MATLAB source : @trigtech/coeffs2vals.m
        Chebfun commit: 7574c77
        """
        return trig_coeffs2vals(coeffs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of Fourier coefficients."""
        return self.coeffs.shape[0]

    @property
    def values(self) -> jax.Array:
        """Function values at equispaced trigonometric points (float64 if real)."""
        v = trig_coeffs2vals(self.coeffs)
        if self.is_real:
            return jnp.real(v).astype(jnp.float64)
        return v

    @property
    def vscale(self) -> float:
        """Vertical scale: max |f(x)| on the grid."""
        return float(jnp.max(jnp.abs(self.values)))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        """Compact display.

        Examples
        --------
        >>> f = Trigtech.from_function(lambda x: jnp.sin(jnp.pi * x))
        >>> repr(f)
        'Trigtech(n=3, is_real=True, vscale=1.000e+00)'
        """
        return f"Trigtech(n={self.n}, is_real={self.is_real}, vscale={self.vscale:.4g})"

    # ------------------------------------------------------------------
    # Prolong / Simplify
    # ------------------------------------------------------------------

    def prolong(self, n: int) -> "Trigtech":
        """Return a new Trigtech with n Fourier coefficients.

        Zero-pads symmetrically if n > self.n; truncates if n < self.n.

        Provenance
        ----------
        MATLAB source : @trigtech/prolong.m
        Chebfun commit: 7574c77
        """
        if n == self.n:
            return self
        new_coeffs = _trig_prolong_coeffs(self.coeffs, n)
        return Trigtech(coeffs=new_coeffs, is_real=self.is_real, ishappy=self.ishappy)

    def simplify(self, tol: float | None = None) -> "Trigtech":
        """Return a new Trigtech with small trailing Fourier coefficients removed.

        Uses ``standard_chop`` on the paired coefficient magnitudes to find
        a suitable cutoff.

        Parameters
        ----------
        tol : float or None
            Tolerance for ``standard_chop``. Default: machine epsilon.

        Returns
        -------
        Trigtech
            Simplified instance.

        Provenance
        ----------
        MATLAB source : @trigtech/simplify.m
        Chebfun commit: 7574c77
        """
        if not self.ishappy:
            return self

        nold = self.n
        N = max(17, round(nold * 1.25 + 5))
        prolonged = self.prolong(N)

        # Round-trip to create slight noise on the plateau
        v = trig_coeffs2vals(prolonged.coeffs)
        c_noisy = trig_vals2coeffs(v)

        chop_in = _trig_abs_coeffs_for_chop(c_noisy)
        cutoff = standard_chop(chop_in, tol)
        cutoff = min(cutoff, len(chop_in))

        n_keep = _chop_cutoff_to_ncoeffs(cutoff, N)
        n_keep = min(n_keep, nold)
        if n_keep % 2 == 0:
            n_keep = max(1, n_keep - 1)

        new_coeffs = _trig_prolong_coeffs(self.coeffs, n_keep)
        return Trigtech(coeffs=new_coeffs, is_real=self.is_real, ishappy=self.ishappy)

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, k: int = 1) -> "Trigtech":
        r"""Return the k-th derivative.

        Multiplies each Fourier coefficient c_j by (i*pi*j)^k.

        Parameters
        ----------
        k : int, default 1
            Differentiation order (static).

        Returns
        -------
        Trigtech
            k-th derivative.

        Notes
        -----
        JIT-safe: yes (k must be static).

        Provenance
        ----------
        MATLAB source : @trigtech/diff.m
        Chebfun commit: 7574c77
        """
        if k == 0:
            return self
        dc = _trig_diff_coeffs(self.coeffs, k)
        # Derivative of a real function is real-valued
        return Trigtech(coeffs=dc, is_real=self.is_real, ishappy=self.ishappy)

    def cumsum(self) -> "Trigtech":
        r"""Return the antiderivative with F(-1) = 0.

        Requires zero mean (c_0 = 0).

        Returns
        -------
        Trigtech
            Antiderivative.

        Raises
        ------
        ValueError
            If the function does not have zero mean.

        Provenance
        ----------
        MATLAB source : @trigtech/cumsum.m
        Chebfun commit: 7574c77
        """
        n = self.n
        c0_idx = n // 2
        c0_mag = float(jnp.abs(self.coeffs[c0_idx]))
        vs = self.vscale if self.vscale > 0 else 1.0
        if c0_mag > 10.0 * vs * _EPS:
            raise ValueError(
                f"Trigtech.cumsum: function does not have zero mean "
                f"(|c_0| = {c0_mag:.3e}). The antiderivative of a non-zero-mean "
                f"periodic function is not periodic."
            )
        bc = _trig_cumsum_coeffs(self.coeffs)
        return Trigtech(coeffs=bc, is_real=self.is_real, ishappy=self.ishappy)

    def sum(self) -> jax.Array:
        r"""Definite integral over [-1, 1].

        Returns 2 * c_0 (real if ``is_real`` is True).

        Returns
        -------
        jax.Array scalar

        Notes
        -----
        JIT-safe: yes.

        Provenance
        ----------
        MATLAB source : @trigtech/sum.m
        Chebfun commit: 7574c77
        """
        s = _trig_definite_integral(self.coeffs)
        if self.is_real:
            return jnp.real(s).astype(jnp.float64)
        return s

    # ------------------------------------------------------------------
    # Roots
    # ------------------------------------------------------------------

    def roots(self) -> jax.Array:
        """Find real roots in [-1, 1].

        Converts to a Chebyshev representation and calls Chebyshev
        rootfinding (NOT JIT-safe).

        Returns
        -------
        jax.Array, shape (r,) float64
            Roots in [-1, 1], sorted.

        Provenance
        ----------
        MATLAB source : @trigtech/roots.m
        Chebfun commit: 7574c77
        """
        return _trig_roots(self.coeffs)

    # ------------------------------------------------------------------
    # Happiness check
    # ------------------------------------------------------------------

    @staticmethod
    def happiness_check(
        coeffs: jax.Array,
        values: jax.Array,
        tol: float | None = None,
        vscale: float = 0.0,
    ) -> tuple[bool, int]:
        """Standard happiness check for trigonometric adaptive construction.

        Parameters
        ----------
        coeffs : jax.Array, shape (N,) complex
        values : jax.Array, shape (N,)
        tol : float or None
        vscale : float, default 0.0

        Returns
        -------
        (ishappy, cutoff) : (bool, int)
        """
        if tol is None:
            tol = _EPS

        coeffs.shape[0]
        vscale_local = float(jnp.max(jnp.abs(values)))
        vscale = max(vscale, vscale_local)

        if vscale_local > 0:
            scaled_tol = tol * max(1.0, vscale / vscale_local)
        else:
            scaled_tol = tol

        chop_in = _trig_abs_coeffs_for_chop(coeffs)
        cutoff = standard_chop(chop_in, scaled_tol)
        ishappy = cutoff < len(chop_in)
        return ishappy, cutoff

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other) -> "Trigtech":
        """Add a Trigtech or scalar.

        Provenance
        ----------
        MATLAB source : @chebtech/plus.m (analogous)
        Chebfun commit: 7574c77
        """
        if isinstance(other, Trigtech):
            nf, ng = self.n, other.n
            n = max(nf, ng)
            fc = _trig_prolong_coeffs(self.coeffs, n)
            gc = _trig_prolong_coeffs(other.coeffs, n)
            new_is_real = self.is_real and other.is_real
            return Trigtech(
                coeffs=fc + gc,
                is_real=new_is_real,
                ishappy=self.ishappy and other.ishappy,
            )
        else:
            # Scalar: add to the constant mode c_0
            c = self.coeffs.copy()
            n = self.n
            c0_idx = n // 2
            c = c.at[c0_idx].add(jnp.complex128(other))
            return Trigtech(coeffs=c, is_real=self.is_real, ishappy=self.ishappy)

    def __radd__(self, other) -> "Trigtech":
        return self.__add__(other)

    def __sub__(self, other) -> "Trigtech":
        """Subtract a Trigtech or scalar.

        Provenance
        ----------
        MATLAB source : @chebtech/minus.m (analogous)
        """
        return self + (-other)

    def __rsub__(self, other) -> "Trigtech":
        return -(self - other)

    def __neg__(self) -> "Trigtech":
        return Trigtech(coeffs=-self.coeffs, is_real=self.is_real, ishappy=self.ishappy)

    def __pos__(self) -> "Trigtech":
        return self

    def __mul__(self, other) -> "Trigtech":
        """Pointwise multiplication via physical-space grid.

        Provenance
        ----------
        MATLAB source : @chebtech/times.m (analogous)
        """
        if isinstance(other, Trigtech):
            # Multiply in physical space to avoid aliasing
            n = self.n + other.n
            if n % 2 == 0:
                n += 1
            x = trigpts(n)
            fv = _trig_eval(self.coeffs, x, self.is_real)
            gv = _trig_eval(other.coeffs, x, other.is_real)
            new_is_real = self.is_real and other.is_real
            pv = fv * gv
            c = trig_vals2coeffs(pv.astype(jnp.complex128))
            return Trigtech(coeffs=c, is_real=new_is_real, ishappy=self.ishappy and other.ishappy)
        else:
            s = jnp.asarray(other, dtype=jnp.complex128)
            return Trigtech(
                coeffs=self.coeffs * s,
                is_real=self.is_real and jnp.isrealobj(jnp.asarray(other)),
                ishappy=self.ishappy,
            )

    def __rmul__(self, other) -> "Trigtech":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Trigtech":
        """Division by scalar or Trigtech.

        Provenance
        ----------
        MATLAB source : @chebtech/rdivide.m (analogous)
        """
        if isinstance(other, Trigtech):
            n = self.n + other.n
            if n % 2 == 0:
                n += 1
            x = trigpts(n)
            fv = _trig_eval(self.coeffs, x, self.is_real)
            gv = _trig_eval(other.coeffs, x, other.is_real)
            new_is_real = self.is_real and other.is_real
            c = trig_vals2coeffs((fv / gv).astype(jnp.complex128))
            return Trigtech(coeffs=c, is_real=new_is_real, ishappy=self.ishappy and other.ishappy)
        else:
            s = jnp.asarray(other, dtype=jnp.complex128)
            return Trigtech(
                coeffs=self.coeffs / s,
                is_real=self.is_real,
                ishappy=self.ishappy,
            )

    def __rtruediv__(self, other) -> "Trigtech":
        """scalar / Trigtech."""
        n = max(self.n, 17)
        if n % 2 == 0:
            n += 1
        x = trigpts(n)
        fv = _trig_eval(self.coeffs, x, self.is_real)
        pv = jnp.asarray(other, dtype=jnp.float64 if self.is_real else jnp.complex128) / fv
        c = trig_vals2coeffs(pv.astype(jnp.complex128))
        return Trigtech(coeffs=c, is_real=self.is_real, ishappy=self.ishappy)

    def __pow__(self, exponent) -> "Trigtech":
        """Raise to a power."""
        if isinstance(exponent, int) and exponent >= 0:
            if exponent == 0:
                c = jnp.zeros(1, dtype=jnp.complex128)
                c = c.at[0].set(1.0 + 0j)
                return Trigtech(coeffs=c, is_real=True, ishappy=True)
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        else:
            n = max(2 * self.n, 17)
            if n % 2 == 0:
                n += 1
            x = trigpts(n)
            fv = _trig_eval(self.coeffs, x, self.is_real)
            pv = fv ** jnp.asarray(exponent, dtype=jnp.float64)
            c = trig_vals2coeffs(pv.astype(jnp.complex128))
            return Trigtech(coeffs=c, is_real=self.is_real, ishappy=self.ishappy)

    def __abs__(self) -> "Trigtech":
        """Absolute value via grid evaluation."""
        n = max(2 * self.n, 17)
        if n % 2 == 0:
            n += 1
        x = trigpts(n)
        fv = jnp.abs(_trig_eval(self.coeffs, x, self.is_real))
        c = trig_vals2coeffs(fv.astype(jnp.complex128))
        return Trigtech(coeffs=c, is_real=True, ishappy=self.ishappy)

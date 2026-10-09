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

# uses-numpy: concrete-array fast path for the trig transforms -- eager JAX
# dispatch (or per-shape XLA compiles) dominates the tens of thousands of
# small transform calls in ballfun/spherefun construction; numpy mirrors
# the exact same algorithm at C speed.  Tracers use the jnp implementation.
import warnings
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.misc import standard_chop

# Machine epsilon for float64.
_EPS = float(jnp.finfo(jnp.float64).eps)


# ============================================================================
# FFT-based transforms (JIT-safe)
# ============================================================================


def _is_double(x) -> bool:
    """True if ``x`` is a real/complex numeric array or Python number
    (MATLAB ``isa(x, 'double')``), i.e. not a bool and not a Trigtech."""
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float, complex)):
        return True
    try:
        arr = jnp.asarray(x)
    except (TypeError, ValueError):
        return False
    return jnp.issubdtype(arr.dtype, jnp.number) and not jnp.issubdtype(
        arr.dtype, jnp.bool_)


def _scale_real(c: jax.Array, r: jax.Array) -> jax.Array:
    """Scale a complex array ``c`` by a real array ``r`` component-wise.

    ``c * r`` promotes ``r`` to complex and does a full complex multiply,
    which injects ``inf * 0 = nan`` when ``c`` has infinite parts.  Scaling
    the real and imaginary components separately preserves Inf/NaN, so
    ``isinf``/``isnan`` on FFT-built coefficients stay meaningful.
    """
    return jnp.real(c) * r + 1j * (jnp.imag(c) * r)


def trig_vals2coeffs(values: jax.Array) -> jax.Array:
    """Dispatching wrapper -- see _trig_vals2coeffs_impl.

    The symmetry-preserving transform is ~14 array ops; ballfun/
    spherefun constructions call it tens of thousands of times on small
    CONCRETE arrays, where per-op JAX dispatch dominates (measured
    1.4 ms/call eager; a jax.jit variant instead paid one XLA compile
    per distinct shape -- 416 compiles in one Helmholtz solve).
    Concrete inputs therefore run a numpy mirror of the same algorithm
    (C-speed, no dispatch, same pocketfft, bit-identical); tracers keep
    the traceable jnp path.
    """
    if isinstance(values, jax.core.Tracer):
        if values.shape[0] <= 1:
            return values.astype(jnp.complex128)
        return _trig_vals2coeffs_impl(values)
    v = np.asarray(values)
    if v.shape[0] <= 1:
        return jnp.asarray(v, dtype=jnp.complex128)
    if not np.all(np.isfinite(v)):
        # Non-finite data: keep the jnp path, whose Inf/NaN propagation
        # the isinf/isnan ports pin (rfft vs fft differ on it).
        return _trig_vals2coeffs_impl(jnp.asarray(values))
    return jnp.asarray(_trig_vals2coeffs_np(v))


def _trig_vals2coeffs_np(values):
    """numpy mirror of _trig_vals2coeffs_impl (kept in lockstep)."""
    n = values.shape[0]
    input_real = not np.iscomplexobj(values)
    vals = values.astype(np.complex128)
    v2 = vals if vals.ndim == 2 else vals[:, None]
    aug = np.concatenate([v2, v2[:1]], axis=0)
    aug_flip = np.conj(aug[::-1])
    is_herm = np.max(np.abs(aug - aug_flip), axis=0) == 0
    is_skew = np.max(np.abs(aug + aug_flip), axis=0) == 0
    if input_real:
        Xr = np.fft.rfft(np.real(vals), axis=0)
        mirror = np.conj(Xr[1:(n + 1) // 2][::-1])
        X = np.concatenate([Xr, mirror], axis=0)
        coeffs = np.fft.fftshift(X, axes=0) / n
    else:
        coeffs = np.fft.fftshift(np.fft.fft(vals, axis=0), axes=0) / n
    c2 = coeffs if coeffs.ndim == 2 else coeffs[:, None]
    c2[:, is_herm] = np.real(c2[:, is_herm])
    c2[:, is_skew] = 1j * np.imag(c2[:, is_skew])
    coeffs = c2 if coeffs.ndim == 2 else c2[:, 0]
    if n % 2 == 1:
        ks = np.arange(-(n - 1) // 2, (n - 1) // 2 + 1)
    else:
        ks = np.arange(-(n // 2), n // 2)
    fix = np.where(ks % 2 == 0, 1.0, -1.0).reshape(
        (n,) + (1,) * (values.ndim - 1))
    return np.real(coeffs) * fix + 1j * (np.imag(coeffs) * fix)


def _trig_vals2coeffs_impl(values: jax.Array) -> jax.Array:
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
    input_real = not jnp.iscomplexobj(jnp.asarray(values))
    values = jnp.asarray(values, dtype=jnp.complex128)
    n = values.shape[0]

    if n <= 1:
        return values

    # Test the value symmetries the FFT does not preserve bit-exactly
    # (MATLAB @trigtech/vals2coeffs.m): Hermitian values -> exactly real
    # coeffs, skew-Hermitian values -> exactly imaginary coeffs.
    v2 = values if values.ndim == 2 else values[:, None]
    aug = jnp.concatenate([v2, v2[:1]], axis=0)
    aug_flip = jnp.conj(jnp.flip(aug, axis=0))
    is_herm = jnp.max(jnp.abs(aug - aug_flip), axis=0) == 0
    is_skew = jnp.max(jnp.abs(aug + aug_flip), axis=0) == 0

    # coeffs = (1/n) * fftshift(fft(values))
    # (axis=0 keeps array-valued (n, m) inputs column-wise correct)
    #
    # For REAL input the spectrum is built from rfft with an explicit
    # conjugate mirror, X[n-k] := conj(X[k]).  MATLAB inherits this
    # bit-exact conjugate symmetry from FFTW's real-input transform;
    # numpy/JAX's complex FFT does not guarantee it, and the MATLAB
    # test suite pins EXACT (== 0) even/odd coefficient symmetry for
    # even/odd real inputs, which follows from it.
    if input_real:
        Xr = jnp.fft.rfft(jnp.real(values), axis=0)  # (n//2 + 1, ...)
        mirror = jnp.conj(jnp.flip(Xr[1:(n + 1) // 2], axis=0))
        X = jnp.concatenate([Xr, mirror], axis=0)
        coeffs = jnp.fft.fftshift(X, axes=0) / n
    else:
        coeffs = jnp.fft.fftshift(jnp.fft.fft(values, axis=0), axes=0) / n

    c2 = coeffs if coeffs.ndim == 2 else coeffs[:, None]
    c2 = jnp.where(is_herm[None, :], jnp.real(c2).astype(jnp.complex128), c2)
    c2 = jnp.where(is_skew[None, :],
                   1j * jnp.imag(c2).astype(jnp.complex128), c2)
    coeffs = c2 if coeffs.ndim == 2 else c2[:, 0]

    # The FFT is for [0, 2) but we want [-1, 1).
    # Fix: multiply c_k by (-1)^k.
    if n % 2 == 1:
        half = (n - 1) // 2
        ks = jnp.arange(-half, half + 1, dtype=jnp.float64)
    else:
        half = n // 2
        ks = jnp.arange(-half, half, dtype=jnp.float64)

    # Exactly real (+/-1): a complex power leaves ~1e-16 imaginary noise
    # that would break the bit-exact symmetry above.
    even_odd_fix = jnp.where(
        (ks.astype(jnp.int64) % 2) == 0, 1.0, -1.0)
    even_odd_fix = even_odd_fix.reshape(
        (n,) + (1,) * (values.ndim - 1))
    return _scale_real(coeffs, even_odd_fix)


def _trig_coeffs2vals_np(coeffs):
    """numpy mirror of _trig_coeffs2vals_impl (kept in lockstep)."""
    c0 = coeffs.astype(np.complex128)
    n = c0.shape[0]
    if n % 2 == 1:
        ks = np.arange(-((n - 1) // 2), (n - 1) // 2 + 1)
    else:
        ks = np.arange(-(n // 2), n // 2)
    fix = np.where(ks % 2 == 0, 1.0, -1.0).reshape(
        (n,) + (1,) * (c0.ndim - 1))
    c = np.real(c0) * fix + 1j * (np.imag(c0) * fix)
    values = np.fft.ifft(np.fft.ifftshift(n * c, axes=0), axis=0)
    c2 = c if c.ndim == 2 else c[:, None]
    v2 = values if values.ndim == 2 else values[:, None]
    is_herm = np.max(np.abs(np.imag(c2)), axis=0) == 0
    is_skew = np.max(np.abs(np.real(c2)), axis=0) == 0
    aug = np.concatenate([v2, v2[:1]], axis=0)
    flipped = np.conj(aug[::-1])
    herm = ((aug + flipped) / 2)[:-1]
    skew = ((aug - flipped) / 2)[:-1]
    v2[:, is_herm] = herm[:, is_herm]
    v2[:, is_skew] = skew[:, is_skew]
    return v2 if values.ndim == 2 else v2[:, 0]


def trig_coeffs2vals(coeffs: jax.Array) -> jax.Array:
    """Dispatching wrapper -- see _trig_coeffs2vals_impl (same concrete/
    tracer rationale as trig_vals2coeffs)."""
    if isinstance(coeffs, jax.core.Tracer):
        if coeffs.shape[0] <= 1:
            return coeffs.astype(jnp.complex128)
        return _trig_coeffs2vals_impl(coeffs)
    c = np.asarray(coeffs)
    if c.shape[0] <= 1:
        return jnp.asarray(c, dtype=jnp.complex128)
    if not np.all(np.isfinite(c)):
        return _trig_coeffs2vals_impl(jnp.asarray(coeffs))
    return jnp.asarray(_trig_coeffs2vals_np(c))


def _trig_coeffs2vals_impl(coeffs: jax.Array) -> jax.Array:
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
    # (axis=0 keeps array-valued (n, m) inputs column-wise correct)
    # Exactly real (+/-1) so the symmetry tests above stay bit-exact
    # Exactly real (+/-1): a complex power leaves ~1e-16 imaginary noise.
    even_odd_fix = jnp.where(
        (ks.astype(jnp.int64) % 2) == 0, 1.0, -1.0)
    even_odd_fix = even_odd_fix.reshape(
        (n,) + (1,) * (coeffs.ndim - 1))
    c = _scale_real(coeffs, even_odd_fix)

    values = jnp.fft.ifft(jnp.fft.ifftshift(n * c, axes=0), axis=0)

    # Enforce the symmetries that the FFT does not preserve bit-exactly
    # (MATLAB @trigtech/coeffs2vals.m): real coeffs -> Hermitian values,
    # imaginary coeffs -> skew-Hermitian values.  Done per column.
    c2 = c if c.ndim == 2 else c[:, None]
    v2 = values if values.ndim == 2 else values[:, None]
    is_herm = jnp.max(jnp.abs(jnp.imag(c2)), axis=0) == 0
    is_skew = jnp.max(jnp.abs(jnp.real(c2)), axis=0) == 0
    aug = jnp.concatenate([v2, v2[:1]], axis=0)
    flipped = jnp.flip(jnp.conj(aug), axis=0)
    herm = ((aug + flipped) / 2)[:-1]
    skew = ((aug - flipped) / 2)[:-1]
    v2 = jnp.where(is_herm[None, :], herm, v2)
    v2 = jnp.where(is_skew[None, :], skew, v2)
    return v2 if values.ndim == 2 else v2[:, 0]


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

    Provenance
    ----------
    MATLAB source : @trigtech/trigpts.m; R2025b linspace.m symmetric branch
    Chebfun commit: 7574c77
    The static technology API uses normalized linspace points directly.
    Adaptive refinement instead calls the global PI-based trigpts API.
    """
    from chebfunjax.utils._trigpts import static_trigpts_nodes
    return static_trigpts_nodes(n)


# ============================================================================
# Evaluation (JIT-safe, grad-safe, vmap-safe)
# ============================================================================


def _sample_as_trig_dtype(f, x):
    """Sample f preserving complexness (mirrors Chebtech's dtype handling).

    Returns (values_complex128, is_real): the constructor previously cast
    every sample to float64 unconditionally, silently discarding the
    imaginary part of complex-valued functions.
    """
    raw = jnp.asarray(f(x))
    is_real = not jnp.iscomplexobj(raw)
    if is_real:
        raw = raw.astype(jnp.float64)
    return raw.astype(jnp.complex128), is_real


def _sample_callable_trig_grid(f, n, *, source_global=False):
    """Apply source callable-grid endpoint averaging, preserving columns.

    Provenance
    ----------
    MATLAB source : @trigtech/refine.m, refineResampling
    Chebfun commit: 7574c77

    Fixed-n uses static normalized technology points; adaptive refinement
    opts into global PI-based source points. Both average endpoints once.
    User-supplied values and off-grid probes do not
    pass through this helper.
    """
    from chebfunjax.utils._trigpts import global_trigpts_nodes
    nodes = global_trigpts_nodes(n) if source_global else trigpts(n)
    points = jnp.concatenate((nodes, jnp.ones((1,), dtype=jnp.float64)))
    values, is_real = _sample_as_trig_dtype(f, points)
    values = values.at[0].set(0.5 * (values[0] + values[-1]))
    return values[:-1], is_real


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
    x = jnp.asarray(x)
    x = x.astype(jnp.complex128 if jnp.iscomplexobj(x) else jnp.float64)
    scalar_input = x.ndim == 0
    x_1d = jnp.atleast_1d(x)

    n = coeffs.shape[0]
    coeffs_cx = jnp.asarray(coeffs, dtype=jnp.complex128)

    # Array-valued (n, m) coefficients evaluate column-wise: the output
    # gains a trailing column axis (matching chebtech's _clenshaw).
    out_shape = x_1d.shape + coeffs.shape[1:]

    if n == 0:
        dt = jnp.float64 if is_real else jnp.complex128
        result = jnp.zeros(out_shape, dtype=dt)
        return result[0] if scalar_input else result

    if n == 1:
        c0 = coeffs_cx[0]
        if is_real:
            val = jnp.real(c0).astype(jnp.float64)
        else:
            val = c0.astype(jnp.complex128)
        result = jnp.broadcast_to(val, out_shape)
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
    # Array-valued: x gains a trailing singleton axis so the (p, 1)
    # point axis broadcasts against per-column amplitudes a[k] of
    # shape (m,).
    xE = x.reshape(x.shape + (1,) * (coeffs_cx.ndim - 1))
    out_shape = x.shape + coeffs_cx.shape[1:]
    u = jnp.cos(jnp.pi * xE)
    v = jnp.sin(jnp.pi * xE)

    if n_h == 1:
        return jnp.broadcast_to(a[0], out_shape)

    # Horner recurrence: start from the highest-frequency pair and work down
    # Initialize with the highest-k term (index n_h-1)
    # Source real-coefficient Horner still returns complex values when x
    # is complex. Initialize loop carries in that dtype before multiplication.
    carry_dtype = jnp.result_type(a.dtype, x.dtype)
    co = jnp.broadcast_to(a[n_h - 1].astype(carry_dtype), out_shape)
    si = jnp.broadcast_to(b[n_h - 1].astype(carry_dtype), out_shape)

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
    # Array-valued: trailing singleton point axis broadcasts against
    # per-column coefficients.
    xE = x.reshape(x.shape + (1,) * (coeffs_cx.ndim - 1))
    out_shape = x.shape + coeffs_cx.shape[1:]
    z = jnp.exp(1j * jnp.pi * xE)

    # Horner from highest wavenumber (index N-1) down
    q = jnp.broadcast_to(coeffs_cx[n - 1].astype(jnp.complex128),
                         out_shape)

    def body(i, q_):
        j = n - 2 - i  # goes from n-2 down to 1
        return coeffs_cx[j] + z * q_

    q = jax.lax.fori_loop(0, n - 2, body, q)

    # Apply lowest-mode prefactor
    if n % 2 == 1:
        # Odd N: q = exp(-i*pi*(N-1)/2 * x) * (c[0] + z*q)
        prefactor = jnp.exp(-1j * jnp.pi * ((n - 1) / 2) * xE)
        return prefactor * (coeffs_cx[0] + z * q)
    else:
        # Even N: q = exp(-i*pi*(N/2-1)*x)*q + cos(N*pi*x/2)*c[0]
        prefactor = jnp.exp(-1j * jnp.pi * (n / 2 - 1) * xE)
        return prefactor * q + jnp.cos(n / 2 * jnp.pi * xE) * coeffs_cx[0]


def _trig_eval_np(coeffs, x, is_real: bool = True):
    """numpy mirror of :func:`_trig_eval` (kept in lockstep).

    Same Horner schemes as ``_trig_eval_real``/``_trig_eval_complex``,
    C-speed and free of JAX tracing.  Used for concrete inputs: every
    distinctly-shaped Trigtech otherwise compiles its own XLA program,
    and long chains of constructed objects (e.g. rank-100 spherefun
    vorticity) exhaust the LLVM JIT code arena ("Unable to allocate
    section memory").
    """
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    x1 = np.atleast_1d(x)
    c = np.asarray(coeffs, dtype=np.complex128)
    n = c.shape[0]
    out_shape = x1.shape + c.shape[1:]
    if n == 0:
        res = np.zeros(out_shape,
                       dtype=np.float64 if is_real else np.complex128)
        return res[0] if scalar_input else res
    if n == 1:
        val = np.real(c[0]) if is_real else c[0]
        res = np.broadcast_to(val, out_shape)
        return res[0] if scalar_input else res

    xE = x1.reshape(x1.shape + (1,) * (c.ndim - 1))
    if is_real:
        c0_idx = n // 2
        c_slice = c[c0_idx::-1]
        a = np.real(c_slice).copy()
        b = np.imag(c_slice).copy()
        if n % 2 == 0:
            a[-1] = a[-1] / 2.0
            b[-1] = 0.0
        n_h = a.shape[0]
        u = np.cos(np.pi * xE)
        v = np.sin(np.pi * xE)
        if n_h == 1:
            res = np.broadcast_to(a[0], out_shape)
            return res[0] if scalar_input else res
        co = np.broadcast_to(a[n_h - 1], out_shape).astype(np.float64)
        si = np.broadcast_to(b[n_h - 1], out_shape).astype(np.float64)
        for k in range(n_h - 2, 0, -1):
            temp = a[k] + u * co + v * si
            si = b[k] + u * si - v * co
            co = temp
        res = a[0] + 2.0 * (u * co + v * si)
    else:
        z = np.exp(1j * np.pi * xE)
        q = np.broadcast_to(c[n - 1], out_shape).astype(np.complex128)
        for j in range(n - 2, 0, -1):
            q = c[j] + z * q
        if n % 2 == 1:
            pref = np.exp(-1j * np.pi * ((n - 1) / 2) * xE)
            res = pref * (c[0] + z * q)
        else:
            pref = np.exp(-1j * np.pi * (n / 2 - 1) * xE)
            res = pref * q + np.cos(n / 2 * np.pi * xE) * c[0]
    return res[0] if scalar_input else res


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
    factor = factor.reshape((n,) + (1,) * (coeffs_cx.ndim - 1))
    return coeffs_cx * factor


# ============================================================================
# Spectral antiderivative (JIT-safe)
# ============================================================================


def _trig_cumsum_coeffs(coeffs: jax.Array, m: int = 1) -> jax.Array:
    r"""Antiderivative of a trigonometric series (F(-1) = 0).

    Given c_k, returns b_k = c_k * (-i/(pi*k))**m for k != 0.
    The zero mode is determined by the source endpoint condition F(-1) = 0.

    The function must have zero mean (c_0 = 0) for the antiderivative to be
    periodic.

    Parameters
    ----------
    coeffs : jax.Array, shape (N,) complex
        Fourier coefficients in increasing wavenumber order, from negative
        modes through the constant mode to positive modes.
    m : int, default 1
        Static order of integration, as required by the JAX trace.

    Returns
    -------
    jax.Array, shape (N,) complex128

    Notes
    -----
    JIT-safe for a static Python integer ``m``.

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

    # Source integration factor: (-i/(pi*k))**m for k != 0.
    # (trailing singleton axes broadcast over array-valued columns)
    safe_wn = jnp.where(wavenumbers == 0, 1.0, wavenumbers)
    int_factor = jnp.where(
        wavenumbers == 0,
        0.0 + 0j,
        (-1j / safe_wn / jnp.pi) ** m,
    )
    int_factor = int_factor.reshape(
        (n_exp,) + (1,) * (c_work.ndim - 1))
    b = c_work * int_factor

    # MATLAB zeros the expanded Nyquist endpoints only for odd-order
    # integration, where the corresponding sine mode vanishes on the grid.
    if is_even and m % 2 == 1:
        b = b.at[0].set(0.0 + 0j)
        b = b.at[-1].set(0.0 + 0j)

    # Determine b_0 from F(-1) = 0:
    # F(-1) = sum_k b_k * exp(-i*pi*k) = sum_k b_k * (-1)^k = 0
    # => b_0 = -sum_{k != 0} b_k * (-1)^k
    integer_wavenumbers = wavenumbers.astype(jnp.int64)
    signs = jnp.where(integer_wavenumbers % 2 == 0, 1.0, -1.0)
    b_no_const = b.at[c0_idx].set(0.0 + 0j)
    b = b.at[c0_idx].set(-jnp.tensordot(signs, b_no_const, axes=(0, 0)))

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
    if n_out < 0 or coeffs.shape[0] == 0:
        # Preserve inherited negative/empty adapters. For nonempty input,
        # native prolong.m58-67 truncation to zero reaches coeffs(1,:)
        # after deleting every row and raises an indexing error.
        return jnp.zeros((max(n_out, 0),) + coeffs.shape[1:],
                         dtype=coeffs.dtype)
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
        cols = coeffs_cx.shape[1:]
        coeffs_cx = jnp.concatenate([
            jnp.zeros((k_up,) + cols, dtype=jnp.complex128),
            coeffs_cx,
            jnp.zeros((k_down,) + cols, dtype=jnp.complex128),
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


def _alias_trigtech(coeffs: jax.Array, m: int) -> jax.Array:
    """Alias Fourier columns with JAX arithmetic and source-ordered folding.

    m is a static positive integer. Even input/output lengths retain the
    source split/collapse of the Nyquist cosine coefficient.

    Provenance
    ----------
    MATLAB source: @trigtech/alias.m
    Chebfun commit: 7574c77
    """
    if not isinstance(m, int) or m < 1:
        raise ValueError("trigtech alias requires a positive static integer length")
    original = jnp.asarray(coeffs)
    twod = original.ndim == 2
    c = original.astype(jnp.complex128)
    if not twod:
        c = c.reshape(-1, 1)
    n, columns = c.shape
    if m == n:
        return original
    if m > n:
        k = (m - n + 1) // 2
        zeros = jnp.zeros((k, columns), dtype=c.dtype)
        if n % 2 == 0:
            c = jnp.concatenate((c[:1] / 2, c[1:], c[:1] / 2))
            c = jnp.concatenate((zeros, c, zeros[:-1]))
            if m % 2:
                c = c[1:]
        else:
            c = jnp.concatenate((zeros, c, zeros))
            if m % 2 == 0:
                c = c[:-1]
    else:
        if n % 2 == 0:
            c = c.at[0].multiply(0.5)
            c = jnp.concatenate((c, c[:1]))
            n += 1
        n2 = (n - 1) // 2
        if m == 1:
            signs = jnp.where(jnp.arange(n2) % 2 == 0, -1., 1.)
            c = (c[n2] + (signs @ c[n2-1::-1] + signs @ c[n2+1:])).reshape(1, columns)
        elif m % 2:
            m2 = (m - 1) // 2
            initial = c[n2-m2:n2+m2+1]

            def fold(j, result):
                k = jnp.mod(j + m2 + 1, -m) + m2
                sign = jnp.where(jnp.mod(j + k, 2) == 0, 1., -1.)
                result = result.at[k+m2].add(sign*c[j+n2])
                return result.at[-k+m2].add(sign*c[-j+n2])

            c = jax.lax.fori_loop(-n2, -m2, fold, initial)
        else:
            m2 = m // 2
            initial = c[n2-m2:n2+m2]
            initial = jnp.concatenate((initial, -initial[:1]))

            def fold(j, result):
                k = jnp.mod(j + m2, -m) + m2
                result = result.at[k+m2].add(c[j+n2])
                return result.at[-k+m2].add(c[-j+n2])

            c = jax.lax.fori_loop(-n2, -m2+1, fold, initial)
            c = c.at[0].add(c[-1])[:-1]
    return c if twod else c.reshape(-1)


def _trigcoeffs_trigtech(coeffs: jax.Array, N: int) -> jax.Array:
    """Return exactly ``N`` trigonometric coefficients of a trigtech.

    Direct port of ``@trigtech/trigcoeffs.m``: pads symmetrically when ``N``
    exceeds the stored length and, when truncating to an even ``N``, folds
    the highest retained mode back onto the ``cos(N/2)`` coefficient (rather
    than the plain ``prolong`` scaling).  Not JIT-safe (Python-int
    branching).
    """
    import numpy as np

    if N is None or N <= 0:
        return jnp.array([], dtype=jnp.complex128)

    orig = jnp.asarray(coeffs)
    twod = orig.ndim == 2
    c = np.asarray(orig).astype(np.complex128)
    if not twod:
        c = c.reshape(-1, 1)
    num = c.shape[0]
    cols = c.shape[1]

    if num < N:
        k = int(np.ceil((N - num) / 2))
        z = np.zeros((k, cols), dtype=c.dtype)
        c = np.concatenate([z, c, z], axis=0)
        num = c.shape[0]

    f_is_even = num % 2 == 0
    const_index = num // 2 if f_is_even else (num - 1) // 2  # 0-based

    if N % 2 == 0:
        start = const_index - N // 2
        end = const_index + (N // 2 - 1)
        out = c[start:end + 1].copy()
        if end < num - 1:
            out[0] = out[0] + c[end + 1]
    else:
        start = const_index - (N - 1) // 2
        end = const_index + (N - 1) // 2
        out = c[start:end + 1].copy()

    out = jnp.asarray(out, dtype=jnp.complex128)
    return out if twod else out.reshape(-1)


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
    # standardChop returns a count, not a zero-based array index.
    # Source standardCheck rounds that count UP to odd before prolongation.
    return min(2 * (chop_cutoff // 2) + 1, n_full)


def _trig_cutoff_decision(raw_cutoff: int, sample_count: int) -> tuple[bool, int]:
    """Return source happiness and retained count for one expanded cutoff.

    Provenance
    ----------
    MATLAB source : @trigtech/standardCheck.m
    Chebfun commit: 7574c77
    Compare against original sample count, not expanded envelope length.
    """
    return raw_cutoff < sample_count, 2 * (raw_cutoff // 2) + 1


def _trig_standard_check(coeffs, values, tol, vscale):
    """Literal per-column source standardCheck on raw Fourier coefficients.

    Provenance
    ----------
    MATLAB source : @trigtech/standardCheck.m
    Chebfun commit: 7574c77
    Distinct from simplify's absolute-value FFT round-trip and signed pairing.
    """
    columns = coeffs[:, None] if coeffs.ndim == 1 else coeffs
    samples = values[:, None] if values.ndim == 1 else values
    n, m = columns.shape
    # For one row MATLAB any() reduces along columns (the first
    # nonsingleton dimension). For multirow data the if condition requires
    # every column's any-result to be true.
    nan_mask = jnp.isnan(columns)
    source_nan = (jnp.any(nan_mask) if n == 1
                  else jnp.all(jnp.any(nan_mask, axis=0)))
    if bool(source_nan):
        raise ValueError("Trigtech standardCheck: function returned NaN")
    tolerance_input = jnp.asarray(tol)
    # MATLAB tests size(tol,2), not numel(tol). Python vectors represent
    # MATLAB rows; preserve explicit matrix shapes and later linear indexing.
    tolerances = (tolerance_input.reshape(1, -1) if tolerance_input.ndim < 2
                  else tolerance_input)
    if tolerances.shape[1] != m:
        maximum = (jnp.max(tolerances).reshape(1, 1) if tolerances.shape[0] == 1
                   else jnp.max(tolerances, axis=0, keepdims=True))
        # Source ones(1,m)*max(tol) is scalar scaling or matrix multiplication;
        # incompatible shapes must not be repaired by flattening/global max.
        tolerances = (jnp.ones((1, m))*maximum.reshape(()) if maximum.size == 1
                      else jnp.ones((1, m)) @ maximum)
    local = jnp.max(jnp.abs(samples), axis=0)
    scales = jnp.maximum(jnp.asarray(vscale), local)
    scaled = tolerances * scales / local
    linear_scaled = jnp.reshape(scaled.T, (-1,))
    happy = True
    retained = 1
    for column in range(m):
        paired = _trig_abs_coeffs_for_chop(columns[:, column])
        raw = standard_chop(paired, float(linear_scaled[column]))
        happy, keep = _trig_cutoff_decision(raw, n)
        retained = max(retained, keep)
        if not happy:
            break
    return happy, retained


def _trig_source_pairs_for_chop(coeffs: jax.Array) -> jax.Array:
    """Pair raw round-tripped coefficients as ``@trigtech/simplify.m`` does.

    In contrast to the adaptive/multiplication helper above, source simplify
    takes absolute values before the FFT round-trip, then sums opposite modes
    without taking their individual magnitudes. ``standard_chop`` applies the
    magnitude to each resulting pair.
    """
    n = len(coeffs)
    if n % 2:
        center = n // 2
        pairs = coeffs[:center][::-1] + coeffs[center + 1:]
        sequence = jnp.concatenate((coeffs[center:center + 1], pairs))
    else:
        half = n // 2
        # After MATLAB's initial coefficient reversal, even-length storage
        # has its center at half-1 and its unpaired mode at the final index.
        # These are the literal @trigtech/simplify.m slices before flipud.
        source_rows = jnp.concatenate(
            (
                coeffs[-1:],
                coeffs[half:n - 1][::-1] + coeffs[:half - 1],
                coeffs[half - 1:half],
            )
        )
        sequence = source_rows[::-1]
    if sequence.shape[0] > 1:
        sequence = jnp.concatenate(
            (sequence[:1], jnp.repeat(sequence[1:], 2))
        )
    return sequence


def _trig_simplify_cutoff(
    coeffs: jax.Array, tol: float | jax.Array | None,
) -> tuple[int, int]:
    """Source pipeline for ``@trigtech/simplify.m`` column cutoffs.

    MATLAB reverses and takes absolute values of the prolonged coefficients
    before its noisy FFT round-trip. This is distinct from the raw-coefficient
    round-trip used by other callers of ``_trig_chop_cutoff``.
    """
    source_coeffs = jnp.abs(coeffs[::-1, ...])
    noisy_coeffs = trig_vals2coeffs(trig_coeffs2vals(source_coeffs))
    ncols = 1 if noisy_coeffs.ndim == 1 else noisy_coeffs.shape[1]

    if tol is None:
        tolerances = [None] * ncols
    else:
        tol_array = jnp.asarray(tol)
        tol_values = jnp.ravel(tol_array)
        if (tol_values.size != ncols
                or (tol_array.ndim > 1 and tol_array.shape[-1] != ncols)):
            tol_values = jnp.full((ncols,), jnp.max(tol_values))
        tolerances = [float(tol_values[j]) for j in range(ncols)]

    if noisy_coeffs.ndim == 1:
        chop_input = _trig_source_pairs_for_chop(noisy_coeffs)
        cutoff = standard_chop(chop_input, tolerances[0])
        return cutoff, len(chop_input)

    cutoff = 1
    chop_length = 0
    for column, column_tol in enumerate(tolerances):
        chop_input = _trig_source_pairs_for_chop(noisy_coeffs[:, column])
        cutoff = max(cutoff, standard_chop(chop_input, column_tol))
        chop_length = len(chop_input)
    return cutoff, chop_length


def _trig_chop_cutoff(coeffs: jax.Array,
                      tol: float | None = None) -> tuple[int, int]:
    """standard_chop on paired trig magnitudes; the cutoff is the max
    across columns for array-valued coeffs (MATLAB @trigtech/simplify.m
    loops the columns and keeps the largest).  Returns (cutoff, chop
    array length)."""
    if coeffs.ndim == 1:
        chop_in = _trig_abs_coeffs_for_chop(coeffs)
        return standard_chop(chop_in, tol), len(chop_in)
    cutoff = 1
    length = 0
    for j in range(coeffs.shape[1]):
        chop_in = _trig_abs_coeffs_for_chop(coeffs[:, j])
        cutoff = max(cutoff, standard_chop(chop_in, tol))
        length = len(chop_in)
    return cutoff, length


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
    import numpy as np

    from chebfunjax.tech.chebtech import Chebtech2
    from chebfunjax.utils.quadrature import chebpts

    n = coeffs.shape[0]
    if n == 0:
        return jnp.array([], dtype=jnp.float64)

    # Sample on Chebyshev-2 points and call Chebtech2.roots()
    n_sample = max(2 * n + 1, 33)
    x_cheb = chebpts(n_sample, kind=2)
    vals = _trig_eval(coeffs, x_cheb, is_real=False)
    vals = jnp.real(vals)

    g = Chebtech2.from_values(vals.astype(jnp.float64))
    r = np.asarray(g.roots())
    if r.size == 0:
        return jnp.asarray(r, dtype=jnp.float64)
    # Polish with Newton on Re(f)(x) = 0 using the exact trig derivative:
    # the Chebyshev resampling of a high-frequency series can leave the
    # roots ~1e-10 off, but each root is simple, so Newton recovers
    # machine precision.
    d1 = _trig_diff_coeffs(coeffs, 1)
    xr = jnp.asarray(r, dtype=jnp.float64)
    for _ in range(2):
        fv = jnp.real(_trig_eval(coeffs, xr, is_real=False))
        fp = jnp.real(_trig_eval(d1, xr, is_real=False))
        step = jnp.where(jnp.abs(fp) > 1e-30, fv / fp, 0.0)
        xr = xr - step
    rp = np.asarray(xr)
    # Discard any polished root that left [-1, 1] (spurious) and re-sort.
    rp = rp[(rp >= -1.0 - 1e-12) & (rp <= 1.0 + 1e-12)]
    return jnp.asarray(np.sort(rp), dtype=jnp.float64)


def _trig_roots_complex(coeffs: jax.Array, prune: bool = True) -> jax.Array:
    """Roots of a trigonometric series via the companion-matrix (MATLAB
    built-in ``roots``) applied to the flipped coefficients, mapping the
    variable ``z = exp(i pi x)`` back through ``x = -i/pi log(z)``.

    When ``prune`` is True, keep only the roots inside the estimated strip
    of analyticity, matching the ``'complex'`` flag of @trigtech/roots.m.

    NOT JIT-safe.

    Provenance
    ----------
    MATLAB source : @trigtech/roots.m (useMatlabsRootsCommand branch)
    Chebfun commit: 7574c77
    """
    import numpy as np

    c = np.asarray(coeffs, dtype=np.complex128).ravel()
    # Simplify: strip leading/trailing negligible modes symmetrically is
    # handled by the caller via simplify(); here just drop the padding.
    if c.size == 0:
        return jnp.array([], dtype=jnp.complex128)
    # Flip coeffs to match MATLAB's roots (descending powers of z).
    r = np.roots(c[::-1])
    r = -1j / np.pi * np.log(r)
    # Polish with complex Newton on f(x) = sum_k c_k e^{i pi k x} = 0
    # (companion-matrix roots of a long series can carry ~1e-13 error).
    n = c.size
    if n % 2 == 1:
        ks = np.arange(-(n - 1) // 2, (n - 1) // 2 + 1)
    else:
        ks = np.arange(-n // 2, n // 2)
    ck = c
    dk = (1j * np.pi * ks) * c
    for _ in range(2):
        E = np.exp(1j * np.pi * np.outer(r, ks))
        fv = E @ ck
        fp = E @ dk
        with np.errstate(invalid="ignore", divide="ignore"):
            step = np.where(np.abs(fp) > 1e-300, fv / fp, 0.0)
        r = r - step
    # f is 2-periodic in x (e^{i pi k (x+2)} = e^{i pi k x}), so wrap the
    # real part to (-1, 1]; this fixes the log branch that sends z = -1 to
    # x = -1 rather than MATLAB's x = 1.
    rr = np.real(r) - 2.0 * np.ceil((np.real(r) - 1.0) / 2.0)
    r = rr + 1j * np.imag(r)
    if prune:
        nnz = np.nonzero(np.abs(c) > 1e-13 * max(np.max(np.abs(c)), 1e-300))[0]
        if nnz.size == 0:
            return jnp.array([], dtype=jnp.complex128)
        N = int(np.ceil(c.size / 2) - 1)
        N = max(N, 1)
        a = 1.0 / N / np.pi * np.log(4.0 / (10 * _EPS) + 1.0)
        r = r[np.abs(np.imag(r)) <= a]
    return jnp.asarray(r, dtype=jnp.complex128)


def _trig_minandmax_scalar(f) -> tuple:
    """Global min/max of a scalar-valued Trigtech via critical points.

    Returns ``((min_val, min_pos), (max_val, max_pos))``.  For a complex
    tech the extrema of ``|f|`` are located (via ``|f|^2`` to avoid the abs
    singularity) and the reported values are ``f`` at those positions.

    NOT JIT-safe (rootfinding has variable output size).

    Provenance
    ----------
    MATLAB source : @trigtech/minandmax.m
    Chebfun commit: 7574c77
    """
    import numpy as np

    is_real = f.is_real
    n = f.n
    if n <= 1:
        val = _trig_eval(f.coeffs, jnp.zeros((1,), jnp.float64),
                         is_real=is_real)[0]
        pos = jnp.array(-1.0, dtype=jnp.float64)
        v = jnp.real(val).astype(jnp.float64) if is_real else val
        return (v, pos), (v, pos)

    if is_real:
        objc = f.coeffs
    else:
        m = max(2 * n + 1, 65)
        x = trigpts(m)
        v = _trig_eval(f.coeffs, x, is_real=False)
        objc = trig_vals2coeffs((jnp.abs(v) ** 2).astype(jnp.complex128))
    # Critical points are the roots of the objective's derivative.
    d1 = _trig_diff_coeffs(objc, 1)
    d2 = _trig_diff_coeffs(objc, 2)
    crit = np.asarray(_trig_roots(d1))
    # Polish with Newton on obj'(x) = 0 (the chebyshev-sampled roots of a
    # high-frequency derivative can be off by ~1e-7; the exact trig
    # derivatives recover machine precision).
    if crit.size:
        xc = jnp.asarray(crit, dtype=jnp.float64)
        for _ in range(2):
            g1 = _trig_eval(d1, xc, is_real=True)
            g2 = _trig_eval(d2, xc, is_real=True)
            step = jnp.where(jnp.abs(g2) > 1e-30, g1 / g2, 0.0)
            xc = xc - step
        crit = np.asarray(xc)
    # Include a periodic reference point so a constant/near-constant
    # objective (empty critical set) still yields a valid extremum.
    if crit.size:
        cand = jnp.asarray(np.concatenate([[-1.0], crit]), dtype=jnp.float64)
    else:
        cand = jnp.array([-1.0], dtype=jnp.float64)
    fv = np.asarray(_trig_eval(f.coeffs, cand, is_real=is_real))
    cand_np = np.asarray(cand)

    if is_real:
        fr = np.real(fv)
        imn = int(np.argmin(fr))
        imx = int(np.argmax(fr))
        return ((jnp.asarray(fr[imn]), jnp.asarray(cand_np[imn])),
                (jnp.asarray(fr[imx]), jnp.asarray(cand_np[imx])))
    mag = np.abs(fv)
    imn = int(np.argmin(mag))
    imx = int(np.argmax(mag))
    return ((jnp.asarray(fv[imn]), jnp.asarray(cand_np[imn])),
            (jnp.asarray(fv[imx]), jnp.asarray(cand_np[imx])))


# ============================================================================
# Trigtech class
# ============================================================================



def _trig_column_mask(values, vscale=0.):
    """Native populate3eps per-column predicate; tracing is conservative.

    Source @trigtech/populate.m48-53, pin7574c77. Static metadata cannot be
    inferred from traced complex values; callers can supply a known mask.
    """
    values = jnp.asarray(values)
    if values.shape[0] == 0:
        return ()
    columns = 1 if values.ndim == 1 else values.shape[1]
    scale = jnp.maximum(jnp.asarray(vscale), jnp.max(jnp.abs(values), axis=0))
    flags = jnp.max(jnp.abs(jnp.imag(values)), axis=0) <= 3 * (_EPS * scale)
    if isinstance(flags, jax.core.Tracer):
        return (not jnp.iscomplexobj(values),) * columns
    return tuple(bool(x) for x in jnp.ravel(flags))


def _trig_nonadaptive_real_flag(values):
    return all(_trig_column_mask(values))


def _trig_project_values(values, mask):
    """Project columns only at native source assignment points (7574c77)."""
    if not mask:
        return values
    if all(mask):
        return jnp.real(values)
    flags = jnp.asarray(mask)
    return jnp.where(flags[0] if len(mask) == 1 else flags,
                     jnp.real(values), values)


def _trig_mask_width(mask, width):
    if len(mask) == width:
        return mask
    if len(mask) == 1:
        return mask * width
    if width == 0:
        return ()
    raise ValueError("Trigtech real_columns width mismatch")


def _trig_mask_and(left, right, width):
    return tuple(a and b for a, b in zip(_trig_mask_width(left, width),
                                        _trig_mask_width(right, width)))


def _trig_probe_mask(probe):
    # MATLAB isreal tests storage, not an imag==0 predicate. JAX arrays use
    # homogeneous dtype, including complex-zero scalars (adapter contract).
    probe = jnp.asarray(probe)
    return (not jnp.iscomplexobj(probe),) * int(probe.size)


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
    real_columns : tuple of bool
        Static source isReal metadata, one flag per represented column.
        Explicit masks take precedence over the legacy is_real argument.
        Without a mask, a raw constructor asserts uniform is_real metadata;
        from_values/from_coeffs infer concrete per-column metadata instead.
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

    coeffs: jax.Array = eqx.field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.complex128)
    )  # complex128, shape (N,)
    is_real: bool = eqx.field(static=True, default=True)
    ishappy: bool = eqx.field(static=True, default=True)
    # Source stores grid values independently of coefficients. Keep exact
    # supplied/scaled values when available; coefficient transforms construct
    # fresh instances without this optional dynamic JAX leaf.
    _values: jax.Array | None = None
    real_columns: tuple[bool, ...] | None = eqx.field(static=True, default=None)

    def __post_init__(self):
        width = 0 if self.coeffs.shape[0] == 0 else (1 if self.coeffs.ndim == 1 else self.coeffs.shape[1])
        mask = self.real_columns
        if mask is None:
            # Backward-compatible raw constructor: aggregate flag
            # asserts a uniform mask. Existing-object factories pass masks.
            mask = (bool(self.is_real),) * width
        else:
            mask = tuple(bool(value) for value in mask)
            if len(mask) != width:
                raise ValueError("Trigtech real_columns must match coefficient columns")
        object.__setattr__(self, "real_columns", mask)
        object.__setattr__(self, "is_real", all(mask))

    @property
    def isReal(self):
        """Source per-column metadata; is_real remains its scalar aggregate."""
        return jnp.asarray(self.real_columns, dtype=jnp.bool_)

    @staticmethod
    def horner(x, coeffs, is_real=False):
        """Direct source horner mask API; public feval passes its aggregate.

        @trigtech/horner.m55-63 and feval.m34, pin7574c77.

        Python metadata adapter: accept scalar flags, static tuples/lists,
        or concrete one-dimensional logical arrays (including f.isReal).
        A traced mask is unsupported: pass the static f.real_columns tuple
        when tracing evaluation. This method does not infer dynamic realness.
        """
        width = 1 if coeffs.ndim == 1 else coeffs.shape[1]
        if any(isinstance(value, jax.core.Tracer)
               for value in jax.tree_util.tree_leaves(is_real)):
            raise TypeError("Trigtech.horner requires a static realness mask; "
                            "pass f.real_columns when tracing evaluation")
        if isinstance(is_real, (bool, int)):
            # Python scalar metadata stays concrete inside a traced evaluator.
            mask = (bool(is_real),) * width
        elif isinstance(is_real, (tuple, list)):
            mask = tuple(bool(v) for v in is_real)
        else:
            flags = jnp.asarray(is_real)
            if flags.ndim == 0:
                mask = (bool(flags),) * width
            elif flags.ndim == 1 and flags.dtype == jnp.bool_:
                mask = tuple(bool(value) for value in flags)
            else:
                raise ValueError("Trigtech.horner mask must be a scalar flag "
                                 "or a one-dimensional logical array")
        mask = _trig_mask_width(mask, width)
        result = _trig_eval(coeffs, x, all(mask))
        return result if all(mask) else _trig_project_values(result, mask)

    # ------------------------------------------------------------------
    # Empty representation (MATLAB trigtech() with no arguments)
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> "Trigtech":
        """The empty Trigtech (MATLAB ``trigtech()``).

        Provenance
        ----------
        MATLAB source : @trigtech/isempty.m
        Chebfun commit: 7574c77
        """
        return cls()

    def isempty(self) -> bool:
        """True for the empty Trigtech (MATLAB ``isempty``).

        Provenance
        ----------
        MATLAB source : @trigtech/isempty.m
        Chebfun commit: 7574c77
        """
        return self.coeffs.size == 0

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
        real_columns: tuple[bool, ...] | None = None,
        pref=None,
        data=None,
    ) -> "Trigtech":
        """Construct from Fourier coefficients using source realness threshold.

        Eager inference tests source-grid values against3eps*vscale. Under
        JAX tracing, value-dependent static metadata is unavailable, so default
        inference conservatively retains complex arithmetic. Callers with a
        known real representation can provide explicit ``is_real=True``;
        real-valued differentiation objectives should state that contract.

        Provenance
        ----------
        MATLAB source : @trigtech/populate.m (nonadaptive construction),
                        @trigtech/vscale.m
        Chebfun commit: 7574c77
        """
        coeffs = jnp.atleast_1d(jnp.asarray(coeffs, dtype=jnp.complex128))
        if pref is not None or data is not None:
            from chebfunjax.tech._trig_constructor import construct

            result = construct(coeffs, pref=pref, data=data, coefficients=True)
            if real_columns is None and is_real is not None:
                real_columns = (bool(is_real),) * len(result.real_columns)
            if real_columns is None:
                real_columns = result.real_columns
                values = result.values
            else:
                # Explicit static metadata is the existing Python adapter;
                # it overrides inference, including under source preferences.
                values = _trig_project_values(_trig_coeffs2vals_impl(result.coeffs), real_columns)
            return cls(coeffs=result.coeffs, real_columns=real_columns,
                       ishappy=ishappy, _values=values)
        if real_columns is None:
            if is_real is None:
                real_columns = _trig_column_mask(_trig_coeffs2vals_impl(coeffs))
            else:
                width = 0 if coeffs.shape[0] == 0 else (1 if coeffs.ndim == 1 else coeffs.shape[1])
                real_columns = (bool(is_real),) * width
        return cls(coeffs=coeffs, real_columns=real_columns, ishappy=ishappy)

    @classmethod
    def from_values(
        cls,
        values: jax.Array,
        *,
        ishappy: bool = True,
        pref=None,
        data=None,
    ) -> "Trigtech":
        """Construct from source-grid values with literal3eps classification.

        Eager realness follows nonadaptive MATLAB populate. Traced complex
        input retains complex arithmetic because realness is static metadata.
        The pure JAX FFT preserves input dtype for source symmetry handling.

        Provenance
        ----------
        MATLAB source : @trigtech/populate.m, @trigtech/vals2coeffs.m,
                        @trigtech/vscale.m
        Chebfun commit: 7574c77
        """
        if pref is not None or data is not None:
            from chebfunjax.tech._trig_constructor import construct

            result = construct(values, pref=pref, data=data)
            return cls(coeffs=result.coeffs, real_columns=result.real_columns,
                       ishappy=ishappy, _values=result.values)
        values = jnp.atleast_1d(jnp.asarray(values))
        mask = _trig_column_mask(values)
        coeffs = _trig_vals2coeffs_impl(values)
        return cls(coeffs=coeffs, real_columns=mask, ishappy=ishappy,
                   _values=_trig_project_values(values, mask))

    @classmethod
    def from_function(
        cls,
        f: Callable[[jax.Array], jax.Array],
        *,
        n: int | None = None,
        maxpow2: int | None = None,
        pref=None,
        data=None,
    ) -> "Trigtech":
        """Source fixed/adaptive constructor with complete Tech preferences.

        MATLAB @trigtech/trigtech.m, populate.m and refine.m, pin7574c77.
        ``n`` overrides fixedLength. Legacy explicit maxpow2 supplies a cap
        only absent an explicit maxLength; omitted defaults use source65536.
        Construction/callbacks are eager; the resulting Tech supports JAX.
        Public Chebfun routing remains a separate pending R2 adapter.
        """
        from chebfunjax.tech._trig_constructor import construct

        result = construct(f, pref=pref, data=data, n=n, maxpow2=maxpow2)
        if maxpow2 is not None and not result.ishappy:
            # Legacy Python maxpow2 adapter retains its warning; native pref
            # construction itself returns unhappy without emitting one.
            warnings.warn(f"Trigtech.from_function: function did not converge with "
                          f"{result.n} points. Returning unhappy representation.", stacklevel=2)
        return result

    @classmethod
    def _fixed_construct(cls, f: Callable, n: int) -> "Trigtech":
        """Fixed-size construction."""
        if n <= 0:
            return cls(coeffs=jnp.array([], dtype=jnp.complex128), is_real=True)
        values, _ = _sample_callable_trig_grid(f, n)
        return cls.from_values(values)

    @classmethod
    def _adaptive_construct(
        cls, f: Callable, maxpow2: int = 16, start_pow2: int = 4,
        real_columns: tuple[bool, ...] | None = None,
    ) -> "Trigtech":
        """Legacy private shape adapter into the shared source constructor."""
        result = cls.from_function(f, pref={"minSamples": 2**start_pow2+1}, maxpow2=maxpow2)
        if real_columns is None:
            return result
        return cls(coeffs=result.coeffs, real_columns=real_columns,
                   ishappy=result.ishappy,
                   _values=_trig_project_values(result.values, real_columns))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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
        JIT-safe: yes. vmap-safe: yes. grad-safe: yes.  Concrete inputs
        (with concrete coefficients) run a numpy mirror of the same
        Horner scheme -- every distinctly-shaped Trigtech otherwise
        compiles its own XLA program, and object-heavy pipelines
        exhaust the LLVM JIT code arena.

        Provenance
        ----------
        MATLAB source : @trigtech/feval.m, @trigtech/horner.m
        Chebfun commit: 7574c77
        """
        if not jnp.iscomplexobj(jnp.asarray(x)) and \
                not isinstance(x, jax.core.Tracer) and \
                not isinstance(self.coeffs, jax.core.Tracer) and \
                self.coeffs.shape[0] <= 1024:
            # The numpy Horner mirror loops once per coefficient in
            # Python; past ~1k coefficients the XLA scan (compiled once
            # per length) is far faster, and huge-length Trigtechs are
            # rare enough not to threaten the JIT code arena.
            return jnp.asarray(_trig_eval_np(self.coeffs, np.asarray(x),
                                             is_real=self.is_real))
        return self._call_traced(x)

    @eqx.filter_jit
    def _call_traced(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        x = x.astype(jnp.complex128 if jnp.iscomplexobj(x) else jnp.float64)
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

    @staticmethod
    def alias(coeffs: jax.Array, m: int) -> jax.Array:
        """Alias Fourier coefficients on the equispaced grid to length ``m``.

        ``ALIAS(C, M)`` zero-pads (``M > len(C)``) or frequency-folds
        (``M < len(C)``) the coefficients ``C``.  Aliasing to length ``M``
        gives exactly the coefficients of the interpolant through the
        underlying function on the ``M``-point equispaced grid.

        Provenance
        ----------
        MATLAB source : @trigtech/alias.m
        Chebfun commit: 7574c77
        """
        return _alias_trigtech(coeffs, m)

    @staticmethod
    def quadwts(n: int) -> jax.Array:
        """Quadrature (trapezoid-rule) weights for ``n`` equispaced points.

        ``QUADWTS(N)`` returns ``2/n`` repeated ``n`` times: the weights for
        the periodic trapezoid rule on ``n`` points of ``[-1, 1)``.

        Provenance
        ----------
        MATLAB source : @trigtech/quadwts.m
        Chebfun commit: 7574c77
        """
        if n == 0:
            return jnp.array([], dtype=jnp.float64)
        return jnp.full((n,), 2.0 / n, dtype=jnp.float64)

    def trigcoeffs(self, N: int | None = None) -> jax.Array:
        """Trigonometric (complex-exponential) coefficients of the trigtech.

        ``trigcoeffs(f)`` returns the stored Fourier coefficients;
        ``trigcoeffs(f, N)`` returns exactly ``N`` of them, padding
        symmetrically or truncating with the correct even-``N`` Nyquist fold.

        Provenance
        ----------
        MATLAB source : @trigtech/trigcoeffs.m
        Chebfun commit: 7574c77
        """
        if N is None:
            N = len(self)
        return _trigcoeffs_trigtech(self.coeffs, N)

    def sample(self, n: int | None = None):
        """Sample the trigtech at ``n`` equispaced points on ``[-1, 1)``.

        Returns ``(values, points)``; ``n = len(self)`` if omitted.  When
        ``n == len(self)`` the stored values are returned directly,
        otherwise the coefficients are aliased to length ``n`` first.

        Provenance
        ----------
        MATLAB source : @trigtech/sample.m
        Chebfun commit: 7574c77
        """
        if n is None:
            n = len(self)
        if n == len(self):
            values = self.values
        else:
            values = trig_coeffs2vals(_alias_trigtech(self.coeffs, n))
            values = _trig_project_values(values, self.real_columns)
        points = trigpts(n)
        return values, points

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
        # Native populate stores supplied values even for an empty array.
        if self._values is not None:
            return self._values
        if self.isempty():
            return jnp.empty(self.coeffs.shape, dtype=jnp.float64 if self.is_real
                             else jnp.complex128)
        return _trig_project_values(trig_coeffs2vals(self.coeffs), self.real_columns)

    @property
    def vscale(self) -> float:
        """Aggregate vertical scale: max |f(x)| on the grid.

        Provenance
        ----------
        MATLAB source : @trigtech/vscale.m
        Chebfun commit: 7574c77

        The source empty case returns zero. For array-valued inputs this
        property retains the existing aggregate adapter; the source's public
        per-column result is a separate unresolved API gap.
        """
        if self.isempty():
            # MATLAB @trigtech/vscale.m returns zero when coeffs are empty.
            return 0.0
        return float(jnp.max(jnp.abs(self.values)))

    def normest(self):
        """Native @trigtech/normest.m7574c77: max of column value scales."""
        from chebfunjax.tech._trig_constructor import _values

        return 0.0 if self.isempty() else float(jnp.max(jnp.abs(_values(self))))

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
        return Trigtech(coeffs=new_coeffs, is_real=self.is_real, real_columns=self.real_columns, ishappy=self.ishappy)

    def simplify(self, tol: float | jax.Array | None = None) -> "Trigtech":
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
        if self.isempty():
            return self
        if not self.ishappy:
            return self

        nold = self.n
        if nold == 0:
            # MATLAB @trigtech/simplify.m leaves an empty trigtech alone.
            return self
        N = max(17, int(jnp.floor(nold * 1.25 + 5 + 0.5)))
        prolonged = self.prolong(N)

        cutoff, chop_len = _trig_simplify_cutoff(prolonged.coeffs, tol)
        cutoff = min(cutoff, chop_len)

        # MATLAB: cutoff = min(cutoff, nold); an even cutoff keeps
        # cutoff/2 + 1 modes on each side (length cutoff + 1), an odd one
        # (cutoff - 1)/2 + 1 (length cutoff).  In particular an
        # unchoppable EVEN-length tech becomes the odd length nold + 1
        # with its Nyquist coefficient split over +-N/2 -- never nold - 1,
        # which silently dropped the Nyquist mode (spin's KdV output).
        cutoff = min(cutoff, nold)
        n_keep = cutoff + 1 if cutoff % 2 == 0 else cutoff
        n_keep = max(1, n_keep)

        new_coeffs = _trig_prolong_coeffs(self.coeffs, n_keep)
        return Trigtech(coeffs=new_coeffs, is_real=self.is_real, real_columns=self.real_columns, ishappy=self.ishappy)

    # ------------------------------------------------------------------
    # Calculus
    # ------------------------------------------------------------------

    def diff(self, k: int = 1, dim: int = 1) -> "Trigtech":
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
        if dim == 2:
            if k == 0:
                return self
            if k >= self.num_columns:
                return Trigtech.empty()
            mask = self.real_columns
            for _ in range(k):
                mask = tuple(a == b for a, b in zip(mask[:-1], mask[1:]))
            return Trigtech(coeffs=jnp.diff(self.coeffs, n=k, axis=1),
                            real_columns=mask, ishappy=self.ishappy,
                            _values=jnp.diff(self.values, n=k, axis=1))
        if k == 0:
            return self
        dc = _trig_diff_coeffs(self.coeffs, k)
        # Derivative of a real function is real-valued
        return Trigtech(coeffs=dc, is_real=self.is_real, real_columns=self.real_columns, ishappy=self.ishappy)

    def cumsum(self, m: int | None = 1, dim: int = 1) -> "Trigtech":
        r"""Return the antiderivative with F(-1) = 0.

        ``m`` selects the order and ``dim`` selects the dimension. For
        ``dim=1`` this follows MATLAB's continuous Fourier antiderivative;
        for any other dimension it performs repeated cumulative sums over
        coefficient columns.

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

        Notes
        -----
        The continuous branch ports MATLAB's direct order-``m`` integration
        factor and its post-integration simplify/left-value adjustment. The
        finite-dimensional branch applies the source column cumulative sum
        ``m`` times. MATLAB updates both its cached values and coefficient
        row after subtracting ``lval``; both stored arrays are preserved.
        The Python method accepts scalar integer ``m``
        and ``dim``.
        """
        if self.isempty():
            return self
        if m is None:
            m = 1
        m = int(m)
        if m == 0:
            return self
        if dim != 1:
            if self.coeffs.ndim == 1:
                return self
            coeffs, values = self.coeffs, self.values
            for _ in range(max(m, 0)):
                coeffs = jnp.cumsum(coeffs, axis=1)
                values = jnp.cumsum(values, axis=1)
            return Trigtech(coeffs=coeffs, real_columns=self.real_columns,
                            ishappy=self.ishappy, _values=values)

        n = self.n
        c0_idx = n // 2
        # Check every array-valued column against its own mean tolerance.
        c0_mag = jnp.abs(self.coeffs[c0_idx])
        # MATLAB's vscale(f) is column-wise for array-valued trigtechs.
        # A large neighboring column must not hide a nonzero mean in a
        # smaller column, and a zero column keeps its exact zero tolerance.
        vs = self.vscale_columns()
        if bool(jnp.any(c0_mag > 10.0 * vs * _EPS)):
            raise ValueError(
                "CHEBFUN:TRIGTECH:cumsum:meanNotZero: "
                "Indefinite integrals are only possible for TRIGTECH objects "
                "with zero mean."
            )
        bc = _trig_cumsum_coeffs(self.coeffs, m=m)
        result = Trigtech(coeffs=bc, is_real=self.is_real, real_columns=self.real_columns, ishappy=self.ishappy)
        result = result.simplify()
        # MATLAB @trigtech/cumsum.m subtracts lval from coefficient row 1
        # after simplify. Preserve this literal source operation; it is not
        # rewritten as a central (constant-mode) correction here.
        lval = result(jnp.asarray(-1.0))
        corrected = result.coeffs.at[0].add(-lval)
        return Trigtech(coeffs=corrected, is_real=result.is_real, real_columns=result.real_columns,
                        ishappy=result.ishappy, _values=result.values - lval)

    def innerProduct(self, other: "Trigtech") -> jax.Array:
        r"""L^2 inner product <f, g> = \int_{-1}^{1} conj(f) g dx.

        For Fourier series f = sum a_k e^{i pi k x},
        g = sum b_k e^{i pi k x}: <f, g> = 2 sum conj(a_k) b_k
        (orthogonality of the modes on [-1, 1]).  MATLAB forces
        <f, f> real-nonnegative (isequal branch).  Added by Claude
        Fable 5 (trigtech method gap).

        Provenance
        ----------
        MATLAB source : @trigtech/innerProduct.m
        Chebfun commit: 7574c77
        """
        n = max(self.n, other.n)
        fc = _trig_prolong_coeffs(self.coeffs, n)
        gc = _trig_prolong_coeffs(other.coeffs, n)
        both_1d = (fc.ndim == 1) and (gc.ndim == 1)
        # Fourier-mode orthogonality on [-1, 1]:
        # <e^{i pi k x}, e^{i pi m x}> = 2 delta_{km}, hence
        # <f, g>_{ij} = 2 sum_k conj(a_{k,i}) b_{k,j}.
        fc2 = fc if fc.ndim == 2 else fc[:, None]
        gc2 = gc if gc.ndim == 2 else gc[:, None]
        out = 2.0 * (jnp.conj(fc2).T @ gc2)  # (mf, mg) matrix
        same = other is self
        if not same and self.coeffs.shape == other.coeffs.shape:
            if not isinstance(self.coeffs, jax.core.Tracer) and \
                    not isinstance(other.coeffs, jax.core.Tracer):
                same = bool(jnp.all(self.coeffs == other.coeffs))
        real_pairs = jnp.asarray(self.real_columns)[:, None] & jnp.asarray(other.real_columns)[None, :]
        out = jnp.where(real_pairs, jnp.real(out), out)
        if same:
            # Force a non-negative real diagonal (MATLAB isequal branch).
            d = jnp.diag(out)
            out = out - jnp.diag(d) + jnp.diag(jnp.abs(d))
        if both_1d:
            # Scalar-valued inputs: return a scalar (legacy behaviour).
            val = out[0, 0]
            if same:
                return jnp.abs(val)
            if self.is_real and other.is_real:
                return jnp.real(val)
            return val
        return out

    inner = innerProduct

    def compose(self, op, g=None, data=None, pref=None) -> "Trigtech":
        """Native composition through the shared eager Trig constructor.

        MATLAB @trigtech/compose.m, Chebfun7574c77. Python shape adapters
        align scalar-column operand arrays; constructor numerics are JAX.
        """
        from chebfunjax.tech._trig_constructor import construct, resolve_pref

        prefs = resolve_pref(pref)
        prefs["minSamples"] = max(prefs["minSamples"], self.n)
        prefs["chebfuneps"] = jnp.maximum(jnp.asarray(prefs["chebfuneps"]), _EPS)
        prefs["sampleTest"] = False

        def columns(u):
            return 1 if u.coeffs.ndim == 1 else u.coeffs.shape[1]

        # Python None and an empty technology represent the omitted G slot.
        if isinstance(g, Trigtech) and g.isempty():
            g = None
        if g is not None:
            if not isinstance(g, Trigtech) or columns(self) != columns(g):
                raise ValueError("CHEBFUN:TRIGTECH:compose:dim: Matrix dimensions must agree.")
            prefs["minSamples"] = max(prefs["minSamples"], g.n)
        elif isinstance(op, Trigtech):
            if columns(self) > 1 and columns(op) > 1:
                raise ValueError("CHEBFUN:TRIGTECH:compose:arrval: Cannot compose two array-valued TRIGTECH objects.")
            if bool(jnp.max(jnp.abs(self.values)) > 1 + 2 * _EPS):
                raise ValueError("CHEBFUN:TRIGTECH:compose:range: The range of f is not contained in the domain of g.")
            prefs["minSamples"] = max(prefs["minSamples"], op.n)

        function = op
        if isinstance(prefs["refinementFunction"], str):
            if g is None:
                def function(x):
                    return op(self(x))
            else:
                def function(x):
                    left, right = self(x), g(x)
                    if left.ndim < right.ndim:
                        left = left[..., None]
                    elif right.ndim < left.ndim:
                        right = right[..., None]
                    return op(left, right)

        result = construct(function, data=data, pref=prefs)
        if not result.ishappy:
            warnings.warn("TRIGTECH:TRIGTECH:compose:convfail: "
                          f"Composition failed to converge with {result.n} points.", stacklevel=2)
        return result

    def restrict(self, a: float, b: float):
        """Restriction to [a, b] within [-1, 1].

        A restricted periodic function is generally NOT periodic, so
        (like MATLAB) the result is a Chebyshev representation on the
        subinterval: returns a Chebtech2 of f|_[a,b] mapped to [-1,1].
        Added by Claude Fable 5.

        Provenance
        ----------
        MATLAB source : @trigtech/restrict.m (output is cheb-based)
        Chebfun commit: 7574c77
        """
        from chebfunjax.tech.chebtech import Chebtech2
        if self.isempty():
            return Chebtech2.empty()
        a = float(a)
        b = float(b)
        if not (-1.0 <= a < b <= 1.0):
            raise ValueError("restrict: need -1 <= a < b <= 1")

        def g(t):
            x = a + (b - a) * (jnp.asarray(t) + 1.0) / 2.0
            return self(x)

        return Chebtech2.from_function(g)

    def sum(self, dim: int = 1) -> "jax.Array | Trigtech":
        r"""Definite integral over [-1, 1].

        Returns 2 * c_0 (real if ``is_real`` is True); one integral per
        column for array-valued techs.  ``dim=2`` sums ACROSS the
        columns and returns a scalar-column Trigtech (MATLAB
        ``sum(f, 2)``, a no-op for scalar-valued input).

        Returns
        -------
        jax.Array (scalar or (m,)) or Trigtech

        Notes
        -----
        JIT-safe: yes (dim=1).

        Provenance
        ----------
        MATLAB source : @trigtech/sum.m
        Chebfun commit: 7574c77
        """
        if dim == 2:
            if self.coeffs.ndim == 1:
                return self
            return Trigtech(coeffs=jnp.sum(self.coeffs, axis=1),
                            real_columns=(self.is_real,), ishappy=self.ishappy,
                            _values=jnp.sum(self.values, axis=1))
        s = _trig_definite_integral(self.coeffs)
        return _trig_project_values(s, self.real_columns)

    # ------------------------------------------------------------------
    # Roots
    # ------------------------------------------------------------------

    def roots(self, complex: bool = False) -> jax.Array:
        """Find roots in [-1, 1].

        By default converts to a Chebyshev representation and calls
        Chebyshev rootfinding, returning the real roots in [-1, 1].  With
        ``complex=True`` (MATLAB ``roots(f, 'complex', 1)``) returns all
        roots -- including complex ones outside [-1, 1] -- via the
        companion-matrix method, pruned to the strip of analyticity.

        NOT JIT-safe (variable output size).

        Returns
        -------
        jax.Array
            Roots (float64 for the default real path, complex128 for the
            ``complex=True`` path); array-valued techs return one
            NaN-padded column per column of ``f``.

        Provenance
        ----------
        MATLAB source : @trigtech/roots.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        def _one(col, mask):
            if complex:
                simp = Trigtech.from_coeffs(col, real_columns=mask).simplify()
                return _np.asarray(_trig_roots_complex(simp.coeffs, prune=True))
            return _np.asarray(_trig_roots(col))

        if self.coeffs.ndim == 2:
            cols = [_one(self.coeffs[:, j], (self.real_columns[j],))
                    for j in range(self.coeffs.shape[1])]
            nmax = max((len(c) for c in cols), default=0)
            dtype = _np.complex128 if complex else _np.float64
            out = _np.full((nmax, len(cols)), _np.nan, dtype=dtype)
            for j, c in enumerate(cols):
                out[: len(c), j] = c
            return jnp.asarray(out)
        return jnp.asarray(_one(self.coeffs, self.real_columns))

    # ------------------------------------------------------------------
    # Happiness check
    # ------------------------------------------------------------------

    @staticmethod
    def happiness_check(
        coeffs: jax.Array,
        values: jax.Array,
        op: Callable | None = None,
        tol: float | jax.Array | None = None,
        vscale: float | jax.Array = 0.0,
    ) -> tuple[bool, int]:
        """Source standard happiness check for Fourier interpolation.

        Coefficient checking uses per-column running/local scale and returns
        a source odd retained count. When ``op`` is supplied, the FULL
        interpolant is compared at the two source points before any chop.
        Its threshold is ``sqrt(max(tol, eps))*max(local_column_scales)``;
        a larger historical ``vscale`` does not loosen this sample test.
        Sample rejection returns ``(False, original_sample_count)``.

        Parameters
        ----------
        coeffs : jax.Array, shape (N,) or (N, M)
        values : jax.Array, same sample/column shape as coeffs
        op : callable or None
            None skips source sampleTest. Otherwise op accepts an array
            of two canonical sample points and preserves output columns.
        tol : float, jax.Array, or None
            None selects binary64 epsilon. A one-dimensional array is
            interpreted as a MATLAB row tolerance. StandardCheck accepts
            one entry per column; otherwise it broadcasts the maximum.
            For an explicit two-dimensional tolerance, MATLAB's final
            dimension rule is used: an (M,1) column tolerance for M>1
            broadcasts its maximum during coefficient checking. Source
            sampleTest retains the original tolerance shape/broadcasting.
        vscale : float or one-dimensional jax.Array
            Running scale, scalar or one entry per function column.
            Two-dimensional scale arrays are outside this API contract.

        Returns
        -------
        ishappy : bool
        cutoff : int
            Source odd count after coefficient checking, including an
            unhappy coefficient result; sample rejection returns N.

        Provenance
        ----------
        MATLAB source : @trigtech/happinessCheck.m, @trigtech/standardCheck.m,
            @trigtech/sampleTest.m
        Chebfun commit: 7574c77
        """
        if tol is None:
            tol = _EPS
        n = coeffs.shape[0]
        local = jnp.max(jnp.abs(values), axis=0)
        effective_scale = jnp.maximum(jnp.asarray(vscale), local)
        ishappy, cutoff = _trig_standard_check(coeffs, values, tol, effective_scale)
        if ishappy and op is not None:
            xeval = jnp.array([-0.357998918959666, 0.036785641195074],
                             dtype=jnp.float64)
            v_fun = _trig_eval(coeffs, xeval, is_real=False)
            v_op = jnp.asarray(op(xeval), dtype=jnp.complex128)
            errors = jnp.max(jnp.abs(v_op - v_fun), axis=0)
            # Source sampleTest uses THIS interpolant's scale, not the
            # possibly larger constructor running/global scale.
            sample_tol = jnp.sqrt(jnp.maximum(_EPS, jnp.asarray(tol))) * jnp.max(local)
            if not bool(jnp.all(errors <= sample_tol)):
                ishappy = False
                cutoff = n
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
        if self.isempty() or (isinstance(other, Trigtech)
                              and other.isempty()):
            # MATLAB @trigtech/plus.m: empty argument -> empty result.
            return Trigtech.empty()
        if isinstance(other, Trigtech):
            nf, ng = self.n, other.n
            n = max(nf, ng)
            fc = _trig_prolong_coeffs(self.coeffs, n)
            gc = _trig_prolong_coeffs(other.coeffs, n)
            # scalar-column + array-valued broadcasts via a trailing
            # column axis (MATLAB R2016b+ implicit expansion)
            if fc.ndim != gc.ndim:
                if fc.ndim == 1:
                    fc = fc[:, None]
                if gc.ndim == 1:
                    gc = gc[:, None]
            coeffs = fc + gc
            width = 1 if coeffs.ndim == 1 else coeffs.shape[1]
            mask = _trig_mask_and(self.real_columns, other.real_columns, width)
            return Trigtech(
                coeffs=coeffs,
                real_columns=mask,
                ishappy=self.ishappy and other.ishappy,
                _values=_trig_project_values(
                    (self.prolong(n).values[:, None] if self.coeffs.ndim == 1
                     and coeffs.ndim == 2 else self.prolong(n).values)
                    + (other.prolong(n).values[:, None] if other.coeffs.ndim == 1
                       and coeffs.ndim == 2 else other.prolong(n).values), mask),
            )
        else:
            # Scalar (or row of per-column scalars): add to the
            # constant mode c_0.  A length-m row expands a
            # scalar-valued tech to m columns (MATLAB implicit
            # expansion), and a complex scalar clears is_real -- the
            # imaginary part was silently dropped before (Fable 5,
            # flip-roots audit).
            s = jnp.asarray(other, dtype=jnp.complex128)
            c = self.coeffs
            if s.ndim == 1 and s.size > 1 and c.ndim == 1:
                c = jnp.broadcast_to(c[:, None],
                                     (c.shape[0], s.size)).copy()
            n = self.n
            c0_idx = n // 2
            c = c.at[c0_idx].add(s)
            width = 1 if c.ndim == 1 else c.shape[1]
            mask = tuple(flag and jnp.isrealobj(jnp.asarray(other))
                         for flag in _trig_mask_width(self.real_columns, width))
            values = self.values
            if c.ndim == 2 and values.ndim == 1:
                values = values[:, None]
            return Trigtech(coeffs=c, real_columns=mask,
                            ishappy=self.ishappy, _values=values + jnp.asarray(other))

    def __radd__(self, other) -> "Trigtech":
        return self.__add__(other)

    def __sub__(self, other) -> "Trigtech":
        """Subtract a Trigtech or scalar.

        Provenance
        ----------
        MATLAB source : @chebtech/minus.m (analogous)
        """
        if self.isempty() or (isinstance(other, Trigtech)
                              and other.isempty()):
            return Trigtech.empty()
        return self + (-other)

    def __rsub__(self, other) -> "Trigtech":
        return -(self - other)

    def __neg__(self) -> "Trigtech":
        if self.isempty():
            return Trigtech.empty()
        return Trigtech(coeffs=-self.coeffs, real_columns=self.real_columns,
                        ishappy=self.ishappy, _values=-self.values)

    def __pos__(self) -> "Trigtech":
        return self

    def __mul__(self, other) -> "Trigtech":
        """Multiply using the source prolong, simplify, positivity sequence.

        Provenance
        ----------
        MATLAB source : @trigtech/times.m, @trigtech/isequal.m,
                        @trigtech/conj.m
        Chebfun commit: 7574c77

        Equality uses coefficients and grid values. Supplied and scalar-scaled
        grid values are retained as a dynamic JAX leaf; coefficient transforms
        discard that cache. Realness propagates per column with a scalar
        aggregate used only where native feval requests all(isReal).
        Construction and value-dependent branch selection remain eager.
        """
        if self.isempty():
            return Trigtech.empty()
        if not isinstance(other, Trigtech):
            scalar = jnp.asarray(other)
            if scalar.size == 0:
                return Trigtech.empty()
            if scalar.ndim > 2 or (scalar.ndim == 2 and scalar.shape[0] != 1):
                raise ValueError("Trigtech times requires a scalar or row vector")
            row = scalar.reshape(-1)
            if row.size == 1:
                # Source scalar path scales coefficients directly.
                coeffs = self.coeffs * row[0]
            else:
                values = self.values
                ncols = 1 if values.ndim == 1 else values.shape[1]
                if ncols not in (1, row.size):
                    raise ValueError("Trigtech times: matrix dimensions must agree")
                if values.ndim == 1:
                    values = values[:, None]
                coeffs = trig_vals2coeffs(values * row[None, :])
            values = self.values
            if row.size > 1 and values.ndim == 1:
                values = values[:, None]
            width = 1 if coeffs.ndim == 1 else coeffs.shape[1]
            mask = tuple(flag and jnp.isrealobj(scalar)
                         for flag in _trig_mask_width(self.real_columns, width))
            return Trigtech(coeffs=coeffs, real_columns=mask, ishappy=self.ishappy,
                            _values=values * (row[0] if row.size == 1 else row[None, :]))
        if other.isempty():
            return Trigtech.empty()
        if self.n == 1:
            # Source constant-tech paths use values and the numeric branch.
            return other * self.values.reshape(-1)
        if other.n == 1:
            return self * other.values.reshape(-1)

        def columns(a):
            return a[:, None] if a.ndim == 1 else a

        fc, gc = columns(self.coeffs), columns(other.coeffs)
        fv, gv = columns(self.values), columns(other.values)
        fm, gm = fc.shape[1], gc.shape[1]
        if fm != gm and fm != 1 and gm != 1:
            raise ValueError(
                "CHEBFUN:TRIGTECH:times:dim2: matrix dimensions must agree")
        n = self.n + other.n - 1
        fnew = columns(self.prolong(n).values)
        if fm != gm:
            if fm == 1:
                fnew = jnp.broadcast_to(fnew, (n, gm))
            else:
                # Source broadcasts g before the equality checks.
                gc = jnp.broadcast_to(gc, (other.n, fm))
                gv = jnp.broadcast_to(gv, (other.n, fm))

        same = fc.shape == gc.shape and bool(
            jnp.array_equal(fc, gc) & jnp.array_equal(fv, gv))
        pos = False
        if same:
            values = fnew ** 2
            pos = self.is_real
        else:
            # Follow the source relational operation through the public
            # conjugation adapter. Its representation limitations remain
            # separate from this multiplication policy.
            conjugate = self.conj()
            conjugates = fc.shape == gc.shape and bool(
                jnp.array_equal(columns(conjugate.coeffs), gc)
                & jnp.array_equal(columns(conjugate.values), gv))
            if conjugates:
                values = jnp.conj(fnew) * fnew
                pos = True
            else:
                gnew = columns(other.prolong(n).values)
                values = fnew * gnew
        # Preserve the public scalar-valued shape convention.
        if self.coeffs.ndim == other.coeffs.ndim == 1:
            values = values[:, 0]
        h = Trigtech(
            coeffs=trig_vals2coeffs(values),
            real_columns=_trig_mask_and(
                self.real_columns, other.real_columns, max(fm, gm)),
            ishappy=self.ishappy and other.ishappy,
            _values=_trig_project_values(values, _trig_mask_and(
                self.real_columns, other.real_columns, max(fm, gm))),
        ).simplify()
        if pos:
            # Source enforces grid positivity after simplification. This is
            # not a guarantee on rounded Horner evaluations at other points.
            values = jnp.abs(trig_coeffs2vals(h.coeffs))
            h = Trigtech(
                coeffs=trig_vals2coeffs(values), is_real=True, ishappy=h.ishappy)
        return h

    def __rmul__(self, other) -> "Trigtech":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Trigtech":
        """Division by scalar or Trigtech.

        Provenance
        ----------
        MATLAB source : @chebtech/rdivide.m (analogous)
        """
        if isinstance(other, Trigtech):
            # Adaptive re-construction so the quotient is fully resolved
            # (MATLAB: compose(f, @rdivide, g)).
            return Trigtech.from_function(
                lambda x: _trig_eval(self.coeffs, x, self.is_real)
                / _trig_eval(other.coeffs, x, other.is_real)
            )
        else:
            arr = jnp.asarray(other)
            ncols = self.coeffs.shape[1] if self.coeffs.ndim == 2 else 1
            # MATLAB @trigtech/rdivide.m size validation: a column-shaped
            # divisor, or a row whose width mismatches the columns, is
            # rejected ('Matrix dimensions must agree').
            if arr.ndim >= 2 and arr.shape[0] > 1:
                raise ValueError(
                    "Trigtech rdivide: matrix dimensions must agree "
                    "(divisor must be a scalar or a row of one entry "
                    "per column).")
            row = arr.reshape(-1)
            if row.shape[0] > 1 and row.shape[0] != ncols:
                raise ValueError(
                    "Trigtech rdivide: matrix dimensions must agree "
                    "(divisor width must match the column count).")
            if row.shape[0] and not bool(jnp.any(row != 0)):
                # MATLAB: division by an all-zero divisor gives the NaN
                # trigtech (one NaN value row).
                nan_c = jnp.full(
                    (1, ncols) if self.coeffs.ndim == 2 else (1,),
                    jnp.nan, dtype=jnp.complex128)
                return Trigtech.from_values(nan_c)
            # A complex divisor clears is_real (the imaginary part was
            # silently dropped before -- Fable 5, flip-roots audit).
            s = arr.astype(jnp.complex128)
            mask = tuple(flag and jnp.isrealobj(arr)
                         for flag in self.real_columns)
            if row.shape[0] > 1:
                # MATLAB divides the VALUES row-wise and sets the
                # zero-divisor columns' values to NaN, whose transform
                # is an all-NaN coefficient column.
                q = self.coeffs / s.reshape(1, -1)
                zero_cols = row == 0
                if bool(jnp.any(zero_cols)):
                    q = jnp.where(zero_cols[None, :], jnp.nan, q)
                values = self.values / s.reshape(1, -1)
                values = jnp.where(zero_cols[None, :], jnp.nan, values)
                return Trigtech(coeffs=q, real_columns=mask,
                                ishappy=self.ishappy, _values=values)
            # Native numel(c)==1 is scalar division regardless of storage
            # shape. Rank-zero adaptation avoids (1,1)/(n,) broadcasting to
            # (1,n), which would reinterpret Fourier modes as columns.
            scalar = s.reshape(())
            return Trigtech(
                coeffs=self.coeffs / scalar,
                real_columns=mask,
                ishappy=self.ishappy,
                _values=self.values / scalar,
            )

    def __rtruediv__(self, other) -> "Trigtech":
        """scalar / Trigtech (adaptive, like MATLAB compose).  A
        complex numerator keeps its imaginary part (Fable 5,
        flip-roots audit: the float64 cast raised on complex input)."""
        use_complex = (not self.is_real) or bool(
            jnp.iscomplexobj(jnp.asarray(other)))
        s = jnp.asarray(
            other,
            dtype=jnp.complex128 if use_complex else jnp.float64)
        return Trigtech.from_function(
            lambda x: s / _trig_eval(self.coeffs, x, self.is_real)
        )

    def __pow__(self, exponent) -> "Trigtech":
        """Pointwise power through source composition.

        MATLAB source : @trigtech/power.m
        Chebfun commit: 7574c77
        """
        if isinstance(exponent, Trigtech):
            return self.compose(jnp.power, exponent)
        return self.compose(lambda x: jnp.power(x, exponent))

    def __abs__(self) -> "Trigtech":
        """Absolute value via grid evaluation."""
        n = max(2 * self.n, 17)
        if n % 2 == 0:
            n += 1
        x = trigpts(n)
        fv = jnp.abs(_trig_eval(self.coeffs, x, self.is_real))
        c = trig_vals2coeffs(fv.astype(jnp.complex128))
        return Trigtech(coeffs=c, is_real=True, ishappy=self.ishappy)

    # ------------------------------------------------------------------
    # Array-valued column operations and elementwise parts
    # (Fable 5, Big-Three array-valued epic)
    # ------------------------------------------------------------------

    def __matmul__(self, other) -> "Trigtech":
        """MATLAB mtimes ``f * A``: right-multiply an array-valued tech
        by a matrix, mixing its columns (coeffs @ A).

        Provenance
        ----------
        MATLAB source : @trigtech/mtimes.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return Trigtech.empty()
        try:
            A = jnp.asarray(other)
        except (TypeError, ValueError) as exc:
            # MATLAB @trigtech/mtimes.m: 'Use OP(f, c) to multiply a
            # TRIGTECH by an object of class <cls>.'
            raise TypeError(
                "Trigtech mtimes: cannot multiply a Trigtech by an "
                f"object of class {type(other).__name__}.") from exc
        if not jnp.issubdtype(A.dtype, jnp.number):
            raise TypeError(
                "Trigtech mtimes: cannot multiply a Trigtech by an "
                f"object of class {type(other).__name__}.")
        if A.size == 0:
            # MATLAB @trigtech/mtimes.m: f * [] is empty.
            return Trigtech.empty()
        c = self.coeffs if self.coeffs.ndim == 2 else self.coeffs[:, None]
        return Trigtech(coeffs=c @ A.astype(jnp.complex128),
                        is_real=self.is_real and bool(jnp.isrealobj(A)),
                        ishappy=self.ishappy,
                        _values=(self.values if self.values.ndim == 2
                                 else self.values[:, None]) @ A)

    def fliplr(self) -> "Trigtech":
        """Reverse the column order of an array-valued tech (no-op for
        scalar-valued input).

        Provenance
        ----------
        MATLAB source : @trigtech/fliplr.m
        Chebfun commit: 7574c77
        """
        if self.coeffs.ndim == 1:
            return self
        # Native fliplr reverses stored arrays but deliberately leaves isReal.
        return Trigtech(coeffs=self.coeffs[:, ::-1],
                        real_columns=self.real_columns, ishappy=self.ishappy,
                        _values=self.values[:, ::-1])

    def flipud(self) -> "Trigtech":
        """Return g with g(x) = f(-x).  Odd length flips the
        coefficients; even length keeps the c_{-N/2} mode in place
        (conjugated) and flips the rest.

        Provenance
        ----------
        MATLAB source : @trigtech/flipud.m
        Chebfun commit: 7574c77
        """
        c = self.coeffs
        if c.shape[0] % 2 == 1:
            new_c = c[::-1]
        else:
            new_c = jnp.concatenate(
                [jnp.conj(c[:1]), c[:0:-1]])
        values = self.values
        return Trigtech(coeffs=new_c, real_columns=self.real_columns,
                        ishappy=self.ishappy,
                        _values=jnp.concatenate([values[:1], values[:0:-1]]))

    def real(self) -> "Trigtech":
        """Real part (a zero tech if the input was purely imaginary).

        Provenance
        ----------
        MATLAB source : @trigtech/real.m
        Chebfun commit: 7574c77
        """
        # MATLAB returns f unchanged when the isReal flag is set.
        if self.is_real:
            return self
        # Source real.m extracts values, tests exact zero, and simplifies.
        values = jnp.real(self.values)
        if not bool(jnp.any(values)):
            z = jnp.zeros((1,) + self.coeffs.shape[1:],
                          dtype=jnp.complex128)
            return Trigtech(coeffs=z, is_real=True, ishappy=True)
        return Trigtech(coeffs=trig_vals2coeffs(values), is_real=True,
                        ishappy=self.ishappy).simplify()

    def imag(self) -> "Trigtech":
        """Imaginary part, preserving source length for complex inputs.

        Provenance
        ----------
        MATLAB source : @trigtech/imag.m
        Chebfun commit: 7574c77
        """
        if self.is_real:
            z = jnp.zeros((1,) + self.coeffs.shape[1:],
                          dtype=jnp.complex128)
            return Trigtech(coeffs=z, is_real=True, ishappy=True)
        # Unlike real.m, source imag.m does not simplify this result.
        return Trigtech(coeffs=trig_vals2coeffs(jnp.imag(self.values)),
                        is_real=True, ishappy=self.ishappy)

    def conj(self) -> "Trigtech":
        """Source conjugation of only columns not marked real.

        Provenance
        ----------
        MATLAB source : @trigtech/conj.m
        Chebfun commit: 7574c77
        """
        if self.is_real:
            return self
        # Literal @trigtech/conj.m: modify only columns not marked real.
        flags = jnp.asarray(self.real_columns)
        flags = flags[0] if self.coeffs.ndim == 1 else flags[None, :]
        coeffs = jnp.where(flags, self.coeffs, jnp.conj(self.coeffs[::-1]))
        values = jnp.where(flags, self.values, jnp.conj(self.values))
        return Trigtech(coeffs=coeffs, real_columns=self.real_columns,
                        ishappy=self.ishappy, _values=values)

    def extract_column(self, j: int) -> "Trigtech":
        """Return column ``j`` (0-based) of an array-valued tech as a
        scalar-valued Trigtech (MATLAB ``extractColumns``)."""
        c = self.coeffs if self.coeffs.ndim == 2 else self.coeffs[:, None]
        values = self.values
        values = values[:, None] if values.ndim == 1 else values
        return Trigtech(coeffs=c[:, j], real_columns=(self.real_columns[j],),
                        ishappy=self.ishappy, _values=values[:, j])

    def minandmax(self):
        """Native full-array adaptive C1 conversion followed by extrema.

        Source @trigtech/minandmax.m33–35, Chebfun7574c77. Python technology
        return adapter is ((min_value,min_position),(max_value,max_position)).
        Empty/complex/array behavior delegates to the same C1 source path;
        no per-column preconversion or direct-Fourier derivative shortcut.
        """
        from chebfunjax.tech.chebtech import Chebtech1
        return Chebtech1.from_function(lambda x: self(x)).minandmax()

    def min(self):
        """Global minimum (value, position) via minandmax."""
        (mn, mnp), _ = self.minandmax()
        return mn, mnp

    def max(self):
        """Global maximum (value, position) via minandmax."""
        _, (mx, mxp) = self.minandmax()
        return mx, mxp

    def mat2cell(self, sizes) -> list:
        """Split an array-valued tech by column counts (MATLAB
        ``mat2cell(f, 1, sizes)``).

        Provenance
        ----------
        MATLAB source : @trigtech/mat2cell.m
        Chebfun commit: 7574c77
        """
        c = self.coeffs if self.coeffs.ndim == 2 \
            else self.coeffs[:, None]
        out = []
        j = 0
        for s in sizes:
            block = c[:, j:j + s]
            values = self.values
            values = values[:, None] if values.ndim == 1 else values
            vb = values[:, j:j + s]
            out.append(Trigtech(
                coeffs=block[:, 0] if s == 1 else block,
                real_columns=self.real_columns[j:j + s], ishappy=self.ishappy,
                _values=vb[:, 0] if s == 1 else vb))
            j += s
        return out

    @classmethod
    def cell2mat(cls, techs) -> "Trigtech":
        """Horizontally concatenate techs into one array-valued tech.

        Provenance
        ----------
        MATLAB source : @trigtech/cell2mat.m
        Chebfun commit: 7574c77
        """
        techs = list(techs)
        if len(techs) == 1:
            # MATLAB @trigtech/cell2mat.m returns its singleton unchanged.
            # In particular, horzcat dropping empty inputs must not turn a
            # scalar coefficient vector into an array-valued representation.
            return techs[0]
        n = max(t.n for t in techs)
        cols = []
        for t in techs:
            c = _trig_prolong_coeffs(t.coeffs, n)
            cols.append(c if c.ndim == 2 else c[:, None])
        return cls(coeffs=jnp.concatenate(cols, axis=1),
                   real_columns=tuple(flag for t in techs for flag in t.real_columns),
                   ishappy=all(t.ishappy for t in techs),
                   _values=jnp.concatenate([
                       t.prolong(n).values[:, None] if t.coeffs.ndim == 1
                       else t.prolong(n).values for t in techs], axis=1))

    def assign_columns(self, cols, g) -> "Trigtech":
        """Overwrite the columns ``cols`` (0-based) with the columns of
        ``g`` (MATLAB assignColumns); ``g=None`` deletes them.

        Provenance
        ----------
        MATLAB source : @trigtech/assignColumns.m
        Chebfun commit: 7574c77
        """
        fc = self.coeffs if self.coeffs.ndim == 2 \
            else self.coeffs[:, None]
        cols = [cols] if isinstance(cols, int) else list(cols)
        if g is None:
            keep = [j for j in range(fc.shape[1]) if j not in cols]
            values = self.values
            values = values[:, None] if values.ndim == 1 else values
            return Trigtech(coeffs=fc[:, keep],
                            real_columns=tuple(self.real_columns[j] for j in keep),
                            ishappy=self.ishappy, _values=values[:, keep])
        n = max(fc.shape[0], g.n)
        fc = _trig_prolong_coeffs(fc, n)
        gc = _trig_prolong_coeffs(g.coeffs, n)
        gc = gc if gc.ndim == 2 else gc[:, None]
        out = fc.at[:, jnp.asarray(cols)].set(gc)
        mask = list(self.real_columns)
        for j, flag in zip(cols, _trig_mask_width(g.real_columns, len(cols)), strict=True):
            mask[j] = flag
        fv, gv = self.prolong(n).values, g.prolong(n).values
        fv = fv[:, None] if fv.ndim == 1 else fv
        gv = gv[:, None] if gv.ndim == 1 else gv
        values = fv.at[:, jnp.asarray(cols)].set(gv)
        return Trigtech(coeffs=out, real_columns=tuple(mask),
                        ishappy=self.ishappy and g.ishappy, _values=values)

    # ------------------------------------------------------------------
    # Size / scale introspection (array-valued)
    # ------------------------------------------------------------------

    @property
    def num_columns(self) -> int:
        """Number of columns (1 for scalar-valued techs)."""
        return 1 if self.coeffs.ndim == 1 else self.coeffs.shape[1]

    def size(self, dim: int | None = None):
        """MATLAB ``size(f)``: (n_rows, n_cols); ``size(f, 2)`` is the
        column count. ``n_rows`` is the number of Fourier coefficients.

        Provenance
        ----------
        MATLAB source : @trigtech/size.m
        Chebfun commit: 7574c77

        Python scalar coefficient vectors represent MATLAB single columns.
        """
        # Recomputed values have the same shape as coefficient storage,
        # including explicit empty matrices; size requires no FFT.
        value_shape = self.coeffs.shape
        if len(value_shape) < 2:
            shape = (value_shape[0], 1) if value_shape else (1, 1)
        else:
            shape = (value_shape[0], value_shape[1])
        if dim is None:
            return shape
        if dim < 1:
            raise ValueError("Dimension argument must be a positive integer.")
        return shape[dim - 1] if dim <= len(shape) else 1

    def any(self, dim: int | None = None):
        """MATLAB ``@trigtech/any`` reduction.

        With no dimension, and for ``dim=1``, follow the literal source
        implementation's ``any(f.values)`` reduction along MATLAB's first
        nonsingleton dimension (which collapses a 1-by-m row across columns).
        ``dim=2`` returns a constant Trigtech whose value is the source's
        sampled-any result at ``0.1273881594``.

        Provenance
        ----------
        MATLAB source : @trigtech/any.m and @trigtech/feval.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and
            The Chebfun Developers.
        """
        if dim is not None and dim not in (1, 2):
            raise ValueError("TRIGTECH:any:dim: DIM input must be 1 or 2.")

        if dim == 2:
            # The MATLAB implementation samples at this fixed interior point
            # and stores the result as a constant coefficient.
            if self.isempty():
                # MATLAB feval(empty, x) is [], and any([]) is scalar false.
                return Trigtech(
                    coeffs=jnp.asarray([[False]]),
                    is_real=True,
                    ishappy=self.ishappy,
                )
            sampled = jnp.asarray(self(jnp.asarray(0.1273881594)))
            flags = (sampled != 0) & ~jnp.isnan(sampled)
            flag = jnp.any(flags)
            return Trigtech(
                coeffs=jnp.asarray([[flag]], dtype=jnp.bool_),
                is_real=True,
                ishappy=self.ishappy,
            )

        values = self.values
        if values.ndim == 0:
            axis = None
        elif values.shape == (0, 0):
            # MathWorks documents any(0-by-0) as scalar logical false.
            return jnp.asarray(False)
        else:
            axis = next((i for i, n in enumerate(values.shape) if n != 1), 0)

        flags = (values != 0) & ~jnp.isnan(values)
        result = jnp.any(flags, axis=axis)
        if values.ndim > 1 and axis == 1:
            result = jnp.squeeze(result)
        return result

    def vscale_columns(self) -> jax.Array:
        """Per-column vertical scale (MATLAB ``vscale`` returns a 1xN row
        for an array-valued tech).  ``vscale`` (the scalar property) is the
        max over all columns.

        Provenance
        ----------
        MATLAB source : @trigtech/vscale.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return jnp.empty((0,), dtype=jnp.float64)
        v = jnp.abs(self.values)
        if v.ndim == 1:
            return jnp.max(v, keepdims=True)
        return jnp.max(v, axis=0)

    # ------------------------------------------------------------------
    # Logical predicates on the values (array-valued)
    # ------------------------------------------------------------------

    def iszero(self) -> jax.Array:
        """Per-column test: identically zero (and free of NaN).

        MATLAB ``@trigtech/iszero.m``::

            out = ~any(f.values, 1) & ~any(isnan(f.values), 1);

        Provenance
        ----------
        MATLAB source : @trigtech/iszero.m
        Chebfun commit: 7574c77
        """
        v = self.values
        v2 = v if v.ndim == 2 else v[:, None]
        zero = ~jnp.any(v2 != 0, axis=0)
        no_nan = ~jnp.any(jnp.isnan(v2), axis=0)
        out = zero & no_nan
        return out[0] if v.ndim == 1 else out

    def isnan(self) -> bool:
        """True if the tech has any NaN value (MATLAB ``isnan``).

        Mirrors ``@trigtech/isnan.m`` (``any(isnan(f.values(:)))``): the
        function values are recovered from the coefficients and tested for
        NaN.  A NaN Fourier coefficient (e.g. from ``f ./ 0``, which yields
        ``0/0 = NaN`` in the non-constant coefficients) makes the inverse
        transform NaN everywhere, so this flags it -- including the case
        where Inf coefficients are also present.

        Provenance
        ----------
        MATLAB source : @trigtech/isnan.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        values = trig_coeffs2vals(self.coeffs)
        return bool(jnp.any(jnp.isnan(jnp.asarray(values))))

    def isinf(self) -> bool:
        """True if the tech has any infinite value (MATLAB ``isinf``).

        An Inf value maps to Inf Fourier coefficients under the FFT.

        Provenance
        ----------
        MATLAB source : @trigtech/isinf.m
        Chebfun commit: 7574c77
        """
        return bool(jnp.any(jnp.isinf(jnp.asarray(self.coeffs))))

    def isfinite(self) -> bool:
        """True if the tech is everywhere finite (MATLAB ``isfinite``).

        Provenance
        ----------
        MATLAB source : @trigtech/isfinite.m
        Chebfun commit: 7574c77
        """
        return bool(jnp.all(jnp.isfinite(jnp.asarray(self.coeffs))))

    def isreal(self) -> bool:
        """True if the underlying function is real-valued (MATLAB
        ``isreal``); for array-valued techs, True only if every column is
        real.

        Provenance
        ----------
        MATLAB source : @trigtech/isreal.m
        Chebfun commit: 7574c77
        """
        return bool(self.is_real)

    # ------------------------------------------------------------------
    # Sign / poly
    # ------------------------------------------------------------------

    def sign(self) -> "Trigtech":
        """Signum of a root-free TRIGTECH.

        For a real-valued tech, samples at ``[-1, x0, 1]`` and returns the
        sign of the column mean as a constant tech.  For a complex-valued
        tech, returns ``f ./ |f|`` (re-approximated).

        Provenance
        ----------
        MATLAB source : @trigtech/sign.m
        Chebfun commit: 7574c77
        """
        if self.is_real:
            arbitrary = 0.1273881594
            x = jnp.array([-1.0, arbitrary, 1.0], dtype=jnp.float64)
            fx = jnp.asarray(self(x))
            meanfx = jnp.mean(jnp.real(fx), axis=0)
            s = jnp.sign(meanfx)
            c = jnp.atleast_1d(s.astype(jnp.complex128))
            if self.coeffs.ndim == 2 and c.ndim == 1:
                c = c[None, :]
            return Trigtech(coeffs=c, is_real=True, ishappy=True)
        return Trigtech.from_function(
            lambda t: (lambda v: v / jnp.abs(v))(self(t)))

    def poly(self) -> jax.Array:
        """Polynomial (Laurent) coefficients — the transpose of the
        Fourier coefficients.  For an array-valued tech the rows of the
        output correspond to the columns of ``f``.

        Provenance
        ----------
        MATLAB source : @trigtech/poly.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return jnp.array([], dtype=jnp.complex128)
        return self.coeffs.T

    # ------------------------------------------------------------------
    # Circular convolution
    # ------------------------------------------------------------------

    def circconv(self, other: "Trigtech") -> "Trigtech":
        """Circular (periodic) convolution of two scalar-valued techs.

        Convolution is multiplication of the Fourier coefficients; the
        two techs are first prolonged to a common length.

        Provenance
        ----------
        MATLAB source : @trigtech/circconv.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or other.isempty():
            return Trigtech.empty()
        if self.num_columns > 1 or other.num_columns > 1:
            raise ValueError(
                "CHEBFUN:TRIGTECH:conv:array "
                "No support for array-valued TRIGTECH objects.")
        # Fourier-mode orthogonality on [-1, 1] gives
        # (f * g)(x) = sum_k 2 a_k b_k e^{i pi k x}, i.e. the convolution
        # is coefficient multiplication scaled by the mode norm 2.
        n = max(self.n, other.n)
        fp = _trig_prolong_coeffs(self.coeffs, n)
        gp = _trig_prolong_coeffs(other.coeffs, n)
        c = 2.0 * fp * gp
        new_is_real = self.is_real and other.is_real
        if new_is_real:
            # Enforce the conjugate symmetry of a real result.
            v = jnp.real(trig_coeffs2vals(c)).astype(jnp.complex128)
            c = trig_vals2coeffs(v)
        h = Trigtech(coeffs=c, is_real=new_is_real,
                     ishappy=self.ishappy and other.ishappy)
        return h.simplify()

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    @classmethod
    def horzcat(cls, *techs) -> "Trigtech":
        """Horizontally concatenate techs into one array-valued tech,
        dropping empty inputs (MATLAB ``[A B ...]``).

        Provenance
        ----------
        MATLAB source : @trigtech/horzcat.m
        Chebfun commit: 7574c77
        """
        techs = list(techs)
        nonempty = [t for t in techs if not t.isempty()]
        if not nonempty:
            return techs[0]
        return cls.cell2mat(nonempty)

    # ------------------------------------------------------------------
    # QR factorisation (array-valued)
    # ------------------------------------------------------------------

    def qr(self, mode: str = "matrix", want_e: bool = False):
        """QR factorisation of an array-valued tech: ``f = Q R`` with ``Q``
        orthonormal in the continuous L^2 inner product on [-1, 1] and
        ``R`` upper-triangular.

        Parameters
        ----------
        mode : {'matrix', 'vector'}
            Form of the optional permutation output ``E`` (identity here,
            as JAX lacks column-pivoted QR).
        want_e : bool
            If True, also return the permutation ``E`` (identity).

        Returns
        -------
        (Q, R) or (Q, R, E)

        Provenance
        ----------
        MATLAB source : @trigtech/qr.m (built-in / weighted discrete QR)
        Chebfun commit: 7574c77
        """
        if self.isempty():
            empty = jnp.empty((0, 0), dtype=jnp.float64)
            return (self, empty, empty) if want_e else (self, empty)
        if self.num_columns == 1:
            R = jnp.sqrt(self.innerProduct(self))
            Q, R = self / R, jnp.reshape(R, (1, 1))
        else:
            Q, R = _trig_qr_builtin_jax(self)
        if want_e:
            # Inherited Python compatibility only: native three-output QR
            # pivots columns. This identity adapter does not qualify that API.
            E = jnp.arange(self.num_columns) if mode == "vector" else jnp.eye(self.num_columns)
            return Q, R, E
        return Q, R

    # ------------------------------------------------------------------
    # Left / right matrix division (least squares)
    # ------------------------------------------------------------------

    @staticmethod
    def mldivide(A, B) -> jax.Array:
        """``A \\ B``: continuous-L^2 least-squares solution of ``A X = B``
        for two techs, returning the numeric coefficient matrix ``X``.

        Provenance
        ----------
        MATLAB source : @trigtech/mldivide.m
        Chebfun commit: 7574c77
        """
        if not (isinstance(A, Trigtech) and isinstance(B, Trigtech)):
            raise ValueError(
                "CHEBFUN:TRIGTECH:mldivide:trigtechMldivideUnknown")
        Q, R = A.qr()
        ip = Q.innerProduct(B)
        ip = jnp.reshape(jnp.asarray(ip, dtype=jnp.complex128),
                         (R.shape[0], -1))
        X = jnp.linalg.solve(R, ip)
        if A.is_real and B.is_real:
            X = jnp.real(X)
        return X

    @staticmethod
    def mrdivide(A, B):
        """``A / B``: right matrix divide.  Divides a tech ``A`` by a scalar
        or matrix ``B`` (least squares), or a numeric ``A`` by a tech ``B``.

        Provenance
        ----------
        MATLAB source : @trigtech/mrdivide.m
        Chebfun commit: 7574c77
        """
        A_is_tech = isinstance(A, Trigtech)
        B_is_tech = isinstance(B, Trigtech)
        if A_is_tech and B_is_tech:
            raise ValueError("CHEBFUN:TRIGTECH:mrdivide:trigtechDivTrigtech")

        if A_is_tech and not B_is_tech:
            if not _is_double(B):
                raise ValueError("CHEBFUN:TRIGTECH:mrdivide:badArg")
            Bd = jnp.asarray(B)
            if Bd.ndim < 2:
                Bd_cols = Bd.size
            else:
                Bd_cols = Bd.shape[1]
            if Bd.size > 1 and Bd_cols != A.num_columns:
                raise ValueError("CHEBFUN:TRIGTECH:mrdivide:size")
            if not bool(jnp.any(Bd != 0)):
                z = jnp.full((1, A.num_columns), jnp.nan, dtype=jnp.complex128)
                return Trigtech.from_values(z)
            if Bd.size == 1:
                return A * (1.0 / Bd.reshape(()))
            # Matrix least squares: X = Q * (R / B).
            Q, R = A.qr()
            Bm = Bd.astype(jnp.complex128)
            # R / B  ==  (B.' \ R.').'
            Y = jnp.linalg.lstsq(Bm.T, R.T)[0].T
            return Q @ Y

        if not A_is_tech and B_is_tech:
            if not _is_double(A):
                raise ValueError("CHEBFUN:TRIGTECH:mrdivide:badArg")
            Am = jnp.atleast_2d(jnp.asarray(A, dtype=jnp.complex128))
            Q, R = B.qr()
            # A / R  ==  (R.' \ A.').'
            AR = jnp.linalg.lstsq(R.T, Am.T)[0].T
            return Q @ AR.T
        raise ValueError("CHEBFUN:TRIGTECH:mrdivide:badArg")


@eqx.filter_jit
def _trig_qr_builtin_jax(f: Trigtech):
    """Two-output @trigtech/qr.m at 7574c77, including source cache order.

    JAX QR supplies the dense factorization. The source multiplies BOTH
    factors by sign(diag(R)); do not substitute a conjugated phase here.
    The one-column zero division is intentionally distinct from Chebfun QR.
    """
    nf, mf = f.n, f.num_columns
    n = max(nf, mf)
    prolonged = f.prolong(n)
    values = prolonged.values
    if f.is_real:
        values = jnp.real(values)
    q, r = jnp.linalg.qr(values, mode="reduced")
    diagonal = jnp.diag(r)
    phase = jnp.sign(diagonal)
    phase = jnp.where(phase == 0, 1, phase)
    q = q * phase[None, :]
    r = phase[:, None] * r
    weight = jnp.sqrt(jnp.asarray(2.0 / n))
    q = q / weight
    r = weight * r
    result = Trigtech(coeffs=_trig_vals2coeffs_impl(q),
                     real_columns=(f.is_real,) * mf, ishappy=f.ishappy,
                     _values=q)
    result = result.prolong(nf)
    return Trigtech(coeffs=result.coeffs, real_columns=result.real_columns,
                    ishappy=result.ishappy, _values=result.values), r


@eqx.filter_jit
def _trig_qr_collate(techs):
    """Native @trigtech/horzcat.m 7574c77 for nonempty QR columns.

    Preserve the first column's happiness, every realness flag, and each
    prolonged value cache. This avoids the unrelated cell2mat policy.
    """
    n = max(t.n for t in techs)
    prolonged = tuple(t.prolong(n) for t in techs)
    return Trigtech(
        coeffs=jnp.column_stack([t.coeffs for t in prolonged]),
        real_columns=tuple(flag for t in techs for flag in t.real_columns),
        ishappy=techs[0].ishappy,
        _values=jnp.column_stack([t.values for t in prolonged]))

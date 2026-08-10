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

    Notes
    -----
    Symmetry is enforced exactly, ``x = (x - flip(x))/2``, as in MATLAB
    trigpts.m -- downstream bit-exact symmetry detection (Hermitian /
    skew-Hermitian value tests in vals2coeffs) depends on sampled values
    of even/odd functions being exactly palindromic.
    """
    x = jnp.linspace(-1.0, 1.0, n + 1, dtype=jnp.float64)
    x = (x - x[::-1]) / 2.0
    return x[:-1]


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
    co = jnp.broadcast_to(a[n_h - 1], out_shape)
    si = jnp.broadcast_to(b[n_h - 1], out_shape)

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
    xE = x.astype(jnp.float64).reshape(
        x.shape + (1,) * (coeffs_cx.ndim - 1))
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
    # (trailing singleton axes broadcast over array-valued columns)
    safe_wn = jnp.where(wavenumbers == 0, 1.0, wavenumbers)
    int_factor = jnp.where(
        wavenumbers == 0,
        0.0 + 0j,
        1.0 / (1j * jnp.pi * safe_wn + 0j),
    )
    int_factor = int_factor.reshape(
        (n_exp,) + (1,) * (c_work.ndim - 1))
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
    if n_out <= 0 or coeffs.shape[0] == 0:
        # MATLAB: trigcoeffs(f, 0) and prolong of an empty are empty.
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
    """Alias Fourier coefficients on the equispaced grid to length ``m``.

    Direct port of ``@trigtech/alias.m``.  If ``m`` exceeds ``len(coeffs)``
    the coefficients are zero-padded (with the correct even-``n`` Nyquist
    symmetry handling); otherwise higher modes are folded down onto the
    modes indistinguishable from them on the ``m``-point grid.  Aliasing to
    length ``m`` reproduces exactly the coefficients of the interpolant on
    the ``m``-point equispaced grid.

    Not JIT-safe (Python-int branching + loop-based accumulation); uses
    numpy for the folding, mirroring the coefficient-surgery helpers in
    ``chebtech``.
    """
    import numpy as np

    orig = jnp.asarray(coeffs)
    twod = orig.ndim == 2
    c = np.asarray(orig).astype(np.complex128)
    if not twod:
        c = c.reshape(-1, 1)
    else:
        c = c.copy()
    n = c.shape[0]
    cols = c.shape[1]

    if m == n:
        # Same length: identity (the fold-down path assumed m < n and
        # produced empty accumulators for n == m == 1).
        return orig

    if m > n:
        k = int(np.ceil((m - n) / 2))
        z = np.zeros((k, cols), dtype=c.dtype)
        if n % 2 == 0:
            # Account for the even-n asymmetry (the cos(N/2) coeff) using
            # the symmetry of the complex exponential.
            c = np.concatenate([c[:1] / 2, c[1:n], c[:1] / 2], axis=0)
            c = np.concatenate([z, c, z[: z.shape[0] - 1]], axis=0)
            if m % 2 == 1:
                c = c[1:]
        else:
            c = np.concatenate([z, c, z], axis=0)
            if m % 2 == 0:
                c = c[:-1]
        out = jnp.asarray(c, dtype=jnp.complex128)
        return out if twod else out.reshape(-1)

    # Make n odd by exploiting symmetry, which simplifies the cases below.
    if n % 2 == 0:
        c = c.copy()
        c[0] = 0.5 * c[0]
        c = np.concatenate([c, c[:1]], axis=0)
        n = n + 1

    if m % 2 == 1:
        if m == 1:
            n2 = (n - 1) // 2
            const = c[n2]
            pos = c[n2 - 1::-1]
            neg = c[n2 + 1:n]
            e = np.ones(int(np.ceil((n - 1) / 2)), dtype=c.dtype)
            e[0::2] = -1
            c = (const + (e @ pos + e @ neg)).reshape(1, cols)
        else:
            m2 = (m - 1) // 2
            n2 = (n - 1) // 2
            al = c[n2 - m2:n2 + m2 + 1].copy()
            for j in range(-n2, -m2):
                k = int(np.mod(j + m2 + 1, -m)) + m2
                sgn = (-1) ** ((j + k) % 2)
                al[k + m2] = al[k + m2] + sgn * c[j + n2]
                al[-k + m2] = al[-k + m2] + sgn * c[-j + n2]
            c = al
    else:
        m2 = m // 2
        n2 = (n - 1) // 2
        al = c[n2 - m2:n2 + m2].copy()
        al = np.concatenate([al, -al[:1]], axis=0)
        for j in range(-n2, -m2 + 1):
            k = int(np.mod(j + m2, -m)) + m2
            al[k + m2] = al[k + m2] + c[j + n2]
            al[-k + m2] = al[-k + m2] + c[-j + n2]
        # Collapse the +m/2 exp term back onto the -m/2 one and drop the tail.
        al[0] = al[0] + al[-1]
        al = al[:-1]
        c = al

    out = jnp.asarray(c, dtype=jnp.complex128)
    return out if twod else out.reshape(-1)


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
    if chop_cutoff <= 1:
        return 1
    # Reverse the kron expansion: (cutoff - 1) / 2 pairs after the first element
    paired_idx = (chop_cutoff + 1) // 2  # = ceil(chop_cutoff / 2)
    # paired_idx modes (including constant) -> n_keep = 2*paired_idx - 1 (odd, centered)
    n_keep = max(1, 2 * paired_idx - 1)
    return min(n_keep, n_full)


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
        obj = object.__new__(cls)
        object.__setattr__(obj, "_is_empty_object", True)
        return obj

    def isempty(self) -> bool:
        """True for the empty Trigtech (MATLAB ``isempty``).

        Provenance
        ----------
        MATLAB source : @trigtech/isempty.m
        Chebfun commit: 7574c77
        """
        return getattr(self, "_is_empty_object", False)

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
            # A real-valued function has conjugate-symmetric Fourier
            # coefficients: c_{-k} = conj(c_k). The previous hardcoded
            # True made every complex trig function evaluate to its real
            # part after any coefficient-space rebuild (e.g. diff).
            n = coeffs.shape[0]
            if n % 2 == 1:
                sym_err = jnp.max(jnp.abs(coeffs - jnp.conj(coeffs[::-1])))
            else:
                # even length: modes [-n/2, ..., n/2-1]; the unpaired
                # -n/2 (Nyquist) mode must itself be real.
                sym_err = jnp.maximum(
                    jnp.max(jnp.abs(coeffs[1:] - jnp.conj(coeffs[1:][::-1]))),
                    jnp.abs(jnp.imag(coeffs[0])),
                )
            scale = jnp.maximum(jnp.max(jnp.abs(coeffs)), 1e-300)
            flag = sym_err <= 1e-13 * scale
            if isinstance(flag, jax.core.Tracer):
                # Under jit tracing the value is unavailable; keep the
                # legacy default (real). Library-internal rebuilds that
                # need accurate inference (piece diff/cumsum) run eagerly.
                is_real = True
            else:
                is_real = bool(flag)
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
        values, is_real = _sample_as_trig_dtype(f, x)
        c = trig_vals2coeffs(values)
        return cls(coeffs=c, is_real=is_real, ishappy=True)

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
            values, is_real = _sample_as_trig_dtype(f, x)
            c = trig_vals2coeffs(values)
            vscale = max(vscale, float(jnp.max(jnp.abs(values))))

            # Check happiness using paired coefficient magnitudes
            # (per-column max cutoff for array-valued sampling)
            cutoff, chop_len = _trig_chop_cutoff(c)
            ishappy = cutoff < chop_len

            if ishappy:
                # Map cutoff back to number of Fourier modes
                n_keep = _chop_cutoff_to_ncoeffs(cutoff, n)
                # Ensure odd (symmetric spectrum)
                if n_keep % 2 == 0:
                    n_keep = max(1, n_keep - 1)
                c_keep = _trig_prolong_coeffs(c, n_keep)
                candidate = cls(coeffs=c_keep, is_real=is_real,
                                ishappy=True)
                # Sample test: guard against coarse-grid aliasing (a
                # sparse high-frequency spectrum can alias to a
                # low-frequency one on the current grid and chop early).
                # Evaluate f and the candidate at the grid MIDPOINTS
                # (off-grid); if they disagree, the grid is too coarse.
                # Fix by Claude Opus 4.8.
                # Irrational fraction of the grid spacing (2/n) so the
                # test points never coincide with an aliasing pattern.
                x_test = x + (2.0 / n) * 0.414213562373095
                f_test, _ = _sample_as_trig_dtype(f, x_test)
                cand_test = candidate(x_test)
                tol_abs = 1e6 * _EPS * max(vscale, 1.0)
                err = float(jnp.max(jnp.abs(
                    jnp.asarray(cand_test) - jnp.asarray(f_test))))
                if err <= tol_abs:
                    return candidate

        # Did not converge
        warnings.warn(
            f"Trigtech.from_function: function did not converge with "
            f"{2**maxpow2} points. Returning unhappy representation.",
            stacklevel=2,
        )
        values, is_real = _sample_as_trig_dtype(f, trigpts(2**maxpow2))
        c_final = trig_vals2coeffs(values)
        return cls(coeffs=c_final, is_real=is_real, ishappy=False)

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
        if not isinstance(x, jax.core.Tracer) and \
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
            if self.is_real:
                values = jnp.real(values).astype(jnp.float64)
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
        if nold == 0:
            # MATLAB @trigtech/simplify.m leaves an empty trigtech alone.
            return self
        N = max(17, round(nold * 1.25 + 5))
        prolonged = self.prolong(N)

        # Round-trip to create slight noise on the plateau
        v = trig_coeffs2vals(prolonged.coeffs)
        c_noisy = trig_vals2coeffs(v)

        cutoff, chop_len = _trig_chop_cutoff(c_noisy, tol)
        cutoff = min(cutoff, chop_len)

        n_keep = _chop_cutoff_to_ncoeffs(cutoff, N)
        n_keep = min(n_keep, nold)
        if n_keep % 2 == 0:
            n_keep = max(1, n_keep - 1)

        new_coeffs = _trig_prolong_coeffs(self.coeffs, n_keep)
        return Trigtech(coeffs=new_coeffs, is_real=self.is_real, ishappy=self.ishappy)

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
            # k-th finite differences ACROSS the columns of an
            # array-valued tech (MATLAB diff(f, k, 2)); empty for
            # scalar-valued input.
            if self.coeffs.ndim == 1:
                return Trigtech(
                    coeffs=jnp.zeros((0,), dtype=self.coeffs.dtype),
                    is_real=self.is_real, ishappy=self.ishappy)
            return Trigtech(coeffs=jnp.diff(self.coeffs, n=k, axis=1),
                            is_real=self.is_real, ishappy=self.ishappy)
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
        # max over columns for array-valued techs (every column must
        # have zero mean)
        c0_mag = float(jnp.max(jnp.abs(self.coeffs[c0_idx])))
        vs = self.vscale if self.vscale > 0 else 1.0
        if c0_mag > 10.0 * vs * _EPS:
            raise ValueError(
                f"Trigtech.cumsum: function does not have zero mean "
                f"(|c_0| = {c0_mag:.3e}). The antiderivative of a non-zero-mean "
                f"periodic function is not periodic."
            )
        bc = _trig_cumsum_coeffs(self.coeffs)
        return Trigtech(coeffs=bc, is_real=self.is_real, ishappy=self.ishappy)

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
        if self.is_real and other.is_real:
            out = jnp.real(out).astype(jnp.complex128)
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

    def compose(self, op) -> "Trigtech":
        """Re-approximate op(f) adaptively (MATLAB @trigtech/compose.m;
        added by Claude Fable 5)."""
        return Trigtech.from_function(lambda x: op(self(x)))

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
                            is_real=self.is_real, ishappy=self.ishappy)
        s = _trig_definite_integral(self.coeffs)
        if self.is_real:
            return jnp.real(s).astype(jnp.float64)
        return s

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

        def _one(col):
            if complex:
                simp = Trigtech.from_coeffs(col).simplify()
                return _np.asarray(_trig_roots_complex(simp.coeffs, prune=True))
            return _np.asarray(_trig_roots(col))

        if self.coeffs.ndim == 2:
            cols = [_one(self.coeffs[:, j])
                    for j in range(self.coeffs.shape[1])]
            nmax = max((len(c) for c in cols), default=0)
            dtype = _np.complex128 if complex else _np.float64
            out = _np.full((nmax, len(cols)), _np.nan, dtype=dtype)
            for j, c in enumerate(cols):
                out[: len(c), j] = c
            return jnp.asarray(out)
        return jnp.asarray(_one(self.coeffs))

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
    ) -> tuple[bool, int]:
        """Standard happiness check for trigonometric adaptive construction.

        Optionally performs a sample test (MATLAB ``pref.sampleTest``):
        evaluates the operator ``op`` and the trigonometric interpolant at two
        off-grid points and, if they disagree by more than
        ``sqrt(max(tol, eps)) * vscale``, declares the representation unhappy
        and reverts the cutoff to the full length.  This rejects the
        aliasing-fooled "happy" case that the coefficient chop alone misses.

        Parameters
        ----------
        coeffs : jax.Array, shape (N,) complex
        values : jax.Array, shape (N,)
        op : callable or None, optional
            Original function handle for the sample test.  When ``None`` the
            sample test is skipped (MATLAB ``pref.sampleTest = 0``).
        tol : float or None
        vscale : float, default 0.0

        Returns
        -------
        (ishappy, cutoff) : (bool, int)

        Provenance
        ----------
        MATLAB source : @trigtech/happinessCheck.m, @trigtech/standardCheck.m,
            @trigtech/sampleTest.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        if tol is None:
            tol = _EPS

        n = coeffs.shape[0]
        vscale_local = float(jnp.max(jnp.abs(values)))
        vscale = max(vscale, vscale_local)

        if vscale_local > 0:
            scaled_tol = tol * max(1.0, vscale / vscale_local)
        else:
            scaled_tol = tol

        cutoff, chop_len = _trig_chop_cutoff(coeffs, scaled_tol)
        ishappy = cutoff < chop_len

        # Sample test (MATLAB @trigtech/sampleTest.m): compare the full
        # interpolant against the operator at two fixed off-grid points.
        if ishappy and op is not None:
            xeval = jnp.array(
                [-0.357998918959666, 0.036785641195074], dtype=jnp.float64
            )
            v_fun = _trig_eval(coeffs, xeval, is_real=False)
            v_op = jnp.asarray(op(xeval), dtype=jnp.complex128)
            err = float(jnp.max(jnp.abs(v_op - v_fun)))
            sample_tol = _np.sqrt(max(_EPS, tol)) * vscale
            if err > sample_tol:
                ishappy = False
                cutoff = n  # revert to size(f.values, 1)
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
            new_is_real = self.is_real and other.is_real
            return Trigtech(
                coeffs=fc + gc,
                is_real=new_is_real,
                ishappy=self.ishappy and other.ishappy,
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
            new_is_real = self.is_real and bool(
                jnp.isrealobj(jnp.asarray(other))
                or not jnp.any(jnp.imag(s)))
            return Trigtech(coeffs=c, is_real=new_is_real,
                            ishappy=self.ishappy)

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
        return Trigtech(coeffs=-self.coeffs, is_real=self.is_real, ishappy=self.ishappy)

    def __pos__(self) -> "Trigtech":
        return self

    def __mul__(self, other) -> "Trigtech":
        """Pointwise multiplication via physical-space grid.

        Provenance
        ----------
        MATLAB source : @chebtech/times.m (analogous)
        """
        if self.isempty() or (isinstance(other, Trigtech)
                              and other.isempty()):
            # MATLAB @trigtech/times.m: empty argument -> empty result.
            return Trigtech.empty()
        if isinstance(other, Trigtech):
            # Multiply in physical space to avoid aliasing
            n = self.n + other.n
            if n % 2 == 0:
                n += 1
            x = trigpts(n)
            fv = _trig_eval(self.coeffs, x, self.is_real)
            gv = _trig_eval(other.coeffs, x, other.is_real)
            new_is_real = self.is_real and other.is_real
            # scalar-column * array-valued broadcasts via a trailing
            # column axis (MATLAB @chebtech/times.m semantics)
            if fv.ndim != gv.ndim:
                if fv.ndim == 1:
                    fv = fv[:, None]
                if gv.ndim == 1:
                    gv = gv[:, None]
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
            # Adaptive re-construction so the quotient is fully resolved
            # (MATLAB: compose(f, @rdivide, g)).
            return Trigtech.from_function(
                lambda x: _trig_eval(self.coeffs, x, self.is_real)
                / _trig_eval(other.coeffs, x, other.is_real)
            )
        else:
            # A complex divisor clears is_real (the imaginary part was
            # silently dropped before -- Fable 5, flip-roots audit).
            s = jnp.asarray(other, dtype=jnp.complex128)
            new_is_real = self.is_real and bool(
                jnp.isrealobj(jnp.asarray(other))
                or not jnp.any(jnp.imag(s)))
            return Trigtech(
                coeffs=self.coeffs / s,
                is_real=new_is_real,
                ishappy=self.ishappy,
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
        """Raise to a power."""
        if isinstance(exponent, int) and exponent >= 0:
            if exponent == 0:
                # ones with the same column count (array-valued f**0
                # keeps m columns, MATLAB power.m)
                c = jnp.ones((1,) + self.coeffs.shape[1:],
                             dtype=jnp.complex128)
                return Trigtech(coeffs=c, is_real=True, ishappy=True)
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        else:
            # Fractional power: adaptive re-construction (MATLAB compose)
            e = jnp.asarray(exponent, dtype=jnp.float64)
            return Trigtech.from_function(
                lambda x: _trig_eval(self.coeffs, x, self.is_real) ** e
            )

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
        A = jnp.asarray(other)
        if A.size == 0:
            # MATLAB @trigtech/mtimes.m: f * [] is empty.
            return Trigtech.empty()
        c = self.coeffs if self.coeffs.ndim == 2 else self.coeffs[:, None]
        return Trigtech(coeffs=c @ A.astype(jnp.complex128),
                        is_real=self.is_real and bool(jnp.isrealobj(A)),
                        ishappy=self.ishappy)

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
        return Trigtech(coeffs=self.coeffs[:, ::-1],
                        is_real=self.is_real, ishappy=self.ishappy)

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
        return Trigtech(coeffs=new_c, is_real=self.is_real,
                        ishappy=self.ishappy)

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
        # Exact coefficient-space real part: conj(f) has coeffs
        # flip(conj(c)) (see conj), so Re(f) = (f + conj(f))/2 has coeffs
        # (c + flip(conj(c)))/2.  Avoids an FFT round-trip.
        c = self.coeffs
        cr = 0.5 * (c + jnp.flip(jnp.conj(c), axis=0))
        scale = max(float(jnp.max(jnp.abs(trig_coeffs2vals(c)))), 1.0)
        if float(jnp.max(jnp.abs(cr))) <= 1e2 * _EPS * scale:
            z = jnp.zeros((1,) + self.coeffs.shape[1:],
                          dtype=jnp.complex128)
            return Trigtech(coeffs=z, is_real=True, ishappy=True)
        out = Trigtech(coeffs=cr, is_real=True, ishappy=self.ishappy)
        # Simplify: the imaginary part may have inflated the length.
        return out.simplify()

    def imag(self) -> "Trigtech":
        """Imaginary part (a zero tech if the input was real).

        Provenance
        ----------
        MATLAB source : @trigtech/imag.m
        Chebfun commit: 7574c77
        """
        # MATLAB checks the isReal FLAG (not values): a real tech has
        # an exactly-zero imaginary part by definition.
        if self.is_real:
            z = jnp.zeros((1,) + self.coeffs.shape[1:],
                          dtype=jnp.complex128)
            return Trigtech(coeffs=z, is_real=True, ishappy=True)
        # Exact coefficient-space imaginary part: Im(f) = (f - conj(f))/2i
        # has coeffs (c - flip(conj(c)))/(2i).
        c = self.coeffs
        ci = (c - jnp.flip(jnp.conj(c), axis=0)) / (2j)
        out = Trigtech(coeffs=ci, is_real=True, ishappy=self.ishappy)
        return out.simplify()

    def conj(self) -> "Trigtech":
        """Complex conjugate (via conjugated grid values).

        Provenance
        ----------
        MATLAB source : @trigtech/conj.m
        Chebfun commit: 7574c77
        """
        if self.is_real:
            return self
        v = jnp.conj(trig_coeffs2vals(self.coeffs))
        return Trigtech(coeffs=trig_vals2coeffs(v),
                        is_real=self.is_real, ishappy=self.ishappy)

    def extract_column(self, j: int) -> "Trigtech":
        """Return column ``j`` (0-based) of an array-valued tech as a
        scalar-valued Trigtech (MATLAB ``extractColumns``)."""
        c = self.coeffs if self.coeffs.ndim == 2 else self.coeffs[:, None]
        return Trigtech(coeffs=c[:, j], is_real=self.is_real,
                        ishappy=self.ishappy)

    def minandmax(self):
        """Global minimum and maximum on [-1, 1].

        Extrema of a smooth periodic function occur at the roots of its
        derivative.  We locate those critical points by finding the real
        roots of the derivative trig series (for a complex-valued tech, of
        ``|f|^2``) and evaluate ``f`` there, matching MATLAB's chebtech
        delegation while avoiding the expensive adaptive re-construction.
        Array-valued techs return a 2 x m result, one column per column.

        Provenance
        ----------
        MATLAB source : @trigtech/minandmax.m
        Chebfun commit: 7574c77
        """
        if self.coeffs.ndim == 2:
            per_col = [_trig_minandmax_scalar(self.extract_column(j))
                       for j in range(self.coeffs.shape[1])]
            min_val = jnp.stack([p[0][0] for p in per_col])
            min_pos = jnp.stack([p[0][1] for p in per_col])
            max_val = jnp.stack([p[1][0] for p in per_col])
            max_pos = jnp.stack([p[1][1] for p in per_col])
            return (min_val, min_pos), (max_val, max_pos)
        return _trig_minandmax_scalar(self)

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
            j += s
            out.append(Trigtech(
                coeffs=block[:, 0] if s == 1 else block,
                is_real=self.is_real, ishappy=self.ishappy))
        return out

    @classmethod
    def cell2mat(cls, techs) -> "Trigtech":
        """Horizontally concatenate techs into one array-valued tech.

        Provenance
        ----------
        MATLAB source : @trigtech/cell2mat.m
        Chebfun commit: 7574c77
        """
        n = max(t.n for t in techs)
        cols = []
        for t in techs:
            c = _trig_prolong_coeffs(t.coeffs, n)
            cols.append(c if c.ndim == 2 else c[:, None])
        return cls(coeffs=jnp.concatenate(cols, axis=1),
                   is_real=all(t.is_real for t in techs),
                   ishappy=all(t.ishappy for t in techs))

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
            return Trigtech(coeffs=fc[:, keep],
                            is_real=self.is_real, ishappy=self.ishappy)
        n = max(fc.shape[0], g.n)
        fc = _trig_prolong_coeffs(fc, n)
        gc = _trig_prolong_coeffs(g.coeffs, n)
        gc = gc if gc.ndim == 2 else gc[:, None]
        out = fc.at[:, jnp.asarray(cols)].set(gc)
        return Trigtech(coeffs=out,
                        is_real=self.is_real and g.is_real,
                        ishappy=self.ishappy and g.ishappy)

    # ------------------------------------------------------------------
    # Size / scale introspection (array-valued)
    # ------------------------------------------------------------------

    @property
    def num_columns(self) -> int:
        """Number of columns (1 for scalar-valued techs)."""
        return 1 if self.coeffs.ndim == 1 else self.coeffs.shape[1]

    def size(self, dim: int | None = None):
        """MATLAB ``size(f)``: (n_rows, n_cols); ``size(f, 2)`` is the
        column count.  ``n_rows`` is the number of Fourier coefficients."""
        shape = (self.n, self.num_columns)
        if dim is None:
            return shape
        return shape[dim - 1]

    def vscale_columns(self) -> jax.Array:
        """Per-column vertical scale (MATLAB ``vscale`` returns a 1xN row
        for an array-valued tech).  ``vscale`` (the scalar property) is the
        max over all columns.

        Provenance
        ----------
        MATLAB source : @trigtech/vscale.m
        Chebfun commit: 7574c77
        """
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
        import numpy as np
        mf = self.num_columns
        nf = self.n
        if mf == 1:
            R = jnp.sqrt(self.innerProduct(self))
            Q = self / R
            if want_e:
                E = jnp.array([0]) if mode == "vector" else jnp.eye(1)
                return Q, jnp.reshape(R, (1, 1)), E
            return Q, jnp.reshape(R, (1, 1))

        isreal = self.is_real
        n = max(nf, mf)
        fp = _trig_prolong_coeffs(self.coeffs, n)  # (n, mf)
        vals = np.asarray(trig_coeffs2vals(fp))
        if isreal:
            vals = np.real(vals)
        Qm, Rm = np.linalg.qr(vals, mode="reduced")  # (n, mf), (mf, mf)
        # Enforce diag(R) >= 0.
        s = np.sign(np.diag(Rm))
        s[s == 0] = 1
        Qm = Qm * s[np.newaxis, :]
        Rm = s[:, np.newaxis] * Rm
        # Scale by the trapezoid weight sqrt(2/n).
        W = np.sqrt(2.0 / n)
        Qm = Qm / W
        Rm = W * Rm
        Qc = trig_vals2coeffs(jnp.asarray(Qm, dtype=jnp.complex128))
        Q = Trigtech(coeffs=Qc, is_real=isreal, ishappy=self.ishappy)
        Q = Q.prolong(nf)
        R = jnp.asarray(Rm, dtype=jnp.complex128)
        if isreal:
            R = jnp.real(R).astype(jnp.complex128)
        if want_e:
            E = jnp.arange(mf) if mode == "vector" else jnp.eye(mf)
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
                return Trigtech(coeffs=z, is_real=A.is_real, ishappy=True)
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

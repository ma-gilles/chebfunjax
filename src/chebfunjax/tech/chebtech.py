"""Chebyshev technology — smooth function approximation on [-1, 1].

Translated from MATLAB Chebfun class @chebtech2 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np  # uses-numpy: concrete-input eval fast path

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


def _expand_coeff_pair(fc: jax.Array, gc: jax.Array):
    """Reshape a scalar-valued coefficient array to one column when the other
    operand is array-valued.

    MATLAB (R2016b+) applies implicit expansion in ``@chebtech/plus.m``, so an
    ``n x m`` coefficient matrix plus an ``n x 1`` one broadcasts over the
    columns.  chebfunjax stores scalar-valued techs with 1-D coefficients, so
    the 1-D operand is reshaped to ``(n, 1)`` to reproduce that.
    """
    if fc.ndim == 1 and gc.ndim == 2:
        return fc[:, None], gc
    if fc.ndim == 2 and gc.ndim == 1:
        return fc, gc[:, None]
    return fc, gc


def _check_rdivide_shape(coeffs: jax.Array, other) -> None:
    """Raise MATLAB's ``rdivide:size`` for an incompatible numeric divisor.

    MATLAB ``@chebtech/rdivide.m`` accepts only a scalar or a row vector whose
    column count matches the tech's; a column vector (or a mismatched row) is
    an error.  Without this check JAX would broadcast the divisor against the
    coefficient rows and silently return a transposed result.
    """
    try:
        arr = jnp.asarray(other)
    except (TypeError, ValueError):
        return
    if arr.ndim == 0 or arr.size == 1:
        return
    ncols = coeffs.shape[1] if coeffs.ndim == 2 else 1
    ok = (arr.ndim == 1 and arr.shape[0] == ncols) or (
        arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == ncols)
    if not ok:
        raise ValueError(
            "CHEBFUN:CHEBTECH:rdivide:size: matrix dimensions must agree; "
            f"cannot divide a tech with {ncols} column(s) by an array of "
            f"shape {tuple(arr.shape)}.")


def _defers_binary(other) -> bool:
    """True when a tech binary op should defer to the other operand
    (Singfun wraps a tech and owns the mixed-operand arithmetic)."""
    return type(other).__name__ == "Singfun"


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

    # MATLAB clenshaw_scl/clenshaw_vec take two consecutive recurrence
    # steps per loop, keeping the intermediate state inside the loop body.
    # The degree and remainder are static, including under JIT/AD.
    degree = n - 1
    def body(i, state):
        bk1, bk2 = state
        k = degree - 2 * i
        bk2 = coeffs[k] + x2 * bk1 - bk2
        bk1 = coeffs[k - 1] + x2 * bk2 - bk1
        return (bk1, bk2)

    # Carry dtype must match the series dtype (complex chebfuns give a
    # complex recurrence; a float64 carry breaks the lax scan/loop typing).
    out_dtype = jnp.result_type(coeffs.dtype, x.dtype)
    carry_shape = jnp.broadcast_shapes(
        x.shape, coeffs.shape[1:] if multi else ())
    init = (
        jnp.zeros(carry_shape, dtype=out_dtype),
        jnp.zeros(carry_shape, dtype=out_dtype),
    )
    bk1, bk2 = jax.lax.fori_loop(0, degree // 2, body, init)
    if degree % 2:
        bk1, bk2 = coeffs[1] + x2 * bk1 - bk2, bk1

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


def _collapse_if_zero(coeffs: jax.Array) -> jax.Array:
    """Return a single zero coefficient when the vertical scale is zero.

    MATLAB @chebtech/mtimes.m does exactly this after scaling ("If the
    vertical scale is zero, set the CHEBTECH to zero"), and
    @chebtech/plus.m builds a length-1 zero when the sum cancels.
    Without it ``0*r`` keeps its operand's length, and every downstream
    length is wrong by that factor -- ode-nonlin/Logistic starts from
    ``0.5 + 0*r`` and prints length(x) at each step, which came out 2x
    MATLAB's 1, 2, 4, 8, ... all the way to 2^19.
    """
    if coeffs.size and not bool(jnp.any(coeffs != 0)):
        return jnp.zeros((1,) + coeffs.shape[1:], dtype=coeffs.dtype)
    return coeffs


def _prolong_coeffs(coeffs: jax.Array, n: int) -> jax.Array:
    """Zero-pad or truncate Chebyshev coefficients to length *n*."""
    m = coeffs.shape[0]
    if m >= n:
        return coeffs[:n]
    pad = jnp.zeros((n - m,) + coeffs.shape[1:], dtype=coeffs.dtype)
    return jnp.concatenate([coeffs, pad])


def _as_matrix(coeffs):
    original = jnp.asarray(coeffs)
    if original.ndim not in (1, 2):
        raise ValueError("alias expects a coefficient vector or matrix")
    return original, original[:, None] if original.ndim == 1 else original


def _restore_shape(result, original):
    return result[:, 0] if original.ndim == 1 else result


@partial(jax.jit, static_argnames=("m",))
def _alias_chebtech2(coeffs, m: int):
    """Alias Chebtech2 coefficients to *m* rows using pure JAX operations.

    The large-fold case uses indexed scatter-add; the small-fold case uses a
    compiled sequential loop to preserve MATLAB's increasing-j accumulation
    order when multiple high modes fold onto one retained coefficient. ``m``
    must be a static nonnegative integer when this function is JIT compiled.

    Provenance
    ----------
    MATLAB source : ``@chebtech2/alias.m``, especially the distinct m==1,
        vectorized-fold, and loop-fold branches
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    original, c = _as_matrix(coeffs)
    if not isinstance(m, int) or m < 0:
        raise ValueError("m must be a nonnegative static integer")
    n, ncols = c.shape
    if m > n:
        out = jnp.concatenate(
            (c, jnp.zeros((m - n, ncols), dtype=c.dtype)), axis=0
        )
    elif m == 0:
        out = c[:0]
    elif m == 1:
        weights = jnp.where(jnp.arange((n + 1) // 2) % 2 == 0, 1, -1)
        out = jnp.sum(c[::2] * weights[:, None], axis=0, keepdims=True)
    elif m == n:
        out = c
    else:
        out = c[:m]
        if m > n / 2:
            # MATLAB vectorizes this branch because each source mode has a
            # unique destination. Source j is 1-based; r=j-1 is its JAX row.
            r = jnp.arange(m, n)
            target = jnp.abs(jnp.mod(r + m - 2, 2 * m - 2) - m + 2)
            out = out.at[target, :].add(c[m:, :], mode="drop")
        else:
            def add_one(i, acc):
                r = i + m
                target = jnp.abs(jnp.mod(r + m - 2, 2 * m - 2) - m + 2)
                return acc.at[target, :].add(c[r, :], mode="drop")

            out = jax.lax.fori_loop(0, n - m, add_one, out)
    return _restore_shape(out, original)


@partial(jax.jit, static_argnames=("m",))
def _alias_chebtech1(coeffs, m: int):
    """Alias Chebtech1 coefficients to *m* rows using pure JAX operations.

    Chebtech1 differs from Chebtech2 in both its fold period and the sign
    applied to modes that cross an even number of half-periods. As above, the
    high-fold branch uses scatter-add and the low-fold branch is a compiled
    source-order loop. ``m`` must be a static nonnegative integer under JIT.

    Provenance
    ----------
    MATLAB source : ``@chebtech1/alias.m``, especially the distinct m==1,
        vectorized-fold, and loop-fold branches
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    original, c = _as_matrix(coeffs)
    if not isinstance(m, int) or m < 0:
        raise ValueError("m must be a nonnegative static integer")
    n, ncols = c.shape
    if m > n:
        out = jnp.concatenate(
            (c, jnp.zeros((m - n, ncols), dtype=c.dtype)), axis=0
        )
    elif m == 0:
        out = c[:0]
    elif m == 1:
        weights = jnp.where(jnp.arange((n + 1) // 2) % 2 == 0, 1, -1)
        out = jnp.sum(c[::2] * weights[:, None], axis=0, keepdims=True)
    elif m == n:
        out = c
    else:
        out = c[:m]
        if m > n / 2:
            r = jnp.arange(m, n)
            target = jnp.abs(jnp.mod(r + m - 1, 2 * m) - m + 1)
            sign = jnp.where(jnp.floor((r + m) / (2 * m)) % 2 == 0, 1, -1)
            out = out.at[target, :].add(sign[:, None] * c[m:, :], mode="drop")
        else:
            def add_one(i, acc):
                r = i + m
                target = jnp.abs(jnp.mod(r + m - 1, 2 * m) - m + 1)
                sign = jnp.where(jnp.floor((r + m) / (2 * m)) % 2 == 0, 1, -1)
                return acc.at[target, :].add(sign * c[r, :], mode="drop")

            out = jax.lax.fori_loop(0, n - m, add_one, out)
    return _restore_shape(out, original)


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
    # Sum over the contour is the length-K DFT along the sample axis (axis 0);
    # for an array-valued ``op`` the columns are independent.  The rho^k
    # weights broadcast along the leading axis.
    powers = rho ** jnp.arange(K, dtype=jnp.float64)
    if g.ndim > 1:
        powers = powers.reshape((K,) + (1,) * (g.ndim - 1))
    c = jnp.fft.fft(g, axis=0) / K / powers
    return jnp.concatenate([c[:1], 2.0 * c[1:n]], axis=0)


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
    # (MATLAB @chebtech/constructorTurbo.m: real(c) / imag(c) / c).  MATLAB
    # stores ``imag(c)`` (real) for a pure-imaginary function and carries the
    # 1i at the chebfun layer; chebfunjax's tech is self-contained, so we keep
    # the pure-imaginary coefficients ``1i*imag(c)`` -- exactly what the plain
    # (general-complex) path produced before fcb4831 made the plain coeffs
    # bit-exactly pure-imaginary and thus triggered this branch.  Dropping the
    # 1i here collapsed 1i*exp(x) onto the real axis.
    if not bool(jnp.iscomplexobj(plain_coeffs)):
        return jnp.real(c)
    if float(jnp.max(jnp.abs(jnp.real(plain_coeffs)))) == 0.0:
        return jnp.asarray(1j * jnp.imag(c), dtype=plain_coeffs.dtype)
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


def _chop_columns(coeffs: jax.Array, tol: float | jax.Array | None) -> int:
    """standard_chop applied column-wise; the cutoff is the max across
    columns (MATLAB @chebtech/simplify.m and standardCheck.m loop over
    the columns of an array-valued chebtech and keep the largest).

    A tolerance vector supplies one tolerance per column. MATLAB replaces a
    vector of the wrong length with its maximum replicated across columns.
    """
    if coeffs.ndim == 1:
        if tol is None:
            return standard_chop(coeffs, None)
        tol_values = jnp.ravel(jnp.asarray(tol))
        scalar_tol = tol_values[0] if tol_values.size == 1 else jnp.max(tol_values)
        return standard_chop(coeffs, float(scalar_tol))

    ncols = coeffs.shape[1]
    if tol is None:
        tolerances = [None] * ncols
    else:
        tol_array = jnp.asarray(tol)
        tol_values = jnp.ravel(tol_array)
        if (tol_values.size != ncols
                or (tol_array.ndim > 1 and tol_array.shape[-1] != ncols)):
            tol_values = jnp.full((ncols,), jnp.max(tol_values))
        tolerances = [float(tol_values[j]) for j in range(ncols)]
    return max(standard_chop(coeffs[:, j], tolerances[j])
               for j in range(ncols))


def _round_half_away(x: float) -> int:
    """MATLAB ``round`` (round-half-away-from-zero), not Python banker's."""
    return int(jnp.floor(jnp.abs(x) + 0.5) * jnp.sign(x)) if x != 0 else 0


class _HappinessError(ValueError):
    """Source NaN diagnostic with MATLAB's identifier and message."""

    def __init__(self, checker):
        self.identifier = f"CHEBFUN:CHEBTECH:{checker}:nanEval"
        super().__init__("Function returned NaN when evaluated.")


def _columns(values: jnp.ndarray) -> jnp.ndarray:
    values = jnp.asarray(values)
    return values[:, None] if values.ndim == 1 else values


def _tol_columns(tol, ncols: int) -> jnp.ndarray:
    result = jnp.ravel(jnp.asarray(tol, dtype=jnp.float64))
    if result.size == 1:
        return jnp.full((ncols,), result[0])
    if result.size != ncols:
        return jnp.full((ncols,), jnp.max(result))
    return result


def _scale_columns(vscale, ncols: int) -> jnp.ndarray:
    result = jnp.ravel(jnp.asarray(vscale, dtype=jnp.float64))
    if result.size == 1:
        return jnp.full((ncols,), result[0])
    if result.size != ncols:
        return jnp.full((ncols,), jnp.max(result))
    return result


def _strict_check(coeffs, vscale, epslevel) -> tuple[bool, int | None]:
    """Literal JAX translation of MATLAB ``strictCheck`` for numeric arrays.

    Source cutoff uses ``find(max(f.coeffs,[],2)>0,'last')`` after zeroing the
    accepted tail. For a real all-negative function this source expression
    yields an empty cutoff. Return ``None`` for that source result: downstream
    Python ``coeffs[:None]`` is the full coefficient array, matching MATLAB
    ``prolong(f, [])``, which leaves the coefficients unchanged.

    Provenance
    ----------
    MATLAB source : ``@chebtech/strictCheck.m`` lines 31–88;
        ``@chebtech/prolong.m`` lines 20–40
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    c = _columns(jnp.asarray(coeffs))
    n, m = c.shape
    scales = _scale_columns(vscale, m)
    tolerances = _tol_columns(epslevel, m)
    if n < 2:
        return False, n
    if bool(jnp.max(scales) == 0):
        return True, 1
    if bool(jnp.any(jnp.isinf(scales))):
        return False, n
    if bool(jnp.any(jnp.isnan(c))):
        raise _HappinessError("strictCheck")

    test_length = min(n, max(5, _round_half_away((n - 1) / 8)))
    ac = jnp.abs(c) / scales[None, :]
    truncated = jnp.where(ac <= tolerances[None, :], 0.0, c)
    tail = truncated[n - test_length :, :]
    if bool(jnp.any(tail != 0)):
        return False, n

    # Source uses max(f.coeffs,[],2), not max(abs(f.coeffs),[],2).
    # For complex data MATLAB breaks equal-magnitude ties by the larger phase
    # angle; relational > 0 then compares the selected value's real part.
    if jnp.iscomplexobj(truncated):
        magnitude = jnp.abs(truncated)
        tied = magnitude == jnp.max(magnitude, axis=1, keepdims=True)
        phase = jnp.where(tied, jnp.angle(truncated), -jnp.inf)
        winner = jnp.argmax(phase, axis=1)
        row_max = jnp.take_along_axis(truncated, winner[:, None], axis=1)[:, 0]
        positive = jnp.real(row_max) > 0
    else:
        positive = jnp.max(truncated, axis=1) > 0
    rows = jnp.where(positive, jnp.arange(n, dtype=jnp.int32) + 1, 0)
    last = int(jnp.max(rows))
    return True, last if last > 0 else None


def _happiness_requirements(values, coeffs, points, vscale, hscale, epslevel):
    """Port the ``classicCheck`` local happinessRequirements subfunction.

    Returns ``(test_length, eps_per_column)``. The ``coeffs`` argument is kept
    because MATLAB's nested helper receives it, but source ``%#ok<INUSL>``
    confirms it is unused.

    Provenance
    ----------
    MATLAB source : ``@chebtech/classicCheck.m`` lines 162–191
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    del coeffs
    v = _columns(jnp.asarray(values))
    x = jnp.ravel(jnp.asarray(points, dtype=jnp.float64))
    n, m = v.shape
    scales = _scale_columns(vscale, m)
    eps_values = _tol_columns(epslevel, m)
    test_length = min(n, max(5, _round_half_away((n - 1) / 8)))
    tail_error = min(float(_EPS) * test_length, 1e-4)

    dy = jnp.diff(v, axis=0)
    dx = jnp.diff(x)[:, None] * jnp.ones((1, m), dtype=jnp.float64)
    grad_est = jnp.max(jnp.abs(dy / dx), axis=0)
    spacing = jnp.spacing(jnp.abs(jnp.asarray(hscale, dtype=jnp.float64)))
    cond_est = jnp.minimum(spacing / scales * grad_est, 1e-4)
    eps_values = jnp.maximum(jnp.maximum(eps_values, cond_est), tail_error)
    return test_length, eps_values


def _classic_check(coeffs, values, points, vscale, hscale, epslevel) -> tuple[bool, int]:
    """JAX translation of ``classicCheck`` and its local requirements routine.

    Coefficient/value array columns are handled together with source per-column
    scales and tolerances. Loops over the reversed tail use JAX immutable arrays;
    only convergence decisions and final integer cutoff leave the device.

    Provenance
    ----------
    MATLAB source : ``@chebtech/classicCheck.m`` lines 47–160 and local
        ``happinessRequirements`` lines 162–191
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    c = _columns(jnp.asarray(coeffs))
    v = _columns(jnp.asarray(values))
    x = jnp.ravel(jnp.asarray(points, dtype=jnp.float64))
    n, m = c.shape
    scales = _scale_columns(vscale, m)
    eps_values = _tol_columns(epslevel, m)

    if n < 2:
        return False, n
    if bool(jnp.any(jnp.isnan(c))):
        raise _HappinessError("classicCheck")
    if bool(jnp.max(scales) == 0):
        return True, 1
    if bool(jnp.any(jnp.isinf(scales))):
        return False, n

    safe_scales = jnp.where(scales == 0, 1.0, scales)
    ac = jnp.abs(c) / safe_scales[None, :]
    test_length, eps_values = _happiness_requirements(
        v, c, x, safe_scales, hscale, eps_values
    )

    tail = ac[n - test_length :, :]
    if not bool(jnp.all(jnp.max(tail, axis=0) < eps_values)):
        return False, 0

    rows_large = jnp.any(ac >= eps_values[None, :], axis=1)
    nonzero_rows = jnp.where(rows_large, jnp.arange(n, dtype=jnp.int32), -1)
    last = int(jnp.max(nonzero_rows))
    if last < 0:
        return True, 1
    tloc = last + 2  # MATLAB find index plus its explicit +1

    reversed_tail = ac[::-1][: n - tloc + 1, :]
    acr = jnp.maximum(
        jax.lax.associative_scan(jnp.maximum, reversed_tail, axis=0),
        0.25 * _EPS,
    )

    bang = jnp.log(1e3 * eps_values[None, :] / acr)
    buck = jnp.arange(n - 1, tloc - 2, -1, dtype=jnp.float64)[:, None]
    tbpb = bang / buck
    sub = tbpb[2 : n - tloc + 1, :]
    if sub.shape[0] == 0:
        return True, n
    per_column = jnp.argmax(sub, axis=0) + 1
    tchop = int(jnp.min(per_column))
    cutoff = n - tchop - 2
    return True, cutoff


def _happiness_check_impl(
    tech_cls,
    kind: int,
    coeffs: jax.Array,
    values: jax.Array,
    op: Callable | None,
    tol: float | None,
    vscale: float,
    hscale: float,
    check: str,
    sample_test: bool = True,
) -> tuple[bool, int | None]:
    """Shared happiness-check dispatch for Chebtech1/Chebtech2.

    Dispatches to the requested happiness variant (``'standard'``,
    ``'strict'``, ``'classic'``) and then applies the common sample test
    (MATLAB ``@chebtech/happinessCheck.m``), reverting the cutoff to the full
    length if the sample test fails. Strict may return None for the source
    empty cutoff; retaining the full series then matches prolong(f,[]).

    Provenance
    ----------
    MATLAB source : @chebtech/happinessCheck.m, @chebtech/sampleTest.m
    Chebfun commit: 7574c77
    """
    if tol is None:
        tol = _EPS

    n = coeffs.shape[0]
    # MATLAB standardCheck/sampleTest scale each column independently.
    # Keep the sample axis when updating a running piecewise/global scale.
    local_scales = jnp.max(jnp.abs(values), axis=0)
    global_scales = jnp.maximum(jnp.asarray(vscale), local_scales)
    column_count = 1 if coeffs.ndim == 1 else coeffs.shape[1]
    tolerances = jnp.atleast_1d(jnp.asarray(tol))
    if tolerances.size != column_count:
        tolerances = jnp.full((column_count,), jnp.max(tolerances))

    if check == "plateau":
        from chebfunjax.utils.plateau import _plateau_check
        ishappy, cutoff = _plateau_check(
            coeffs, values, global_scales, float(jnp.max(tolerances)))
    elif coeffs.ndim == 2 and check in {"strict", "classic"}:
        scales = jnp.broadcast_to(global_scales, (column_count,))
        if check == "strict":
            ishappy, cutoff = _strict_check(coeffs, scales, tolerances)
        else:
            points = chebpts(n, kind)
            ishappy, cutoff = _classic_check(
                coeffs, values, points, scales, hscale, tolerances
            )
    elif coeffs.ndim == 2:
        scales = jnp.broadcast_to(global_scales, (column_count,))
        results = [
            _happiness_check_impl(
                tech_cls, kind, coeffs[:, column], values[:, column], None,
                float(tolerances[column]), float(scales[column]), hscale,
                check, sample_test=False,
            )
            for column in range(column_count)
        ]
        ishappy = all(result[0] for result in results)
        cutoff = max(result[1] for result in results)
    else:
        vscale_local = float(local_scales)
        vscale = float(global_scales)
        scalar_tol = float(tolerances[0])
        if check == "strict":
            ishappy, cutoff = _strict_check(coeffs, vscale, scalar_tol)
        elif check == "classic":
            points = chebpts(n, kind)
            ishappy, cutoff = _classic_check(
                coeffs, values, points, vscale, hscale, scalar_tol
            )
        elif check == "standard":
            if vscale_local > 0:
                scaled_tol = scalar_tol * max(hscale, vscale / vscale_local)
            else:
                scaled_tol = scalar_tol * hscale
            cutoff = standard_chop(coeffs, scaled_tol)
            ishappy = cutoff < n
        else:
            raise ValueError(
                f"unknown happiness check {check!r} "
                "(expected 'standard', 'strict', 'classic', or 'plateau')"
            )

    if ishappy and op is not None and sample_test:
        xeval = jnp.array(
            [-0.357998918959666, 0.036785641195074], dtype=jnp.float64
        )
        keep = n if cutoff is None else max(1, min(int(cutoff), n))
        f_test = tech_cls(coeffs=coeffs[:keep])
        v_fun = f_test(xeval)
        v_op = _as_fun_dtype(op(xeval))
        errors = jnp.max(jnp.abs(v_op - v_fun), axis=0)
        sample_tolerances = jnp.sqrt(jnp.maximum(_EPS, tolerances)) * jnp.maximum(
            hscale * local_scales, global_scales
        )
        if not bool(jnp.all(errors <= sample_tolerances)):
            ishappy = False
            cutoff = n

    return ishappy, cutoff


def _extrapolate_values(
    values: jax.Array,
    x: jax.Array,
    w: jax.Array,
):
    """Barycentric extrapolation of NaN/Inf sample rows (MATLAB extrapolate.m).

    Replaces every row of ``values`` that contains a NaN or Inf (in any
    column) by the value of the barycentric interpolant built from the finite
    rows.  Rows with only finite entries are left bit-for-bit unchanged, so a
    clean sample is returned identically.  Uses NumPy internally (data-dependent
    masks make this unsuitable for tracing); it is a construction-time helper,
    not a JIT hot path.

    Parameters
    ----------
    values : jax.Array, shape (n,) or (n, m)
        Sampled function values (possibly containing NaN/Inf).
    x : jax.Array, shape (n,)
        The Chebyshev points at which ``values`` were sampled.
    w : jax.Array, shape (n,)
        The barycentric weights for the ``n``-point grid.

    Returns
    -------
    new_values : jax.Array
        ``values`` with masked rows replaced (same shape/dtype as input).
    mask_nan : numpy.ndarray, shape (n,)
        Column vector flagging rows that held a NaN.
    mask_inf : numpy.ndarray, shape (n,)
        Column vector flagging rows that held an Inf.

    Provenance
    ----------
    MATLAB source : @chebtech/extrapolate.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    import numpy as _np

    v = _np.asarray(values)
    was_1d = v.ndim == 1
    if was_1d:
        v = v[:, None]
    xr = _np.asarray(x).reshape(-1)
    wr = _np.asarray(w).reshape(-1)
    n, m = v.shape

    mask_nan = _np.any(_np.isnan(v), axis=1)
    mask_inf = _np.any(_np.isinf(v), axis=1)
    mask = mask_nan | mask_inf

    if _np.any(mask):
        good = ~mask
        xgood = xr[good]
        if xgood.size == 0:
            raise ValueError(
                "CHEBFUN:CHEBTECH:extrapolate:nansInfs: "
                "Too many NaNs/Infs to handle."
            )
        xbad = xr[mask]

        # Modified barycentric weights for the good points (MATLAB loop):
        # w_k <- w_k * prod_j (xgood_k - xbad_j).
        wmod = wr[good].astype(_np.float64).copy()
        for xb in xbad:
            wmod = wmod * (xgood - xb)

        vgood = v[good, :]
        newvals = _np.zeros((xbad.size, m), dtype=v.dtype)
        for k in range(xbad.size):
            # Barycentric formula of the second kind at the bad point.
            w2 = wmod / (xbad[k] - xgood)
            newvals[k, :] = (w2 @ vgood) / _np.sum(w2)

        v = v.copy()
        v[mask, :] = newvals

    out = v[:, 0] if was_1d else v
    return jnp.asarray(out), mask_nan, mask_inf


def _sample_extrapolate(
    f: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    extrapolate: bool,
) -> jax.Array:
    """Sample ``f`` on grid ``x``, optionally skipping the endpoints.

    With ``extrapolate=False`` this is just ``f(x)``.  With ``extrapolate=True``
    (MATLAB ``pref.extrapolate``) the operator is evaluated only at the
    interior points ``x[1:-1]`` and the two endpoint rows are set to NaN, so
    the constructor's extrapolation step fills them from the interior samples
    -- this avoids ever evaluating ``f`` at ``x = +/-1``.

    Provenance
    ----------
    MATLAB source : @chebtech2/refine.m (refineResampling / refineNested,
        ``pref.extrapolate`` branch), @chebtech/populate.m
    Chebfun commit: 7574c77
    """
    if not extrapolate:
        return _as_fun_dtype(f(x))
    v_int = _as_fun_dtype(f(x[1:-1]))
    nan_row = jnp.full((1,) + v_int.shape[1:], jnp.nan, dtype=v_int.dtype)
    return jnp.concatenate([nan_row, v_int, nan_row], axis=0)


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


def _cumsum_lval(coeffs: jax.Array) -> jax.Array:
    """Evaluate coefficient columns at x=-1, matching MATLAB ``lval``."""
    c = jnp.asarray(coeffs)
    signs = jnp.where(jnp.arange(c.shape[0]) % 2 == 0, 1.0, -1.0)
    if c.ndim == 1:
        return jnp.sum(c * signs)
    return jnp.sum(c * signs[:, None], axis=0)


@partial(jax.jit, static_argnames=("dim",))
def _cumsum_coeffs_by_dim(
    coeffs: jax.Array, *, dim: int = 1
) -> jax.Array:
    """JIT adapter for continuous integration or coefficient-column sums.

    ``dim`` is static because it selects source branches with different
    coefficient lengths and meanings.
    """
    if dim == 1:
        return _cumsum_coeffs(coeffs)
    if coeffs.ndim == 1:
        return coeffs
    return jnp.cumsum(coeffs, axis=1)


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


def _prune_spurious_roots(r, rho):
    """Prune 'spurious' complex roots by Boyd's radius test.

    Keeps roots whose Bernstein-ellipse radius ``|r + sqrt(r^2 - 1)|``
    (reflected to be >= 1) does not exceed ``rho``, mirroring the
    ``'prune'`` branch of MATLAB ``@chebtech/roots.m``.  ``r`` is treated
    as complex so the square root is well defined for real ``|r| < 1``.
    """
    import numpy as np
    r = np.asarray(r)
    rho_roots = np.abs(r + np.sqrt(r.astype(np.complex128) ** 2 - 1.0))
    rho_roots = np.where(rho_roots < 1.0, 1.0 / rho_roots, rho_roots)
    return r[rho_roots <= rho]


def _roots_colleague(coeffs: jax.Array, qz: bool = False,
                     all_roots: bool = False, prune: bool = False,
                     recurse: bool = True) -> jax.Array:
    import numpy as np
    """Find roots of a Chebyshev expansion in [-1, 1].

    Uses recursive subdivision for degree > 50 and colleague matrix
    eigenvalue computation for degree <= 50.

    Parameters mirror the option surface of MATLAB ``@chebtech/roots.m``:

    * ``qz`` — solve the colleague *matrix pencil* via the QZ / generalized
      eigenvalue algorithm instead of plain QR ([Nakatsukasa & Noferini]).
    * ``all_roots`` — return every root the linearization produces
      (complex, and outside [-1, 1]) instead of only real roots in [-1, 1]
      (MATLAB ``'all'``).
    * ``prune`` — when ``all_roots`` holds, discard 'spurious' roots by the
      Bernstein-radius test (MATLAB ``'prune'``).
    * ``recurse`` — when ``False``, never subdivide; solve one colleague
      problem for the whole series (MATLAB ``'recurse'``, 0).

    MATLAB's ``'complex'`` flag is exactly ``all_roots=True, prune=True``.

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

    # length(f) for the top-level prune radius (MATLAB uses numel(coeffs)).
    length_f = c.shape[0]

    # Normalize
    vscl = np.max(np.abs(c))
    if vscl == 0.0:
        return jnp.array([0.0], dtype=jnp.float64)
    c_scaled = c / vscl

    r = _roots_main(c_scaled, htol, qz=qz, all_roots=all_roots,
                    prune=prune, recurse=recurse)

    # Prune the roots if requested (MATLAB prunes at the top level only when
    # recurse is off; with recursion the per-leaf prune already ran).
    if prune and not recurse and length_f > 0:
        rho = np.sqrt(np.finfo(np.float64).eps) ** (-1.0 / length_f)
        r = _prune_spurious_roots(r, rho)

    if all_roots:
        # Keep complex roots; sort deterministically (real, then imag).
        r = np.asarray(r)
        order = np.lexsort((np.imag(r), np.real(r)))
        return jnp.asarray(r[order])
    r = np.sort(np.real(r))
    return jnp.asarray(r, dtype=jnp.float64)


def _roots_main(c, htol: float, qz: bool = False, all_roots: bool = False,
                prune: bool = False, recurse: bool = True):
    import numpy as np
    """Recursive root-finding engine (numpy, NOT JIT-safe).

    Follows MATLAB Chebfun's roots.m strategy:
    - Trim trailing small coefficients.
    - If ``recurse`` and degree > 50, subdivide at a slightly off-center
      point and recurse.
    - Otherwise form the colleague matrix and compute eigenvalues, then
      filter (real roots in [-1, 1]), prune, or keep all per the options.
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
        if not all_roots:
            mask_im = np.abs(np.imag(r)) < htol
            r = np.real(r[mask_im])
            r = r[(r >= -(1.0 + htol)) & (r <= (1.0 + htol))]
            r = np.clip(r, -1.0, 1.0)
        return r

    if (not recurse) or (n - 1 <= MAX_EIG_SIZE):
        # Form the colleague matrix
        c_adj = -0.5 * c[:-1] / c[-1]
        c_adj[-2] += 0.5

        nn = n - 1
        oh = 0.5 * np.ones(nn - 1)
        A = np.diag(oh, 1) + np.diag(oh, -1)
        if np.iscomplexobj(c_adj) or (qz and np.iscomplexobj(c)):
            A = A.astype(np.complex128)
        A[-2, -1] = 1.0
        A[:, 0] = c_adj[::-1]

        if qz:
            # Colleague matrix *pencil* (A, B) solved by the QZ / GEP
            # algorithm for extra numerical stability, mirroring the
            # scaled generalized eigenproblem of MATLAB
            # @chebtech/roots.m ('qz' branch).
            c_old = c.copy()
            c_old = c_old / np.linalg.norm(c_old, np.inf)
            B = np.eye(nn)
            if np.iscomplexobj(c_old):
                B = B.astype(np.complex128)
            B[0, 0] = c_old[-1]
            c_scaled = -0.5 * c_old[:-1]
            c_scaled[-2] = c_scaled[-2] + 0.5 * B[0, 0]
            A[:, 0] = c_scaled[::-1]
            import scipy.linalg as _sla
            rts = _sla.eig(A, B, right=False)
        else:
            rts = np.linalg.eigvals(A)

        if not all_roots:
            # Keep roots with small imaginary part and inside [-1, 1].
            mask = np.abs(np.imag(rts)) < htol
            rts = np.real(rts[mask])
            rts = rts[np.abs(rts) <= 1.0 + htol]
            rts = np.sort(rts)
            if rts.size > 0:
                rts[0] = max(rts[0], -1.0)
                rts[-1] = min(rts[-1], 1.0)
        elif prune:
            # Prune spurious roots by the Bernstein-radius test (local n).
            rho = np.sqrt(np.finfo(np.float64).eps) ** (-1.0 / n)
            rts = _prune_spurious_roots(rts, rho)
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

    # Convert values to coefficients with a pure-numpy transform: this
    # recursion is NOT JIT-safe anyway, and routing through the jitted
    # vals2coeffs compiled one XLA program per distinct piece length
    # (~20 ms each), dominating multi-piece roots() wall time.
    def _v2c_np(v):
        nn = v.shape[0]
        if nn <= 1:
            return v.copy()
        tmp = np.concatenate([v[nn - 1:0:-1], v[:nn - 1]])
        if np.iscomplexobj(v):
            if np.all(np.real(v) == 0):
                cc = 1j * np.real(np.fft.ifft(np.imag(tmp)))
            else:
                cc = np.fft.ifft(tmp)
        else:
            cc = np.real(np.fft.ifft(tmp))
        cc = cc[:nn]
        cc[1:nn - 1] *= 2.0
        vflip = v[::-1]
        k = np.arange(nn)
        if np.max(np.abs(v - vflip)) == 0:
            cc[k % 2 == 1] = 0.0
        if np.max(np.abs(v + vflip)) == 0:
            cc[k % 2 == 0] = 0.0
        return cc

    c_left = _v2c_np(v_left)
    c_right = _v2c_np(v_right)

    # Recurse
    r_left = _roots_main(c_left, 2.0 * htol, qz=qz, all_roots=all_roots,
                         prune=prune, recurse=recurse)
    r_right = _roots_main(c_right, 2.0 * htol, qz=qz, all_roots=all_roots,
                          prune=prune, recurse=recurse)

    # Map back to original interval
    r_left_mapped = 0.5 * (SPLIT_POINT - 1.0) + 0.5 * (SPLIT_POINT + 1.0) * r_left
    r_right_mapped = 0.5 * (SPLIT_POINT + 1.0) + 0.5 * (1.0 - SPLIT_POINT) * r_right

    return np.concatenate([r_left_mapped, r_right_mapped])


# ============================================================================
# Chebtech2 — the core class
# ============================================================================


def _is_empty_tech(obj) -> bool:
    """Source isempty.m, including zero-sized coefficient arrays."""
    if getattr(obj, "_is_empty_object", False):
        return True
    return isinstance(obj, (Chebtech1, Chebtech2)) and obj.coeffs.size == 0


def _poly_coeffs(coeffs: jax.Array) -> jax.Array:
    r"""Monomial (power-basis) coefficients of a Chebyshev-T expansion.

    Given coefficients ``c`` of ``f(x) = sum_k c[k] T_k(x)`` returns the
    power-basis coefficients ``p`` with the *highest* degree first, so
    ``f(x) = p[0]*x^(n-1) + ... + p[n-2]*x + p[n-1]`` (mirroring MATLAB
    ``poly``).  For an ``(n, m)`` array-valued input the result has shape
    ``(m, n)``: row ``j`` holds the power coefficients of column ``j`` of
    ``f`` (this transposition matches MATLAB ``@chebtech/poly.m``).

    Provenance
    ----------
    MATLAB source : @chebtech/poly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Reference: Section 3.3, Mason & Handscomb, "Chebyshev Polynomials",
        Chapman & Hall/CRC (2003).
    """
    c = jnp.atleast_1d(coeffs)
    if c.ndim == 2:
        return jnp.stack(
            [_poly_coeffs(c[:, j]) for j in range(c.shape[1])], axis=0
        )
    n = c.shape[0]
    dt = jnp.result_type(c.dtype, jnp.float64)
    c = c.astype(dt)
    if n == 0:
        return c
    # Accumulate p(x) = sum_k c[k] T_k(x) in the power basis (ascending),
    # using T_0 = 1, T_1 = x, T_k = 2 x T_{k-1} - T_{k-2}.
    tkm2 = jnp.zeros(n, dtype=dt).at[0].set(1.0)          # T_0
    p = c[0] * tkm2
    if n == 1:
        return p[::-1]
    tkm1 = jnp.zeros(n, dtype=dt).at[1].set(1.0)          # T_1
    p = p + c[1] * tkm1
    for k in range(2, n):
        # x * tkm1  == shift power coeffs up by one index
        xt = jnp.concatenate([jnp.zeros(1, dt), tkm1[:-1]])
        tk = 2.0 * xt - tkm2
        p = p + c[k] * tk
        tkm2 = tkm1
        tkm1 = tk
    return p[::-1]                                        # highest degree first


def _chebT_to_chebU_coeffs(cT: jax.Array) -> jax.Array:
    r"""Convert Chebyshev-T coefficients to Chebyshev-U coefficients.

    Uses the identity ``T_n = (1/2)(U_n - U_{n-2})`` (with ``U_{-1}=0``,
    ``T_0 = U_0``).  A column vector or a matrix (column-wise conversion)
    is accepted.

    Provenance
    ----------
    MATLAB source : @chebtech/chebTcoeffs2chebUcoeffs.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    cT = jnp.asarray(cT)
    if cT.size == 0:
        return cT
    twod = cT.ndim == 2
    c = cT if twod else cT[:, None]
    # Pad two zero rows: coeff of U_n needs T_n and T_{n+2}.
    cU = jnp.concatenate(
        [c, jnp.zeros((2,) + c.shape[1:], dtype=c.dtype)], axis=0
    )
    cU = cU.at[0].set(2.0 * c[0])
    top = 0.5 * (cU[:-2] - cU[2:])
    cU = jnp.concatenate([top, cU[-2:]], axis=0)
    cU = cU[:-2]
    return cU if twod else cU[:, 0]


def _boundary_end_values(c):
    """Return ``|f(-1)|`` and ``|f(1)|`` as a JAX array of shape ``(2,m)``.

    Row 0 is x = -1 (``T_k(-1) = (-1)^k``); row 1 is x = +1 (``T_k(1) = 1``).
    """
    c = jnp.asarray(c)
    n = c.shape[0]
    ncols = 1 if c.ndim == 1 else c.shape[1]
    cm = c.reshape((n, ncols))
    signs = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
    fm1 = jnp.sum(cm * signs[:, None], axis=0)
    fp1 = jnp.sum(cm, axis=0)
    return jnp.abs(jnp.stack((fm1, fp1), axis=0))


def _boundary_root_tolerance(vscale, ncols: int) -> jax.Array:
    scales = jnp.ravel(jnp.asarray(vscale, dtype=jnp.float64))
    if scales.size == 1:
        scales = jnp.broadcast_to(scales, (ncols,))
    elif scales.size != ncols:
        raise ValueError("vscale must be scalar or have one value per column")
    return 1e3 * scales * _EPS


def _has_no_boundary_roots(coeffs, vscale) -> jax.Array:
    """Apply the source's initial endpoint/tolerance early-return test."""
    c = jnp.asarray(coeffs)
    if c.size == 0:
        return jnp.asarray(True)
    ncols = 1 if c.ndim == 1 else c.shape[1]
    cm = c.reshape((c.shape[0], ncols))
    tol = _boundary_root_tolerance(vscale, ncols)
    return jnp.all(jnp.min(_boundary_end_values(cm), axis=0) > tol)


def _peel_boundary_factor(c, active, sgn):
    """Divide selected coefficient columns by ``1 + sgn*x``.

    This is the upper-banded backward substitution from MATLAB's sparse
    ``D\\c(2:end,:)``. The three nonzero diagonals are evaluated directly, so
    no dense matrix or NumPy solve is needed.
    """
    from jax import lax

    n, m = c.shape
    size = n - 1
    rhs = c[1:, :]
    work = jnp.zeros((size + 2, m), dtype=c.dtype).at[:size].set(rhs)

    def solve_row(k, solution):
        i = size - 1 - k
        diagonal = jnp.where(i == 0, 1.0, 0.5).astype(c.real.dtype)
        value = (work[i] - sgn * solution[i + 1]
                 - 0.5 * solution[i + 2]) / diagonal
        return solution.at[i].set(value)

    solution = lax.fori_loop(
        0, size, solve_row, jnp.zeros_like(work)
    )[:size]
    divided = sgn * solution
    updated = c.at[:-1, :].set(
        jnp.where(active[None, :], divided, c[:-1, :])
    )
    return updated.at[-1, :].set(
        jnp.where(active, jnp.zeros((), dtype=c.dtype), c[-1, :])
    )


def _extract_boundary_roots(coeffs, vscale, num_roots=None):
    """Peel roots at the boundary points -1 and +1 in coefficient space.

    Returns ``(new_coeffs, rootsLeft, rootsRight)`` where ``new_coeffs`` is
    free of boundary roots (up to ``num_roots`` when supplied) and the
    multiplicity vectors give the number of roots removed at ``x = -1`` and
    ``x = +1`` per column.  Not simplified — the calling method simplifies.

    Provenance
    ----------
    MATLAB source : @chebtech/extractBoundaryRoots.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    from jax import lax

    c0 = jnp.asarray(coeffs)
    scalar = c0.ndim == 1
    if c0.ndim not in (1, 2):
        raise ValueError("coeffs must be a vector or a 2-D column matrix")
    if c0.shape[0] == 0:
        m = 1 if scalar else c0.shape[1]
        zeros = jnp.zeros((m,), dtype=jnp.int32)
        return c0, zeros[0] if scalar else zeros, zeros[0] if scalar else zeros
    m = 1 if scalar else c0.shape[1]
    c = c0.reshape((c0.shape[0], m))
    rootsLeft = jnp.zeros((m,), dtype=jnp.int32)
    rootsRight = jnp.zeros((m,), dtype=jnp.int32)
    tol = _boundary_root_tolerance(vscale, m)

    nr = None if num_roots is None else jnp.reshape(
        jnp.asarray(num_roots, dtype=jnp.float64), (2, m)
    )

    endValues = _boundary_end_values(c)
    no_roots = jnp.all(jnp.min(endValues, axis=0) > tol)

    def extract(_):
        if c.shape[0] == 1:
            # A nonzero singleton takes the source no-root return. The
            # scalar-zero case is not source-qualified; this guard avoids a
            # zero-length recurrence whose MATLAB behavior is unestablished.
            return c, rootsLeft, rootsRight

        if nr is None:
            def cond(state):
                _c, _left, _right, ends, _tol = state
                return jnp.any(jnp.min(ends, axis=0) <= _tol)

            def body(state):
                coeff, left, right, ends, current_tol = state
                do_left = jnp.any(ends[0] <= current_tol)
                sgn = jnp.where(do_left, 1.0, -1.0)
                active = jnp.where(
                    do_left, ends[0] <= current_tol, ends[1] <= current_tol
                )
                coeff = _peel_boundary_factor(coeff, active, sgn)
                left = left + (active & do_left).astype(left.dtype)
                right = right + (active & ~do_left).astype(right.dtype)
                return coeff, left, right, _boundary_end_values(coeff), current_tol * 1e2

            coeff, left, right, _ends, _tol = lax.while_loop(
                cond, body, (c, rootsLeft, rootsRight, endValues, tol)
            )
            return coeff, left, right

        def cond(state):
            _c, _left, _right, _ends, _tol, requested = state
            return jnp.any(requested > 0)

        def body(state):
            coeff, left, right, ends, current_tol, requested = state
            do_left = jnp.any(requested[0] > 0)

            def peel_side(data, side):
                coeff, left, right, ends, current_tol, requested = data
                sgn = jnp.where(side == 0, 1.0, -1.0)
                active = ends[side] <= current_tol
                requested_active = requested[side] > 0
                matches = jnp.all(active == requested_active)

                def matched(d):
                    coeff, left, right, ends, current_tol, requested = d
                    coeff = _peel_boundary_factor(coeff, active, sgn)
                    # MATLAB increments the entire count row here, including
                    # zero-request columns; retain this source quirk.
                    left = left + (side == 0).astype(left.dtype)
                    right = right + (side == 1).astype(right.dtype)
                    requested = requested.at[side].set(
                        requested[side] - requested_active.astype(requested.dtype)
                    )
                    ends = _boundary_end_values(coeff)
                    return coeff, left, right, ends, current_tol * 1e2, requested

                def mismatched(d):
                    coeff, left, right, ends, current_tol, requested = d
                    return (
                        coeff, left, right, ends, current_tol,
                        requested.at[side].set(jnp.zeros_like(requested[side])),
                    )

                return lax.cond(matches, matched, mismatched, data)

            side = jnp.where(do_left, 0, 1)
            return peel_side((coeff, left, right, ends, current_tol, requested), side)

        coeff, left, right, _ends, _tol, _requested = lax.while_loop(
            cond, body, (c, rootsLeft, rootsRight, endValues, tol, nr)
        )
        return coeff, left, right

    c, rootsLeft, rootsRight = lax.cond(
        no_roots,
        lambda _: (c, rootsLeft, rootsRight),
        extract,
        operand=None,
    )

    c_out = c[:, 0] if scalar else c
    l = rootsLeft[0] if scalar else rootsLeft
    r = rootsRight[0] if scalar else rootsRight
    return jnp.asarray(c_out), l, r


def _initial_resampling_n(min_samples: int = 17) -> int:
    """MATLAB refineResampling initial point count."""
    # MATLAB log2(0)=-Inf, so minSamples=1 yields a single point.
    if min_samples == 1:
        return 1
    return 2 ** math.ceil(math.log2(min_samples - 1)) + 1


def _next_resampling_n(previous_n: int) -> int:
    """MATLAB's approximately sqrt(2)-spaced resampling length update."""
    if previous_n == 1:
        return 1
    power = math.log2(previous_n - 1)
    if power == math.floor(power) and power > 5:
        proposed = math.floor(2 ** (math.floor(power) + 0.5) + 0.5) + 1
        return proposed - proposed % 2 + 1
    return 2 ** (math.floor(power) + 1) + 1


def _refine_sample(op, x, *, extrapolate: bool):
    """Sample Chebtech2, optionally retaining NaN endpoint sentinels."""
    if not extrapolate:
        return _as_fun_dtype(op(x))
    values = _as_fun_dtype(op(x[1:-1]))
    nan_rows = jnp.full(
        (1,) + values.shape[1:], jnp.nan, dtype=values.dtype
    )
    return jnp.concatenate((nan_rows, values, nan_rows), axis=0)


def _refine_chebtech2_resampling(
    op, values=None, *, max_length=65537, min_samples=17, extrapolate=False
):
    """Refine Chebtech2 by resampling the complete grid at each length."""
    if values is None or values.shape[0] == 0:
        n = _initial_resampling_n(min_samples)
        if n > max_length:
            n = max_length
    else:
        n = _next_resampling_n(values.shape[0])
        # Python robustness: source resampling remains at n=1 forever.
        if n <= values.shape[0] or n > max_length:
            return values, True
    return _refine_sample(op, chebpts(n, kind=2), extrapolate=extrapolate), False


def _refine_chebtech2_nested(
    op, values=None, *, max_length=65537, min_samples=17, extrapolate=False
):
    """Refine Chebtech2 with nested points, reusing old values exactly."""
    if values is None or values.shape[0] == 0:
        return _refine_chebtech2_resampling(
            op, None, max_length=max_length, min_samples=min_samples,
            extrapolate=extrapolate,
        )
    old_n = values.shape[0]
    n = 2 * old_n - 1
    # Python robustness for the source's non-growing single-point grid.
    if n <= old_n or n > max_length:
        return values, True
    # MATLAB x(2:2:end-1), translated from 1-based indexing.
    new_values = _as_fun_dtype(op(chebpts(n, kind=2)[1:-1:2]))
    out = jnp.empty((n,) + values.shape[1:], dtype=jnp.result_type(values, new_values))
    out = out.at[::2].set(values)
    out = out.at[1:-1:2].set(new_values)
    return out, False


def _refine_chebtech1_resampling(
    op, values=None, *, max_length=65537, min_samples=17
):
    """Refine Chebtech1 by resampling the complete grid at each length."""
    if values is None or values.shape[0] == 0:
        n = _initial_resampling_n(min_samples)
        if n > max_length:
            n = max_length
    else:
        n = _next_resampling_n(values.shape[0])
        # Python robustness: source resampling remains at n=1 forever.
        if n <= values.shape[0] or n > max_length:
            return values, True
    return _as_fun_dtype(op(chebpts(n, kind=1))), False


def _refine_chebtech1_nested(
    op, values=None, *, max_length=65537, min_samples=17
):
    """Refine Chebtech1 on 17*3^q grids, reusing old samples between calls."""
    if values is None or values.shape[0] == 0:
        return _refine_chebtech1_resampling(
            op, None, max_length=max_length, min_samples=min_samples
        )
    old_n = values.shape[0]
    if old_n < max_length and 3 * old_n > max_length:
        # MATLAB uses maxLength once more, with a full resampling.
        return _as_fun_dtype(op(chebpts(max_length, kind=1))), False
    if old_n < max_length:
        n = 3 * old_n
        x = chebpts(n, kind=1)
        # Preserve MATLAB's exact callback order: x(1:3:end-2), then
        # x(3:3:end), followed by placement of old values in x(2:3:end).
        left_new = _as_fun_dtype(op(x[::3]))
        right_new = _as_fun_dtype(op(x[2::3]))
        out = jnp.empty(
            (n,) + values.shape[1:], dtype=jnp.result_type(values, left_new, right_new)
        )
        out = out.at[::3].set(left_new)
        out = out.at[2::3].set(right_new)
        out = out.at[1::3].set(values)
        return out, False
    return values, True


def _update_running_vscale(values, vscale):
    """@chebtech/populate.m scale update, before replacing nonfinite rows."""
    values = jnp.asarray(values)
    sampled_finite = jnp.where(jnp.isfinite(values), values, 0.0)
    return jnp.maximum(vscale, jnp.max(jnp.abs(sampled_finite), axis=0))


def _adaptive_refine_construct(
    tech_cls, kind, op, *, max_length, min_samples, refinement_function,
    tol, check, vscale, sample_test, hscale, extrapolate=False,
    use_turbo=False, fixed_length=None,
):
    """Populate from source nested/resampling batches, including sentinels.

    MATLAB source: @chebtech/populate.m and @chebtech{1,2}/refine.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    if isinstance(refinement_function, str):
        strategy = refinement_function.lower()
        refiners = {
            (1, "nested"): _refine_chebtech1_nested,
            (1, "resampling"): _refine_chebtech1_resampling,
            (2, "nested"): _refine_chebtech2_nested,
            (2, "resampling"): _refine_chebtech2_resampling,
        }
        if (kind, strategy) not in refiners:
            raise ValueError("refinement_function must be nested or resampling")
        refine = refiners[kind, strategy]
    else:
        # MATLAB passes a full techPref struct. Python uses a dict with its
        # source field names, and None for the initial empty sample array.
        if not callable(refinement_function):
            raise TypeError("refinement_function must be a string or callable")
        def refine(function, values, **prefs):
            return refinement_function(function, values, {
                "maxLength": prefs["max_length"],
                "minSamples": prefs["min_samples"],
                "extrapolate": prefs.get("extrapolate", False),
                "chebfuneps": _EPS if tol is None else tol,
                "fixedLength": math.nan if fixed_length is None else fixed_length,
                "sampleTest": sample_test,
                "refinementFunction": refinement_function,
                "happinessCheck": check,
                "useTurbo": use_turbo,
            })
    kwargs = {"max_length": max_length, "min_samples": min_samples}
    if kind == 2:
        kwargs["extrapolate"] = extrapolate
    old_values = None
    scales = jnp.asarray(vscale, dtype=jnp.float64)
    coeffs = None
    while True:
        sampled, gave_up = refine(op, old_values, **kwargs)
        if gave_up:
            if coeffs is None:
                # MATLAB leaves ishappy undefined for this invalid refiner.
                # Give Python callers a controlled error before any samples.
                raise ValueError("custom refiner gave up before sampling the function")
            break
        sampled = _as_fun_dtype(sampled)
        scales = _update_running_vscale(sampled, scales)
        # MATLAB extrapolate flags whole rows, even when only one column
        # contains a nonfinite sample; populate restores every column there.
        bad_nan = jnp.isnan(sampled)
        bad_inf = jnp.isinf(sampled)
        if sampled.ndim == 2:
            bad_nan = jnp.any(bad_nan, axis=1, keepdims=True)
            bad_inf = jnp.any(bad_inf, axis=1, keepdims=True)
        values = sampled
        n = values.shape[0]
        if not bool(jnp.all(jnp.isfinite(values))):
            values = _extrapolate_values(
                values, chebpts(n, kind=kind), tech_cls.barywts(n))[0]
        coeffs = tech_cls.vals2coeffs(values)
        happy, cutoff = tech_cls.happiness_check(
            coeffs, values, op=op, tol=tol, vscale=scales, check=check,
            hscale=hscale, sample_test=sample_test)
        if happy:
            return tech_cls(coeffs=coeffs[:cutoff], ishappy=True)
        # Source populate restores NaN/Inf flags after an unhappy check, so
        # nested refinement reuses original bad samples, never extrapolations.
        old_values = jnp.where(bad_nan, jnp.nan, values)
        old_values = jnp.where(bad_inf, jnp.inf, old_values)
    warnings.warn(
        f"{tech_cls.__name__}.from_function: function did not converge with "
        f"{coeffs.shape[0]} points. Returning unhappy representation.",
        stacklevel=2,
    )
    return tech_cls(coeffs=coeffs, ishappy=False)


class _TechOperationError(ValueError):
    """Source Chebtech diagnostic with separate identifier and message."""

    def __init__(self, identifier, message):
        super().__init__(message)
        self.identifier = identifier


def _numeric_array(value):
    """Convert Python numeric literals as MATLAB doubles, preserving typed arrays."""
    try:
        if isinstance(value, (list, tuple, int, float, complex)) and not isinstance(value, bool):
            return _as_fun_dtype(value)
        return jnp.asarray(value)
    except (TypeError, ValueError):
        return None


def _matlab_numeric_class(value, array):
    """Source diagnostic names for explicit numeric dtypes and Python literals."""
    if isinstance(value, str):
        return "char"
    if isinstance(value, dict):
        return "struct"
    if callable(value):
        return "function_handle"
    if array is not None:
        if array.dtype == jnp.bool_:
            return "logical"
        if array.dtype == jnp.float32 or array.dtype == jnp.complex64:
            return "single"
        return str(array.dtype)
    # An unsupported Python class has no MATLAB class counterpart.
    return type(value).__name__


def _mtimes_unknown(value, array):
    name = _matlab_numeric_class(value, array)
    raise _TechOperationError(
        "CHEBFUN:CHEBTECH:mtimes:chebtechMtimesUnknown",
        f"mtimes does not know how to multiply a CHEBTECH and a {name}.",
    )


def _tech_mtimes(tech, other):
    """MATLAB @chebtech/mtimes.m right-operand source dispatch, JAX arithmetic.

    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Python rank-one multipliers represent MATLAB column vectors. Python
    literals represent MATLAB doubles; explicit array dtypes retain source
    class diagnostics. Traced results keep their static coefficient length.
    """
    cls = type(tech)
    if _is_empty_tech(tech) or _is_empty_tech(other):
        return cls.empty()
    if isinstance(other, (Chebtech1, Chebtech2)):
        raise _TechOperationError(
            "CHEBFUN:CHEBTECH:mtimes:chebtechMtimesChebtech",
            "Use .* to multiply CHEBTECH objects.",
        )
    array = _numeric_array(other)
    if array is not None and array.size == 0:
        return cls.empty()
    if array is None or array.dtype not in (jnp.float64, jnp.complex128):
        _mtimes_unknown(other, array)
    scalar_valued = tech.coeffs.ndim == 1
    if array.size == 1:
        out = tech.coeffs * array.reshape(-1)[0]
    else:
        if array.ndim not in (1, 2):
            raise ValueError("matrix multiplier must be a scalar, vector or matrix")
        coefficients = tech.coeffs if not scalar_valued else tech.coeffs[:, None]
        if coefficients.shape[1] != array.shape[0]:
            raise _TechOperationError(
                "CHEBFUN:CHEBTECH:mtimes:size2",
                "Inner matrix dimensions must agree.",
            )
        out = coefficients @ array
        if scalar_valued and out.ndim == 2 and out.shape[1] == 1:
            out = out[:, 0]
    if not isinstance(out, jax.core.Tracer):
        out = _collapse_if_zero(out)
    return cls(coeffs=out, ishappy=tech.ishappy)



def _tech_numeric_times(tech, other):
    """Source numeric TIMES: scalar scaling, bsxfun rows, or column contraction.

    MATLAB source: @chebtech/times.m, Chebfun commit: 7574c77.
    Python one-dimensional arrays spell MATLAB row-vector column weights.
    Traced output keeps its static coefficient count, as for MTIMES.
    """
    array = _numeric_array(other)
    if array is not None and array.size == 0:
        return type(tech).empty()
    if array is None or array.dtype not in (jnp.float64, jnp.complex128):
        raise _TechOperationError(
            "CHEBFUN:CHEBTECH:times:typeMismatch",
            "Incompatible operation between objects.\nMake sure functions are of the same type.",
        )
    if array.size == 1:
        out = tech.coeffs * array.reshape(-1)[0]
    else:
        coefficients = _columns(tech.coeffs)
        if array.ndim == 1:
            out = coefficients * array[None, :]
        elif array.ndim == 2 and array.shape[1] > 1:
            out = coefficients * array
        elif array.ndim == 2:
            out = coefficients @ array
            if tech.coeffs.ndim == 1:
                out = out[:, 0]
        else:
            raise ValueError("pointwise multiplier must be a scalar, vector or matrix")
    if not isinstance(out, jax.core.Tracer):
        out = _collapse_if_zero(out)
    return type(tech)(coeffs=out, ishappy=tech.ishappy)


def _tech_rmtimes(tech, other):
    """Source scalar-left mtimes dispatch, leaving pointwise * unchanged.

    MATLAB source: @chebtech/mtimes.m, Chebfun commit: 7574c77.
    """
    if _is_empty_tech(tech) or _is_empty_tech(other):
        return type(tech).empty()
    array = _numeric_array(other)
    if array is not None and array.size == 0:
        return type(tech).empty()
    if array is not None and array.size > 1:
        raise _TechOperationError(
            "CHEBFUN:CHEBTECH:mtimes:size",
            "Inner matrix dimensions must agree.",
        )
    return _tech_mtimes(tech, other)


def _tech_object_times(tech, other):
    """Source constant recursion, dimension diagnostics and positive products.

    MATLAB source: @chebtech/times.m, Chebfun commit: 7574c77.
    Construction and simplify choose coefficient lengths outside JAX tracing.
    """
    if tech.n == 1:
        return _tech_numeric_times(other, _columns(tech.coeffs))
    if other.n == 1:
        return _tech_numeric_times(tech, _columns(other.coeffs))
    left_columns = 1 if tech.coeffs.ndim == 1 else tech.coeffs.shape[1]
    right_columns = 1 if other.coeffs.ndim == 1 else other.coeffs.shape[1]
    if left_columns != right_columns and left_columns != 1 and right_columns != 1:
        raise _TechOperationError(
            "CHEBFUN:CHEBTECH:times:dim2", "Inner matrix dimensions must agree.",
        )
    n = max(tech.n, other.n)
    left = _columns(_prolong_coeffs(tech.coeffs, n))
    right = _columns(_prolong_coeffs(other.coeffs, n))
    # Source's real-square and conjugate-product branches have the same
    # condition after prolongation and scalar-column broadcasting.
    positive = bool(jnp.all(left == jnp.conj(right)))
    coeffs = _coeff_multiply(tech.coeffs, other.coeffs)
    result = type(tech).from_coeffs(
        coeffs, ishappy=tech.ishappy and other.ishappy).simplify()
    if positive:
        values = jnp.abs(type(result).coeffs2vals(result.coeffs))
        result = type(result).from_coeffs(
            type(result).vals2coeffs(values), ishappy=result.ishappy)
    return result


def _tech_cell2mat(cls, techs):
    """Source cell2mat concatenation, with horzcat empty-argument removal.

    MATLAB source: @chebtech/{cell2mat,horzcat}.m, Chebfun commit: 7574c77.
    Python's flat sequence spells the horizontal-concatenation input.
    """
    if isinstance(techs, (Chebtech1, Chebtech2)):
        return techs
    items = tuple(techs)
    if not items:
        return cls.empty()
    kept = []
    for item in items:
        if _is_empty_tech(item):
            continue
        array = None if isinstance(item, (Chebtech1, Chebtech2)) else _numeric_array(item)
        if array is not None and array.size == 0:
            continue
        if not isinstance(item, (Chebtech1, Chebtech2)):
            raise _TechOperationError(
                "CHEBFUN:CHEBTECH:horzcat:typeMismatch",
                "Incompatible concatenation. Ensure discretizations are of the same type.",
            )
        kept.append(item)
    if not kept:
        return items[0] if isinstance(items[0], (Chebtech1, Chebtech2)) else cls.empty()
    if len(kept) == 1:
        return kept[0]
    n = max(t.n for t in kept)
    columns = []
    for t in kept:
        coefficients = t.prolong(n).coeffs
        columns.append(coefficients if coefficients.ndim == 2 else coefficients[:, None])
    return cls(coeffs=jnp.concatenate(columns, axis=1),
               ishappy=all(t.ishappy for t in kept))


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
        return _is_empty_tech(self)

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
        extrapolate: bool = False,
        start_pow2: int = 4,
        check: str = "standard",
        vscale: float = 0.0,
        sample_test: bool = True,
        hscale: float = 1.0,
        refinement_function: str | Callable = "nested",
        max_length: int | None = None,
        min_samples: int | None = None,
    ) -> "Chebtech2":
        """Construct a Chebtech2 from a callable.

        If ``n`` is given, evaluates the function on an ``n``-point 2nd-kind
        Chebyshev grid and forms the interpolant directly (non-adaptive).

        If ``n`` is ``None`` (the default), uses an adaptive algorithm that
        refines a nested grid while reusing sampled values until the Chebyshev coefficients decay
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

        refinement_function : str or callable, optional
            "nested" reuses old samples; "resampling" evaluates a full grid.
            A callable receives ``(f, old_values, preferences)``. Initially
            ``old_values`` is None; preferences is a dict with MATLAB techPref
            field names. Giving up before sampling raises ValueError.
        max_length, min_samples : int or None, optional
            Maximum grid length and minimum initial sample count. The default
            limits are 65537 and 17; the initial count rounds up to 2**q+1.

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
            plain = cls._adaptive_construct(f, maxpow2, tol=tol,
                                            check=check, vscale=vscale,
                                            extrapolate=extrapolate,
                                            start_pow2=start_pow2,
                                            sample_test=sample_test,
                refinement_function=refinement_function, max_length=max_length,
                min_samples=min_samples, hscale=hscale,
                use_turbo=True, fixed_length=n)
            num = n if n is not None else 2 * len(plain)
            c = _turbo_coeffs(f, plain.coeffs, num)
            return cls(coeffs=c, ishappy=plain.ishappy)
        if n is not None:
            return cls._fixed_construct(f, n, extrapolate=extrapolate)
        return cls._adaptive_construct(f, maxpow2, tol=tol, check=check,
                                       vscale=vscale,
                                       extrapolate=extrapolate,
                                       start_pow2=start_pow2,
                                       sample_test=sample_test,
                refinement_function=refinement_function, max_length=max_length,
                min_samples=min_samples, hscale=hscale)

    @classmethod
    def _fixed_construct(
        cls, f: Callable[[jax.Array], jax.Array], n: int,
        extrapolate: bool = False,
    ) -> "Chebtech2":
        """Fixed-length construction on an n-point Chebyshev-2 grid."""
        if n <= 0:
            return cls(coeffs=jnp.array([], dtype=jnp.float64))
        x = chebpts(n, kind=2)
        values = _sample_extrapolate(f, x, extrapolate)
        if not bool(jnp.all(jnp.isfinite(values))):
            # Extrapolate NaN/Inf samples (MATLAB @chebtech/populate.m).
            values = _extrapolate_values(values, x, cls.barywts(n))[0]
        c = vals2coeffs(values)
        return cls(coeffs=c)

    @classmethod
    def _adaptive_construct(
        cls, f, maxpow2=16, start_pow2=4, tol=None, check="standard",
        vscale=0.0, sample_test=True, hscale=1.0,
        extrapolate: bool = False,
        refinement_function="nested", max_length=None, min_samples=None,
        use_turbo=False, fixed_length=None,
    ) -> "Chebtech2":
        """Source-shaped nested/resampling adaptive population.

        MATLAB source: @chebtech/populate.m, @chebtech2/refine.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return _adaptive_refine_construct(
            cls, 2, f,
            max_length=2**maxpow2+1 if max_length is None else max_length,
            min_samples=2**start_pow2+1 if min_samples is None else min_samples,
            refinement_function=refinement_function,
            tol=tol, check=check, vscale=vscale, sample_test=sample_test,
            hscale=hscale, extrapolate=extrapolate,
            use_turbo=use_turbo, fixed_length=fixed_length,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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
        This method is JIT-safe, grad-safe, and vmap-safe.  Concrete
        inputs (with concrete coefficients) evaluate via numpy Clenshaw
        (``np.polynomial.chebyshev.chebval``): the jitted path compiles
        one XLA program per (coefficient length, point shape) pair, and
        object-heavy pipelines -- rootfinding curve fits, adaptive
        constructors -- otherwise accumulate thousands of compiles.

        Provenance
        ----------
        MATLAB source : @chebtech/clenshaw.m, @chebtech/feval.m
        Chebfun commit: 7574c77
        """
        if not isinstance(x, jax.core.Tracer) and \
                not isinstance(self.coeffs, jax.core.Tracer) and \
                self.coeffs.shape[0] <= 4096:
            # chebval loops once per coefficient in Python; past a few
            # thousand coefficients the compiled Clenshaw is faster, and
            # huge-length Chebtechs are rare enough not to threaten the
            # JIT code arena.
            import numpy.polynomial.chebyshev as _ncheb

            xn = np.asarray(x)
            xn = xn.astype(np.complex128 if np.iscomplexobj(xn)
                           else np.float64)
            c = np.asarray(self.coeffs)
            if c.shape[0] == 0:
                return jnp.zeros(xn.shape + c.shape[1:],
                                 dtype=c.dtype if c.size else np.float64)
            if xn.ndim == 0 and c.ndim == 1 and c.shape[0] <= 4096:
                # Scalar point, 1-D coefficients: Clenshaw with Python
                # scalars.  chebval's per-coefficient numpy scalar ops
                # cost ~0.13 ms for a length-40 series — the ODE marcher
                # evaluates coefficient chebfuns one scalar at a time.
                cl = c.tolist()
                xf = complex(xn) if (np.iscomplexobj(xn)
                                     or np.iscomplexobj(c)) else float(xn)
                x2 = 2.0 * xf
                b1 = b2 = 0.0
                for ck in cl[:0:-1]:
                    b1, b2 = x2 * b1 - b2 + ck, b1
                return jnp.asarray(xf * b1 - b2 + cl[0])
            val = _ncheb.chebval(xn, c, tensor=True)
            # chebval returns c.shape[1:] + x.shape; the traced path
            # returns x.shape + c.shape[1:].
            if c.ndim > 1:
                val = np.moveaxis(val, range(c.ndim - 1),
                                  range(-(c.ndim - 1), 0))
            return jnp.asarray(val)
        return self._call_traced(x)

    @eqx.filter_jit
    def _call_traced(self, x: jax.Array) -> jax.Array:
        # Preserve a complex argument (MATLAB evaluates a real Chebyshev
        # series at complex points via Clenshaw with a complex recurrence);
        # everything else is promoted to float64.
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            x = x.astype(jnp.complex128)
        else:
            x = x.astype(jnp.float64)
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
    def extrapolate(values: jax.Array) -> jax.Array:
        """Extrapolate NaN/Inf sample rows via barycentric interpolation.

        Replaces every row of ``values`` holding a NaN or Inf (in any column)
        by the barycentric interpolant of the finite rows, at the 2nd-kind
        Chebyshev points.  Finite rows -- including the endpoints -- are
        returned unchanged, so this reverts to the identity on clean data.

        Parameters
        ----------
        values : jax.Array, shape (n,) or (n, m)
            Sampled function values (may contain NaN/Inf).

        Returns
        -------
        jax.Array
            ``values`` with the masked rows replaced.

        Provenance
        ----------
        MATLAB source : @chebtech/extrapolate.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        v = jnp.asarray(values)
        n = v.shape[0]
        new_values, _, _ = _extrapolate_values(
            v, chebpts(n, kind=2), Chebtech2.barywts(n)
        )
        return new_values

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

    @property
    def vscale_columns(self) -> jax.Array:
        """Vertical scale per array-valued column, as a JAX vector.

        Provenance
        ----------
        MATLAB source : @chebtech/vscale.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and
            The Chebfun Developers.

        For the no-field ``Tech.empty()`` sentinel this returns a length-zero
        vector, matching its zero-column helper shape; MATLAB's public
        ``vscale`` returns scalar zero for an empty tech.
        """
        if getattr(self, "_is_empty_object", False):
            return jnp.empty((0,), dtype=jnp.float64)
        if self.coeffs.size == 0:
            ncols = 1 if self.coeffs.ndim == 1 else self.coeffs.shape[1]
            return jnp.zeros((ncols,), dtype=jnp.float64)
        values = self.values
        if values.ndim == 1:
            values = values[:, None]
        return jnp.max(jnp.abs(values), axis=0)

    def normest(self):
        """Estimate the infinity norm from values on the representation grid.

        Provenance
        ----------
        MATLAB source : @chebtech/normest.m
        Chebfun commit: 7574c77
        JAX contract: JIT and differentiation preserve the scalar array.
        """
        return jnp.max(jnp.abs(self.values))

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

    def simplify(self, tol: float | jax.Array | None = None) -> "Chebtech2":
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
        if self.isempty() or self.n == 0:
            return self
        if not self.ishappy:
            return self

        nold = self.n
        # Prolong to give standard_chop room for plateau detection
        N = max(17, _round_half_away(nold * 1.25 + 5))
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
        extrapolate: bool = False,
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
        extrapolate : bool, default False
            Apply MATLAB's extrapolation policy while populating samples.
            The default preserves the existing compose behavior.

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
            if self.coeffs.ndim == 2 and op.coeffs.ndim == 2:
                # MATLAB @chebtech/compose.m: composing two array-valued
                # techs g(f) is unsupported ('CHEBTECH:compose:arrval').
                # Previously this failed incidentally inside the traced
                # evaluation; the numpy eval path broadcasts instead, so
                # enforce the contract explicitly.
                raise ValueError(
                    "Chebtech2.compose: cannot compose two array-valued "
                    "techs g(f) (MATLAB CHEBTECH:compose:arrval).")
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
            # MATLAB @chebtech/compose.m sets sampleTest=false after raising
            # minSamples to cover every operand.
            sample_test=False,
            extrapolate=extrapolate,
        )

    # ------------------------------------------------------------------
    # Restriction
    # ------------------------------------------------------------------

    def restrict(self, a, b: float | None = None):
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
        if b is None or not isinstance(a, (int, float)):
            # MATLAB restrict(f, s) with a breakpoint vector s: one tech per
            # sub-interval, returned as a list when s has more than two
            # entries (MATLAB returns a cell array there).
            brk = [float(t) for t in jnp.asarray(a).reshape(-1)]
            if len(brk) < 2:
                raise ValueError(
                    "CHEBFUN:CHEBTECH:restrict:badInterval: "
                    "need at least two breakpoints.")
            if len(brk) == 2:
                return self.restrict(brk[0], brk[1])
            return [self.restrict(brk[i], brk[i + 1])
                    for i in range(len(brk) - 1)]
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
        check: str = "standard",
        sample_test: bool = True,
    ) -> tuple[bool, int | None]:
        """Happiness check for adaptive construction.

        Tests whether a Chebyshev coefficient sequence has converged.  With
        ``check='standard'`` (the default) it calls ``standard_chop`` with the
        tolerance scaled by ``max(hscale, vscale / vscale_local)`` (matching
        MATLAB's ``@chebtech/standardCheck.m``).  The ``'strict'`` and
        ``'classic'`` variants dispatch to the corresponding MATLAB happiness
        checks (``@chebtech/strictCheck.m`` / ``classicCheck.m``).

        Optionally performs a sample test: evaluates the operator ``op``
        and the Chebyshev interpolant at two off-grid points and checks
        that they agree to within ``sqrt(tol) * vscale``.

        Parameters
        ----------
        coeffs : jax.Array, shape (n,) or (n, m)
            Chebyshev coefficients.
        values : jax.Array, shape (n,) or (n, m)
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
        check : {'standard', 'strict', 'classic'}, default 'standard'
            Which happiness definition to use (MATLAB ``pref.happinessCheck``).

        Returns
        -------
        ishappy : bool
            True if the representation has converged.
        cutoff : int or None
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
            @chebtech/strictCheck.m, @chebtech/classicCheck.m,
            @chebtech/sampleTest.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        standard_chop
        """
        return _happiness_check_impl(
            Chebtech2, 2, coeffs, values, op, tol, vscale, hscale, check,
            sample_test,
        )

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
        if _defers_binary(other):
            return NotImplemented
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech2.empty()
        if isinstance(other, Chebtech2):
            # Prolong to the same length (zero-pad shorter one)
            nf = self.n
            ng = other.n
            n = max(nf, ng)
            fc = _prolong_coeffs(self.coeffs, n)
            gc = _prolong_coeffs(other.coeffs, n)
            fc, gc = _expand_coeff_pair(fc, gc)
            return Chebtech2.from_coeffs(
                _collapse_if_zero(fc + gc),
                ishappy=self.ishappy and other.ishappy)
        else:
            # Scalar addition: only the c_0 coefficient changes. Promote the
            # coefficient dtype first — scattering a complex scalar into a
            # float64 buffer silently drops the imaginary part.
            s = _as_scalar(other)
            c = self.coeffs.astype(jnp.result_type(self.coeffs.dtype, s.dtype))
            # MATLAB plus.m performs singleton expansion of f when the double
            # is a row vector with more columns than f has.
            ncols = s.shape[-1] if s.ndim else 1
            if ncols > 1 and c.ndim == 1:
                c = jnp.tile(c[:, None], (1, ncols))
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
        if _defers_binary(other):
            return NotImplemented
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
        if _defers_binary(other):
            return NotImplemented
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech2.empty()
        if isinstance(other, (Chebtech1, Chebtech2)):
            return _tech_object_times(self, other)
        else:
            return _tech_numeric_times(self, other)

    def __rmul__(self, other) -> "Chebtech2":
        """Numeric-left pointwise TIMES (column scaling), not matrix MTIMES.

        Provenance
        ----------
        MATLAB source : @chebtech/times.m
        Chebfun commit: 7574c77
        """
        return self.__mul__(other)

    def __rmatmul__(self, other):
        """MATLAB numeric-left MTIMES; only a scalar multiplier is accepted.

        Provenance
        ----------
        MATLAB source : @chebtech/mtimes.m
        Chebfun commit: 7574c77
        """
        return _tech_rmtimes(self, other)

    def __matmul__(self, other) -> "Chebtech2":
        """MATLAB mtimes ``f * A``: right-multiply an array-valued tech
        by a matrix, mixing its columns (coeffs @ A).

        Provenance
        ----------
        MATLAB source : @chebtech/mtimes.m
        Chebfun commit: 7574c77
        """
        return _tech_mtimes(self, other)

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

    def qr(self, mode: str = "matrix", method: str = "built-in",
           want_e: bool = False):
        """QR factorisation ``f = Q R`` of an array-valued tech.

        ``Q`` is a tech with the same number of columns as ``f`` whose
        columns are orthonormal in the continuous L2 inner product on
        [-1, 1]; ``R`` is ``m x m`` upper-triangular with a non-negative
        diagonal.  A single-column tech is simply normalised.

        Parameters
        ----------
        mode : {'matrix', 'vector'}, default 'matrix'
            Form of the optional permutation output ``E``.
        method : {'built-in', 'householder'}, default 'built-in'
            'built-in' orthogonalises a Gauss-Legendre-weighted matrix of
            nodal values with a dense QR; 'householder' uses Trefethen's
            Householder triangularisation of a quasimatrix.
        want_e : bool, default False
            If True, also return ``E``.  Neither method pivots, so ``E``
            is the identity (as a vector or a matrix, per ``mode``).

        Returns
        -------
        (Q, R) or (Q, R, E)

        NOT JIT-safe (dense linear algebra with data-dependent shapes).

        Provenance
        ----------
        MATLAB source : @chebtech/qr.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm:
            L. N. Trefethen, "Householder triangularization of a
            quasimatrix", IMA J. Numer. Anal., 30(4):887-897, 2010.

        See Also
        --------
        mldivide, mrdivide
        """
        return _tech_qr(self, mode=mode, method=method, want_e=want_e)

    def mldivide(self, other) -> jax.Array:
        """``A \\ B``: continuous-L2 least-squares solution of ``A X = B``.

        Both operands must be techs of the same type.  Returns the numeric
        coefficient matrix ``X`` (a vector when ``B`` has one column).

        Parameters
        ----------
        other : Chebtech2
            Right-hand side.

        Returns
        -------
        jax.Array

        NOT JIT-safe (calls :meth:`qr`).

        Provenance
        ----------
        MATLAB source : @chebtech/mldivide.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        qr, mrdivide
        """
        return _tech_mldivide(self, other)

    def mrdivide(self, other):
        """``A / B``: right matrix divide by a scalar or matrix.

        Dividing by a scalar rescales the coefficients.  Dividing by a
        matrix gives the continuous-L2 least-squares solution of
        ``X B = A``.  ``mrdivide`` between two techs is an error (use
        ``/`` elementwise division instead).

        Parameters
        ----------
        other : float or jax.Array
            Scalar or matrix divisor.

        Returns
        -------
        Chebtech2

        NOT JIT-safe (calls :meth:`qr`).

        Provenance
        ----------
        MATLAB source : @chebtech/mrdivide.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        qr, mldivide
        """
        return _tech_mrdivide(self, other)

    @staticmethod
    def rmrdivide(numeric, tech):
        """``A / B`` with a numeric ``A`` and a tech ``B`` (least squares).

        Parameters
        ----------
        numeric : float or jax.Array
            Numerator.
        tech : Chebtech2
            Denominator.

        Returns
        -------
        Chebtech2

        NOT JIT-safe (calls :meth:`qr`).

        Provenance
        ----------
        MATLAB source : @chebtech/mrdivide.m (``double / chebtech`` branch)
        Chebfun commit: 7574c77

        See Also
        --------
        mrdivide, qr
        """
        return _tech_mrdivide(numeric, tech)

    def isequal(self, other) -> bool:
        """True when two techs have the same coefficient array.

        Parameters
        ----------
        other : Chebtech2
            Tech to compare against.

        Returns
        -------
        bool

        Provenance
        ----------
        MATLAB source : @chebtech/isequal.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        return _tech_isequal(self, other)

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
        return _tech_cell2mat(cls, techs)

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
        if _defers_binary(other):
            return NotImplemented
        if isinstance(other, Chebtech2):
            # MATLAB: compose(f, @rdivide, g) — adaptive re-construction so
            # the quotient is resolved to machine precision (a fixed grid
            # silently under-resolves, e.g. 1/(1+25x^2) needs ~185 coeffs).
            return self.compose(lambda a, b: a / b, other)
        else:
            _check_rdivide_shape(self.coeffs, other)
            return Chebtech2.from_coeffs(self.coeffs / _as_scalar(other), ishappy=self.ishappy)

    def __rtruediv__(self, other) -> "Chebtech2":
        """Scalar / Chebtech2 (adaptive, like MATLAB compose)."""
        return self.compose(lambda y: _as_scalar(other) / y)

    def __pow__(self, exponent) -> "Chebtech2":
        """Raise to a power.

        Integer powers of at least three use adaptive composition.
        Lower nonnegative integer powers use coefficient arithmetic.
        Non-integer powers via evaluation on a grid and re-interpolation.

        Provenance
        ----------
        MATLAB source : @chebtech/power.m
        Chebfun commit: 7574c77
        """
        if isinstance(exponent, int) and exponent >= 3:
            # @chebtech/power.m composes the operator. Repeated TIMES
            # accumulates avoidable error in singular-function cubes.
            return self.compose(lambda y: y ** exponent)
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

    def cumsum(self, dim: int = 1) -> "Chebtech2":
        """Indefinite integral (antiderivative with F(-1) = 0).

        ``dim=1`` integrates the continuous variable. All other dimensions
        cumulatively sum coefficient columns, matching MATLAB's branch.

        Parameters
        ----------
        dim : int, default 1
            Static dimension selector. Values other than 1 select the source
            coefficient-column cumulative sum.

        Returns
        -------
        Chebtech2
            The antiderivative for ``dim=1`` or column-prefix tech for other
            dimensions.

        JIT and differentiation
        -----------------------
        ``dim`` is a static Python integer. ``_cumsum_coeffs_by_dim`` is a
        JIT-safe fixed-shape coefficient adapter. The eager continuous path
        also applies MATLAB's adaptive simplify and final lval correction;
        under tracing, it uses the JIT-safe coefficient recurrence without
        host-adaptive simplify.

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
        if self.isempty() or self.coeffs.size == 0:
            return self
        if dim != 1:
            if self.coeffs.ndim == 1:
                return self
            new_coeffs = _cumsum_coeffs_by_dim(self.coeffs, dim=dim)
            return Chebtech2(coeffs=new_coeffs, ishappy=self.ishappy)

        new_coeffs = _cumsum_coeffs(self.coeffs)
        if isinstance(self.coeffs, jax.core.Tracer):
            return Chebtech2.from_coeffs(new_coeffs, ishappy=self.ishappy)
        result = Chebtech2.from_coeffs(
            new_coeffs, ishappy=self.ishappy
        ).simplify()
        if result.isempty() or result.coeffs.size == 0:
            return result
        corrected = result.coeffs.at[0].add(-_cumsum_lval(result.coeffs))
        return eqx.tree_at(lambda tech: tech.coeffs, result, corrected)

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

    def roots(self, qz: bool = False, *, complex_roots: bool = False,
              all_roots: bool = False, prune: bool = False,
              recurse: bool = True) -> jax.Array:
        """Roots in [-1, 1] via colleague matrix eigenvalues.

        NOT JIT-safe (variable output size, recursive subdivision).

        Parameters
        ----------
        qz : bool, default False
            When True, compute the small-degree roots from the colleague
            matrix *pencil* via the QZ (generalized eigenvalue) algorithm
            for extra numerical stability, mirroring ``roots(f, 'qz', 1)``
            in MATLAB.  When False, the standard colleague matrix and the
            QR algorithm are used.
        complex_roots : bool, default False
            MATLAB's ``'complex'`` flag: equivalent to setting both
            ``all_roots`` and ``prune`` True.  Returns the (pruned) complex
            roots, including those off the real axis.
        all_roots : bool, default False
            MATLAB's ``'all'``: return every root the linearization yields
            (complex, and outside [-1, 1]) rather than only real roots in
            [-1, 1].
        prune : bool, default False
            MATLAB's ``'prune'``: when ``all_roots`` holds, discard
            'spurious' roots by the Bernstein-radius test.
        recurse : bool, default True
            MATLAB's ``'recurse'``: when False, never subdivide; solve one
            colleague eigenproblem for the whole series.

        Returns
        -------
        jax.Array, shape (n_roots,)
            Roots (sorted, real by default; complex when ``all_roots``).

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
            [3] Y. Nakatsukasa and V. Noferini, "On the stability of
                polynomial rootfinding via linearizations in nonmonomial
                bases" (the QZ variant).

        See Also
        --------
        diff, sum
        """
        if complex_roots:
            all_roots = True
            prune = True
        kw = dict(qz=qz, all_roots=all_roots, prune=prune, recurse=recurse)
        if self.coeffs.ndim == 2:
            # Array-valued: roots per column, NaN-padded to equal length
            # (MATLAB @chebtech/roots.m does exactly this)
            import numpy as _np
            cols = [_np.asarray(_roots_colleague(self.coeffs[:, j], **kw))
                    for j in range(self.coeffs.shape[1])]
            nmax = max((len(c) for c in cols), default=0)
            dt = (_np.complex128
                  if any(_np.iscomplexobj(c) for c in cols) else float)
            out = _np.full((nmax, len(cols)), _np.nan, dtype=dt)
            for j, c in enumerate(cols):
                out[: len(c), j] = c
            return jnp.asarray(out)
        return _roots_colleague(self.coeffs, **kw)

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

    # ------------------------------------------------------------------
    # Sign, monomial coefficients, boundary-root extraction, T->U
    # ------------------------------------------------------------------

    def sign(self) -> "Chebtech2":
        """Signum of the function (assumes no interior roots).

        For a real ``f`` with no roots in ``[-1, 1]`` this returns the
        constant ``+1`` or ``-1`` matching the sign of ``f`` (evaluated as
        the mean over the endpoints and an arbitrary interior point).  For
        complex ``f`` it returns ``f / |f|`` via ``compose``.  As in MATLAB
        ``@chebtech/sign.m``, no warning is issued when ``f`` has roots.

        Provenance
        ----------
        MATLAB source : @chebtech/sign.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        if _is_empty_tech(self):
            return Chebtech2.empty()
        if not jnp.iscomplexobj(self.coeffs):
            arbitrary_point = 0.1273881594
            fx = self(jnp.array([-1.0, arbitrary_point, 1.0],
                                dtype=jnp.float64))
            meanfx = jnp.mean(fx, axis=0)
            s = jnp.sign(meanfx)
            if self.coeffs.ndim == 2:
                return Chebtech2(coeffs=jnp.atleast_1d(s)[None, :])
            return Chebtech2.from_coeffs(jnp.atleast_1d(s))
        return self.compose(lambda x: x / jnp.abs(x))

    def poly(self) -> jax.Array:
        """Monomial (power-basis) coefficients of the function.

        Returns ``C`` so that ``f(x) = C[0]*x^(n-1) + ... + C[n-1]``.  For
        an array-valued tech the rows of ``C`` correspond to the columns of
        ``f`` (as in MATLAB ``poly``).

        Provenance
        ----------
        MATLAB source : @chebtech/poly.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self):
            return jnp.array([], dtype=jnp.float64)
        return _poly_coeffs(self.coeffs)

    def extractBoundaryRoots(self, num_roots=None):
        """Extract roots at the boundary points -1 and +1.

        Returns ``(g, rootsLeft, rootsRight)`` where ``g`` is a Chebtech2
        free of boundary roots and ``rootsLeft`` / ``rootsRight`` are the
        multiplicities removed at ``x = -1`` and ``x = +1``.

        Provenance
        ----------
        MATLAB source : @chebtech/extractBoundaryRoots.m
        Chebfun commit: 7574c77
        """
        if getattr(self, "_is_empty_object", False):
            counts = jnp.zeros((0,), dtype=jnp.int32)
            return self, counts, counts
        if bool(_has_no_boundary_roots(self.coeffs, self.vscale_columns)):
            ncols = 1 if self.coeffs.ndim == 1 else self.coeffs.shape[1]
            zeros = jnp.zeros((ncols,), dtype=jnp.int32)
            if self.coeffs.ndim == 1:
                return self, zeros[0], zeros[0]
            return self, zeros, zeros
        c, l, r = _extract_boundary_roots(
            self.coeffs, self.vscale_columns, num_roots
        )
        g = Chebtech2(coeffs=c, ishappy=self.ishappy).simplify()
        return g, l, r

    @staticmethod
    def chebTcoeffs2chebUcoeffs(cT: jax.Array) -> jax.Array:
        """Convert Chebyshev-T coefficients to Chebyshev-U coefficients.

        Provenance
        ----------
        MATLAB source : @chebtech/chebTcoeffs2chebUcoeffs.m
        Chebfun commit: 7574c77
        """
        return _chebT_to_chebU_coeffs(cT)


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
        return _is_empty_tech(self)

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
        check: str = "standard",
        sample_test: bool = True,
        refinement_function: str | Callable = "nested",
        max_length: int | None = None,
        min_samples: int | None = None,
        tol: float | None = None,
        vscale: float = 0.0,
        hscale: float = 1.0,
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
            Legacy grid cap is 2**maxpow2+1; max_length overrides it.

        refinement_function : str or callable, optional
            "nested" reuses old samples; "resampling" evaluates a full grid.
            A callable receives ``(f, old_values, preferences)``. Initially
            ``old_values`` is None; preferences is a dict with MATLAB techPref
            field names. Giving up before sampling raises ValueError.
        max_length, min_samples : int or None, optional
            Maximum grid length and minimum initial sample count. The default
            limits are 65537 and 17; the initial count rounds up to 2**q+1.

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
            plain = cls._adaptive_construct(
                f, maxpow2, check=check, sample_test=sample_test,
                refinement_function=refinement_function, max_length=max_length,
                min_samples=min_samples, tol=tol, vscale=vscale, hscale=hscale,
                use_turbo=True, fixed_length=n)
            num = n if n is not None else 2 * len(plain)
            c = _turbo_coeffs(f, plain.coeffs, num)
            return cls(coeffs=c, ishappy=plain.ishappy)
        if n is not None:
            return cls._fixed_construct(f, n)
        return cls._adaptive_construct(
            f, maxpow2, check=check, sample_test=sample_test,
                refinement_function=refinement_function, max_length=max_length,
                min_samples=min_samples, tol=tol, vscale=vscale, hscale=hscale)

    @classmethod
    def _fixed_construct(
        cls, f: Callable[[jax.Array], jax.Array], n: int
    ) -> "Chebtech1":
        """Fixed-length construction on an n-point Chebyshev-1 grid."""
        if n <= 0:
            return cls(coeffs=jnp.array([], dtype=jnp.float64))
        x = chebpts(n, kind=1)
        values = _as_fun_dtype(f(x))
        if not bool(jnp.all(jnp.isfinite(values))):
            # Extrapolate NaN/Inf samples (MATLAB @chebtech/populate.m).
            values = _extrapolate_values(values, x, cls.barywts(n))[0]
        c = _chebtech1_vals2coeffs(values)
        return cls(coeffs=c)

    @classmethod
    def _adaptive_construct(
        cls, f, maxpow2=16, start_pow2=4, tol=None, check="standard",
        vscale=0.0, sample_test=True, hscale=1.0,
        refinement_function="nested", max_length=None, min_samples=None,
        use_turbo=False, fixed_length=None,
    ) -> "Chebtech1":
        """Source-shaped nested/resampling adaptive population.

        MATLAB source: @chebtech/populate.m, @chebtech1/refine.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return _adaptive_refine_construct(
            cls, 1, f,
            max_length=2**maxpow2+1 if max_length is None else max_length,
            min_samples=2**start_pow2+1 if min_samples is None else min_samples,
            refinement_function=refinement_function,
            tol=tol, check=check, vscale=vscale, sample_test=sample_test,
            hscale=hscale, use_turbo=use_turbo, fixed_length=fixed_length,
        )

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
        # Preserve a complex argument (MATLAB evaluates a real Chebyshev
        # series at complex points via Clenshaw with a complex recurrence);
        # everything else is promoted to float64.
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            x = x.astype(jnp.complex128)
        else:
            x = x.astype(jnp.float64)
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
    def extrapolate(values: jax.Array) -> jax.Array:
        """Extrapolate NaN/Inf sample rows via barycentric interpolation.

        Replaces every row of ``values`` holding a NaN or Inf (in any column)
        by the barycentric interpolant of the finite rows, at the 1st-kind
        Chebyshev points.  Finite rows are returned unchanged.

        Provenance
        ----------
        MATLAB source : @chebtech/extrapolate.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        v = jnp.asarray(values)
        n = v.shape[0]
        new_values, _, _ = _extrapolate_values(
            v, chebpts(n, kind=1), Chebtech1.barywts(n)
        )
        return new_values

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

    @property
    def vscale_columns(self) -> jax.Array:
        """Vertical scale per array-valued column, as a JAX vector.

        Provenance
        ----------
        MATLAB source : @chebtech/vscale.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and
            The Chebfun Developers.

        For the no-field ``Tech.empty()`` sentinel this returns a length-zero
        vector, matching its zero-column helper shape; MATLAB's public
        ``vscale`` returns scalar zero for an empty tech.
        """
        if getattr(self, "_is_empty_object", False):
            return jnp.empty((0,), dtype=jnp.float64)
        if self.coeffs.size == 0:
            ncols = 1 if self.coeffs.ndim == 1 else self.coeffs.shape[1]
            return jnp.zeros((ncols,), dtype=jnp.float64)
        values = self.values
        if values.ndim == 1:
            values = values[:, None]
        return jnp.max(jnp.abs(values), axis=0)

    def normest(self):
        """Estimate the infinity norm from values on the representation grid.

        Provenance
        ----------
        MATLAB source : @chebtech/normest.m
        Chebfun commit: 7574c77
        JAX contract: JIT and differentiation preserve the scalar array.
        """
        return jnp.max(jnp.abs(self.values))

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

    def simplify(self, tol: float | jax.Array | None = None) -> "Chebtech1":
        """Return a new Chebtech1 with trailing coefficients chopped.

        Provenance
        ----------
        MATLAB source : @chebtech/simplify.m
        Chebfun commit: 7574c77
        """
        if self.isempty() or self.n == 0:
            return self
        if not self.ishappy:
            return self
        nold = self.n
        N = max(17, _round_half_away(nold * 1.25 + 5))
        prolonged_c = jnp.concatenate(
            [self.coeffs,
             jnp.zeros((N - nold,) + self.coeffs.shape[1:],
                       dtype=self.coeffs.dtype)]
        )
        # MATLAB @chebtech/simplify.m uses the Chebtech2 transforms for this
        # plateau round-trip, including for Chebtech1 inputs.
        c = vals2coeffs(coeffs2vals(prolonged_c))
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
        if _defers_binary(other):
            return NotImplemented
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech1.empty()
        if isinstance(other, Chebtech1):
            n = max(self.n, other.n)
            fc = _prolong_coeffs(self.coeffs, n)
            gc = _prolong_coeffs(other.coeffs, n)
            fc, gc = _expand_coeff_pair(fc, gc)
            return Chebtech1.from_coeffs(
                _collapse_if_zero(fc + gc),
                ishappy=self.ishappy and other.ishappy)
        else:
            # Scalar addition changes only c_0.  Promote the coefficient
            # dtype first — scattering a complex scalar into a float64
            # buffer silently drops the imaginary part.
            s = _as_scalar(other)
            c = self.coeffs.astype(jnp.result_type(self.coeffs.dtype, s.dtype))
            # MATLAB plus.m performs singleton expansion of f when the double
            # is a row vector with more columns than f has.
            ncols = s.shape[-1] if s.ndim else 1
            if ncols > 1 and c.ndim == 1:
                c = jnp.tile(c[:, None], (1, ncols))
            c = c.at[0].add(s)
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
        if _defers_binary(other):
            return NotImplemented
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
        if _defers_binary(other):
            return NotImplemented
        if _is_empty_tech(self) or _is_empty_tech(other):
            return Chebtech1.empty()
        if isinstance(other, (Chebtech1, Chebtech2)):
            return _tech_object_times(self, other)
        else:
            return _tech_numeric_times(self, other)

    def __rmul__(self, other) -> "Chebtech1":
        """Numeric-left pointwise TIMES (column scaling), not matrix MTIMES.

        Provenance
        ----------
        MATLAB source : @chebtech/times.m
        Chebfun commit: 7574c77
        """
        return self.__mul__(other)

    def __rmatmul__(self, other):
        """MATLAB numeric-left MTIMES; only a scalar multiplier is accepted.

        Provenance
        ----------
        MATLAB source : @chebtech/mtimes.m
        Chebfun commit: 7574c77
        """
        return _tech_rmtimes(self, other)

    def __matmul__(self, other) -> "Chebtech1":
        """MATLAB mtimes ``f * A``: right-multiply an array-valued tech
        by a matrix, mixing its columns (coeffs @ A).

        Provenance
        ----------
        MATLAB source : @chebtech/mtimes.m
        Chebfun commit: 7574c77
        """
        return _tech_mtimes(self, other)

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

    def qr(self, mode: str = "matrix", method: str = "built-in",
           want_e: bool = False):
        """QR factorisation ``f = Q R`` of an array-valued tech.

        ``Q`` is a tech with the same number of columns as ``f`` whose
        columns are orthonormal in the continuous L2 inner product on
        [-1, 1]; ``R`` is ``m x m`` upper-triangular with a non-negative
        diagonal.  A single-column tech is simply normalised.

        Parameters
        ----------
        mode : {'matrix', 'vector'}, default 'matrix'
            Form of the optional permutation output ``E``.
        method : {'built-in', 'householder'}, default 'built-in'
            'built-in' orthogonalises a Gauss-Legendre-weighted matrix of
            nodal values with a dense QR; 'householder' uses Trefethen's
            Householder triangularisation of a quasimatrix.
        want_e : bool, default False
            If True, also return ``E``.  Neither method pivots, so ``E``
            is the identity (as a vector or a matrix, per ``mode``).

        Returns
        -------
        (Q, R) or (Q, R, E)

        NOT JIT-safe (dense linear algebra with data-dependent shapes).

        Provenance
        ----------
        MATLAB source : @chebtech/qr.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm:
            L. N. Trefethen, "Householder triangularization of a
            quasimatrix", IMA J. Numer. Anal., 30(4):887-897, 2010.

        See Also
        --------
        mldivide, mrdivide
        """
        return _tech_qr(self, mode=mode, method=method, want_e=want_e)

    def mldivide(self, other) -> jax.Array:
        """``A \\ B``: continuous-L2 least-squares solution of ``A X = B``.

        Both operands must be techs of the same type.  Returns the numeric
        coefficient matrix ``X`` (a vector when ``B`` has one column).

        Parameters
        ----------
        other : Chebtech1
            Right-hand side.

        Returns
        -------
        jax.Array

        NOT JIT-safe (calls :meth:`qr`).

        Provenance
        ----------
        MATLAB source : @chebtech/mldivide.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        qr, mrdivide
        """
        return _tech_mldivide(self, other)

    def mrdivide(self, other):
        """``A / B``: right matrix divide by a scalar or matrix.

        Dividing by a scalar rescales the coefficients.  Dividing by a
        matrix gives the continuous-L2 least-squares solution of
        ``X B = A``.  ``mrdivide`` between two techs is an error (use
        ``/`` elementwise division instead).

        Parameters
        ----------
        other : float or jax.Array
            Scalar or matrix divisor.

        Returns
        -------
        Chebtech1

        NOT JIT-safe (calls :meth:`qr`).

        Provenance
        ----------
        MATLAB source : @chebtech/mrdivide.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        qr, mldivide
        """
        return _tech_mrdivide(self, other)

    @staticmethod
    def rmrdivide(numeric, tech):
        """``A / B`` with a numeric ``A`` and a tech ``B`` (least squares).

        Parameters
        ----------
        numeric : float or jax.Array
            Numerator.
        tech : Chebtech1
            Denominator.

        Returns
        -------
        Chebtech1

        NOT JIT-safe (calls :meth:`qr`).

        Provenance
        ----------
        MATLAB source : @chebtech/mrdivide.m (``double / chebtech`` branch)
        Chebfun commit: 7574c77

        See Also
        --------
        mrdivide, qr
        """
        return _tech_mrdivide(numeric, tech)

    def isequal(self, other) -> bool:
        """True when two techs have the same coefficient array.

        Parameters
        ----------
        other : Chebtech1
            Tech to compare against.

        Returns
        -------
        bool

        Provenance
        ----------
        MATLAB source : @chebtech/isequal.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        return _tech_isequal(self, other)

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
        return _tech_cell2mat(cls, techs)

    def __truediv__(self, other) -> "Chebtech1":
        """Division.

        Provenance
        ----------
        MATLAB source : @chebtech/rdivide.m
        Chebfun commit: 7574c77
        """
        if _defers_binary(other):
            return NotImplemented
        if isinstance(other, Chebtech1):
            # Adaptive re-construction so the quotient is fully resolved
            # (MATLAB: compose(f, @rdivide, g)).
            return self.compose(lambda a, b: a / b, other)
        else:
            _check_rdivide_shape(self.coeffs, other)
            return Chebtech1.from_coeffs(self.coeffs / _as_scalar(other), ishappy=self.ishappy)

    def __rtruediv__(self, other) -> "Chebtech1":
        return self.compose(lambda y: _as_scalar(other) / y)

    def __pow__(self, exponent) -> "Chebtech1":
        """Raise to a power.

        Integer powers of at least three use source adaptive composition.

        Provenance
        ----------
        MATLAB source : @chebtech/power.m
        Chebfun commit: 7574c77
        """
        if isinstance(exponent, int) and exponent >= 3:
            return self.compose(lambda y: y ** exponent)
        if isinstance(exponent, int) and exponent >= 0:
            if exponent == 0:
                return Chebtech1.from_coeffs(jnp.array([1.0], dtype=jnp.float64))
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        elif isinstance(exponent, Chebtech1):
            # @chebtech/power.m delegates tech-valued exponents to compose.
            return self.compose(lambda a, b: a ** b, exponent)
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

    def cumsum(self, dim: int = 1) -> "Chebtech1":
        """Integrate continuously or cumulatively sum coefficient columns.

        ``dim=1`` gives the antiderivative with F(-1)=0. All other dimensions
        use a prefix sum over coefficient columns, matching MATLAB's branch.

        Parameters
        ----------
        dim : int, default 1
            Static dimension selector. Values other than 1 select the source
            coefficient-column cumulative sum.

        ``dim`` is a static Python integer under JIT. The pure coefficient
        adapter ``_cumsum_coeffs_by_dim`` retains JIT and AD support; the eager
        continuous path also applies source adaptive simplify and its final
        lval correction. Under tracing, that path uses the unsimplified
        coefficient recurrence because simplify is host-adaptive.

        Provenance
        ----------
        MATLAB source : @chebtech/cumsum.m
        Chebfun commit: 7574c77
        Algorithm: Pages 32-33 of Mason & Handscomb, "Chebyshev Polynomials".
        """
        if self.isempty() or self.coeffs.size == 0:
            return self
        if dim != 1:
            if self.coeffs.ndim == 1:
                return self
            new_coeffs = _cumsum_coeffs_by_dim(self.coeffs, dim=dim)
            return Chebtech1(coeffs=new_coeffs, ishappy=self.ishappy)

        new_coeffs = _cumsum_coeffs(self.coeffs)
        if isinstance(self.coeffs, jax.core.Tracer):
            return Chebtech1.from_coeffs(new_coeffs, ishappy=self.ishappy)
        result = Chebtech1.from_coeffs(
            new_coeffs, ishappy=self.ishappy
        ).simplify()
        if result.isempty() or result.coeffs.size == 0:
            return result
        corrected = result.coeffs.at[0].add(-_cumsum_lval(result.coeffs))
        return eqx.tree_at(lambda tech: tech.coeffs, result, corrected)

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

    def roots(self, qz: bool = False, *, complex_roots: bool = False,
              all_roots: bool = False, prune: bool = False,
              recurse: bool = True) -> jax.Array:
        """Roots in [-1, 1] via colleague matrix eigenvalues.

        NOT JIT-safe.  See :meth:`Chebtech2.roots` for the full option
        surface (``qz``, ``complex_roots``, ``all_roots``, ``prune``,
        ``recurse``) mirroring MATLAB ``@chebtech/roots.m``.

        Provenance
        ----------
        MATLAB source : @chebtech/roots.m
        Chebfun commit: 7574c77
        """
        if complex_roots:
            all_roots = True
            prune = True
        kw = dict(qz=qz, all_roots=all_roots, prune=prune, recurse=recurse)
        if self.coeffs.ndim == 2:
            # Array-valued: roots per column, NaN-padded to equal length
            # (MATLAB @chebtech/roots.m), same as Chebtech2.roots.
            import numpy as _np
            cols = [_np.asarray(_roots_colleague(self.coeffs[:, j], **kw))
                    for j in range(self.coeffs.shape[1])]
            nmax = max((len(c) for c in cols), default=0)
            dt = (_np.complex128
                  if any(_np.iscomplexobj(c) for c in cols) else float)
            out = _np.full((nmax, len(cols)), _np.nan, dtype=dt)
            for j, c in enumerate(cols):
                out[: len(c), j] = c
            return jnp.asarray(out)
        return _roots_colleague(self.coeffs, **kw)

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
        check: str = "standard",
        sample_test: bool = True,
    ) -> tuple[bool, int | None]:
        """Happiness check for adaptive construction.

        Same logic as Chebtech2.happiness_check but sample-tests at
        1st-kind off-grid points.  Supports the ``'standard'``, ``'strict'``,
        and ``'classic'`` happiness variants (MATLAB ``pref.happinessCheck``).

        Provenance
        ----------
        MATLAB source : @chebtech/happinessCheck.m, @chebtech/standardCheck.m,
            @chebtech/strictCheck.m, @chebtech/classicCheck.m
        Chebfun commit: 7574c77
        """
        return _happiness_check_impl(
            Chebtech1, 1, coeffs, values, op, tol, vscale, hscale, check,
            sample_test,
        )

    # ------------------------------------------------------------------
    # Composition, restriction, extrema (mirror Chebtech2)
    # ------------------------------------------------------------------

    def compose(
        self,
        op: Callable,
        g: "Chebtech1 | None" = None,
        *,
        maxpow2: int = 16,
    ) -> "Chebtech1":
        """Compose an operator with this Chebtech1.

        ``self.compose(op)`` returns ``op(self(x))``; with a second tech
        ``g`` it returns ``op(self(x), g(x))``; ``self.compose(g)`` for a
        Chebtech1 ``g`` returns ``g(self(x))``.  Result is adaptively
        constructed on a 1st-kind grid.  NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebtech/compose.m, @chebtech1/compose.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        """
        import math

        if isinstance(op, Chebtech1):
            op_cheb = op
            composed_func = lambda x: op_cheb(self(x))  # noqa: E731
            min_n = max(self.n, op_cheb.n)
        elif g is not None:
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
            composed_func = lambda x: op(self(x))  # noqa: E731
            min_n = self.n

        start_pow2 = max(4, math.ceil(math.log2(max(min_n - 1, 1))))
        return Chebtech1._adaptive_construct(
            composed_func,
            maxpow2=maxpow2,
            start_pow2=start_pow2,
            # MATLAB @chebtech1/compose.m uses complete-grid resampling.
            refinement_function="resampling",
            sample_test=False,
        )

    def restrict(self, a, b: float | None = None):
        """Restrict this Chebtech1 to a sub-interval [a, b] of [-1, 1].

        Returns a new Chebtech1 on [-1, 1] representing ``self`` on
        ``[a, b]`` via the affine map ``y = (b-a)/2 x + (a+b)/2``.

        Provenance
        ----------
        MATLAB source : @chebtech/restrict.m
        Chebfun commit: 7574c77
        """
        if b is None or not isinstance(a, (int, float)):
            # MATLAB restrict(f, s) with a breakpoint vector s: one tech per
            # sub-interval, returned as a list when s has more than two
            # entries (MATLAB returns a cell array there).
            brk = [float(t) for t in jnp.asarray(a).reshape(-1)]
            if len(brk) < 2:
                raise ValueError(
                    "CHEBFUN:CHEBTECH:restrict:badInterval: "
                    "need at least two breakpoints.")
            if len(brk) == 2:
                return self.restrict(brk[0], brk[1])
            return [self.restrict(brk[i], brk[i + 1])
                    for i in range(len(brk) - 1)]
        a = float(a)
        b = float(b)
        if a < -1.0 - 10 * _EPS or b > 1.0 + 10 * _EPS or a >= b:
            raise ValueError(
                f"[a, b] = [{a}, {b}] is not a valid sub-interval of [-1, 1]. "
                f"Require -1 <= a < b <= 1."
            )
        if abs(a - (-1.0)) < 10 * _EPS and abs(b - 1.0) < 10 * _EPS:
            return Chebtech1(coeffs=self.coeffs.copy(), ishappy=self.ishappy)
        n = self.n
        x = chebpts(n, kind=1)
        y = 0.5 * (b - a) * x + 0.5 * (a + b)
        new_values = self(y)
        new_coeffs = _chebtech1_vals2coeffs(new_values)
        return Chebtech1(coeffs=new_coeffs, ishappy=self.ishappy)

    def minandmax(self):
        """Global minimum and maximum on [-1, 1] with their positions.

        NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebtech/minandmax.m
        Chebfun commit: 7574c77
        """
        import numpy as _np

        if jnp.iscomplexobj(self.coeffs):
            realf = self.real()
            imagf = self.imag()
            h = (realf * realf + imagf * imagf).simplify()
            (_, min_pos), (_, max_pos) = h.minandmax()
            if self.coeffs.ndim == 2:
                min_val = jnp.diagonal(self(jnp.atleast_1d(min_pos)))
                max_val = jnp.diagonal(self(jnp.atleast_1d(max_pos)))
            else:
                min_val = self(min_pos)
                max_val = self(max_pos)
            return (min_val, min_pos), (max_val, max_pos)

        if self.coeffs.ndim == 2:
            per_col = [
                Chebtech1(coeffs=self.coeffs[:, j],
                          ishappy=self.ishappy).minandmax()
                for j in range(self.coeffs.shape[1])
            ]
            min_val = jnp.stack([p[0][0] for p in per_col])
            min_pos = jnp.stack([p[0][1] for p in per_col])
            max_val = jnp.stack([p[1][0] for p in per_col])
            max_pos = jnp.stack([p[1][1] for p in per_col])
            return (min_val, min_pos), (max_val, max_pos)

        fp = self.diff()
        r = fp.roots()
        endpoints = jnp.array([-1.0, 1.0], dtype=jnp.float64)
        if r.shape[0] > 0:
            candidates = jnp.concatenate([endpoints, r])
        else:
            candidates = endpoints
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

    def min(self):
        """Global minimum on [-1, 1]. NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebtech/min.m
        Chebfun commit: 7574c77
        """
        (min_val, min_pos), _ = self.minandmax()
        return min_val, min_pos

    def max(self):
        """Global maximum on [-1, 1]. NOT JIT-safe.

        Provenance
        ----------
        MATLAB source : @chebtech/max.m
        Chebfun commit: 7574c77
        """
        _, (max_val, max_pos) = self.minandmax()
        return max_val, max_pos

    def sign(self) -> "Chebtech1":
        """Signum of the function (assumes no interior roots).

        Provenance
        ----------
        MATLAB source : @chebtech/sign.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self):
            return Chebtech1.empty()
        if not jnp.iscomplexobj(self.coeffs):
            arbitrary_point = 0.1273881594
            fx = self(jnp.array([-1.0, arbitrary_point, 1.0],
                                dtype=jnp.float64))
            meanfx = jnp.mean(fx, axis=0)
            s = jnp.sign(meanfx)
            if self.coeffs.ndim == 2:
                return Chebtech1(coeffs=jnp.atleast_1d(s)[None, :])
            return Chebtech1.from_coeffs(jnp.atleast_1d(s))
        return self.compose(lambda x: x / jnp.abs(x))

    def poly(self) -> jax.Array:
        """Monomial (power-basis) coefficients (highest degree first).

        Provenance
        ----------
        MATLAB source : @chebtech/poly.m
        Chebfun commit: 7574c77
        """
        if _is_empty_tech(self):
            return jnp.array([], dtype=jnp.float64)
        return _poly_coeffs(self.coeffs)

    def extractBoundaryRoots(self, num_roots=None):
        """Extract roots at the boundary points -1 and +1.

        Provenance
        ----------
        MATLAB source : @chebtech/extractBoundaryRoots.m
        Chebfun commit: 7574c77
        """
        if getattr(self, "_is_empty_object", False):
            counts = jnp.zeros((0,), dtype=jnp.int32)
            return self, counts, counts
        if bool(_has_no_boundary_roots(self.coeffs, self.vscale_columns)):
            ncols = 1 if self.coeffs.ndim == 1 else self.coeffs.shape[1]
            zeros = jnp.zeros((ncols,), dtype=jnp.int32)
            if self.coeffs.ndim == 1:
                return self, zeros[0], zeros[0]
            return self, zeros, zeros
        c, l, r = _extract_boundary_roots(
            self.coeffs, self.vscale_columns, num_roots
        )
        g = Chebtech1(coeffs=c, ishappy=self.ishappy).simplify()
        return g, l, r

    @staticmethod
    def chebTcoeffs2chebUcoeffs(cT: jax.Array) -> jax.Array:
        """Convert Chebyshev-T coefficients to Chebyshev-U coefficients.

        Provenance
        ----------
        MATLAB source : @chebtech/chebTcoeffs2chebUcoeffs.m
        Chebfun commit: 7574c77
        """
        return _chebT_to_chebU_coeffs(cT)


# ============================================================================
# QR factorisation and matrix division for array-valued techs
# ============================================================================


def _tech_kind(cls) -> int:
    """1 for Chebtech1, 2 for Chebtech2."""
    return 1 if cls.__name__ == "Chebtech1" else 2


def _unit_sign(d: jax.Array) -> jax.Array:
    """MATLAB ``sign`` on a (possibly complex) vector, mapping 0 to 1.

    For real input this is ``+-1``; for complex input it is ``d/|d|``, the
    unimodular phase, so that ``conj(s)*d = |d| >= 0``.
    """
    mag = jnp.abs(d)
    return jnp.where(mag == 0, jnp.ones_like(d), d / jnp.where(mag == 0, 1.0, mag))


def _tech_qr_builtin(f, want_e: bool, mode: str):
    """Weighted Gauss-Legendre QR, including the source fast branch.

    Provenance: ``@chebtech/qr.m``, Chebfun commit 7574c77.
    The three-output adapter currently returns identity permutation; MATLAB's
    built-in three-output QR pivots columns, which remains a parity gap.
    """
    from chebfunjax.utils.interpolation import barymat
    from chebfunjax.utils.quadrature import legpts

    cls = type(f)
    kind = _tech_kind(cls)
    coeffs = f.coeffs if f.coeffs.ndim == 2 else f.coeffs[:, None]
    n, m = coeffs.shape
    if n < m:
        coeffs = jnp.concatenate(
            [coeffs, jnp.zeros((m - n, m), dtype=coeffs.dtype)], axis=0)
        n = m

    if n <= 4000:
        xc = chebpts(n, kind=kind)
        vc = cls.barywts(n)
        xl, wl, vl = legpts(n, bary=True)
        sqrt_wl = jnp.sqrt(wl)
        WP = sqrt_wl[:, None] * barymat(xl, xc, vc)
        invWP = barymat(xc, xl, vl) * (1.0 / sqrt_wl)[None, :]
        values = cls.coeffs2vals(coeffs)
        Qd, R = jnp.linalg.qr(WP @ values, mode="reduced")
        s = _unit_sign(jnp.diagonal(R))
        Q_coeffs = cls.vals2coeffs(invWP @ (Qd * s[None, :]))
    else:
        # The source avoids both n-by-n interpolation matrices above 4000.
        # Accurate ASY angles are passed directly; acos(xl) loses endpoint
        # precision required by the large-degree transform.
        from chebfunjax.utils.transforms import _legendre_idlt, leg2cheb, ndct

        xl, wl, _vl, theta = legpts(n, newtheta=True)
        sqrt_wl = jnp.sqrt(wl)
        converted = sqrt_wl[:, None] * ndct(xl, coeffs, theta)
        Qd, R = jnp.linalg.qr(converted, mode="reduced")
        s = _unit_sign(jnp.diagonal(R))
        legendre_values = (Qd * s[None, :]) / sqrt_wl[:, None]
        Q_coeffs = leg2cheb(_legendre_idlt(legendre_values))

    # Enforce a nonnegative real diagonal while preserving Q @ R.
    R = jnp.conj(s)[:, None] * R
    Q = cls(coeffs=Q_coeffs, ishappy=f.ishappy)
    if want_e:
        return Q, R, _tech_qr_perm(m, mode)
    return Q, R


def _tech_qr_householder(f, want_e: bool, mode: str):
    """Trefethen's Householder triangularisation of a quasimatrix."""
    from chebfunjax.utils.misc import abstract_qr
    from chebfunjax.utils.quadrature import chebweights

    cls = type(f)
    kind = _tech_kind(cls)
    coeffs = f.coeffs if f.coeffs.ndim == 2 else f.coeffs[:, None]
    n, m = coeffs.shape
    tol = _EPS * f.vscale

    new_n = 2 * max(n, m)
    padded = jnp.concatenate(
        [coeffs, jnp.zeros((new_n - n, m), dtype=coeffs.dtype)], axis=0) \
        if new_n > n else coeffs[:new_n]
    A = cls.coeffs2vals(padded)

    x = chebpts(new_n, kind=kind)
    w = chebweights(new_n, kind=kind)

    def ip(u, v):
        return jnp.sum(w * jnp.conj(u) * v)

    # Legendre-Chebyshev-Vandermonde basis (three-term recurrence).
    cols = [jnp.ones_like(x)]
    if m > 1:
        cols.append(x)
    for k in range(3, m + 1):
        cols.append(((2 * k - 3) * x * cols[k - 2]
                     - (k - 2) * cols[k - 3]) / (k - 1))
    E = jnp.stack(cols, axis=1)
    E = E * jnp.sqrt((2.0 * jnp.arange(1, m + 1) - 1.0) / 2.0)[None, :]
    E = E.astype(A.dtype)

    Qd, R = abstract_qr(A, E, ip, lambda u: float(jnp.max(jnp.abs(u))), tol)
    Q_coeffs = cls.vals2coeffs(jnp.asarray(Qd))[: new_n // 2]

    Q = cls(coeffs=Q_coeffs, ishappy=f.ishappy)
    if want_e:
        return Q, jnp.asarray(R), _tech_qr_perm(m, mode)
    return Q, jnp.asarray(R)


def _tech_qr_perm(m: int, mode: str):
    """The (identity) column permutation returned as a vector or matrix."""
    if mode == "vector" or mode == 0:
        return jnp.arange(m)
    return jnp.eye(m, dtype=jnp.float64)


def _tech_qr(f, mode: str = "matrix", method: str = "built-in",
             want_e: bool = False):
    """Shared implementation of MATLAB ``@chebtech/qr``."""
    if _is_empty_tech(f):
        return (f, jnp.zeros((0, 0)), jnp.zeros((0, 0))) if want_e \
            else (f, jnp.zeros((0, 0)))

    cls = type(f)
    if f.coeffs.ndim == 1 or f.coeffs.shape[1] == 1:
        # Single column: just normalise.
        R = jnp.sqrt(jnp.reshape(jnp.asarray(f.inner(f)), ()))
        Q = f / R
        R = jnp.reshape(R, (1, 1))
        if want_e:
            return Q, R, _tech_qr_perm(1, mode)
        return Q, R

    if isinstance(method, str) and method.lower() == "householder":
        return _tech_qr_householder(f, want_e, mode)
    if isinstance(method, str) and method.lower() not in ("built-in", "builtin"):
        raise ValueError(f"CHEBFUN:CHEBTECH:qr:method: unknown method {method!r}")
    _ = cls
    return _tech_qr_builtin(f, want_e, mode)


def _collapse_single_column(tech):
    """Return a scalar-valued tech when ``tech`` has exactly one column.

    chebfunjax stores scalar-valued techs with 1-D coefficients, so a
    one-column result of a matrix operation is collapsed back to that
    form (the same convention as ``mat2cell``).
    """
    if tech.coeffs.ndim == 2 and tech.coeffs.shape[1] == 1:
        return type(tech)(coeffs=tech.coeffs[:, 0], ishappy=tech.ishappy)
    return tech


def _tech_mldivide(A, B):
    """Shared implementation of MATLAB ``@chebtech/mldivide``."""
    if type(A) is not type(B) or not hasattr(B, "coeffs"):
        raise ValueError(
            "CHEBFUN:CHEBTECH:mldivide:chebtechMldivideUnknown: "
            "arguments to chebtech mldivide must both be chebtech objects "
            "of the same type.")
    Q, R = _tech_qr(A, method="built-in")
    ip = jnp.reshape(jnp.asarray(Q.inner(B)), (R.shape[0], -1))
    X = jnp.linalg.solve(R, ip)
    return X[:, 0] if X.shape[1] == 1 else X


def _tech_mrdivide(A, B):
    """Shared implementation of MATLAB ``@chebtech/mrdivide``."""
    A_tech = hasattr(A, "coeffs") and hasattr(A, "ishappy")
    B_tech = hasattr(B, "coeffs") and hasattr(B, "ishappy")

    if A_tech and B_tech:
        raise ValueError(
            "CHEBFUN:CHEBTECH:mrdivide:chebtechDivChebtech: "
            "use ./ to divide by a chebtech.")

    if A_tech:
        if isinstance(B, bool) or not isinstance(
                B, (int, float, complex, jnp.ndarray, np.ndarray, list, tuple)):
            raise ValueError(
                "CHEBFUN:CHEBTECH:mrdivide:badArg: "
                f"chebtech/{type(B).__name__} is not well-defined.")
        Bd = jnp.asarray(B)
        m = A.coeffs.shape[1] if A.coeffs.ndim == 2 else 1
        b_cols = Bd.shape[-1] if Bd.ndim >= 1 else 1
        if Bd.size > 1 and b_cols != m:
            raise ValueError(
                "CHEBFUN:CHEBTECH:mrdivide:size: matrix dimensions must agree.")
        if not bool(jnp.any(Bd != 0)):
            nan_row = jnp.full((1, m), jnp.nan, dtype=jnp.float64)
            return type(A)(coeffs=nan_row[:, 0] if m == 1 else nan_row,
                           ishappy=True)
        if Bd.size == 1:
            return A * (1.0 / jnp.reshape(Bd, ()))
        Q, R = _tech_qr(A, method="built-in")
        Bm = jnp.atleast_2d(Bd)
        # R / B  ==  (B.' \ R.').'
        Y = jnp.linalg.lstsq(Bm.T, R.T)[0].T
        return _collapse_single_column(Q @ Y)

    if B_tech:
        if isinstance(A, bool) or not isinstance(
                A, (int, float, complex, jnp.ndarray, np.ndarray, list, tuple)):
            raise ValueError(
                "CHEBFUN:CHEBTECH:mrdivide:badArg: "
                f"{type(A).__name__}/chebtech is not well-defined.")
        Am = jnp.atleast_2d(jnp.asarray(A))
        m = B.coeffs.shape[1] if B.coeffs.ndim == 2 else 1
        if Am.shape[-1] != m:
            raise ValueError(
                "CHEBFUN:CHEBTECH:mrdivide:size: matrix dimensions must agree.")
        Q, R = _tech_qr(B, method="built-in")
        # A / R  ==  (R.' \ A.').'
        AR = jnp.linalg.lstsq(R.T, Am.T)[0].T
        return _collapse_single_column(Q @ AR.T)

    raise ValueError("CHEBFUN:CHEBTECH:mrdivide:badArg")


def _tech_isequal(f, g) -> bool:
    """Shared implementation of MATLAB ``@chebtech/isequal``."""
    if type(f) is not type(g):
        return False
    if _is_empty_tech(f) or _is_empty_tech(g):
        return _is_empty_tech(f) and _is_empty_tech(g)
    cf = f.coeffs if f.coeffs.ndim == 2 else f.coeffs[:, None]
    cg = g.coeffs if g.coeffs.ndim == 2 else g.coeffs[:, None]
    if cf.shape != cg.shape:
        return False
    return bool(jnp.all(cf == cg))

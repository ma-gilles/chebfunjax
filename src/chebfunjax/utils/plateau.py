"""MATLAB's plateau-based Chebyshev coefficient happiness test.

This is an adaptive construction-time routine. Numerical arrays and filters
are evaluated with JAX; scalar decisions (cutoff locations and early exits)
are transferred to the host, as in the surrounding adaptive constructor.

Provenance
----------
MATLAB source : @chebtech/plateauCheck.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The Chebfun
    Developers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _lag6_filter(values: jax.Array) -> jax.Array:
    """Apply MATLAB ``filter(LPA, LPB, x)`` for the source lag-six filter."""
    padded = jnp.pad(values, (12, 0))
    forcing = (padded[12:] - 2.0*padded[6:-6] + padded[:-12]) / 36.0

    def step(carry, value):
        previous, previous2 = carry
        current = value + 2.0*previous - previous2
        return (current, previous), current

    zero = jnp.zeros((), dtype=values.dtype)
    initial = (zero, zero)
    _, filtered = jax.lax.scan(step, initial, forcing)
    return filtered


def _plateau_column(abs_coeff: jax.Array, tol: float) -> tuple[bool, int | None]:
    """Source ``checkColumn`` on one normalized absolute coefficient column."""
    n = int(abs_coeff.shape[0])
    over = abs_coeff >= tol / 50.0
    indices = jnp.arange(n, dtype=jnp.int32) + 1
    last = int(jnp.max(jnp.where(over, indices, 0)))
    cutoff = 4 + last if last else None

    if last and cutoff < 0.95*n:
        return True, cutoff

    thresh = max((2.0/3.0)*float(jnp.log(tol)), float(jnp.log(1e-7)))
    floor = float(jnp.log(jnp.finfo(jnp.float64).eps / 1000.0))
    log_abs = jnp.maximum(jnp.log(abs_coeff), floor)

    # Each of the source's six passes extends the maximum window by one
    # coefficient, resulting in a seven-coefficient window.
    win_max = log_abs
    for offset in range(1, 7):
        win_max = jnp.maximum(win_max[:-1], log_abs[offset:])
    n_window = int(win_max.shape[0])
    smooth = _lag6_filter(win_max)

    good = jnp.nonzero(smooth < thresh, size=1, fill_value=-1)[0]
    first_good = int(good[0])
    t_ok = first_good + 1 - 6 if first_good >= 0 else None
    if t_ok is None or n_window - t_ok < 16:
        return False, cutoff

    # MATLAB fits [1, (1:24)/24] to each 24-sample window. The slope below is
    # the closed-form least-squares coefficient for that same two-column fit.
    ls_window = 24
    n_slopes = n_window - ls_window
    if n_slopes <= 0:
        return False, n_window
    starts = jnp.arange(n_slopes, dtype=jnp.int32)
    offsets = jnp.arange(ls_window, dtype=jnp.int32)
    windows = smooth[starts[:, None] + offsets[None, :]]
    x = (jnp.arange(1, ls_window + 1, dtype=smooth.dtype)
         / ls_window)
    x_centered = x - jnp.mean(x)
    slopes_fitted = jnp.sum(
        windows * x_centered[None, :], axis=1
    ) / jnp.sum(x_centered*x_centered)
    slopes = jnp.concatenate((jnp.full((ls_window,), jnp.nan),
                              slopes_fitted))

    slope_min = float(jnp.min(jnp.where(jnp.isnan(slopes), jnp.inf, slopes)))
    starts_decrease = jnp.nonzero(
        slopes < 0.3*slope_min, size=1, fill_value=-1
    )[0]
    if int(starts_decrease[0]) < 0:
        return False, n_window
    t_start = int(jnp.max(jnp.where(
        slopes < 0.3*slope_min, jnp.arange(slopes.size), -1
    )))

    slow = slopes[t_start:] > 0.01*slope_min
    slow_indices = jnp.nonzero(slow)[0] + t_start + 1
    if slow_indices.size < 6:
        return False, n_window
    gaps = slow_indices[5:] - slow_indices[:-5]
    consecutive = jnp.nonzero(gaps == 5, size=1, fill_value=-1)[0]
    first_run = int(consecutive[0])
    if first_run < 0:
        return False, n_window
    cutoff = int(slow_indices[first_run])
    return cutoff < n_window, cutoff


def _plateau_check(
    coeffs: jax.Array,
    values: jax.Array,
    vscale: float | jax.Array,
    tol: float,
) -> tuple[bool, int | jax.Array]:
    """Return MATLAB ``plateauCheck``'s happiness flag and coefficient cutoff.

    Parameters
    ----------
    coeffs : array, shape ``(n,)`` or ``(n, m)``
        Chebyshev coefficients, with coefficient rows on axis zero.
    values : array, shape ``(n,)`` or ``(n, m)``
        Values sampled by the constructor. A zero function is accepted and
        an infinite value prevents chopping, matching the MATLAB source.
    vscale : scalar or length-``m`` array
        Existing global vertical scale from the constructor.
    tol : positive scalar
        Chebfun's requested relative accuracy.

    Returns
    -------
    (ishappy, cutoff)
        The array-valued result is happy only if every column is happy; its
        cutoff is the largest per-column cutoff.
    """
    coeffs = jnp.asarray(coeffs)
    values = jnp.asarray(values)
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]
    if values.ndim == 1:
        values = values[:, None]
    if coeffs.ndim != 2 or values.ndim != 2:
        raise ValueError("coeffs and values must be vectors or 2D arrays")
    if coeffs.shape[0] == 0 or coeffs.shape[1] == 0:
        raise ValueError("coeffs must contain at least one coefficient")
    if values.shape[1] != coeffs.shape[1]:
        raise ValueError("values and coeffs must have the same column count")
    if not bool(jnp.isfinite(tol)) or tol <= 0:
        raise ValueError("tol must be a positive finite scalar")
    if bool(jnp.any(jnp.isnan(coeffs))):
        raise ValueError("Function returned NaN when evaluated.")

    max_values = jnp.max(jnp.abs(values), axis=0)
    if bool(jnp.max(max_values) == 0):
        return True, 1
    n = int(coeffs.shape[0])
    if bool(jnp.any(jnp.isinf(max_values))):
        return False, n

    n90 = int(jnp.ceil(0.90*n))
    abs_coeff = jnp.abs(coeffs[:n90, :])
    local_scales = jnp.max(abs_coeff, axis=0)
    supplied_scales = jnp.asarray(vscale, dtype=jnp.float64).reshape(-1)
    if supplied_scales.size == 1:
        supplied_scales = jnp.broadcast_to(supplied_scales,
                                           local_scales.shape)
    elif supplied_scales.size != coeffs.shape[1]:
        raise ValueError("vscale must be scalar or one value per column")
    scales = jnp.maximum(local_scales, supplied_scales)
    # Source division is unguarded: zero scales can create NaNs here.
    # The nanEval check above concerns input coefficients, not these values.
    normalized = abs_coeff / scales[None, :]

    happy_columns = [False] * coeffs.shape[1]
    cutoffs = [0] * coeffs.shape[1]
    # MATLAB stops at the first unresolved column; remaining cutoffs remain
    # zero before its final max(cutOff).
    for col in range(coeffs.shape[1]):
        happy, cutoff = _plateau_column(normalized[:, col], float(tol))
        happy_columns[col] = happy
        if cutoff is None:
            # MATLAB cutOff(col)=[] deletes that slot, rather than
            # substituting a full-length cutoff on an early return.
            del cutoffs[col]
        else:
            cutoffs[col] = cutoff
        if not happy:
            break
    return (all(happy_columns), max(cutoffs) if cutoffs
            else jnp.empty((0,), dtype=jnp.int32))

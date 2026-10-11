"""Reuse disk standardChop compilation after the source FFT is complete.

Provenance
----------
Chebfun standardChop.m, pin7574c77680d7e82b79626300bf255498271a72df;
the existing utils.misc._standard_chop_kernel translation supplies the
arithmetic and source indices. Only the physical buffer length is separated
from its active length. This helper receives at least17 active magnitudes and
zero padding. It never pads Fourier or Chebyshev coefficients before an FFT.

Maximum envelopes are unchanged by trailing nonnegative zeros. Source n still
bounds lookahead windows, fallback cutoffs and indexed endpoints. Threshold
counts exclude padding explicitly, including when the threshold underflows.
The minimization excludes indices beyond the source window. Thus capacity
controls compilation reuse while n retains the source stopping semantics.
"""

import jax
import jax.numpy as jnp


@jax.jit
def _standard_chop_active(coeffs: jnp.ndarray, tol: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """Use the source active length on post-FFT zero-padded magnitudes."""
    capacity = coeffs.size
    magnitudes = jnp.asarray(jnp.abs(coeffs), dtype=jnp.float64)
    envelope = jax.lax.associative_scan(jnp.maximum, magnitudes, reverse=True)
    scale = envelope[0]
    envelope = envelope / jnp.where(scale == 0, 1.0, scale)

    # MATLAB uses half-away rounding. For positive source indices this exact
    # integer expression is round(1.25*j + 5), including j=6,14,22,... ties.
    j = jnp.arange(2, capacity + 1)
    windows = (5 * j + 22) // 4
    e1 = envelope[j - 1]
    e2 = envelope[jnp.minimum(windows - 1, n - 1)]
    r = 3.0 * (1.0 - jnp.log(e1) / jnp.log(tol))
    plateau = (windows <= n) & ((e1 == 0) | (e2 / e1 > r))
    first = jnp.min(jnp.where(plateau, j, n + 1))
    found = first <= n
    plateau_point = first - 1
    window = jnp.minimum((5 * first + 22) // 4, n)

    threshold = tol ** (7.0 / 6.0)
    j3 = jnp.sum((jnp.arange(capacity) < n) & (envelope >= threshold))
    shortened = j3 < window
    window = jnp.where(shortened, j3 + 1, window)
    envelope = envelope.at[window - 1].set(
        jnp.where(shortened, threshold, envelope[window - 1])
    )
    indices = jnp.arange(capacity)
    bias = indices * ((-1.0 / 3.0) * jnp.log10(tol)) / (window - 1)
    cc = jnp.where(indices < window, jnp.log10(envelope) + bias, jnp.inf)
    cutoff = jnp.maximum(jnp.argmin(cc), 1)
    cutoff = jnp.where(envelope[jnp.minimum(plateau_point - 1, n - 1)] == 0,
                       plateau_point, cutoff)
    return jnp.where(scale == 0, 1, jnp.where(found, cutoff, n))

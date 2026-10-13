"""Selected-technology primitives for the native three-phase constructor.

Provenance
----------
MATLAB source: @chebfun3/chebfun3f.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._binary64 import _divide_binary64_by_positive_integer


def envelope_happiness(values, tech, relative_tol):
    """Check the literal source coefficient envelope, including its vscale.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m, happinessCheck3;
        @chebtech/happinessCheck.m; @trigtech/happinessCheck.m
    Chebfun commit: 7574c77
    """
    coefficients = tech.vals2coeffs(jnp.asarray(values))
    envelope = jnp.sum(jnp.abs(coefficients), axis=1)
    scale = jnp.max(jnp.abs(tech.coeffs2vals(envelope)))
    return tech.happiness_check(envelope, envelope, tol=relative_tol,
                                vscale=scale)[0]


_grid_ratios = jax.jit(_divide_binary64_by_positive_integer, static_argnums=1)


def periodic_points(n, a, b):
    """Return the source half-open physical sampling grid.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m, trigtech chebX/Y/Z branch
    Chebfun commit: 7574c77
    """
    indices = jnp.arange(n, dtype=jnp.float64)
    ratio = _grid_ratios(indices, n)
    return a + ratio * (b - a)


def periodic_factors(values, relative_tol, session_tol):
    """Build numeric Fourier factors and apply both source simplifications.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m, phase3;
        @chebfun/simplify.m; @trigtech/simplify.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.tech.trigtech import Trigtech

    combined = Trigtech.from_values(jnp.asarray(values)).simplify(session_tol)
    scales = jnp.max(jnp.abs(combined.values), axis=0)
    tolerances = relative_tol * jnp.max(scales) / scales
    combined = combined.simplify(tolerances)
    return [Trigtech.from_coeffs(combined.coeffs[:, k],
                                is_real=combined.real_columns[k])
            for k in range(values.shape[1])]

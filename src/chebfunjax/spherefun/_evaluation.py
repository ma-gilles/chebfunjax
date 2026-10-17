"""JAX angular evaluation of genuine Spherefun CDR factors.

Provenance
----------
MATLAB source: @spherefun/feval.m, @separableApprox/feval.m,
@mapping/mapping.m, @trigtech/horner.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d.separable_approx import _phys_to_ref
from chebfunjax.tech.trigtech import _trig_eval


def factor_values(factors, physical_points):
    """Batch equal genuine lengths without aliasing or coefficient padding.

    Static grouping retains each factor's own real/complex Horner contract.
    Only coefficient arrays enter the shared compiled evaluator, avoiding a
    compiled program for every Spherefun object or rank.
    """
    points = _phys_to_ref(physical_points, -jnp.pi, jnp.pi)
    groups = {}
    for index, factor in enumerate(factors):
        key = (factor.coeffs.shape, factor.coeffs.dtype, factor.is_real)
        groups.setdefault(key, []).append(index)
    values = [None] * len(factors)
    for (_, _, is_real), indices in groups.items():
        coeffs = jnp.stack([factors[i].coeffs for i in indices], axis=-1)
        batch = _trig_eval(coeffs, points, is_real=is_real)
        for position, index in enumerate(indices):
            values[index] = batch[..., position]
    return jnp.stack(values, axis=-1)


@jax.jit
def point_values(cols, rows, weights):
    """Source vector expression: (C .* R) * diag(D)."""
    return (cols * rows) @ weights


@jax.jit
def grid_values(cols, rows, weights):
    """Source tensor-grid expression: C * D * R.' (no conjugation)."""
    return (cols * weights) @ rows.T

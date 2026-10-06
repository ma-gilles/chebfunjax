"""Global source trigpts nodes, distinct from static technology points."""
import operator
from functools import partial

import jax
import jax.numpy as jnp

from chebfunjax.utils._binary64 import _divide_binary64_by_positive_integer


@partial(jax.jit, static_argnums=(0,))
def global_trigpts_nodes(n):
    """Global source trigpts canonical nodes with preserved PI arithmetic.

    Provenance
    ----------
    MATLAB source : trigpts.m; installed R2025b linspace.m symmetric branch
    Chebfun commit: 7574c77
    n is a static integer. Requires binary64 mode. Separate source PI arithmetic is retained.
    """
    n = operator.index(n)
    if n > 2**53:
        raise ValueError("source grid integer count must be at most 2**53")
    if n <= 0:
        return jnp.empty((0,), dtype=jnp.float64)
    pi = jnp.asarray(jnp.pi, dtype=jnp.float64)
    # Correctly rounded source PI/n, without backend reciprocal rewrites.
    step = _divide_binary64_by_positive_integer(pi, n)
    multipliers = (2*jnp.arange(n+1, dtype=jnp.int64)-n).astype(jnp.float64)
    angles = jax.lax.optimization_barrier(multipliers*step)
    angles = angles.at[0].set(-pi).at[-1].set(pi)
    if n % 2 == 0:
        angles = angles.at[n//2].set(0.0)
    # PI's exact binary64 identity is 7074237752028440 * 2**-51.
    # Both exact and rounded angle/D remain normal for nonzero grid angles
    # at n<=2**53. Power-of-two scaling therefore adds no second rounding.
    quotient = _divide_binary64_by_positive_integer(angles, 7074237752028440)
    nodes = jnp.ldexp(quotient, 51)
    nodes = jax.lax.optimization_barrier(nodes)
    return ((nodes-nodes[::-1])/2.0)[:-1]


def map_global_nodes(nodes, a, b):
    """Staged source diff*nodes/2+mean mapping with separate rounding.

    Provenance
    ----------
    MATLAB source : trigpts.m (domain mapping)
    Chebfun commit: 7574c77
    """
    a, b = jnp.asarray(a, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64)
    width = jax.lax.optimization_barrier(b-a)
    product = jax.lax.optimization_barrier(width*nodes)
    half = jax.lax.optimization_barrier(product/2.)
    center_sum = jax.lax.optimization_barrier(a+b)
    center = jax.lax.optimization_barrier(center_sum/2.)
    return half+center


@partial(jax.jit, static_argnums=(0,))
def static_trigpts_nodes(n):
    """Static technology points from the source normalized linspace.

    Provenance
    ----------
    MATLAB source : @trigtech/trigpts.m; R2025b linspace.m symmetric branch
    Chebfun commit: 7574c77
    Unlike global trigpts, this API computes normalized points directly.
    Requires binary64 mode and a static integer count at most 2**53.
    """
    n = operator.index(n)
    if n > 2**53:
        raise ValueError("source grid integer count must be at most 2**53")
    if n <= 0:
        return jnp.empty((0,), dtype=jnp.float64)
    step = _divide_binary64_by_positive_integer(jnp.asarray(1., dtype=jnp.float64), n)
    x = (2*jnp.arange(n+1, dtype=jnp.int64)-n).astype(jnp.float64)*step
    x = x.at[0].set(-1.).at[-1].set(1.)
    if n % 2 == 0:
        x = x.at[n//2].set(0.)
    return x[:-1]

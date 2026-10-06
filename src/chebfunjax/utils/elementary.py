"""Real atanh evaluation using general cancellation-resistant identities.

Independently stated logarithmic identity, not MATLAB built-in internals.
"""

import jax.numpy as jnp


def _atanh_log1p_real(x):
    """Compute real-domain atanh by cancellation-resistant log1p identities.

    Real values outside[-1,1] yield NaN so the existing Chebfun elementwise
    wrapper can retry complex arithmetic. Complex inputs keep existing JAX
    arctanh. No approximation/coefficient fit or bound is changed here.

    Provenance
    ----------
    MATLAB source : @chebfun/atanh.m, @chebfun/acoth.m (builtin consumers)
    Chebfun commit: 7574c77
    Independent logarithmic identities; not MATLAB builtin implementation.
    This binary64 numerical adaptation is qualified against high precision
    and original MATLAB Chebfun composition assertions.

    """
    x = jnp.asarray(x)
    if jnp.iscomplexobj(x):
        return jnp.arctanh(x)
    a = jnp.abs(x)
    small = 0.5 * jnp.log1p(2.0 * a + 2.0 * a * a / (1.0 - a))
    large = 0.5 * jnp.log1p(2.0 * a / (1.0 - a))
    magnitude = jnp.where(a < 0.5, small, large)
    result = jnp.copysign(magnitude, x)
    # atanh(x)-x = x^3/3+... is below half a binary64 ulp throughout
    # this general tiny-argument range. Preserve signedzero/subnormal bits.
    return jnp.where(a < 2.0**-28, x, result)

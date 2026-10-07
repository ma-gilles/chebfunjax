"""Source epsilon-normal padding using the shared default JAX normal stream.

JAX normal draws are not MATLAB RNG bitstream equivalents. This eager adapter
shares call-order state with randnfun; explicit randnfun keys/seeds bypass it.
Only insufficient coefficients consume a key. No outer-jit RNG API is claimed.

Provenance
----------
MATLAB source : @chebfun/chebpade.m (Clenshaw-Lord coefficient padding)
Chebfun commit: 7574c77
"""

import jax.numpy as jnp


def _normal_draw(count):
    """Consume one shared key for a real normal column.

    Provenance
    ----------
    MATLAB source : @chebfun/chebpade.m (eps*randn padding), randnfun.m
    Chebfun commit: 7574c77
    JAX stream sharing preserves call ordering, not MATLAB random bits.
    """
    from chebfunjax.utils import _randnfun

    return _randnfun._normal_draw(_randnfun._next_key(), count, 1)[:, 0]


def _epsilon_pad(coefficients, required):
    """Append exactly required-length real normal draws times binary64 eps.

    Provenance
    ----------
    MATLAB source : @chebfun/chebpade.m (Clenshaw-Lord padding before c0 scaling)
    Chebfun commit: 7574c77
    """
    c = jnp.asarray(coefficients)
    missing = required - c.shape[0]
    if missing <= 0:
        return c
    padding = jnp.asarray(2.0**-52, dtype=jnp.float64) * _normal_draw(missing)
    return jnp.concatenate((c, padding))

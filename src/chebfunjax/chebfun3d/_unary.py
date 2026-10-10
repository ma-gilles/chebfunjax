"""Native constructor and sign dispatch for Chebfun3 unary functions.

MATLAB source: @chebfun3/{exp,cos,tanh,abs,sqrt,log}.m and
private/singleSignTest.m, Chebfun commit7574c77.
Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""
import jax.numpy as jnp

from chebfunjax.chebfun3d._power import _single_sign_test


def _principal_value(op, values):
    """MATLAB promotes negative real sqrt/log samples to principal complex."""
    values = jnp.asarray(values)
    if not jnp.iscomplexobj(values) and bool(jnp.any(values < 0)):
        values = values.astype(jnp.complex128)
    return op(values)


def source_unary(f, name):
    """Preserve source branch order; public empty wrappers act first."""
    from chebfunjax.chebfun3d.chebfun3 import chebfun3

    op = getattr(jnp, name)
    if name in ('exp', 'cos', 'tanh'):
        return chebfun3(lambda x, y, z: op(f(x, y, z)), f.domain)
    if f.isreal():
        single, _, positive = _single_sign_test(f)
        if not single:
            raise ValueError(f'CHEBFUN:CHEBFUN3:{name}:notSmooth: '
                             'Sign change detected. Unable to represent the result.')
        if name == 'abs':
            return f if positive else -f
    if name == 'abs':
        return f.compose(jnp.abs)
    return f.compose(lambda values: _principal_value(op, values))

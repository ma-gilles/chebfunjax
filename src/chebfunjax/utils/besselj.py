"""JAX principal-branch Bessel J for static real order.

Provenance
----------
MATLAB source: @chebfun/besselj.m (builtin adapter).
Chebfun commit: 7574c77
DLMF 10.2.1/10.2.2 (ODE/series), 10.6.1 (recurrence), 10.9.6
(Schlaefli integral), 10.17.3 (real-axis asymptotics).
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .bessel_general import _bessel_j_base, _series


def _complex_base(order, z):
    """Continue a small-argument series along a radial path using the ODE."""
    radius = jnp.abs(z)
    start = z/jnp.maximum(radius, 1.)
    value = _series(order, start)
    deriv = order/start*value-_series(order+1, start)
    steps = jnp.maximum(1, jnp.ceil((radius-1)/.4).astype(jnp.int32))
    step = (z-start)/steps

    def advance(i, state):
        center = start+i*step
        value, deriv = state
        # Store scaled Taylor terms, avoiding powers of small centers.
        def term(k, state):
            cm2, cm1, ck, ck1, total, derivative = state
            ck2 = -(center*(k+1)*(2*k+1)*ck1*step
                    +(k*k+center*center-order*order)*ck*step**2
                    +2*center*cm1*step**3+cm2*step**4)/(center**2*(k+2)*(k+1))
            return cm1, ck, ck1, ck2, total+ck2, derivative+(k+2)*ck2
        initial = (0j, 0j, value, deriv*step, value+deriv*step, deriv*step)
        state = lax.fori_loop(0, 28, term, initial)
        # step=0 is handled by the caller's small-argument branch.
        return state[-2], state[-1]/step
    return lax.fori_loop(0, steps, advance, (value, deriv))[0]


def _positive(order, z, complex_input):
    count = math.floor(order)
    fraction = order-count
    base = _complex_base if complex_input else _bessel_j_base
    base0, base1 = base(fraction, z), base(fraction+1, z)
    if count == 0:
        return base0
    if count < 0:
        def downward(k, state):
            current, higher = state
            return 2*(fraction-k)/z*current-higher, current
        return lax.fori_loop(0, -count, downward, (base0, base1))[0]

    def upward(_):
        def step(k, state):
            previous, current = state
            return current, 2*(fraction+k)/z*current-previous
        return lax.fori_loop(1, count, step, (base0, base1))[1]

    def miller(_):
        last = count+math.ceil(math.sqrt(40*count))+50
        if complex_input:
            last = last+jnp.ceil(jnp.abs(z)).astype(jnp.int32)
        def step(i, state):
            higher, current, target = state
            k = last-i
            target = jnp.where(k == count, current, target)
            lower = 2*(fraction+k)/z*current-higher
            rescale = jnp.where(jnp.abs(lower)>1e100, 1e-100, 1.)
            return current*rescale, lower*rescale, target*rescale
        zero = jnp.zeros_like(z)
        one, zero, target = lax.fori_loop(0, last, step, (zero, zero+1, zero))
        scale = jnp.maximum(jnp.abs(zero), jnp.abs(one))
        zero, one, target = zero/scale, one/scale, target/scale
        # Two-component normalization remains safe near either base zero.
        return target*(jnp.conj(zero)*base0+jnp.conj(one)*base1)/(jnp.abs(zero)**2+jnp.abs(one)**2)
    if complex_input:
        return miller(None)
    return lax.cond(jnp.abs(z)>order, upward, miller, operand=None)


def _scalar(order, z, complex_input):
    if order < 0 and order == int(order):
        return (-1.)**int(-order)*_scalar(-order, z, complex_input)
    if not complex_input:
        # Noninteger real negatives are promoted by the public entry point.
        return lax.cond(jnp.abs(z)<4, lambda x: _series(order, x),
                        lambda x: _positive(order, jnp.abs(x), False)
                        *jnp.where(x<0, (-1.)**int(order), 1.), z)
    # Reflect into the right half-plane, retaining the principal branch.
    reflected = jnp.real(z)<0
    right = jnp.where(reflected, -z, z)
    phase = jnp.exp(jnp.where(jnp.imag(z)<0, -1j, 1j)*math.pi*order)
    value = lax.cond(jnp.abs(right)<4, lambda x: _series(order, x),
                     lambda x: _positive(order, x, True), right)
    return jnp.where(reflected, phase*value, value)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def besselj(order, z):
    """Evaluate J_order(z), with analytic argument differentiation.

    Order is a finite real Python scalar. Noninteger orders promote real
    inputs to complex so negative arguments follow MATLAB's principal branch.
    Numerical qualification is bounded; extreme order/argument regimes are
    not covered by an AMOS-style uniform asymptotic implementation.

    Provenance
    ----------
    MATLAB source: @chebfun/besselj.m (builtin adapter).
    Chebfun commit: 7574c77
    """
    if isinstance(order, complex):
        if order.imag != 0:
            raise ValueError("besselj: order must be finite and real")
        order = order.real
    order = float(order)
    if not math.isfinite(order):
        raise ValueError('besselj: order must be finite and real')
    z = jnp.asarray(z)
    complex_input = jnp.iscomplexobj(z) or order != int(order)
    z = z.astype(jnp.complex128 if complex_input else jnp.float64)
    result = jax.vmap(lambda x: _scalar(order, x, complex_input))(z.reshape(-1)).reshape(z.shape)
    if order >= 0 or order == int(order):
        result = jnp.where(z == 0, 1. if order == 0 else 0., result)
    return result


@besselj.defjvp
def _besselj_jvp(order, primals, tangents):
    z, = primals
    dz, = tangents
    result = besselj(order, z)
    derivative = (besselj(order-1, z)-besselj(order+1, z))/2
    return result, derivative*dz


besselj = jax.jit(besselj, static_argnums=(0,))

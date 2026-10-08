"""JAX real-order Bessel evaluator for bounded Laguerre RH diagnostics.

Provenance
----------
MATLAB source: lagpts.m asyBessel calls besselj (special-function adapter).
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
DLMF10.2.2 power series,10.9.6 Schlaefli integral,10.17.3 asymptotic series.
This adapter needs independent qualification; it does not claim MATLAB libm
identity or uniformly accurate evaluation at arbitrary order/argument.
"""
import math

import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import lax

from chebfunjax.utils._bessel_gauss128 import GAUSS128_NODES, GAUSS128_WEIGHTS


def _series(order, z):
    # The order is static, but special-function arithmetic executes in JAX.
    if order < 0 and order == int(order):
        return (-1.0)**int(-order) * _series(-order, z)
    # Large static gamma is represented logarithmically to avoid host overflow.
    term = (jnp.exp(order*jnp.log(z/2)-jsp.gammaln(jnp.asarray(order+1, dtype=jnp.float64))) if order > 11
            else (z/2)**order/jsp.gamma(jnp.asarray(order+1, dtype=jnp.float64)))
    def step(k, state):
        term, total = state
        term = term * (-z*z/4)/(k*(order+k))
        return term, total+term
    return lax.fori_loop(1, 180, step, (term, term))[1]


def _integral(order, x):
    nodes, weights = GAUSS128_NODES, GAUSS128_WEIGHTS
    nodes, weights = jnp.asarray(nodes), jnp.asarray(weights)
    theta = (nodes+1)*(math.pi/2)
    value = jnp.sum(weights*jnp.cos(x*jnp.sin(theta)-order*theta))/2
    # Exponent is already below -70 at the cutoff for the bounded orders.
    cutoff = jnp.arcsinh((80+8*abs(order))/x)
    t = (nodes+1)*(cutoff/2)
    tail = jnp.sum(weights*jnp.exp(-x*jnp.sinh(t)-order*t))*(cutoff/2)
    return value-jnp.sin(jnp.pi*order)/math.pi*tail


def _asymptotic(order, x):
    even, odd, term = jnp.asarray(1.0), jnp.asarray(0.0), jnp.asarray(1.0)
    for k in range(1, 25):
        term = term*(4*order*order-(2*k-1)**2)/(8*k*x)
        if k % 2:
            odd = odd + (-1.0)**((k-1)//2)*term
        else:
            even = even + (-1.0)**(k//2)*term
    phase = math.pi*(order/2+0.25)
    c, s = jnp.cos(phase), jnp.sin(phase)
    cp = jnp.cos(x)*c+jnp.sin(x)*s
    sp = jnp.sin(x)*c-jnp.cos(x)*s
    return jnp.sqrt(2/(math.pi*x))*(cp*even-sp*odd)


def _bessel_j_base(order, x):
    """Positive real argument and static real order; RH special-function adapter."""
    return lax.cond(x < 4, lambda y: _series(order, y),
                    lambda y: lax.cond(y < max(32., 2*order*order),
                        lambda z: _integral(order, z),
                        lambda z: _asymptotic(order, z), y), x)


def _bessel_j_general_complex(order, z):
    """Principal imaginary-axis continuation used by negative RH Newton trials."""
    return _series(order, jnp.asarray(z, dtype=jnp.complex128))


def _bessel_j_general(order, x):
    """Static-order recurrence with bounded-order integral constants.

    DLMF10.6.1 recurrence; downward Miller normalization uses two base values
    to avoid dividing by a base value close to a zero. This adapter remains
    under qualification independently of the source RH coefficient port.
    """
    if order <= 1:
        return _bessel_j_base(order, x)
    count = math.floor(order)
    fraction = order-count
    base0 = _bessel_j_base(fraction, x)
    base1 = _bessel_j_base(fraction+1, x)

    def upward(_):
        def step(k, state):
            previous, current = state
            return current, 2*(fraction+k)/x*current-previous
        return lax.fori_loop(1, count, step, (base0, base1))[1]

    def downward(_):
        last = count+math.ceil(math.sqrt(40*count))+50
        def step(i, state):
            higher, current, target = state
            k = last-i
            target = jnp.where(k == count, current, target)
            lower = 2*(fraction+k)/x*current-higher
            rescale = jnp.where(jnp.abs(lower)>1e100, 1e-100, 1.)
            return current*rescale, lower*rescale, target*rescale
        one, zero, target = lax.fori_loop(0, last, step,
            (jnp.asarray(0.), jnp.asarray(1.), jnp.asarray(0.)))
        scale = jnp.maximum(jnp.abs(zero), jnp.abs(one))
        zero, one, target = zero/scale, one/scale, target/scale
        return target*(zero*base0+one*base1)/(zero*zero+one*one)

    return lax.cond(x > order, upward, downward, operand=None)

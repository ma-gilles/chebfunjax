"""JAX Airy functions using the ODE, saddle integral and asymptotics.

Provenance
----------
MATLAB source: @chebfun/airy.m (replacement for the builtin airy primitive).
Chebfun commit: 7574c77
Numerical formulas: Airy ODE and initial values DLMF 9.2; contour integral
9.5.4; connection formulas 9.2.12 and 9.2.10; expansions 9.7.5--9.7.8.
This is a numerical replacement, not a port of MATLAB builtin internals.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .airy_contour_nodes import NODES, WEIGHTS


def _radial(z):
    """Continue both solutions from zero in short degree24 Taylor panels."""
    n = jnp.maximum(1, jnp.ceil(2*jnp.abs(z)*(1+jnp.sqrt(jnp.abs(z))))).astype(jnp.int32)
    h = z/n
    y = jnp.stack((jnp.full_like(z, .35502805388781723926),
                   jnp.full_like(z, .61492662744600073515)))
    dy = jnp.stack((jnp.full_like(z, -.25881940379280679841),
                    jnp.full_like(z, .44828835735382635791)))

    def panel(i, state):
        y, dy = state
        center = i*h
        # Coefficients include powers of h, avoiding large intermediate
        # coefficients and leaving each term O((h*sqrt(z))**k).
        a0, a1 = y, h*dy
        a2 = center*h*h*y/2
        sy = a0+a1+a2
        sd = a1+2*a2

        def term(k, carry):
            am1, ak, ak1, sy, sd = carry
            ak2 = (center*h*h*ak+h**3*am1)/((k+1)*(k+2))
            return ak, ak1, ak2, sy+ak2, sd+(k+2)*ak2

        _, _, _, sy, sd = lax.fori_loop(1, 23, term, (a0, a1, a2, sy, sd))
        next_dy = sd/jnp.where(h == 0, 1, h)
        return (jnp.where(i < n, sy, y),
                jnp.where((i < n) & (h != 0), next_dy, dy))

    y, dy = lax.fori_loop(0, jnp.max(n), panel, (y, dy))
    return y[0], dy[0], y[1], dy[1]


def _saddle(z):
    """Ai and Ai' from a contour through sqrt(z) in the central sector."""
    root = jnp.sqrt(z)
    phi = jnp.angle(root)
    scale = 1/jnp.sqrt(1+jnp.abs(root))
    u = jnp.asarray(NODES)
    weights = jnp.asarray(WEIGHTS)
    s = scale[..., None]*u
    directions = [jnp.exp(1j*(jnp.pi/3-phi/2)),
                  jnp.exp(1j*(-jnp.pi/3-phi/2))]
    values, derivatives = [], []
    for direction in directions:
        v = direction[..., None]*s
        integrand = jnp.exp(root[..., None]*v*v+v**3/3)*direction[..., None]
        values.append(jnp.sum(weights*integrand, axis=-1)*scale)
        derivatives.append(-jnp.sum(weights*(root[..., None]+v)*integrand, axis=-1)*scale)
    prefactor = jnp.exp(-2*z*root/3)/(2j*jnp.pi)
    return prefactor*(values[0]-values[1]), prefactor*(derivatives[0]-derivatives[1])


def _asymptotic_sector(z):
    """Single exponential expansion in |arg(z)| <= 2pi/3."""
    root = jnp.sqrt(z)
    zeta = (2/3)*z*root
    ones = jnp.ones_like(z)

    def term(k, state):
        previous, u, v = state
        current = -previous*((6*k-5)*(6*k-1)/(72*k))/zeta
        return current, u+current, v+current*((6*k+1)/(1-6*k))

    _, u, v = lax.fori_loop(1, 24, term, (ones, ones, ones))
    amplitude = jnp.exp(-zeta)/(2*jnp.sqrt(jnp.pi))
    quarter = jnp.sqrt(root)
    return amplitude*u/quarter, -amplitude*v*quarter


def _asymptotic_ai(z):
    omega = jnp.exp(2j*jnp.pi/3)
    direct, direct_d = _asymptotic_sector(z)
    upper, upper_d = _asymptotic_sector(omega*z)
    lower, lower_d = _asymptotic_sector(jnp.conj(omega)*z)
    outside = jnp.abs(jnp.angle(z)) > 2*jnp.pi/3
    return (jnp.where(outside, -omega*upper-jnp.conj(omega)*lower, direct),
            jnp.where(outside, -omega**2*upper_d-jnp.conj(omega)**2*lower_d, direct_d))


@jax.jit
def _airy_all_impl(x):
    """Return Ai, Ai', Bi, Bi' for real or complex JAX arrays.

    Provenance
    ----------
    MATLAB source: @chebfun/airy.m (builtin airy dispatch).
    Chebfun commit: 7574c77
    """
    x = jnp.asarray(x)
    real_input = not jnp.issubdtype(x.dtype, jnp.complexfloating)
    z = x.astype(jnp.complex128)
    if z.size == 0:
        empty = jnp.real(z) if real_input else z
        return empty, empty, empty, empty
    finite = jnp.isfinite(z)
    z = jnp.where(finite, z, 0j)
    large = jnp.abs(z) > 24
    local = jnp.where(large, 0j, z)
    ai, aip, bi, bip = _radial(local)
    central = jnp.abs(jnp.angle(local)) <= jnp.pi/3
    sai, saip = _saddle(jnp.where(central, local, 0j))
    ai, aip = jnp.where(central, sai, ai), jnp.where(central, saip, aip)
    far = jnp.where(large, z, 25+0j)
    fai, faip = _asymptotic_ai(far)
    omega = jnp.exp(2j*jnp.pi/3)
    upper, upper_d = _asymptotic_ai(omega*far)
    lower, lower_d = _asymptotic_ai(jnp.conj(omega)*far)
    phase = jnp.exp(1j*jnp.pi/6)
    fbi = phase*upper+jnp.conj(phase)*lower
    fbip = phase*omega*upper_d+jnp.conj(phase*omega)*lower_d
    values = (jnp.where(large, fai, ai), jnp.where(large, faip, aip),
              jnp.where(large, fbi, bi), jnp.where(large, fbip, bip))
    values = tuple(jnp.where(finite, v, complex(float("nan"), float("nan"))) for v in values)
    return tuple(jnp.real(v) for v in values) if real_input else values


@jax.custom_jvp
def airy_all(x):
    """Return Ai, Ai prime, Bi and Bi prime with their analytic JAX tangent.

    Provenance
    ----------
    MATLAB source: @chebfun/airy.m (builtin airy dispatch).
    Chebfun commit: 7574c77
    """
    return _airy_all_impl(x)


@airy_all.defjvp
def _airy_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    ai, aip, bi, bip = airy_all(x)
    return (ai, aip, bi, bip), (aip*dx, x*ai*dx, bip*dx, x*bi*dx)

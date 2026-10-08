"""Source general-alpha Laguerre RH driver; bounded CPU candidate.

Provenance
----------
MATLAB source: lagpts.m407-530; besselroots.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
General special-function adapter requires qualification; no GW fallback.
Source initial guesses, Newton stopping and convergence guards are unchanged.
"""
import math
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from chebfunjax.utils._gradual import gradual_exp_negative, gradual_positive_multiply
from chebfunjax.utils._signed_gradual import flip_sign_bits, gradual_signed_divide
from chebfunjax.utils.bessel_general import _bessel_j_general, _bessel_j_general_complex
from chebfunjax.utils.bessel_roots_general import _bessel_roots_general
from chebfunjax.utils.laguerre_rh import _poly_asy_rh_alpha01
from chebfunjax.utils.laguerre_rh_expansions import (
    _asyairy_general,
    _asybessel_general,
    _asybulk_general,
)


def _rh_initial_guesses_general(n, alpha, comp_repr=False):
    """Source RH starting nodes with source Piessens/McMahon Bessel seeds."""
    mn = min(n, math.ceil(17 * math.sqrt(n))) if comp_repr else n
    itric = math.floor(3.6 * n**0.188 + 0.5)
    igatt = math.floor(mn + 1.31 * n**0.4 - n + 0.5)
    nu = 4.0 * n + 2.0 * alpha + 2.0
    roots = _bessel_roots_general(alpha, itric)
    bes = roots**2
    den = 4.0 * n + 2.0 * alpha + 2.0
    bes = bes / den * (1.0 + (bes + 2.0 * (alpha**2 - 1.0)) / den**2 / 3.0)
    ak = jnp.asarray([-13.69148903521072, -12.828776752865757,
        -11.93601556323626, -11.00852430373326, -10.04017434155809,
        -9.02265085340981, -7.944133587120853, -6.786708090071759,
        -5.520559828095551, -4.08794944413097, -2.338107410459767],
        dtype=jnp.float64)
    t = 1.5 * jnp.pi * (jnp.arange(igatt, 11, -1, dtype=jnp.float64) - 0.25)
    air_coeff = -(t**(2.0/3.0)) * (1.0 + 5.0/48.0/t**2 - 5.0/36.0/t**4 + 77125.0/82944.0/t**6 - 10856875.0/6967296.0/t**8)
    air_coeff = jnp.concatenate((air_coeff, ak[max(0, 11-igatt):]))
    air = (nu + air_coeff * (4.0*nu)**(1.0/3.0) + air_coeff**2 * (nu/16.0)**(-1.0/3.0)/5.0
        + (11.0/35.0 - alpha**2 - air_coeff**3 * 12.0/175.0)/nu
        + (16.0/1575.0*air_coeff + 92.0/7875.0*air_coeff**4)*2.0**(2.0/3.0)*nu**(-5.0/3.0)
        - (15152.0/3031875.0*air_coeff**5 + 1088.0/121275.0*air_coeff**2)*2.0**(1.0/3.0)*nu**(-7.0/3.0))
    zeros = jnp.zeros((mn-itric-max(igatt,0),), dtype=jnp.float64)
    return jnp.concatenate((bes, zeros, air)), itric, igatt

def _rh_factors_general(n, alpha):
    """Literal source normalization factors for general alpha."""
    facts = []
    for a, k in ((alpha, 1.0), (alpha+1.0, 2.0)):
        m = n - k + 1.0
        fact = 0.0
        fact += (1/3840*a**10 - 5/2304*a**9 + 11/2304*a**8 + 7/1920*a**7 - 229/11520*a**6 + 107/34560*a**5 + 2653/103680*a**4 - 989/155520*a**3 - 3481/311040*a**2 + 139/103680*a + 9871/6531840)/m**5
        fact += (1/384*a**8 - 1/96*a**7 + 1/576*a**6 + 43/1440*a**5 - 5/384*a**4 - 23/864*a**3 + 163/25920*a**2 + 31/6480*a - 139/155520)/m**4
        fact += (1/48*a**6 - 1/48*a**5 - 1/24*a**4 + 5/144*a**3 + 1/36*a**2 - 1/144*a - 31/6480)/m**3
        fact += (1/8*a**4 + 1/12*a**3 - 1/24*a**2 + 1/72)/m**2
        fact += (1/2*a**2 + 1/2*a + 1/6)/m
        fact += 1.0
        facts.append(fact)
    factorx = jnp.sqrt(facts[0]/facts[1])/2.0/(1.0-1.0/n)**(1.0+alpha/2.0)
    factorw = -(1.0-1.0/(n+1.0))**(n+1.0+alpha/2.0)*(1.0-1.0/n)**(1.0+alpha/2.0)*jnp.exp(1.0+2.0*jnp.log(2.0))*4.0**(1.0+alpha)*jnp.pi*n**alpha*jnp.sqrt(facts[0]*facts[1])*(1.0+1.0/n)**(alpha/2.0)
    return factorx, factorw

def _poly_asy_rh_general(np, y, alpha, T):
    """Source region selector with static real alpha."""
    if alpha in (0, 1):
        return _poly_asy_rh_alpha01(np, y, alpha, T)
    def near_zero(value):
        return lax.cond(
            value < 0.0,
            lambda yy: _asybessel_general(np, yy, alpha, T, _bessel_j_general_complex,
                                          complex_trial=True),
            lambda yy: _asybessel_general(np, yy, alpha, T, _bessel_j_general), value)
    def far_or_bulk(value):
        return lax.cond(value > 3.7*(np+alpha),
                        lambda yy: _asyairy_general(np, yy, alpha, T),
                        lambda yy: _asybulk_general(np, yy, alpha, T), value)
    return lax.cond(y < math.sqrt(np+alpha), near_zero, far_or_bulk, y)


@partial(jax.jit, static_argnames=("n", "alpha", "comp_repr"))
def _laguerre_rh_general(n, alpha, comp_repr=False):
    """Source full RH rule for static real alpha and n>=3000."""
    if n < 3000 or not math.isfinite(alpha) or alpha <= -1:
        raise ValueError("general RH requires n>=3000 and finite alpha>-1")
    x0, _itric, _igatt = _rh_initial_guesses_general(n, alpha, comp_repr)
    capacity = x0.shape[0]
    weights0 = jnp.zeros((capacity,), dtype=jnp.float64)
    factorx, factorw = _rh_factors_general(n, alpha)
    T = math.ceil(34.0 / math.log(n))
    eps = jnp.finfo(jnp.float64).eps
    extrapolation = jnp.asarray([7., -21., 35., -35., 21., -7., 1.],
                                dtype=jnp.float64)

    def node(k, state):
        x, weights, no_underflow = state
        source_guess = x[k]
        source_indices = jnp.clip(k - 1 - jnp.arange(7), 0, capacity - 1)
        extrapolated = jnp.dot(extrapolation, x[source_indices])
        xk = lax.cond(source_guess == 0.0, lambda _: extrapolated,
                      lambda _: source_guess, operand=None)
        initial_newton = (xk, xk, jnp.inf, xk, jnp.asarray(0, jnp.int32),
                          jnp.asarray(False))

        def newton_condition(ns):
            xcur, step, _old_value, _old_x, count, stalled = ns
            return ((jnp.abs(step) > eps * 400.0 * xcur)
                    & (count < 9) & ~stalled)

        def newton_step(ns):
            xcur, _step, old_value, old_x, count, _stalled = ns
            value = _poly_asy_rh_general(n, xcur, alpha, T)
            derivative_poly = _poly_asy_rh_general(n-1, xcur, alpha+1.0, T)
            step = value / (derivative_poly * factorx - value / 2.0)
            stalled = jnp.abs(value) >= jnp.abs(old_value) * (1.0 - 500.0 * eps)
            next_x = jnp.where(stalled, old_x, xcur - step)
            next_old_x = jnp.where(stalled, old_x, xcur)
            return next_x, step, value, next_old_x, count + 1, stalled

        xk, _step, _value, _old_x, count, _stalled = lax.while_loop(
            newton_condition, newton_step, initial_newton)
        invalid = ((xk < 0.0) | (xk > 4.0*n + 2.0*alpha + 2.0) | (count == 9)
                   | ((k != 0) & (x[k-1] >= xk)))
        xk = eqx.error_if(xk, invalid, "MATLAB lagpts RH Newton convergence guard")
        x = x.at[k].set(xk)

        def compute_weight(_):
            left = _poly_asy_rh_general(n-1, xk, alpha+1.0, T)
            right = _poly_asy_rh_general(n+1, xk, alpha, T)
            return _source_rh_weight(xk, factorw, left, right)

        wk = lax.cond(no_underflow, compute_weight,
                      lambda _: jnp.asarray(0.0, jnp.float64), operand=None)
        weights = weights.at[k].set(wk)
        starts_underflow = no_underflow & (k > 0) & _source_starts_underflow(wk, weights[k-1])
        no_underflow = no_underflow & ~starts_underflow
        return x, weights, no_underflow

    if comp_repr:
        # Source RHW returns immediately before the first zero following a
        # positive weight, or at its smaller heuristic capacity. Returning a
        # live length keeps the numerical kernel JAX; the public eager adapter
        # slices this prefix without solving any discarded tail nodes.
        def condition(state):
            k, _x, _w, active = state
            return (k < capacity) & active

        def advance(state):
            k, x, w, active = state
            x, w, active = node(k, (x, w, active))
            return k + 1, x, w, active

        k, x, weights, active = lax.while_loop(
            condition, advance, (jnp.asarray(0), x0, weights0, jnp.asarray(True)))
        length = k - jnp.asarray(~active, dtype=k.dtype)
        return x, weights, length
    x, weights, _no_underflow = lax.fori_loop(
        0, n, node, (x0, weights0, jnp.asarray(True)))
    return x, weights


def _source_rh_weight(x, factorw, left, right):
    """Source exp, multiply, denominator product, then divide; retain subnormals.

    MATLAB source: lagpts.m493-500, Chebfun7574c77680d7e82b79626300bf255498271a72df.
    factorw is strictly negative in the supported alpha range. Bitwise underflow
    decisions in the driver avoid CPU comparisons flushing a nonzero subnormal.
    Each source arithmetic stage rounds separately; libm bit identity is not claimed.
    """
    numerator = flip_sign_bits(gradual_positive_multiply(
        gradual_exp_negative(-x), -factorw))
    denominator = lax.optimization_barrier(left * right)
    return gradual_signed_divide(numerator, denominator)


def _source_positive_sqrt(value):
    """Decode positive binary64 subnormals before the source square root.

    MATLAB source: lagpts.m156; Chebfun7574c77680d7e82b79626300bf255498271a72df.
    Square roots of positive subnormals are normal binary64 numbers.
    """
    from chebfunjax.utils._gradual import _positive_parts
    mantissa, exponent = _positive_parts(value)
    odd = exponent % 2
    decoded = jnp.ldexp(jnp.sqrt(jnp.ldexp(mantissa, odd)), (exponent - odd) // 2)
    return jnp.where(value >= jnp.finfo(jnp.float64).tiny, jnp.sqrt(value), decoded)


def _source_positive_product(left, right):
    """Commutative source binary64 product with a possibly tiny operand."""
    left, right = jnp.broadcast_arrays(left, right)
    left_bits = lax.bitcast_convert_type(left, jnp.uint64)
    right_bits = lax.bitcast_convert_type(right, jnp.uint64)
    swap = right_bits < left_bits
    small, large = jnp.where(swap, right, left), jnp.where(swap, left, right)
    result = gradual_positive_multiply(small, large)
    # IEEE positive nonzero times infinity remains infinity, including tiny inputs.
    nonzero = lax.bitcast_convert_type(small, jnp.uint64) != 0
    return jnp.where(jnp.isinf(large) & nonzero, jnp.inf, result)


def _source_starts_underflow(current, previous):
    """Source current==0 and previous>0 without flushing binary64 subnormals."""
    magnitude = jnp.uint64(0x7fffffffffffffff)
    current_bits = lax.bitcast_convert_type(current, jnp.uint64)
    previous_bits = lax.bitcast_convert_type(previous, jnp.uint64)
    current_zero = (current_bits & magnitude) == 0
    previous_positive = (previous_bits > 0) & (previous_bits <= magnitude) & ~jnp.isnan(previous)
    return current_zero & previous_positive

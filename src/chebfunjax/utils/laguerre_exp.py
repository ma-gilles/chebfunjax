"""JAX port of explicit Laguerre EXP/EXPW expansions.

Provenance
----------
MATLAB source: lagpts.m1307–1433, laguerreExp.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
The source supplies less accurate Airy-region weights; no Newton/GW fallback.
EXPW retains the prefix through the last nonzero raw weight, using exact bits.
"""
import math
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from chebfunjax.utils._gradual import gradual_exp_negative, gradual_positive_divide
from chebfunjax.utils.airy import _airy_negative
from chebfunjax.utils.bessel_general import _bessel_j_general
from chebfunjax.utils.bessel_roots_general import _bessel_roots_general
from chebfunjax.utils.laguerre_rh_general import _source_positive_product


@partial(jax.jit, static_argnames=('T',))
def _exp_corrections(jak, t, ak, alpha, d, T):
    """Literal source correction polynomials, including every T>=7 term."""
    bes = jak * 0
    wbes = jak * 0
    bulk = t * 0
    wbulk = t * 0
    air = ak * 0
    if ( T >= 7 ):
        bes = bes + (657*jak**6 +36*jak**4*(73*alpha**2-181) +2*jak**2*(2459*alpha**4 -10750*alpha**2 +14051)         + 4*(1493*alpha**6 -9303*alpha**4 +19887*alpha**2 - 12077) )*d**6/2835
        wbes = wbes + (11944*alpha**6 + 5256*jak**6 - (5061*alpha**5 + 5085*alpha**4 + 4830*alpha**3         -22724*alpha**2 - 22932*alpha + 39164)*jak**4 - 74424*alpha**4 + 8*(2459*alpha**4 -10750*alpha**2         + 14051)*jak**2 + 159096*alpha**2 - 96616)/2835/2*d**6
        bulk = bulk -d**5/181440*(9216*(21*alpha**6 - 105*alpha**4 + 147*alpha**2 - 31)*t**10         -69120*(21*alpha**6 - 105*alpha**4 + 147*alpha**2 - 31)*t**9         + 384*(12285*alpha**6 -61320*alpha**4 + 85785*alpha**2 - 18086)*t**8         - 64*(136080*alpha**6 - 675675*alpha**4 +943110*alpha**2 - 198743)*t**7         + 144*(70560*alpha**6 - 345765*alpha**4 +479850*alpha**2 - 101293)*t**6         + 72576*alpha**6 - (8128512*alpha**6 - 38656800*alpha**4+ 52928064*alpha**2 - 13067711)*t**5         + 5*(1016064*alpha**6 - 4581360*alpha**4 +6114528*alpha**2 + 113401)*t**4 - 317520*alpha**4         - 10*(290304*alpha**6 -1245888*alpha**4 + 1620864*alpha**2 - 528065)*t**3         + 5*(290304*alpha**6 -1234800*alpha**4 + 1598688*alpha**2 - 327031)*t**2 + 417312*alpha**2         -5*(96768*alpha**6 - 417312*alpha**4 + 544320*alpha**2 - 111509)*t -85616)/(t-1)**8/t**2
        wbulk = wbulk + d**6/362880*(9216*(21*alpha**6 - 105*alpha**4 + 147*alpha**2 - 31)*t**10         -1536*(945*alpha**6 - 4830*alpha**4 + 6825*alpha**2 - 1444)*t**9         + 384*(11340*alpha**6 -60165*alpha**4 + 86310*alpha**2 - 18289)*t**8         - 2*(2903040*alpha**6 - 17055360*alpha**4+ 25401600*alpha**2 - 5*alpha - 5252997)*t**7         - (11753280*alpha**4 - 23506560*alpha**2+ 67*alpha - 13987519)*t**6 - 290304*alpha**6 +         12*(1016064*alpha**6 -3578400*alpha**4 + 4108608*alpha**2 + 16*alpha + 7100871)*t**5         - 5*(4064256*alpha**6 -16559424*alpha**4 + 20926080*alpha**2 + 61*alpha - 15239393)*t**4         + 1270080*alpha**4 +10*(1741824*alpha**6 - 7386624*alpha**4 + 9547776*alpha**2 + 29*alpha - 1560107)*t**3         - 15*(580608*alpha**6 - 2503872*alpha**4 + 3265920*alpha**2 + 11*alpha - 669051)*t**2- 1669248*alpha**2         + 4*(604800*alpha**6 - 2630880*alpha**4 + 3447360*alpha**2 + 13*alpha- 706850)*t         - 7*alpha + 342463)/(t-1)**9/t**3
    if ( T >= 5 ):
        bes = bes + (11*jak**4 +3*jak**2*(11*alpha**2-19) +46*alpha**4 -140*alpha**2 +94)*d**4/45
        wbes = wbes + (46*alpha**4 + 33*jak**4 +6*jak**2*(11*alpha**2 -19) -140*alpha**2 +94)/45*d**4
        air = air -(15152/3031875*ak**5+1088/121275*ak**2)*2**(1/3)*d**(7/3)
        bulk = bulk - d**3/720*(32*(15*alpha**4 - 30*alpha**2 + 7)*t**6 -144*(15*alpha**4 - 30*alpha**2 + 7)*t**5         + 16*(225*alpha**4 - 450*alpha**2 +104)*t**4 - 240*alpha**4 - 480*(5*alpha**4 - 10*alpha**2 + 1)*t**3         + 480*alpha**2 +45*(16*alpha**4 - 32*alpha**2 + 7)*t + 990*t**2 - 105)/(t-1)**5/t
        wbulk = wbulk + d**4/720*(16*(15*alpha**4 - 30*alpha**2 + 7)*t**6 - 32*(45*alpha**4 - 90*alpha**2 +22)*t**5         + 48*(75*alpha**4 - 150*alpha**2 + 74)*t**4 + 240*alpha**4 - 600*(8*alpha**4- 16*alpha**2 - 5)*t**3         + 45*(80*alpha**4 - 160*alpha**2 + 57)*t**2 - 480*alpha**2 -90*(16*alpha**4 - 32*alpha**2 + 7)*t         + 105)/(t-1)**6/t**2
    if (T >= 3):
        bes = bes + (jak**2 + 2*alpha**2 - 2)*d**2/3
        wbes = wbes + (alpha**2 + jak**2 -1)*2/3*d**2
        air = air +  ak**2*(d*16)**(1/3)/5 + (11/35-alpha**2-12/175*ak**3)*d +         (16/1575*ak+92/7875*ak**4)*2**(2/3)*d**(5/3)
        bulk = bulk - d/12*(4*(3*alpha**2 - 1)*t**2 +12*alpha**2 - 12*(2*alpha**2 - 1)*t - 3)/(t-1)**2
        wbulk = wbulk  + d**2/6*(2*t + 3)/(t-1)**3
    bes = jak**2*d*(1 + bes )
    air = 1/d +ak*(d/4)**(-1/3) + air
    bulk = bulk + t/d
    return bes, bulk, air, wbes, wbulk


def _exp_product(left, right):
    """Source rounded product, preserving subnormals and invalid negative weights."""
    return jnp.where((left >= 0) & (right >= 0),
                     _source_positive_product(left, right), left * right)


def _exp_last_nonzero(weights):
    """MATLAB find(w,1,'last'), without CPU subnormal comparison flushing."""
    bits = lax.bitcast_convert_type(weights, jnp.uint64)
    nonzero = (bits & jnp.uint64(0x7fffffffffffffff)) != 0
    return jnp.max(jnp.where(nonzero, jnp.arange(weights.size) + 1, 0), initial=0)


def _laguerre_exp(n, alpha=0.0, comp_repr=False):
    """Source expansions in staged JAX kernels; EXPW returns a live length.

    Dynamic array inputs keep complete fixed rules out of monolithic compile-time
    evaluation. Every correction and weight expression follows the source.
    """
    if n < 2 or not math.isfinite(alpha) or alpha < -1:
        raise ValueError('lagpts EXP requires n>=2 and finite alpha>=-1')
    mn = min(n, math.ceil(17 * math.sqrt(n))) if comp_repr else n
    ibes = max(math.floor(math.sqrt(n) + 0.5), 7)
    iair = math.floor(0.9 * n)
    bulk_stop = min(mn, iair - 1)
    air_count = max(mn - iair + 1, 0)
    if ibes + max(bulk_stop - ibes, 0) + air_count != mn:
        raise ValueError('lagpts EXP source region geometry is invalid')
    T = math.ceil(34 / math.log(n))
    d = 1 / (4 * n + 2 * alpha + 2)
    k = jnp.arange(ibes + 1, bulk_stop + 1, dtype=jnp.float64)
    pt = (4 * n - 4 * k + 3) * d
    t = jnp.pi**2 / 16 * (pt - 1)**2

    def inverse_step(_, value):
        return value - (pt * jnp.pi + 2 * jnp.sqrt(value - value**2)
                        - jnp.arccos(2 * value - 1)) * jnp.sqrt(value / (1 - value)) / 2

    t = lax.fori_loop(0, 6, inverse_step, t)
    jak = _bessel_roots_general(alpha, ibes)
    ak = jnp.asarray([-13.69148903521072, -12.828776752865757,
        -11.93601556323626, -11.00852430373326, -10.04017434155809,
        -9.02265085340981, -7.944133587120853, -6.786708090071759,
        -5.520559828095551, -4.08794944413097, -2.338107410459767], jnp.float64)
    tair = 3 * jnp.pi / 2 * (jnp.arange(air_count, 11, -1, dtype=jnp.float64) - 0.25)
    asymptotic = -tair**(2/3) * (1 + 5/48/tair**2 - 5/36/tair**4
                                + 77125/82944/tair**6 - 10856875/6967296/tair**8)
    ak = jnp.concatenate((asymptotic, ak[max(0, 11 - air_count):]))
    bes, bulk, air, wbes, wbulk = _exp_corrections(jak, t, ak, alpha, d, T)

    w = _exp_raw_weights(bes, bulk, air, jak, t, ak, wbes, wbulk, alpha, d)
    x = jnp.concatenate((bes, bulk, air))
    invalid = ((jnp.min(x) < 0) | (jnp.max(x) > 4*n + 2*alpha + 2)
               | (jnp.min(jnp.diff(x)) <= 0) | (jnp.min(w) < 0))
    x = eqx.error_if(x, invalid, 'MATLAB lagpts EXP wrong node or weight')
    if comp_repr:
        return x, w, _exp_last_nonzero(w)
    return x, w


@partial(jax.jit, static_argnames=('alpha',))
def _exp_raw_weights(bes, bulk, air, jak, t, ak, wbes, wbulk, alpha, d):
    """Evaluate source weights with dynamic node arrays as compilation inputs."""
    # Retain source arithmetic stages: power, exponential, products, division,
    # then correction. Do not replace the exp with a log-weight reconstruction.
    bessel = jax.vmap(lambda value: _bessel_j_general(alpha - 1, value))(jak)
    wb = _exp_product(4 * d * bes**alpha, gradual_exp_negative(-bes))
    wb = gradual_positive_divide(wb, lax.optimization_barrier(bessel**2))
    wb = _exp_product(wb, 1 + wbes)
    wm = _exp_product(bulk**alpha, gradual_exp_negative(-bulk))
    wm = _exp_product(_exp_product(wm, 2.0), jnp.pi)
    wm = _exp_product(wm, jnp.sqrt(t / (1 - t)))
    wm = _exp_product(wm, 1 + wbulk)
    aip = _airy_negative(ak)[1]
    wa = _exp_product(4**(1/3) * air**(alpha + 1/3), gradual_exp_negative(-air))
    wa = gradual_positive_divide(wa, lax.optimization_barrier(aip**2))
    return jnp.concatenate((wb, wm, wa))

"""Small-order ASY with Python orchestration over pure JAX components.

Provenance: Chebfun hermpts.m, HermiteInitialGuesses, hermpts_asy0 and
hermpoly_asy_airy, commit 7574c77680d7e82b79626300bf255498271a72df.
The source can return complex, nonconverged small-order ASY values. The
source n3 seed-layout failure is preserved. Full AD/API qualification is open.
"""

from functools import partial
from math import ceil, exp, floor, log, ulp

import jax
import jax.numpy as jnp
from jax import lax

from ._amos_airy import _airy_unscaled_dynamic
from ._source_complex import (
    complex_trig,
    ordered_multiply,
    principal_power,
    real_acos_source,
    source_complex_divide,
    source_integer_power,
    source_scale_divide,
)
from ._source_complex import eager_binary as _eager_binary
from ._source_real import source_pow_jit
from .hermite_rec import _AIRY_ROOTS


def _airy_pair(z):
    # Dispatch two ordinary calls to the same scalar JAX executable. No
    # outer map/whole-formula compilation duplicates the component graph.
    ai, ai_status = _airy_unscaled_dynamic(
        z, jnp.asarray(1.0 / 3.0, jnp.float64), jnp.asarray(False)
    )
    aip, aip_status = _airy_unscaled_dynamic(
        z, jnp.asarray(2.0 / 3.0, jnp.float64), jnp.asarray(True)
    )
    values = jnp.stack((ai, aip))
    statuses = jnp.stack((ai_status, aip_status))
    valid = jnp.all((statuses == 0) | (statuses == 3)) & jnp.all(jnp.isfinite(values))
    values = jnp.where(valid, values, jnp.asarray(jnp.nan + 1j * jnp.nan, jnp.complex128))
    return values[0], values[1]


# Stable ordinary JAX component dispatch; no numerical result memoization.
_formula_trig = jax.jit(complex_trig)
_formula_product = jax.jit(ordered_multiply)
_formula_divide = jax.jit(source_complex_divide)
_formula_scale_divide = jax.jit(source_scale_divide)
_small_acos = jax.jit(real_acos_source)


@partial(jax.jit, static_argnames=("n",))
def _small_initial_guesses(n: int) -> jax.Array:
    """Positive-half guesses from MATLAB's Airy/Tricomi blend."""
    if n == 3:
        raise ValueError("hermpts: source small REC/ASY seed concatenation fails at n=3")
    odd = n % 2
    m = (n - 1) // 2 if odd else n // 2
    a = 0.5 if odd else -0.5
    nu = 4.0 * m + 2.0 * a + 2.0
    j = jnp.arange(1, m + 1, dtype=jnp.float64)
    t = _eager_binary(
        _eager_binary(_eager_binary(3.0, jnp.pi, 0), 8.0, 3),
        _eager_binary(_eager_binary(4.0, j, 0), 1.0, 2),
        0,
    )
    t2 = source_pow_jit(t, -2)
    airy_asym = _eager_binary(
        -source_pow_jit(t, 2.0 / 3.0),
        _eager_binary(
            _eager_binary(
                _eager_binary(
                    _eager_binary(
                        _eager_binary(1.0, _eager_binary(5.0 / 48.0, t2, 0), 1),
                        _eager_binary(5.0 / 36.0, source_pow_jit(t2, 2), 0),
                        2,
                    ),
                    _eager_binary(77125.0 / 82944.0, source_pow_jit(t2, 3), 0),
                    1,
                ),
                _eager_binary(108056875.0 / 6967296.0, source_pow_jit(t2, 4), 0),
                2,
            ),
            _eager_binary(162375596875.0 / 334430208.0, source_pow_jit(t2, 5), 0),
            1,
        ),
        0,
    )
    n_exact = min(m, len(_AIRY_ROOTS))
    airy_roots = airy_asym.at[:n_exact].set(jnp.asarray(_AIRY_ROOTS[:n_exact]))
    if m < 10:
        airy_roots = jnp.asarray(_AIRY_ROOTS, dtype=jnp.float64)
    two_four_thirds = exp(log(2.0) * (4.0 / 3.0)) if m < 10 else 2.0 ** (4.0 / 3.0)
    (airy3, airy4, airy5) = (
        source_pow_jit(airy_roots, 3),
        source_pow_jit(airy_roots, 4),
        source_pow_jit(airy_roots, 5),
    )
    if m < 10:
        (airy3, airy4, airy5) = (
            source_pow_jit(airy_roots, lax.optimization_barrier(jnp.float64(k))) for k in (3, 4, 5)
        )
    airy_radicand = _eager_binary(
        _eager_binary(
            _eager_binary(
                _eager_binary(
                    _eager_binary(
                        nu,
                        _eager_binary(
                            _eager_binary(2.0 ** (2.0 / 3.0), airy_roots, 0),
                            source_pow_jit(nu, 1.0 / 3.0),
                            0,
                        ),
                        1,
                    ),
                    _eager_binary(
                        _eager_binary(
                            _eager_binary(1.0 / 5.0, two_four_thirds, 0),
                            source_pow_jit(airy_roots, 2),
                            0,
                        ),
                        source_pow_jit(nu, -1.0 / 3.0),
                        0,
                    ),
                    1,
                ),
                _eager_binary(
                    _eager_binary(
                        _eager_binary(11.0 / 35.0, source_pow_jit(a, 2), 2),
                        _eager_binary(12.0 / 175.0, airy3, 0),
                        2,
                    ),
                    nu,
                    3,
                ),
                1,
            ),
            _eager_binary(
                _eager_binary(
                    _eager_binary(
                        _eager_binary(16.0 / 1575.0, airy_roots, 0),
                        _eager_binary(92.0 / 7875.0, airy4, 0),
                        1,
                    ),
                    2.0 ** (2.0 / 3.0),
                    0,
                ),
                source_pow_jit(nu, -5.0 / 3.0),
                0,
            ),
            1,
        ),
        _eager_binary(
            _eager_binary(
                _eager_binary(
                    _eager_binary(15152.0 / 3031875.0, airy5, 0),
                    _eager_binary(1088.0 / 121275.0, source_pow_jit(airy_roots, 2), 0),
                    1,
                ),
                2.0 ** (1.0 / 3.0),
                0,
            ),
            source_pow_jit(nu, -7.0 / 3.0),
            0,
        ),
        2,
    )
    airy = (
        jnp.sqrt(airy_radicand.astype(jnp.complex128)).real if m < 10 else jnp.sqrt(airy_radicand)
    )[::-1]
    rhs = _eager_binary(
        _eager_binary(
            _eager_binary(_eager_binary(4.0 * m, _eager_binary(4.0, j, 0), 2), 3.0, 1), nu, 3
        ),
        jnp.pi,
        0,
    )
    t0 = jnp.full((m,), _eager_binary(jnp.pi, 2.0, 3), dtype=jnp.float64)

    def tricomi_step(_, t_value):
        residual = t_value - jnp.sin(t_value) - rhs
        derivative = 1.0 - jnp.cos(t_value)
        return t_value - residual / derivative

    t0 = lax.fori_loop(0, 7, tricomi_step, t0)
    tval = source_pow_jit(jnp.cos(_eager_binary(t0, 2.0, 3)), 2)
    sin_guess = jnp.sqrt(
        _eager_binary(
            _eager_binary(nu, tval, 0),
            _eager_binary(
                _eager_binary(
                    _eager_binary(
                        _eager_binary(
                            _eager_binary(
                                5.0,
                                _eager_binary(
                                    4.0, source_pow_jit(_eager_binary(1.0, tval, 2), 2), 0
                                ),
                                3,
                            ),
                            _eager_binary(1.0, _eager_binary(1.0, tval, 2), 3),
                            2,
                        ),
                        1.0,
                        2,
                    ),
                    _eager_binary(3.0, source_pow_jit(a, 2), 0),
                    1,
                ),
                3.0 * nu,
                3,
            ),
            2,
        )
    )
    p = 0.4985 + ulp(1.0)
    cut = floor(p * n)
    airy_start = ceil(p * n) - 1
    positive = jnp.concatenate((sin_guess[:cut], airy[airy_start:]))
    if odd:
        return jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), positive))[: m + 1]
    return positive[:m]


def _small_geometry(n, theta):
    """Source geometry before Airy, shared across scalar orders."""
    t = jnp.asarray(theta, dtype=jnp.complex128)
    musq = jnp.asarray(2 * n + 1, dtype=jnp.float64)
    (cos_t, sin_t) = _formula_trig(t)
    sin_2t = _formula_product(_eager_binary(2.0, cos_t, 0), sin_t)
    eta = _eager_binary(_eager_binary(0.5, t, 0), _eager_binary(0.25, sin_2t, 0), 2)
    chi = -principal_power(_eager_binary(_eager_binary(3.0, eta, 0), 2.0, 3), 2.0 / 3.0)
    phi = principal_power(_eager_binary(-chi, source_integer_power(sin_t, 2), 3), 0.25)
    z = _eager_binary(source_pow_jit(musq, 2.0 / 3.0), chi, 0)
    return musq, cos_t, sin_t, chi, phi, z


def _small_formula(n, theta):
    """Source formula orchestrated over bounded JAX component kernels."""
    musq, cos_t, sin_t, chi, phi, z = _small_geometry(n, theta)
    (ai, aip) = _airy_pair(z)
    (ai, aip) = (jnp.real(ai), jnp.real(aip))
    (a0, b0) = (1.0, 1.0)
    a1 = 15.0 / 144.0
    b1 = -7.0 / 5.0 * a1
    a2 = 5.0 * 7.0 * 9.0 * 11.0 / 2.0 / 144.0**2
    b2 = -13.0 / 11.0 * a2
    a3 = 7.0 * 9.0 * 11.0 * 13.0 * 15.0 * 17.0 / 6.0 / 144.0**3
    b3 = -19.0 / 17.0 * a3
    u0 = 1.0
    u1 = _eager_binary(
        _eager_binary(source_integer_power(cos_t, 3), _eager_binary(6.0, cos_t, 0), 2), 24.0, 3
    )
    u2 = _eager_binary(
        _eager_binary(
            _eager_binary(
                _eager_binary(-9.0, source_integer_power(cos_t, 4), 0),
                _eager_binary(249.0, source_integer_power(cos_t, 2), 0),
                1,
            ),
            145.0,
            1,
        ),
        1152.0,
        3,
    )
    u3 = _eager_binary(
        _eager_binary(
            _eager_binary(
                _eager_binary(
                    _eager_binary(
                        _eager_binary(-4042.0, source_integer_power(cos_t, 9), 0),
                        _eager_binary(18189.0, source_integer_power(cos_t, 7), 0),
                        1,
                    ),
                    _eager_binary(28287.0, source_integer_power(cos_t, 5), 0),
                    2,
                ),
                _eager_binary(151995.0, source_integer_power(cos_t, 3), 0),
                2,
            ),
            _eager_binary(259290.0, cos_t, 0),
            2,
        ),
        414720.0,
        3,
    )
    val_sum = ai
    term = _eager_binary(
        _eager_binary(
            _eager_binary(
                -_eager_binary(
                    _eager_binary(_eager_binary(a0, source_integer_power(phi, 6), 0), u1, 0),
                    a1 * u0,
                    1,
                ),
                source_integer_power(chi, 2),
                3,
            ),
            aip,
            0,
        ),
        source_pow_jit(musq, 4.0 / 3.0),
        3,
    )
    val_sum = _eager_binary(val_sum, term, 1)
    term = _eager_binary(
        _eager_binary(
            _eager_binary(
                _eager_binary(
                    _eager_binary(
                        _eager_binary(_eager_binary(b0, source_integer_power(phi, 12), 0), u2, 0),
                        _eager_binary(_eager_binary(b1, source_integer_power(phi, 6), 0), u1, 0),
                        1,
                    ),
                    b2 * u0,
                    1,
                ),
                source_integer_power(chi, 3),
                3,
            ),
            ai,
            0,
        ),
        source_pow_jit(musq, 2.0),
        3,
    )
    val_sum = _eager_binary(val_sum, term, 1)
    term = _eager_binary(
        _eager_binary(
            _eager_binary(
                -_eager_binary(
                    _eager_binary(
                        _eager_binary(
                            _eager_binary(source_integer_power(phi, 18), u3, 0),
                            _eager_binary(
                                _eager_binary(a1, source_integer_power(phi, 12), 0), u2, 0
                            ),
                            1,
                        ),
                        _eager_binary(_eager_binary(a2, source_integer_power(phi, 6), 0), u1, 0),
                        1,
                    ),
                    a3 * u0,
                    1,
                ),
                source_integer_power(chi, 5),
                3,
            ),
            aip,
            0,
        ),
        source_pow_jit(musq, 4.0 / 3.0 + 2.0),
        3,
    )
    val_sum = _eager_binary(val_sum, term, 1)
    val = _formula_product(
        _eager_binary(
            _eager_binary(
                _eager_binary(2.0, jnp.sqrt(jnp.pi), 0), source_pow_jit(musq, 1.0 / 6.0), 0
            ),
            phi,
            0,
        ),
        val_sum,
    )
    v0 = 1.0
    v1 = _eager_binary(
        _eager_binary(source_integer_power(cos_t, 3), _eager_binary(6.0, cos_t, 0), 1), 24.0, 3
    )
    v2 = _eager_binary(
        _eager_binary(
            _eager_binary(
                _eager_binary(15.0, source_integer_power(cos_t, 4), 0),
                _eager_binary(327.0, source_integer_power(cos_t, 2), 0),
                2,
            ),
            143.0,
            2,
        ),
        1152.0,
        3,
    )
    v3 = _eager_binary(
        _eager_binary(
            _eager_binary(
                _eager_binary(
                    _eager_binary(
                        _eager_binary(259290.0, cos_t, 0),
                        _eager_binary(238425.0, source_integer_power(cos_t, 3), 0),
                        1,
                    ),
                    _eager_binary(36387.0, source_integer_power(cos_t, 5), 0),
                    2,
                ),
                _eager_binary(18189.0, source_integer_power(cos_t, 7), 0),
                1,
            ),
            _eager_binary(4042.0, source_integer_power(cos_t, 9), 0),
            2,
        ),
        414720.0,
        3,
    )
    c0 = _formula_divide(
        -_eager_binary(
            _eager_binary(_eager_binary(b0, source_integer_power(phi, 6), 0), v1, 0), b1 * v0, 1
        ),
        chi,
    )
    dval_sum = _eager_binary(_eager_binary(c0, ai, 0), source_pow_jit(musq, 2.0 / 3.0), 3)
    term = _eager_binary(a0 * v0, aip, 0)
    dval_sum = _eager_binary(dval_sum, term, 1)
    c1 = _formula_divide(
        -_eager_binary(
            _eager_binary(
                _eager_binary(
                    _eager_binary(source_integer_power(phi, 18), v3, 0),
                    _eager_binary(_eager_binary(b1, source_integer_power(phi, 12), 0), v2, 0),
                    1,
                ),
                _eager_binary(_eager_binary(b2, source_integer_power(phi, 6), 0), v1, 0),
                1,
            ),
            b3 * v0,
            1,
        ),
        source_integer_power(chi, 4),
    )
    term = _eager_binary(_eager_binary(c1, ai, 0), source_pow_jit(musq, 2.0 / 3.0 + 2.0), 3)
    dval_sum = _eager_binary(dval_sum, term, 1)
    d1 = _formula_divide(
        _eager_binary(
            _eager_binary(
                _eager_binary(_eager_binary(a0, source_integer_power(phi, 12), 0), v2, 0),
                _eager_binary(_eager_binary(a1, source_integer_power(phi, 6), 0), v1, 0),
                1,
            ),
            a2 * v0,
            1,
        ),
        source_integer_power(chi, 3),
    )
    term = _eager_binary(_eager_binary(d1, aip, 0), source_pow_jit(musq, 2.0), 3)
    dval_sum = _eager_binary(dval_sum, term, 1)
    const = _formula_scale_divide(
        _eager_binary(jnp.sqrt(_eager_binary(2.0, jnp.pi, 0)), source_pow_jit(musq, 1.0 / 3.0), 0),
        phi,
    )
    dval = _formula_product(const, dval_sum)
    dt = _formula_divide(
        -val,
        _formula_product(
            _eager_binary(_eager_binary(jnp.sqrt(2.0), jnp.sqrt(musq), 0), dval, 0), sin_t
        ),
    )
    return (val, dval, dt)


def _hermpts_asy_small(n):
    """Private complete source trajectory for valid explicit ASY n2..20.

    Ordinary Python orchestration preserves source iteration and component order.
    Whole-route JIT and automatic differentiation remain unqualified.
    Nonfatal Airy status3 retains its finite value; unsupported/fatal statuses
    propagate NaN rather than becoming an artificial status0 success.
    """
    if not 2 <= n <= 20:
        raise ValueError("small ASY trajectory requires 2 <= n <= 20")
    if n == 3:
        raise ValueError("hermpts: source small REC/ASY seed concatenation fails at n=3")
    seed = _small_initial_guesses(n)
    musq = jnp.asarray(2 * n + 1, jnp.float64)
    theta = jnp.stack([_small_acos(q) for q in _eager_binary(seed, jnp.sqrt(musq), 3)])
    threshold = jnp.sqrt(jnp.finfo(jnp.float64).eps) / 10.0
    for _ in range(20):
        roots = [_small_formula(n, q) for q in theta]
        value, derivative, dt = (jnp.stack([row[k] for row in roots]) for k in range(3))
        theta = _eager_binary(theta, dt, 2)
        # Only the source loop predicate crosses to Python control. Every
        # arithmetic value, norm and convergence comparison remains in JAX.
        if bool(jnp.max(jnp.abs(dt)) < threshold):
            break
    x = _eager_binary(jnp.sqrt(musq), jnp.cos(theta), 0)
    ders = _eager_binary(_eager_binary(x, value, 0), _eager_binary(jnp.sqrt(2.0), derivative, 0), 1)
    x2 = _eager_binary(x, x, 0)
    d2 = _eager_binary(ders, ders, 0)
    w = jnp.conj(_eager_binary(jnp.exp(-x2), d2, 3))
    v = _eager_binary(jnp.exp(_eager_binary(-x2, 2.0, 3)), ders, 3)
    if n % 2:
        x = jnp.concatenate((-x[::-1], x[1:]))
        w = jnp.concatenate((w[::-1], w[1:]))
        v = jnp.concatenate((v[::-1], v[1:]))
    else:
        x = jnp.concatenate((-x[::-1], x))
        w = jnp.concatenate((w[::-1], w))
        v = jnp.concatenate((v[::-1], -v))
    w = _eager_binary(_eager_binary(jnp.sqrt(jnp.pi), jnp.sum(w), 3), w, 0)
    v = _eager_binary(v, jnp.max(jnp.abs(v)), 3)
    return tuple(_computed_result_storage(result) for result in (x, w, v))


def _computed_result_storage(result):
    """Match source computed-result storage for the qualified finite domain.

    Independent native operations canonicalize exactly zero imaginary results
    to real storage; explicit complex construction has different semantics.
    Nonfinite native storage classification is unqualified and is not inferred.
    This eager value rule has no order table or numerical tolerance.
    """
    real_result = jnp.all(jnp.isfinite(result)) & jnp.all(jnp.imag(result) == 0)
    return jnp.real(result) if bool(real_result) else result

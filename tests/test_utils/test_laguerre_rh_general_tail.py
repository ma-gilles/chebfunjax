"""Source-stage tail checks; NumPy is a host diagnostic reference only."""
# uses-numpy: inspect binary64 bit patterns and frozen source-stage diagnostics.
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.laguerre_rh_general import _source_rh_weight


def test_frozen_source_tail_stages_preserve_nonzero_weights():
    rows = json.loads((Path(__file__).parent / "fixtures" /
        "laguerre_rh_tail_source_stages.json").read_text())["rows"]
    arrays = [jnp.asarray([r[key] for r in rows]) for key in
              ("x", "factor", "left", "right")]
    actual = np.asarray(jax.jit(_source_rh_weight)(*arrays))
    expected = np.asarray([r["reference_weight"] for r in rows])
    assert np.array_equal(actual == 0, expected == 0)
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=5e-324)


def test_full_general_rule_retains_source_tail_after_normalization():
    from chebfunjax.utils.laguerre_rh_general import _laguerre_rh_general
    from chebfunjax.utils.quadrature import lagpts
    _, raw = _laguerre_rh_general(3000, .3)
    _, normalized = lagpts(3000, .3, method="RH")
    raw, normalized = np.asarray(raw), np.asarray(normalized)
    assert np.count_nonzero(raw) == 942
    assert np.count_nonzero(normalized) == 942
    assert np.count_nonzero((raw > 0) & (raw < np.finfo(float).tiny)) > 0


def test_source_sqrt_and_public_bary_interval_tail():
    import math

    from chebfunjax.utils.laguerre_rh_general import _source_positive_sqrt
    from chebfunjax.utils.quadrature import lagpts
    tiny = np.asarray([0., 5e-324, 1e-320, np.finfo(float).tiny, .5, 3.])
    np.testing.assert_allclose(np.asarray(jax.jit(_source_positive_sqrt)(jnp.asarray(tiny))),
                               [math.sqrt(float(v)) for v in tiny], rtol=2e-16, atol=0)
    x, w, v = map(np.asarray, lagpts(3000, .3, bary=True, method="RH"))
    reference = np.asarray([(-1.)**k * math.sqrt(float(a)*float(b))
                            for k, (a, b) in enumerate(zip(w, x, strict=True))])
    reference /= np.max(np.abs(reference))
    assert np.array_equal(v == 0, reference == 0)
    np.testing.assert_allclose(v, reference, rtol=5e-15, atol=0)
    for interval, scale in [((1., math.inf), math.exp(-1)),
                            ((-math.inf, 1.), math.exp(1)),
                            ((709., math.inf), math.exp(-709))]:
        _, mapped, vb = map(np.asarray, lagpts(3000, .3, interval, bary=True, method="RH"))
        expected = np.asarray([float(value)*scale for value in w])
        assert np.array_equal(mapped == 0, expected == 0)
        np.testing.assert_allclose(mapped, expected, rtol=5e-15, atol=5e-324)
        np.testing.assert_array_equal(vb, v)


def test_large_order_bessel_recurrence_without_large_gamma_or_quadrature():
    from scipy.special import jv

    from chebfunjax.utils.bessel_general import _bessel_j_general, _bessel_j_general_complex
    for order in [11.3, 12., 20.3, 50., 100.3, 170., 200.3, 500.]:
        x = np.asarray([.2, 2., 4., 10., order*.5, order-1, order, order+1,
                        order*2, order*5])
        actual = np.asarray(jax.jit(jax.vmap(lambda z: _bessel_j_general(order, z)))(jnp.asarray(x)))
        np.testing.assert_allclose(actual, jv(order, x), rtol=1e-10, atol=1e-13)
        z = 1j*np.asarray([.2, 2., 12., 40.])
        actual = np.asarray(jax.jit(jax.vmap(lambda zz: _bessel_j_general_complex(order, zz)))(jnp.asarray(z)))
        np.testing.assert_allclose(actual, jv(order, z), rtol=1e-10, atol=1e-13)


def test_source_underflow_predicate_signed_zero_and_tiny_previous():
    from chebfunjax.utils.laguerre_rh_general import _source_starts_underflow
    current = np.asarray([0., -0., 5e-324, 0., 0., 0.])
    previous = np.asarray([5e-324, 5e-324, 1., -5e-324, -0., 0.])
    actual = np.asarray(jax.jit(_source_starts_underflow)(jnp.asarray(current), jnp.asarray(previous)))
    np.testing.assert_array_equal(actual, (current == 0) & (previous > 0))


def test_source_positive_product_ieee_infinite_scale():
    from chebfunjax.utils.laguerre_rh_general import _source_positive_product
    left = np.asarray([0., 5e-324, 1., np.inf])
    actual = np.asarray(jax.jit(_source_positive_product)(jnp.asarray(left), jnp.inf))
    with np.errstate(invalid="ignore"):
        expected = left * np.inf
    np.testing.assert_array_equal(actual, expected)


def test_fixed_gauss128_integral_constants():
    from chebfunjax.utils._bessel_gauss128 import GAUSS128_NODES, GAUSS128_WEIGHTS
    x, w = np.asarray(GAUSS128_NODES), np.asarray(GAUSS128_WEIGHTS)
    assert len(x) == len(w) == 128
    assert np.all(w > 0) and np.all(np.abs(x) < 1)
    expected = np.asarray([2/(k+1) if k % 2 == 0 else 0. for k in range(256)])
    actual = np.asarray([np.sum(w*x**k) for k in range(256)])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)

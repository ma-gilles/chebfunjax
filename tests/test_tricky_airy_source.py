"""Independent controls for the source damped Airy example callback."""
# uses-numpy: independent SciPy references and recorded native callback control.
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import airy

SCRIPT = Path(__file__).resolve().parents[1] / 'examples/quad/tricky.py'
spec = importlib.util.spec_from_file_location('tricky_airy_example', SCRIPT)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_product_and_tangent_on_finite_source_domain():
    x = np.unique(np.r_[np.linspace(0., 40., 257),
                        np.nextafter(24., 0.), 24., np.nextafter(24., np.inf)])
    value, tangent = jax.jit(lambda z: jax.jvp(
        example._damped_airy, (z,), (jnp.ones_like(z),)))(jnp.asarray(x))
    ai, aip, _, _ = airy(-x)
    expected = np.exp(-x)*ai
    derivative = -np.exp(-x)*(ai+aip)
    # Existing JAX Airy numerical replacement gate; not a tolerance from
    # the Tricky page, which supplies no accuracy inequality.
    assert np.max(abs(np.asarray(value)-expected)) < 1e-12
    assert np.max(abs(np.asarray(tangent)-derivative)) < 1e-12


def test_native_finite_underflow_and_nonfinite_endpoints():
    # Native R2025b at exact binary64 physical point0x413851226c0108ea:
    # Ai(-x)=0.009303236822568329, exp(-x)=0 and product=0.
    x = jnp.array([1593634.42189079, 1e100, 1e300, np.inf, -np.inf, np.nan])
    value, tangent = jax.jvp(example._damped_airy, (x,), (jnp.ones_like(x),))
    np.testing.assert_array_equal(np.asarray(value)[:3], 0.)
    np.testing.assert_array_equal(np.asarray(tangent)[:3], 0.)
    assert np.isnan(np.asarray(value)[3:]).all()


def test_actual_exponential_underflow_neighbors():
    # Locate the backend's actual transition; no coordinate cutoff in the
    # helper. CPU subnormal flushing differs from native exp by~1e-308.
    lo, hi = 700., 750.
    for _ in range(60):
        mid = (lo+hi)/2
        if float(jnp.exp(jnp.float64(-mid))) == 0:
            hi = mid
        else:
            lo = mid
    x = jnp.array([lo, hi, np.nextafter(hi, np.inf)])
    weight = np.asarray(jnp.exp(-x))
    assert weight[0] > 0 and (weight[1:] == 0).all()
    assert np.isfinite(np.asarray(example._damped_airy(x))).all()
    np.testing.assert_array_equal(np.asarray(example._damped_airy(x))[1:], 0.)


def test_empty_shape():
    assert example._damped_airy(jnp.empty((0,))).shape == (0,)

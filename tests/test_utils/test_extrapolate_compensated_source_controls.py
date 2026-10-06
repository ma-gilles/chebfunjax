"""Independent reduction and source row-semantics contracts (portable fixture).

Source: Chebfun 7574c77 @chebtech/extrapolate.m. Compensation is a
numerical adaptation, not a port of MATLAB BLAS implementation.
"""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from chebfunjax.tech import chebtech as candidate

# Saved binary products are JAX diagnostic inputs, not MATLAB callback values.
# The original source bounds are verified separately in the MATLAB port tests.
FIXTURE = Path(__file__).with_name("data") / "unbnd_n65_weighted_products.json"
FIXTURE_SHA256 = "bd5fb616e61f544b85547012d8eb482873fb3b6383aa0fa74a18c2d88c3b2be3"


def exact_sum(a):
    a = np.asarray(a)
    with mp.workdps(100):
        if np.iscomplexobj(a):
            return complex(
                mp.fsum(mp.mpf(float(v.real)) for v in a), mp.fsum(mp.mpf(float(v.imag)) for v in a)
            )
        return float(mp.fsum(mp.mpf(float(v)) for v in a))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("complex_values", [False, True])
def test_cancellation_matrix(compiled, complex_values):
    a = np.array([[1e16, 1e-100], [1.0, 2.0], [-1e16, -1e-100]])
    if complex_values:
        a = a + 1j * a[::-1, ::-1]
    expected = np.array([exact_sum(a[:, k]) for k in range(a.shape[1])])
    fn = candidate._compensated_sum_axis0
    if compiled:
        fn = jax.jit(fn)
    np.testing.assert_array_equal(np.asarray(fn(jnp.asarray(a))), expected)


@pytest.mark.parametrize("compiled", [False, True])
def test_saved_weighted_binary_products(compiled):
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    fixture = json.loads(raw)
    w2 = np.array([float.fromhex(t) for t in fixture["weights_hex"]])
    values = np.array([float.fromhex(t) for t in fixture["values_hex"]])
    products = np.array([float.fromhex(t) for t in fixture["weighted_products_hex"]])
    np.testing.assert_array_equal(w2 * values, products)
    assert exact_sum(w2) == float.fromhex(fixture["denominator_hex"])
    # Here each multiplication by w2 is exact (signed power-of-two weights).
    expected = exact_sum(products) / exact_sum(w2)

    def ratio(weights, values):
        return candidate._compensated_sum_axis0(
            weights * values
        ) / candidate._compensated_sum_axis0(weights)

    fn = jax.jit(ratio) if compiled else ratio
    actual = float(fn(jnp.asarray(w2), jnp.asarray(values)))
    assert abs(actual - expected) <= abs(np.spacing(expected))


@pytest.mark.parametrize("kind", [1, 2])
def test_multiple_bad_rows_and_array_mask(kind):
    n = 9
    # Host fixtures independently define both Chebyshev grids and baryweights.
    if kind == 2:
        x = -np.cos(np.pi * np.arange(n) / (n - 1))
        w = (-1.0) ** np.arange(n)
        w[[0, -1]] *= 0.5
    else:
        theta = (2 * np.arange(n) + 1) * np.pi / (2 * n)
        x = -np.cos(theta)
        w = (-1.0) ** np.arange(n) * np.sin(theta)
    exact = np.column_stack([x * x + 2 * x + 3, (1 + 2j) * x + 4j])
    values = exact.copy()
    values[0, 0] = np.nan  # The entire row, including finite second column.
    values[4, 1] = np.inf
    actual, nanmask, infmask = candidate._extrapolate_values(values, x, w)
    actual = np.asarray(actual)
    mask = np.asarray(nanmask | infmask)
    np.testing.assert_array_equal(actual[~mask], values[~mask])
    np.testing.assert_allclose(actual[mask], exact[mask], rtol=0, atol=3e-13)
    assert np.flatnonzero(nanmask).tolist() == [0]
    assert np.flatnonzero(infmask).tolist() == [4]


def test_clean_and_all_bad():
    x = np.array([-1.0, 0.0, 1.0])
    w = np.array([0.5, -1.0, 0.5])
    v = np.array([-0.0, 1.0, 2.0])
    out, _, _ = candidate._extrapolate_values(v, x, w)
    np.testing.assert_array_equal(np.asarray(out).view(np.uint64), v.view(np.uint64))
    with pytest.raises(ValueError, match="Too many NaNs/Infs"):
        candidate._extrapolate_values(np.full(3, np.nan), x, w)


@pytest.mark.parametrize("compiled", [False, True])
def test_nonfinite_sum_semantics(compiled):
    a = jnp.array([[jnp.inf, jnp.inf, jnp.nan], [1.0, -jnp.inf, 2.0]])
    fn = candidate._compensated_sum_axis0
    if compiled:
        fn = jax.jit(fn)
    actual = np.asarray(fn(a))
    assert actual[0] == np.inf
    assert np.isnan(actual[1:]).all()

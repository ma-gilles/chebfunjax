"""Supplemental observed native CPU product controls; no native clause count.

Pinned @chebtech/extrapolate.m specifies a row times one column. Stored
observations cover independent cancellation basis, tails/transition, rounded
nonpower products and held-out mapped samples. This is provider-scoped
arithmetic evidence; the constructor limits dispatch more narrowly.
"""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import chebtech as tech
from chebfunjax.utils._cpu_endpoint_dot import _cpu_endpoint_dot

FIXTURE = Path(__file__).with_name("data") / "native_cpu_endpoint_dot_20261011.npz"
SHA256 = "7ff59a62a58d876e689ed72dbe48659b9eb61393c12ddabd351faa4faf98202d"
LENGTHS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 80, 128]


@pytest.mark.parametrize("n", LENGTHS)
def test_observed_native_cpu_products(n):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    with np.load(FIXTURE) as data:
        weights = jnp.asarray(data[f"w{n}"])
        values = jnp.asarray(data[f"v{n}"])
        expected = data[f"e{n}"]
    actual = jax.jit(jax.vmap(_cpu_endpoint_dot))(weights, values)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("columns", [1, 2])
def test_multiple_bad_complex_rows_retain_compensated_route(kind, columns, monkeypatch):
    # Guard against extending the observed scalar endpoint order to other
    # source routes. This supplements the existing numerical mask controls.
    from chebfunjax.utils import _cpu_endpoint_dot as module

    def forbidden(*args):
        raise AssertionError("scalar endpoint adapter reached a general route")

    monkeypatch.setattr(module, "_cpu_endpoint_dot", forbidden)
    x = tech.chebpts(17, kind=kind)
    w = getattr(tech, f"Chebtech{kind}").barywts(17)
    values = (1 + 2j) * x + 4j
    if columns == 2:
        values = jnp.stack([values, 2 * values], axis=-1)
    values = values.at[0].set(jnp.nan).at[4].set(jnp.inf)
    out, mask_nan, mask_inf = tech._extrapolate_values(values, x, w)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert bool(mask_nan[0]) and bool(mask_inf[4])


@pytest.mark.parametrize("case", ["large", "kind1", "matrix", "complex", "interior", "multiple"])
def test_unqualified_routes_keep_compensation(case, monkeypatch):
    from chebfunjax.utils import _cpu_endpoint_dot as module

    def forbidden(*args):
        raise AssertionError("endpoint adapter reached an unqualified route")

    monkeypatch.setattr(module, "_cpu_endpoint_dot", forbidden)
    n = 257 if case == "large" else 17
    kind = 1 if case == "kind1" else 2
    x = tech.chebpts(n, kind=kind)
    w = getattr(tech, f"Chebtech{kind}").barywts(n)
    values = x * x + 3
    if case == "matrix":
        values = jnp.stack([values, 2 * values], axis=-1)
    elif case == "complex":
        values = values + 1j * x
    bad = n // 2 if case == "interior" else 0
    values = values.at[bad].set(jnp.nan)
    if case == "multiple":
        values = values.at[-1].set(jnp.nan)
    out, _, _ = tech._extrapolate_values(values, x, w)
    assert bool(jnp.all(jnp.isfinite(out)))

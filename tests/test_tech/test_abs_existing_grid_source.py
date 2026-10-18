"""Existing-grid ABS representation and happiness, native7574c77.

@chebtech/abs.m changes coefficients through the current representation grid
for real/pure-imaginary data; it does not refine or change ishappy. Positive
smooth factors below have no roots, retaining that source precondition.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import chebtech as module
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize(
    "kind",
    [
        "positive",
        "negative",
        "imaginary",
        "negative_imaginary",
        "real_array",
        "imaginary_array",
        "empty",
        "empty_array",
        "complex_zero",
        "real_inf",
        "imaginary_inf",
    ],
)
def test_existing_grid_coefficients_and_happiness(Tech, kind):
    c = jnp.asarray([2.0, 0.2, 0.03, 0.004, 0.0005])
    if kind == "negative":
        c = -c
    elif kind == "imaginary":
        c = 1j * c
    elif kind == "negative_imaginary":
        c = -1j * c
    elif kind in ("real_array", "imaginary_array"):
        c = jnp.stack((c, -c, 2 * c), axis=-1)
        if kind == "imaginary_array":
            c = 1j * c
    elif kind == "empty":
        c = jnp.empty((0,), dtype=jnp.float64)
    elif kind == "empty_array":
        c = jnp.empty((0, 2), dtype=jnp.complex128)
    elif kind == "complex_zero":
        c = jnp.zeros((5,), dtype=jnp.complex128)
    elif kind in ("real_inf", "imaginary_inf"):
        c = jnp.zeros(5).at[0].set(jnp.inf)
        if kind == "imaginary_inf":
            c = jax.lax.complex(jnp.zeros_like(c), c)
    f = Tech(coeffs=c, ishappy=False)
    # Literal native transform recipe, evaluated independently at original n.
    expected = jnp.abs(c) if c.size == 0 else Tech.vals2coeffs(jnp.abs(Tech.coeffs2vals(c)))
    h = abs(f)
    print(
        {
            "tech": Tech.__name__,
            "kind": kind,
            "input_shape": c.shape,
            "actual_shape": h.coeffs.shape,
            "input_happy": f.ishappy,
            "output_happy": h.ishappy,
        },
        flush=True,
    )
    assert h.coeffs.shape == expected.shape
    assert h.ishappy == f.ishappy
    np.testing.assert_array_equal(
        np.asarray(h.coeffs).view(np.uint64), np.asarray(expected).view(np.uint64)
    )
    np.testing.assert_array_equal(np.asarray(f.coeffs), np.asarray(c))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_empty_object_preserved(Tech):
    f = Tech.empty()
    h = abs(f)
    assert h is f
    assert h.isempty()


_CAPTURE = json.loads(
    (Path(__file__).with_name("fixtures") / "abs_existing_grid_native.json").read_text()
)


@pytest.mark.parametrize("row", _CAPTURE["rows"], ids=lambda r: str(r["type"]) + "_" + r["name"])
def test_native_storage_dispatch(row):
    shape = tuple(row["input_size"])

    def component(name):
        hex_words = [] if shape[0] * shape[1] == 0 else row[name]
        assert len(hex_words) == shape[0] * shape[1]
        if not hex_words:
            assert row[name] in ([], [""])
        words = jnp.asarray([int(x, 16) for x in hex_words], dtype=jnp.uint64)
        values = jax.lax.bitcast_convert_type(words, jnp.float64)
        return values.reshape(shape[::-1]).T

    real = component("input_real_words")
    imag = component("input_imag_words")
    c = real if row["input_storage_real"] else jax.lax.complex(real, imag)
    assert module._abs_uses_existing_grid(c) == row["existing_grid_eligible"]

"""Literal 12 MTIMES groups per Tech; Chebfun pin7574c77.

Python @ adapts MATLAB *. Captured source seed6178 uniform100, complex
normal alpha and two sequential normal3x3 matrices preserve source inputs.
Bounds and source errors are unchanged; Nx1 sample vectors use 1D storage.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

DATA = json.loads((Path(__file__).parent / "fixtures/mtimes_native_rng.json").read_text())
X = jnp.asarray(DATA["x"])
ALPHA = complex(DATA["alpha_real"], DATA["alpha_imag"])
EPS = np.finfo(float).eps


@pytest.fixture(params=[Chebtech1, Chebtech2], ids=lambda t: t.__name__)
def Tech(request):
    return request.param


def column_op(x):
    return jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)


def two_op(x):
    return jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)


def bound(f):
    return 10 * float(jnp.max(f.vscale_columns * EPS))


def test_native_01_empty(Tech):
    f = Tech.from_function(jnp.sin)
    g = Tech.empty()
    assert (f @ []).isempty() and ([] @ f).isempty() and (2 @ g).isempty() and (g @ 2).isempty()


def test_native_02_scalar_commutes(Tech):
    f = Tech.from_function(jnp.sin)
    assert (ALPHA @ f).isequal(f @ ALPHA)


def test_native_03_scalar_values(Tech):
    f = Tech.from_function(jnp.sin)
    g = ALPHA @ f
    assert float(jnp.max(jnp.abs(g(X) - ALPHA * jnp.sin(X)))) < bound(g)


def test_native_04_scalar_zero_coeffs(Tech):
    f = Tech.from_function(jnp.sin)
    assert bool(jnp.all((0 @ f).coeffs == 0))


def test_native_05_array_commutes(Tech):
    f = Tech.from_function(column_op)
    assert (ALPHA @ f).isequal(f @ ALPHA)


def test_native_06_array_values(Tech):
    f = Tech.from_function(column_op)
    g = ALPHA @ f
    assert float(jnp.max(jnp.abs(g(X) - ALPHA * column_op(X)))) < bound(g)


def test_native_07_array_zero_coeffs(Tech):
    f = Tech.from_function(column_op)
    assert bool(jnp.all((0 @ f).coeffs == 0))


def test_native_08_matrix_values(Tech):
    A = jnp.asarray(DATA["matrices"][0 if Tech is Chebtech1 else 1])
    f = Tech.from_function(column_op)
    g = f @ A
    assert float(jnp.max(jnp.abs(g(X) - column_op(X) @ A))) < bound(g)


def test_native_09_numeric_left_shape(Tech):
    f = Tech.from_function(jnp.exp)
    with pytest.raises(ValueError) as caught:
        [1, 2, 3] @ f
    assert caught.value.identifier == "CHEBFUN:CHEBTECH:mtimes:size"
    assert str(caught.value) == "Inner matrix dimensions must agree."


def test_native_10_numeric_right_shape(Tech):
    f = Tech.from_function(two_op)
    with pytest.raises(ValueError) as caught:
        f @ jnp.array([[1.0], [2.0], [3.0]])
    assert caught.value.identifier == "CHEBFUN:CHEBTECH:mtimes:size2"
    assert str(caught.value) == "Inner matrix dimensions must agree."


def test_native_11_tech_product(Tech):
    f = Tech.from_function(two_op)
    g = Tech.from_function(lambda x: x)
    with pytest.raises(ValueError) as caught:
        f @ g
    assert str(caught.value) == "Use .* to multiply CHEBTECH objects."


def test_native_12_uint8(Tech):
    f = Tech.from_function(two_op)
    with pytest.raises(ValueError) as caught:
        f @ jnp.uint8(128)
    assert str(caught.value) == "mtimes does not know how to multiply a CHEBTECH and a uint8."

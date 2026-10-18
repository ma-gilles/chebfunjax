"""Whole binary64 subnormal-imaginary SIN representation and derivative controls."""
import cmath
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.elementary import _sin_cos_numeric

DATA = json.loads((Path(__file__).with_name("data") / "sin_subnormal_native_20261018.json").read_text())


def _array(words):
    return jnp.asarray(np.array([int(w, 16) for w in words], dtype=np.uint64).view(np.float64))


def _words(value, order="C"):
    return [f"{int(w):016x}" for w in np.asarray(value, dtype=np.float64).ravel(order=order).view(np.uint64)]


def _input():
    d = DATA["input"]
    x = _array(d["real"]).reshape(tuple(d["shape"]), order="F")
    y = _array(d["imag"]).reshape(tuple(d["shape"]), order="F")
    assert _words(x, "F") == d["real"] and _words(y, "F") == d["imag"]
    return jax.lax.complex(x, y)


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_sin_whole_subnormal_native_words(mode):
    fn = _sin_cos_numeric if mode == "eager" else jax.jit(_sin_cos_numeric)
    actual = fn(_input())
    mask = DATA["mask_F"]
    assert sum(mask) == 324
    words = _words(jnp.imag(actual), "F")
    assert [w for w, keep in zip(words, mask) if keep] == [
        w for w, keep in zip(DATA["native"]["imag"], mask) if keep]


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_sin_subnormal_only_selected_imaginary_words_change(mode):
    fn = _sin_cos_numeric if mode == "eager" else jax.jit(_sin_cos_numeric)
    actual = fn(_input())
    prior = DATA["prior_modes"][mode]
    assert _words(jnp.real(actual)) == prior["real"]
    mask = np.asarray(DATA["mask_F"]).reshape(tuple(DATA["input"]["shape"]), order="F").ravel()
    imag = _words(jnp.imag(actual))
    assert [w for w, keep in zip(imag, mask) if not keep] == [
        w for w, keep in zip(prior["imag"], mask) if not keep]


@pytest.mark.parametrize("point", [(-2., 0.), (-.4, -0.), (0., 0.), (.4, 0.), (2., 0.), (.4, .3)])
@pytest.mark.parametrize("reverse", [False, True])
def test_sin_subnormal_ordinary_real_coordinate_derivatives(point, reverse):
    def fn(p):
        value = _sin_cos_numeric(jax.lax.complex(p[0], p[1]))
        return jnp.stack((jnp.real(value), jnp.imag(value)))

    z = complex(*point)
    d1, d2 = cmath.cos(z), -cmath.sin(z)
    jac = np.array([[d1.real, -d1.imag], [d1.imag, d1.real]])
    hess = np.array([[[d2.real, -d2.imag], [-d2.imag, -d2.real]],
                     [[d2.imag, d2.real], [d2.real, -d2.imag]]])
    derivative = jax.jacrev if reverse else jax.jacfwd
    p = jnp.asarray(point)
    np.testing.assert_allclose(derivative(fn)(p), jac, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(derivative(derivative(fn))(p), hess, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("tangent", [1.+0j, 1j, .3-.7j])
def test_sin_subnormal_axis_transverse_jvp(tangent):
    z = jnp.asarray(.4+0j)
    _, actual = jax.jvp(_sin_cos_numeric, (z,), (jnp.asarray(tangent),))
    np.testing.assert_allclose(actual, cmath.cos(.4)*tangent, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_sin_signed_gradual_product_independent_fraction(mode):
    from chebfunjax.utils._signed_gradual import gradual_signed_multiply

    oracle = DATA["arithmetic"]
    fn = gradual_signed_multiply if mode == "eager" else jax.jit(gradual_signed_multiply)
    assert len(oracle["expected"]) == 320
    assert _words(fn(_array(oracle["left"]), _array(oracle["right"]))) == oracle["expected"]


@pytest.mark.parametrize("point", [(.7, -.3), (-2., 3.), (.4, 2.)])
def test_sin_signed_gradual_product_ordinary_derivatives(point):
    from chebfunjax.utils._signed_gradual import gradual_signed_multiply

    def fn(p):
        return gradual_signed_multiply(p[0], p[1])

    p = jnp.asarray(point)
    np.testing.assert_allclose(jax.jacfwd(fn)(p), [point[1], point[0]], rtol=0., atol=0.)
    np.testing.assert_allclose(jax.jacrev(jax.jacfwd(fn))(p), [[0., 1.], [1., 0.]], rtol=0., atol=0.)

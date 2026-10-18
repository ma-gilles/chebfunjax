"""Independent native infinite-imaginary TAN and frozen-prior word scope."""

import cmath
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS_RAW

DATA = Path(__file__).with_name("data")
CAPTURE = json.loads((DATA / "tan_infinite_phase_native_20261018.json").read_text())
PRIOR = DATA / "tan_phase_prior_31488.py.txt"
namespace = {"__name__": "_tan_phase_prior_31488", "__file__": str(PRIOR)}
exec(compile(PRIOR.read_text(), str(PRIOR), "exec"), namespace)
PRIOR_TAN = namespace["_tan_numeric"]
GROUPS = [(r["name"], [r]) for r in CAPTURE["rows"] if not r["name"].startswith("scalar_")]
GROUPS.append(("scalar_controls", [r for r in CAPTURE["rows"] if r["name"].startswith("scalar_")]))


def _component(words, shape):
    return (
        np.array([int(w, 16) for w in words], dtype=np.uint64)
        .view(np.float64)
        .reshape(shape, order="F")
    )


def _input(row):
    c = row["input"]
    shape = tuple(c["shape"])
    x, y = jnp.asarray(_component(c["real"], shape)), jnp.asarray(_component(c["imag"], shape))
    z = jax.lax.complex(x, y)
    assert _words(jnp.real(z)) == c["real"] and _words(jnp.imag(z)) == c["imag"]
    return z


def _words(v):
    return [
        f"{int(w):016x}" for w in np.asarray(v, dtype=np.float64).ravel(order="F").view(np.uint64)
    ]


def _selected(row):
    c = row["input"]
    return [
        (
            (int(x, 16) & 0x7FFFFFFFFFFFFFFF) < 0x7FF0000000000000
            and (int(y, 16) & 0x7FFFFFFFFFFFFFFF) == 0x7FF0000000000000
        )
        for x, y in zip(c["real"], c["imag"])
    ]


@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g[0])
def test_tan_infinite_phase_native_components(group, mode):
    fn = _UNARY_FNS_RAW["tan"] if mode == "eager" else jax.jit(_UNARY_FNS_RAW["tan"])
    for row in group[1]:
        actual = fn(_input(row))
        mask = _selected(row)
        assert actual.dtype == jnp.complex128 and actual.shape == tuple(row["output"]["shape"])
        for part in ["real", "imag"]:
            got = jnp.real(actual) if part == "real" else jnp.imag(actual)
            assert [w for w, k in zip(_words(got), mask) if k] == [
                w for w, k in zip(row["output"][part], mask) if k
            ]


@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("group", GROUPS, ids=lambda g: g[0])
def test_tan_infinite_phase_outside_real_and_all_imaginary_words(group, mode):
    fn = _UNARY_FNS_RAW["tan"] if mode == "eager" else jax.jit(_UNARY_FNS_RAW["tan"])
    prior = PRIOR_TAN if mode == "eager" else jax.jit(PRIOR_TAN)
    for row in group[1]:
        z = _input(row)
        actual, expected = fn(z), prior(z)
        mask = _selected(row)
        assert [w for w, k in zip(_words(jnp.real(actual)), mask) if not k] == [
            w for w, k in zip(_words(jnp.real(expected)), mask) if not k
        ]
        assert _words(jnp.imag(actual)) == _words(jnp.imag(expected))


def _fn(p):
    value = _UNARY_FNS_RAW["tan"](jax.lax.complex(p[0], p[1]))
    return jnp.stack((jnp.real(value), jnp.imag(value)))


@pytest.mark.parametrize(
    "point", [(0.4, 0.0), (-0.4, -0.0), (0.0, 0.0), (1.0, 0.0), (-1.0, 0.3), (0.4, 0.3)]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_tan_phase_ordinary_real_coordinate_first_second(point, reverse):
    z = complex(*point)
    d1 = 1 / cmath.cos(z) ** 2
    d2 = 2 * cmath.tan(z) * d1
    jac = np.array([[d1.real, -d1.imag], [d1.imag, d1.real]])
    hess = np.array(
        [[[d2.real, -d2.imag], [-d2.imag, -d2.real]], [[d2.imag, d2.real], [d2.real, -d2.imag]]]
    )
    d = jax.jacrev if reverse else jax.jacfwd
    p = jnp.asarray(point)
    np.testing.assert_allclose(d(_fn)(p), jac, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(d(d(_fn))(p), hess, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("tangent", [1.0 + 0j, 1j, 0.3 - 0.7j])
def test_tan_phase_finite_transverse_jvp(tangent):
    _, actual = jax.jvp(_UNARY_FNS_RAW["tan"], (jnp.asarray(0.4 + 0j),), (jnp.asarray(tangent),))
    np.testing.assert_allclose(actual, tangent / cmath.cos(0.4) ** 2, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64])
def test_tan_phase_other_dtypes_preserve_all_component_words(dtype):
    real = np.array(
        [-0.0, 0.0, -0.4, 0.4, -1.0, 1.0, -np.inf, np.inf, -np.nan, np.nan],
        dtype=np.float32 if dtype in [np.float32, np.complex64] else np.float64,
    )
    if dtype == np.complex64:
        value = jax.lax.complex(
            jnp.asarray(real),
            jnp.asarray(
                np.array(
                    [-np.inf, np.inf, -np.inf, np.inf, -0.0, 0.0, 0.3, -0.3, np.nan, -np.nan],
                    dtype=np.float32,
                )
            ),
        )
    else:
        value = jnp.asarray(real)
    for fn, prior in [
        (_UNARY_FNS_RAW["tan"], PRIOR_TAN),
        (jax.jit(_UNARY_FNS_RAW["tan"]), jax.jit(PRIOR_TAN)),
    ]:
        a, b = np.asarray(fn(value)), np.asarray(prior(value))
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.real.tobytes() == b.real.tobytes() and a.imag.tobytes() == b.imag.tobytes()


NATIVE_PHASE_REAL = {}
for row in CAPTURE["rows"]:
    for xr, yi, ro in zip(row["input"]["real"], row["input"]["imag"], row["output"]["real"]):
        if (int(xr, 16) & 0x7FFFFFFFFFFFFFFF) < 0x7FF0000000000000 and (
            int(yi, 16) & 0x7FFFFFFFFFFFFFFF
        ) == 0x7FF0000000000000:
            assert NATIVE_PHASE_REAL.get((xr, yi), ro) == ro
            NATIVE_PHASE_REAL[xr, yi] = ro


@pytest.mark.parametrize("shape", [(12,), (1, 12), (3, 4), (12, 1)])
@pytest.mark.parametrize("phases", [(0.0, -0.0), (-1.0, 1.0), (-0.4, 0.4), (-2.0, 2.0)])
def test_tan_phase_mixed_genuine_phase_nan_neighbors(shape, phases):
    tiny = np.nextafter(0.0, 1.0)
    value = np.empty(12, dtype=np.complex128)
    value.real = [0.0, -0.0, 0.0, -0.0, -tiny, tiny, -1.0, 1.0, -np.nan, np.nan, *phases]
    value.imag = [
        np.nan,
        np.nan,
        -np.nan,
        -np.nan,
        np.nan,
        -np.nan,
        np.nan,
        -np.nan,
        0.3,
        -0.3,
        np.inf,
        -np.inf,
    ]
    value = value.reshape(shape, order="F")
    z = jnp.asarray(value)
    mask = (np.isfinite(value.real) & np.isinf(value.imag)).ravel(order="F").tolist()
    xr, yi = _words(value.real), _words(value.imag)
    for fn, prior in [
        (_UNARY_FNS_RAW["tan"], PRIOR_TAN),
        (jax.jit(_UNARY_FNS_RAW["tan"]), jax.jit(PRIOR_TAN)),
    ]:
        actual, expected = fn(z), prior(z)
        ar, er = _words(jnp.real(actual)), _words(jnp.real(expected))
        assert [w for w, k in zip(ar, mask) if not k] == [w for w, k in zip(er, mask) if not k]
        assert [w for w, k in zip(ar, mask) if k] == [
            NATIVE_PHASE_REAL[x, y] for x, y, k in zip(xr, yi, mask) if k
        ]
        assert _words(jnp.imag(actual)) == _words(jnp.imag(expected))

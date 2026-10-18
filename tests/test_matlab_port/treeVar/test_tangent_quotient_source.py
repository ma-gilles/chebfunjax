"""Literal native quotient controls and independent frozen-prior component scope."""

import cmath
import gc
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS_RAW
from chebfunjax.utils._source_tangent import _tiny_quotient_component, source_tan, source_tanh

DATA = Path(__file__).with_name("data")
CAPTURE = json.loads((DATA / "tangent_quotient_native_20261018.json").read_text())
PRIOR = DATA / "tangent_prior_4bfd.py.txt"
namespace = {"__name__": "_tangent_prior_4bfd", "__file__": str(PRIOR)}
exec(compile(PRIOR.read_text(), str(PRIOR), "exec"), namespace)
ROWS = [r for r in CAPTURE["rows"] if not r["name"].startswith("provider_")]
PROVIDERS = [r for r in CAPTURE["rows"] if r["name"].startswith("provider_")]


@pytest.fixture(autouse=True)
def _clear_compilations(request):
    large_native = "native_selected_components" in request.node.name
    if large_native:
        jax.clear_caches()
        gc.collect()
    yield
    if large_native:
        jax.clear_caches()
        gc.collect()


def _part(words, shape):
    return (
        np.asarray([int(w, 16) for w in words], dtype=np.uint64)
        .view(np.float64)
        .reshape(shape, order="F")
    )


def _words(value):
    return np.asarray(value, dtype=np.float64).ravel(order="F").view(np.uint64)


def _argument(row):
    c = row["input"]
    shape = tuple(c["shape"])
    z = jax.lax.complex(jnp.asarray(_part(c["real"], shape)), jnp.asarray(_part(c["imag"], shape)))
    assert np.array_equal(
        _words(jnp.real(z)), np.asarray([int(w, 16) for w in c["real"]], dtype=np.uint64)
    )
    assert np.array_equal(
        _words(jnp.imag(z)), np.asarray([int(w, 16) for w in c["imag"]], dtype=np.uint64)
    )
    return z


def _masks(row):
    x = np.asarray([int(w, 16) for w in row["input"]["real"]], dtype=np.uint64) & np.uint64(
        0x7FFFFFFFFFFFFFFF
    )
    y = np.asarray([int(w, 16) for w in row["input"]["imag"]], dtype=np.uint64) & np.uint64(
        0x7FFFFFFFFFFFFFFF
    )
    finite = (x < 0x7FF0000000000000) & (y < 0x7FF0000000000000)
    real = finite & (x > 0) & (x < 0x3C80000000000000)
    imag = finite & (y > 0) & (y < 0x3C80000000000000)
    return real, imag


@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["name"])
def test_tangent_native_selected_components_and_frozen_prior_words(row, mode):
    name = row["method"]
    fn = _UNARY_FNS_RAW[name]
    prior = namespace["_" + name + "_numeric"]
    if mode == "jit":
        fn, prior = jax.jit(fn), jax.jit(prior)
    z = _argument(row)
    actual, old = fn(z), prior(z)
    assert actual.dtype == old.dtype == jnp.complex128
    assert actual.shape == old.shape == tuple(row["output"]["shape"])
    for part, mask in zip(["real", "imag"], _masks(row)):
        a = _words(getattr(jnp, part)(actual))
        b = _words(getattr(jnp, part)(old))
        native = np.asarray([int(w, 16) for w in row["output"][part]], dtype=np.uint64)
        np.testing.assert_array_equal(a[mask], native[mask])
        np.testing.assert_array_equal(a[~mask], b[~mask])


@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("row", PROVIDERS, ids=lambda r: r["name"])
def test_tangent_scalar_source_provider_words(row, mode):
    value = jnp.asarray(_part(row["input"]["real"], row["input"]["shape"]))
    provider = source_tan if row["method"] == "tan" else source_tanh
    fn = lambda v: jax.lax.map(provider, v.reshape(-1)).reshape(v.shape)
    if mode == "jit":
        fn = jax.jit(fn)
    np.testing.assert_array_equal(
        _words(fn(value)), np.asarray([int(w, 16) for w in row["output"]["real"]], dtype=np.uint64)
    )


@pytest.mark.parametrize("name", ["tan", "tanh"])
@pytest.mark.parametrize(
    "point",
    [
        (0.4, 0.0),
        (-0.4, -0.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.3),
        (0.4, 0.3),
        (1e-18, 0.3),
        (0.4, 1e-18),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_tangent_real_coordinate_first_second_derivatives(name, point, reverse):
    z = complex(*point)
    f = cmath.tan(z) if name == "tan" else cmath.tanh(z)
    d1 = 1 + f * f if name == "tan" else 1 - f * f
    d2 = (2 if name == "tan" else -2) * f * d1
    jac = np.array([[d1.real, -d1.imag], [d1.imag, d1.real]])
    hess = np.array(
        [[[d2.real, -d2.imag], [-d2.imag, -d2.real]], [[d2.imag, d2.real], [d2.real, -d2.imag]]]
    )

    def fn(p):
        v = _UNARY_FNS_RAW[name](jax.lax.complex(p[0], p[1]))
        return jnp.stack((jnp.real(v), jnp.imag(v)))

    d = jax.jacrev if reverse else jax.jacfwd
    p = jnp.asarray(point)
    np.testing.assert_allclose(d(fn)(p), jac, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(d(d(fn))(p), hess, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("name", ["tan", "tanh"])
@pytest.mark.parametrize("tangent", [1.0 + 0j, 1j, 0.3 - 0.7j])
def test_tangent_transverse_jvp(name, tangent):
    z = 0.4 + 1e-18j
    f = cmath.tan(z) if name == "tan" else cmath.tanh(z)
    derivative = 1 + f * f if name == "tan" else 1 - f * f
    _, actual = jax.jvp(_UNARY_FNS_RAW[name], (jnp.asarray(z),), (jnp.asarray(tangent),))
    np.testing.assert_allclose(actual, tangent * derivative, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("name", ["tan", "tanh"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64])
def test_tangent_other_dtype_prior_words(name, dtype):
    rd = np.float64 if dtype == np.float64 else np.float32
    real = np.asarray([-0.0, 0.0, -0.4, 0.4, -1.0, 1.0, -np.inf, np.inf, -np.nan, np.nan], dtype=rd)
    value = (
        jax.lax.complex(
            jnp.asarray(real),
            jnp.asarray(
                [0.0, -0.0, 1e-20, -1e-20, 0.3, -0.3, 0.3, -0.3, np.nan, -np.nan], dtype=jnp.float32
            ),
        )
        if dtype == np.complex64
        else jnp.asarray(real)
    )
    for fn, prior in [
        (_UNARY_FNS_RAW[name], namespace["_" + name + "_numeric"]),
        (jax.jit(_UNARY_FNS_RAW[name]), jax.jit(namespace["_" + name + "_numeric"])),
    ]:
        a, b = np.asarray(fn(value)), np.asarray(prior(value))
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.real.tobytes() == b.real.tobytes() and a.imag.tobytes() == b.imag.tobytes()


@pytest.mark.parametrize("name", ["tan", "tanh"])
@pytest.mark.parametrize("shape", [(12,), (1, 12), (3, 4), (12, 1)])
def test_tangent_mixed_tiny_infinite_and_nan_neighbor_scope(name, shape):
    tiny = np.nextafter(0.0, 1.0)
    normal = np.finfo(float).tiny
    value = np.empty(12, dtype=np.complex128)
    value.real = [0.4, -0.4, tiny, -normal, np.inf, -np.inf, 0.0, -0.0, np.nan, -np.nan, 1.0, -1.0]
    value.imag = [tiny, -normal, 0.3, -0.3, 0.3, -0.3, np.nan, -np.nan, 0.3, -0.3, 0.3, -0.3]
    value = value.reshape(shape, order="F")
    z = jnp.asarray(value)
    row = {
        "method": name,
        "input": {
            "real": [f"{w:016x}" for w in _words(value.real)],
            "imag": [f"{w:016x}" for w in _words(value.imag)],
        },
    }
    masks = _masks(row)
    wanted = set(zip(row["input"]["real"], row["input"]["imag"]))
    native = {}
    for control in ROWS:
        if control["method"] != name:
            continue
        for x, y, r, i in zip(
            control["input"]["real"],
            control["input"]["imag"],
            control["output"]["real"],
            control["output"]["imag"],
        ):
            if (x, y) in wanted:
                if (x, y) in native:
                    assert native[x, y] == (r, i)
                native[x, y] = r, i
    for fn, prior in [
        (_UNARY_FNS_RAW[name], namespace["_" + name + "_numeric"]),
        (jax.jit(_UNARY_FNS_RAW[name]), jax.jit(namespace["_" + name + "_numeric"])),
    ]:
        actual, old = fn(z), prior(z)
        for k, (part, mask) in enumerate(zip(["real", "imag"], masks)):
            a, b = _words(getattr(jnp, part)(actual)), _words(getattr(jnp, part)(old))
            np.testing.assert_array_equal(a[~mask], b[~mask])
            expected = [
                int(native[x, y][k], 16)
                for x, y, selected in zip(row["input"]["real"], row["input"]["imag"], mask)
                if selected
            ]
            np.testing.assert_array_equal(a[mask], np.asarray(expected, dtype=np.uint64))


@pytest.mark.parametrize("name", ["tan", "tanh"])
def test_tangent_empty_prior_words(name):
    for shape in [(0,), (0, 2), (2, 0)]:
        value = jnp.empty(shape, dtype=jnp.complex128)
        for fn, prior in [
            (_UNARY_FNS_RAW[name], namespace["_" + name + "_numeric"]),
            (jax.jit(_UNARY_FNS_RAW[name]), jax.jit(namespace["_" + name + "_numeric"])),
        ]:
            a, b = np.asarray(fn(value)), np.asarray(prior(value))
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize(
    "filename",
    ["tangent_smith_fraction_20261018.json", "tangent_smith_normal_fraction_20261018.json"],
)
@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_tangent_quotient_independent_fraction_rounding(filename, mode):
    rows = json.loads((DATA / filename).read_text())["rows"]
    value = jnp.asarray(_part([r["component"] for r in rows], [len(rows)]))
    coefficient = jnp.asarray(_part([r["coefficient"] for r in rows], [len(rows)]))

    def fn(v, t):
        return _tiny_quotient_component(v, t, False), _tiny_quotient_component(v, t, True)

    if mode == "jit":
        fn = jax.jit(fn)
    for part, actual in zip(["plus", "minus"], fn(value, coefficient)):
        np.testing.assert_array_equal(
            _words(actual), np.asarray([int(r[part], 16) for r in rows], dtype=np.uint64)
        )


@pytest.mark.parametrize("first", ["source", "public"])
def test_tangent_cold_traced_import_constants(first, record_property):
    """A prior traced import must not leave phase coefficient globals as tracers."""
    import os
    import subprocess
    import sys

    code = r"""
import hashlib,json,os,sys,subprocess
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
import chebfunjax,jaxlib,ml_dtypes
from chebfunjax.operators.treevar import _UNARY_FNS_RAW
expected=json.loads(sys.argv[2])
assert Path(sys.executable).resolve()==Path(expected['python']).resolve()
assert str(Path(chebfunjax.__file__).resolve())==expected['origin']
assert [jax.__version__,jaxlib.__version__,ml_dtypes.__version__]==expected['versions']
assert sorted(os.sched_getaffinity(0))==expected['affinity']
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not jax.config.jax_enable_compilation_cache
if expected['head'] is not None:
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=expected['repo'],text=True).strip()==expected['head']
assert jax.config.jax_enable_x64 and jax.default_backend()=='cpu'
tiny=np.nextafter(0.,1.);z=jax.lax.complex(jnp.asarray(.4),jnp.asarray(tiny))
if sys.argv[1]=='source':
 def initial(x):
  from chebfunjax.utils._source_complex import source_expm1
  return source_expm1(x)
 jax.jit(initial)(jnp.asarray(.4)).block_until_ready()
else:
 jax.jit(_UNARY_FNS_RAW['sinh'])(z).block_until_ready()
from chebfunjax.utils._source_phase import IP,PI
from chebfunjax.utils._source_scalar_constants import LARGE
assert not isinstance(IP,jax.core.Tracer) and not isinstance(PI,jax.core.Tracer),(type(IP).__name__,type(PI).__name__)
np.testing.assert_array_equal(np.asarray(IP),np.asarray(LARGE['two_over_pi'],dtype=np.int64))
np.testing.assert_array_equal(np.asarray(PI).view(np.uint64),np.asarray(LARGE['PIo2'][:5],dtype=np.float64).view(np.uint64))
for name in ['tan','tanh']:
 for fn in [_UNARY_FNS_RAW[name],jax.jit(_UNARY_FNS_RAW[name])]:
  out=fn(z);assert int(np.asarray(out.imag).view(np.uint64))==1
files={str(Path(m.__file__).resolve()) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).is_file()}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
print(json.dumps({'first':sys.argv[1],'module_files':{p:sha(p) for p in sorted(files)}}))
"""
    from pathlib import Path

    import jax
    import jaxlib
    import ml_dtypes

    import chebfunjax

    expected = {
        "python": sys.executable,
        "origin": str(Path(chebfunjax.__file__).resolve()),
        "versions": [jax.__version__, jaxlib.__version__, ml_dtypes.__version__],
        "affinity": sorted(os.sched_getaffinity(0)),
        "repo": str(Path.cwd()),
        "head": None,
    }
    try:
        expected["head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    child_env = dict(os.environ)
    child_env.update(
        CUDA_VISIBLE_DEVICES="",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        JAX_ENABLE_COMPILATION_CACHE="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    child = subprocess.run(
        [sys.executable, "-c", code, first, json.dumps(expected)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    observed = json.loads(child.stdout)
    assert observed["first"] == first
    record_property("cold_child_module_files", json.dumps(observed["module_files"], sort_keys=True))

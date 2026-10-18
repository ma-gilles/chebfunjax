"""JAX getTol arithmetic and strict native scalar-power controls."""

import ast
import hashlib
import importlib
import inspect
import json
import math
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun._numeric_phase_one import phase_one_numeric
from chebfunjax.diskfun._numeric_tolerance import _gradients, numeric_tolerance
from chebfunjax.diskfun.diskfun import _get_tol
from chebfunjax.utils._source_real import source_pow_jit

RAW = json.loads((Path(__file__).parents[1] / "fixtures" /
                  "disk_gettol_native_20261018.json").read_text())
CASES = RAW["cases"]


def _array(record):
    if not math.prod(record["shape"]):
        return np.empty(record["shape"])
    return np.asarray([struct.unpack(">d", bytes.fromhex(w))[0]
                       for w in record["re_hex"]]).reshape(record["shape"], order="F")


def _arguments(case):
    values = jnp.asarray(_array(case["input"]))
    m, n = values.shape
    return values, 2*math.pi/n, math.pi/max(m-1, 1), _array(case["pseudo"]).item()


def _classes(actual, expected):
    actual = np.asarray(actual, np.float64)
    expected = np.asarray(expected, np.float64)
    assert actual.size == expected.size
    actual = actual.reshape(expected.shape)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    return actual, expected


def _words(actual, expected):
    actual, expected = _classes(actual, expected)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual[finite]).view(np.uint64),
        np.ascontiguousarray(expected[finite]).view(np.uint64))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_actual_production_tolerance_native_words(case):
    tol, _ = _get_tol(*_arguments(case), source_numeric=True)
    _words(tol, _array(case["tolerance"]))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_baseline_words_except_native_required_power_changes(case):
    tol, vscale = _get_tol(*_arguments(case), source_numeric=True)
    assert isinstance(tol, jax.Array) and isinstance(vscale, jax.Array)
    assert tol.dtype == jnp.float64 and vscale.dtype == jnp.float64
    assert tol.shape in ((), (0, 1)) and vscale.shape == ()
    if case["native_pow_gap"]:
        # Exactly these seven old words must change to the captured native
        # result. The unchanged native assertions above remain strict.
        with pytest.raises(AssertionError):
            _words(tol, _array(case["baseline_tolerance"]))
        _words(tol, _array(case["tolerance"]))
    else:
        _words(tol, _array(case["baseline_tolerance"]))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_native_vscale_words_and_tolerance_classes_stay_strict(case):
    tol, vscale = _get_tol(*_arguments(case), source_numeric=True)
    _words(vscale, _array(case["vscale"]))
    _classes(tol, _array(case["tolerance"]))


@pytest.mark.parametrize("case", [c for c in CASES if c["stages"]],
                         ids=lambda c: c["name"])
def test_native_forward_difference_division_words(case):
    values, hx, hy, _ = _arguments(case)
    dx, dy = _gradients(values, jnp.float64(hx), jnp.float64(hy))
    _words(dx, _array(case["stages"]["dfdx"]))
    _words(dy, _array(case["stages"]["dfdy"]))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_native_phase_decision_words_at_computed_tolerance(case):
    values, hx, hy, pseudo = _arguments(case)
    tol, _ = _get_tol(values, hx, hy, pseudo, source_numeric=True)
    actual = phase_one_numeric(values, tol, 100.)
    native = case["phase"]
    assert len(actual.pivots) == native["rank"]
    assert actual.remove_pole == native["remove_poles"]
    assert actual.happy == bool(native["is_happy"])
    _words(actual.pivots, _array(native["pivots"]))
    _words(actual.pivot_array, _array(native["pivot_array"]))
    _words(actual.indices+1, _array(native["pivot_indices"]))
    assert actual.plus == (_array(native["idx_plus"]).reshape(-1)-1).tolist()
    assert actual.minus == (_array(native["idx_minus"]).reshape(-1)-1).tolist()


def test_source_dispatch_executes_without_host_numpy_numerics(monkeypatch):
    module = importlib.import_module("chebfunjax.diskfun.diskfun")
    values = jnp.asarray([[.25, -.5], [.75, 1.]])
    hx, hy, pseudo = jnp.float64(math.pi), jnp.float64(math.pi), jnp.float64(2**-52)

    class ForbidNumpy:
        def __getattr__(self, name):
            raise AssertionError("host NumPy accessed: " + name)

    monkeypatch.setattr(module, "np", ForbidNumpy())
    tol, vscale = module._get_tol(values, hx, hy, pseudo, source_numeric=True)
    tol.block_until_ready()
    assert isinstance(tol, jax.Array) and isinstance(vscale, jax.Array)
    closed = jax.make_jaxpr(lambda a, b, c, d: module._get_tol(
        a, b, c, d, source_numeric=True))(values, hx, hy, pseudo)
    assert "pure_callback" not in str(closed) and "io_callback" not in str(closed)
    # Negative control demonstrates that the gate detects the legacy host path.
    with pytest.raises(AssertionError, match="host NumPy accessed: diff"):
        module._get_tol(values, hx, hy, pseudo)


def test_default_false_body_AST_exact_after_new_source_dispatch():
    node = ast.parse(inspect.getsource(_get_tol)).body[0]
    first = node.body.pop(1)
    assert isinstance(first, ast.If) and isinstance(first.test, ast.Name)
    assert first.test.id == "source_numeric"
    # Python3.12 adds empty type_params to FunctionDef ASTs. The frozen
    # baseline hash uses Python3.10; remove only that empty schema field.
    canonical = ast.dump(node, include_attributes=False).replace(", type_params=[]", "")
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    assert digest == RAW["baseline_get_tol_AST_sha256"]


LARGE = RAW["large_grid_power_controls"]["cases"]


def _scalar(word):
    return struct.unpack(">d", bytes.fromhex(word))[0]


@pytest.mark.parametrize("case", LARGE, ids=lambda c: c["input_hex"])
def test_native_power_across_full_array_grid_domain(case):
    # These are normal positive integral operands, including rounded sys.maxsize.
    exponent = jnp.float64(_scalar(RAW["large_grid_power_controls"]["exponent_hex"]))
    actual = source_pow_jit(jnp.float64(_scalar(case["input_hex"])), exponent)
    _words(actual, _scalar(case["power_hex"]))


def test_power_vector_and_outer_jit_use_same_native_words():
    grids = jnp.asarray([_scalar(c["input_hex"]) for c in LARGE])
    exponent = jnp.float64(_scalar(RAW["large_grid_power_controls"]["exponent_hex"]))
    expected = np.asarray([_scalar(c["power_hex"]) for c in LARGE])
    _words(source_pow_jit(grids, exponent), expected)
    _words(jax.jit(lambda x: source_pow_jit(x, exponent))(grids), expected)


@pytest.mark.parametrize("shape", [(2, 2048), (4097, 2)])
def test_source_tolerance_larger_static_shapes(shape):
    grid = max(shape)
    native = next(c for c in LARGE if _scalar(c["input_hex"]) == grid)
    # Constant values give source gradient0 and vscale1, so only the literal
    # source scalar power/domain/value/eps products remain. Each Python
    # binary64 product rounds separately as in the captured source staging.
    expected = ((_scalar(native["power_hex"]) * math.pi) * 1.) * (2**-52)
    tol, vscale = numeric_tolerance(jnp.ones(shape, jnp.float64),
                                   2*math.pi/shape[1],
                                   math.pi/(shape[0]-1), 2**-52)
    _words(tol, expected)
    _words(vscale, 1.)


def test_every_nonempty_64bit_array_grid_is_in_source_power_domain():
    # Shape limits, rather than observed native sample sizes, establish domain.
    assert np.iinfo(np.intp).max == 2**63-1
    exponent = _scalar(RAW["large_grid_power_controls"]["exponent_hex"])
    assert 0 < exponent < 1 and exponent <= 32
    assert exponent * 63 <= 42 < 1000
    assert float(np.iinfo(np.intp).max) == 2.**63

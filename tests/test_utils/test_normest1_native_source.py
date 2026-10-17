"""Native replay: real dense MATLAB R2025b condest/normest1 native replay.

The fixture directory is produced by capture_normest_cases.m. MAT arrays,
including full 625-word RNG states and operator traces, are the oracles.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.utils import _normest1 as normest_module
from chebfunjax.utils._matlab_twister import MatlabTwisterState
from chebfunjax.utils._matlab_uniform_stream import MatlabUniformStream
from chebfunjax.utils._normest1 import normest1_real
from chebfunjax.utils._normest_triangular import triangular_solve

NATIVE_DIR = Path(__file__).resolve().parents[1] / "references" / "pde_condest_native"
EPS = np.finfo(np.float64).eps


def _load(name):
    # Do not squeeze MATLAB columns or cell arrays: orientation is source data.
    return _unwrap(
        loadmat(
            NATIVE_DIR / f"{name}.mat", squeeze_me=False, struct_as_record=False, mat_dtype=True
        )
    )


def _unwrap(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def _field(record, name):
    record = _unwrap(record)
    if isinstance(record, dict):
        return _unwrap(record[name])
    return _unwrap(getattr(record, name))


def _source_state(record):
    """MATLAB State(1:624) are MT words; State(625) is the output index."""
    state = np.asarray(_field(record, "State"), dtype=np.uint32).reshape(-1)
    assert state.shape == (625,)
    return MatlabTwisterState(
        jnp.asarray(state[:624], dtype=jnp.uint32), jnp.asarray(state[624], dtype=jnp.int32)
    )


def _assert_state(actual, native_record):
    expected = _source_state(native_record)
    np.testing.assert_array_equal(np.asarray(actual.words), np.asarray(expected.words))
    np.testing.assert_array_equal(np.asarray(actual.index), np.asarray(expected.index))


def _assert_source_array(actual, expected):
    """Keep the existing 100*eps scaled source-array comparison policy."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    np.testing.assert_array_equal(
        np.signbit(actual[np.isinf(actual)]), np.signbit(expected[np.isinf(expected)])
    )
    finite = np.isfinite(expected)
    scale = max(1.0, float(np.max(np.abs(expected[finite]))) if finite.any() else 0.0)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=0.0, atol=100.0 * EPS * scale)


def _trace_entries(value):
    cells = np.asarray(value, dtype=object)
    return [_unwrap(item) for item in cells.ravel(order="F")]


def _trace_flag(entry):
    value = _field(entry, "flag")
    if isinstance(value, str):
        return value
    return "".join(str(ch) for ch in np.asarray(value).reshape(-1)).strip()


def _trace_array(entry, name):
    return np.asarray(_field(entry, name), dtype=np.float64)


def _lu_actions(lower, upper, trace):
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    def forward(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        y = triangular_solve(lower, x, lower=True, unit=True)
        y = triangular_solve(upper, y, lower=False)
        trace.append(("notransp", np.asarray(x).copy(), np.asarray(y).copy()))
        return y

    def transpose(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        y = triangular_solve(upper, x, lower=False, transpose=True)
        y = triangular_solve(lower, y, lower=True, unit=True, transpose=True)
        trace.append(("transp", np.asarray(x).copy(), np.asarray(y).copy()))
        return y

    return forward, transpose


def _assert_native_trace(actual_trace, native_trace):
    expected_entries = _trace_entries(native_trace)
    assert len(actual_trace) == len(expected_entries)
    for (actual_flag, actual_input, actual_output), expected in zip(actual_trace, expected_entries):
        assert actual_flag == _trace_flag(expected)
        # The input signs/one-hot vectors and captured rand-start block are
        # exact; equality here also checks MATLAB's column-major draw ordering.
        np.testing.assert_array_equal(actual_input, _trace_array(expected, "input"))
        _assert_source_array(actual_output, _trace_array(expected, "output"))


def _run_normest(case, action, transpose, *, t=2):
    owner = MatlabUniformStream(_source_state(_field(case, "before")))
    result = normest1_real(
        action, transpose, n=int(np.asarray(_field(case, "A")).shape[0]), stream=owner, t=t
    )
    return result, owner


REGULAR_CASES = (
    "reachable_pde6",
    "dense6",
    "positive_inverse6",
    "ties10",
    "hilbert6",
    "index_below6",
    "index_above6",
    "small4",
)


@pytest.mark.parametrize("name", REGULAR_CASES)
def test_native_normest1_full_trace_and_final_state(name):
    case = _load(name)
    trace = []
    forward, transpose = _lu_actions(_field(case, "L"), _field(case, "U"), trace)
    result, owner = _run_normest(case, forward, transpose)

    _assert_source_array(result["estimate"], _field(case, "est"))
    np.testing.assert_array_equal(
        np.asarray(result["v"]).reshape(-1, 1),
        np.asarray(_field(case, "v"), dtype=np.float64),
    )
    _assert_source_array(np.asarray(result["w"]).reshape(-1, 1), _field(case, "w"))
    np.testing.assert_array_equal(
        np.asarray(result["iterations"]).reshape(-1), np.asarray(_field(case, "iter")).reshape(-1)
    )
    _assert_native_trace(trace, _field(case, "trace"))
    _assert_state(owner.state, _field(case, "after"))


@pytest.mark.parametrize("name", REGULAR_CASES + ("singular6",))
def test_native_condest_values_and_index_gt_one_decisions(name):
    case = _load(name)
    before = _source_state(_field(case, "before"))
    owner = MatlabUniformStream(before)
    value = normest_module.condest_real(_field(case, "A"), stream=owner)
    native_value = float(np.asarray(_field(case, "c")).reshape(-1)[0])
    _assert_source_array(value, native_value)

    threshold = EPS * int(np.count_nonzero(_field(case, "A"))) * float(value)
    expected_threshold = float(np.asarray(_field(case, "threshold")).reshape(-1)[0])
    _assert_source_array(threshold, expected_threshold)
    # The source initialization predicate short-circuits for an all-zero
    # Jacobian before evaluating eps*nnz*c (which is NaN for 0*Inf).
    source_index_gt_one = int(np.count_nonzero(_field(case, "A"))) == 0 or threshold > 1.0
    assert bool(source_index_gt_one) is bool(_field(case, "index_gt_one"))
    if name == "singular6" or name == "small4":
        _assert_state(owner.state, _field(case, "before"))
    else:
        _assert_state(owner.state, _field(case, "after"))


@pytest.mark.parametrize("t", (1, 3, 6))
def test_native_block_sizes_and_exact_branch(t):
    case = _load(f"block{t}")
    trace = []
    forward, transpose = _lu_actions(_field(case, "L"), _field(case, "U"), trace)
    result, owner = _run_normest(case, forward, transpose, t=t)
    _assert_source_array(result["estimate"], _field(case, "est"))
    np.testing.assert_array_equal(
        np.asarray(result["v"]).reshape(-1, 1), np.asarray(_field(case, "v"), dtype=np.float64)
    )
    _assert_source_array(np.asarray(result["w"]).reshape(-1, 1), _field(case, "w"))
    np.testing.assert_array_equal(
        np.asarray(result["iterations"]).reshape(-1), np.asarray(_field(case, "iter")).reshape(-1)
    )
    _assert_native_trace(trace, _field(case, "trace"))
    _assert_state(owner.state, _field(case, "after"))
    if t == 6:
        assert len(trace) == 1
        _assert_state(owner.state, _field(case, "before"))


def test_native_zero_operator_first_iteration_and_redraws():
    case = _load("zero_operator6")
    trace = []

    def zero(x):
        x = np.asarray(x, dtype=np.float64)
        y = jnp.zeros_like(x)
        trace.append(("zero", x.copy(), y.copy()))
        return y

    owner = MatlabUniformStream(_source_state(_field(case, "before")))
    result = normest1_real(zero, zero, n=6, stream=owner)
    _assert_source_array(result["estimate"], _field(case, "est"))
    np.testing.assert_array_equal(
        np.asarray(result["v"]).reshape(-1, 1), np.asarray(_field(case, "v"), dtype=np.float64)
    )
    _assert_source_array(np.asarray(result["w"]).reshape(-1, 1), _field(case, "w"))
    np.testing.assert_array_equal(
        np.asarray(result["iterations"]).reshape(-1), np.asarray(_field(case, "iter")).reshape(-1)
    )
    expected_trace = _trace_entries(_field(case, "trace"))
    assert len(trace) == len(expected_trace)
    for (_, actual_input, actual_output), expected in zip(trace, expected_trace):
        np.testing.assert_array_equal(actual_input, _trace_array(expected, "input"))
        np.testing.assert_array_equal(actual_output, _trace_array(expected, "output"))
    _assert_state(owner.state, _field(case, "after"))


def test_native_consecutive_calls_advance_the_same_uniform_owner(monkeypatch):
    case = _load("consecutive")
    owner = MatlabUniformStream(_source_state(_field(case, "before")))
    monkeypatch.setattr(normest_module, "default_uniform_stream", owner)
    first = normest_module.condest_real(_field(case, "A"))
    _assert_source_array(first, _field(case, "c1"))
    _assert_state(owner.state, _field(case, "middle"))
    second = normest_module.condest_real(_field(case, "A"))
    _assert_source_array(second, _field(case, "c2"))
    _assert_state(owner.state, _field(case, "after"))


@pytest.mark.parametrize("name", REGULAR_CASES + ("singular6",))
def test_condition_lu_matches_native_factors(name):
    case = _load(name)
    lower = np.asarray(_field(case, "L"))
    upper = np.asarray(_field(case, "U"))
    expected = np.where(
        np.arange(lower.shape[0])[:, None] > np.arange(lower.shape[1])[None, :], lower, upper
    )
    np.testing.assert_array_equal(
        normest_module._condition_lu(jnp.asarray(_field(case, "A"))), expected
    )

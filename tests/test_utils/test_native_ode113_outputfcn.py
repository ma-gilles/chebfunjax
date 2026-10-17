"""Native-backed MATLAB R2025b ode113 OutputFcn observations.

Capture source and exact options are documented in the sibling capture package.
Native MAT arrays preserve exact callback values and dimensions; JSON supplies
readable metadata. Missing artifacts fail rather than skip.

MATLAB source: R2025b ode113.m, ntrp113.m, odefinalize.m;
@chebfun/ode113.m, @chebfun/constructODEsol.m, tests/chebfun/test_ivp.m.
Chebfun commit: 7574c77.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.chebfun1d.chebfun import ode113 as chebfun_ode113
from chebfunjax.utils.native_ode113 import native_ode113

_NATIVE = Path(__file__).resolve().parents[1] / "references" / "ode113_outputfcn_native"
_EPS = np.finfo(np.float64).eps
# Existing native ode113 exact-polynomial/source-array contract.
_BOUND = 100.0 * _EPS

_CASES = [
    ("forward", [0.0, 0.2], 1, None, False, "none"),
    ("refined_selected", [0.0, 0.2], 4, [2, 1, 2], False, "none"),
    ("reverse", [0.2, 0.0], 4, [2, 1, 2], False, "none"),
    ("requested", [0.0, 0.013, 0.071, 0.19, 0.2], 4, None, False, "none"),
    ("stop", [0.0, 0.2], 4, None, False, "stop2"),
    ("init_return", [0.0, 0.2], 1, None, False, "initstop"),
    ("event_requested", [0.0, 0.04, 0.1, 0.2], 4, None, True, "none"),
    ("throw", [0.0, 0.2], 1, None, False, "throw2"),
    ("requested_stop", [0.0, 0.001, 0.002, 0.003, 0.2], 4, None, False, "stop2"),
]

_WRAPPERS = [
    ("wrapper_normal", [0.0, 0.2], 4, [2, 1, 2], False, True, "none"),
    ("wrapper_stop", [0.0, 0.2], 1, None, False, True, "stop2"),
    ("wrapper_restart_stop", [0.0, 0.1, 0.2], 1, None, False, True, "stop2"),
    ("wrapper_no_restart_stop", [0.0, 0.1, 0.2], 4, None, False, False, "stop2"),
    ("wrapper_event", [0.0, 0.1, 0.2], 4, None, True, True, "none"),
]


class _CallbackFailure(RuntimeError):
    pass


def _rhs(t, y):
    del t
    return jnp.array([-y[0], 2 * y[1]], dtype=y.dtype)


def _mat_callback_entries(name):
    # MAT is the numeric oracle. Keep MATLAB array dimensions by avoiding
    # squeeze_me; the sibling JSON supplies readable case metadata only.
    mat = loadmat(_NATIVE / f"{name}.mat", squeeze_me=False, struct_as_record=False)
    assert {"c", "callback_log", "observation"} <= set(mat)
    cells = np.asarray(mat["callback_log"]).reshape(-1, order="F")
    return [_unwrap_mat_singletons(cell) for cell in cells]


def _unwrap_mat_singletons(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def _mat_struct_field(value, name):
    return getattr(value, name)


def _mat_flag(entry):
    raw = entry.flag
    if isinstance(raw, str):
        return raw
    return "".join(str(v) for v in np.asarray(raw).reshape(-1, order="F"))


def _mat_bool(entry, field):
    return bool(np.asarray(getattr(entry, field)).reshape(-1, order="F")[0])


def _assert_callback_payload(actual, expected, native):
    assert actual["flag"] == expected["flag"]
    assert _mat_flag(native) == expected["flag"]
    t_native = np.asarray(native.t, dtype=np.float64)
    y_native = np.asarray(native.y, dtype=np.float64)
    assert list(t_native.shape) == expected["t_size"]
    assert list(y_native.shape) == expected["y_size"]
    assert list(actual["t"].shape) == list(t_native.shape)
    assert list(actual["y"].shape) == list(y_native.shape)
    assert _mat_bool(native, "return_stop") is expected["return_stop"]
    assert _mat_bool(native, "throws") is expected["throws"]
    for got, want in ((actual["t"], t_native), (actual["y"], y_native)):
        scale = max(1.0, float(np.max(np.abs(want))) if want.size else 0.0)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=_BOUND * scale)


def _assert_same_solver_state(actual, control):
    np.testing.assert_array_equal(np.asarray(actual["x"]), np.asarray(control["x"]))
    np.testing.assert_array_equal(np.asarray(actual["y"]), np.asarray(control["y"]))
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual["idata"])
    control_leaves, control_tree = jax.tree_util.tree_flatten(control["idata"])
    assert actual_tree == control_tree
    assert len(actual_leaves) == len(control_leaves)
    for got, want in zip(actual_leaves, control_leaves):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert actual["stats"] == control["stats"]


@pytest.mark.parametrize("name,span,refine,selection,event,mode", _CASES)
def test_outputfcn_callbacks_match_native_capture(name, span, refine, selection, event, mode):
    native_entries = _mat_callback_entries(name)
    expected = json.loads((_NATIVE / f"{name}_callbacks.json").read_text())
    observation = json.loads((_NATIVE / f"{name}.json").read_text())
    assert observation["case"]["name"] == name

    calls = []
    ordinary_count = 0

    def output(t, y, flag):
        nonlocal ordinary_count
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        should_stop = False
        throws = False
        if flag == "":
            ordinary_count += 1
            should_stop = mode == "stop2" and ordinary_count == 2
            throws = mode == "throw2" and ordinary_count == 2
        elif flag == "init":
            should_stop = mode == "initstop"
        calls.append(
            {
                "flag": flag,
                "t": np.asarray(t),
                "y": np.asarray(y),
                "return_stop": should_stop,
                "throws": throws,
            }
        )
        if throws:
            raise _CallbackFailure("intentional second ordinary callback exception")
        return should_stop

    options = {
        "RelTol": 1e-6,
        "AbsTol": 1e-9,
        "InitialStep": 0.01,
        "MaxStep": 0.05,
        "Refine": refine,
        "OutputFcn": output,
    }
    if selection is not None:
        options["OutputSel"] = jnp.asarray(selection, dtype=jnp.int64)
    if event:
        options["Events"] = lambda t, y: (t - 0.075, True, 0)

    if mode == "throw2":
        with pytest.raises(_CallbackFailure, match="intentional second ordinary"):
            native_ode113(_rhs, span, jnp.array([1.0, 2.0]), options)
    else:
        solved = native_ode113(_rhs, span, jnp.array([1.0, 2.0]), options)
        assert callable(solved["sol"])
        if mode == "stop2":
            mat = loadmat(_NATIVE / f"{name}.mat", squeeze_me=False, struct_as_record=False)
            source_solution = _unwrap_mat_singletons(mat["solution_numeric"])
            source_x = np.asarray(_mat_struct_field(source_solution, "x"))
            source_y = np.asarray(_mat_struct_field(source_solution, "y"))
            np.testing.assert_allclose(
                np.asarray(solved["x"]).reshape(-1),
                source_x.reshape(-1),
                rtol=0.0,
                atol=_BOUND * max(1.0, float(np.max(np.abs(source_x)))),
            )
            candidate_y = np.asarray(solved["y"])
            assert candidate_y.shape == source_y.shape
            np.testing.assert_allclose(
                candidate_y,
                source_y,
                rtol=0.0,
                atol=_BOUND * max(1.0, float(np.max(np.abs(source_y)))),
            )
        elif observation["mesh_equal_control"] is True:
            # The callback observes interpolated output values; it must not
            # perturb accepted Adams history, mesh, or controller statistics.
            control_options = dict(options)
            control_options["OutputFcn"] = None
            control = native_ode113(_rhs, span, jnp.array([1.0, 2.0]), control_options)
            assert observation["state_equal_control"] is True
            _assert_same_solver_state(solved, control)

    assert len(calls) == len(expected)
    assert len(native_entries) == len(expected)
    for actual, descriptive, native in zip(calls, expected, native_entries):
        _assert_callback_payload(actual, descriptive, native)
    assert [v["flag"] for v in calls] == observation["callback_flags"]
    if mode == "throw2":
        assert observation["error_id"] == "Capture:CallbackException"
        assert "done" not in [v["flag"] for v in calls]
    else:
        assert calls[-1]["flag"] == "done"


def test_outputsel_checks_original_integer_before_int32_conversion():
    # Regression input exceeds int32 but is not a valid component index. It
    # must be rejected, rather than wrapping to component 1.
    options = {
        "OutputFcn": lambda t, y, flag: False,
        "OutputSel": jnp.asarray([4294967297], dtype=jnp.int64),
    }
    with pytest.raises(ValueError, match="OutputSel index outside state vector"):
        native_ode113(_rhs, (0.0, 0.2), jnp.array([1.0, 2.0]), options)


@pytest.mark.parametrize("name,span,refine,selection,event,restart,mode", _WRAPPERS)
def test_outputfcn_public_wrapper_matches_native_observation(
    name, span, refine, selection, event, restart, mode
):
    native_entries = _mat_callback_entries(name)
    descriptive = json.loads((_NATIVE / f"{name}_callbacks.json").read_text())
    observation = json.loads((_NATIVE / f"{name}.json").read_text())
    assert observation["case"]["name"] == name
    assert observation["case"]["route"] == "wrapper"
    calls = []
    ordinary_count = 0

    def output(t, y, flag):
        nonlocal ordinary_count
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        if flag == "init":
            ordinary_count = 0
        should_stop = False
        if flag == "":
            ordinary_count += 1
            should_stop = mode == "stop2" and ordinary_count == 2
        calls.append(
            {
                "flag": flag,
                "t": np.asarray(t),
                "y": np.asarray(y),
                "return_stop": should_stop,
                "throws": False,
            }
        )
        return should_stop

    options = {
        "RelTol": 1e-6,
        "AbsTol": 1e-9,
        "InitialStep": 0.01,
        "MaxStep": 0.05,
        "Refine": refine,
        "OutputFcn": output,
        "restartSolver": restart,
    }
    if selection is not None:
        options["OutputSel"] = jnp.asarray(selection, dtype=jnp.int64)
    if event:
        options["Events"] = lambda t, y: (t - 0.075, True, 0)

    if observation["success"]:
        solution = chebfun_ode113(_rhs, span, jnp.array([1.0, 2.0]), options)
        assert observation["construction_success"] is True
        # The original test_ivp pass(3) representation bound is 1e-5.
        mat = loadmat(_NATIVE / f"{name}.mat", squeeze_me=False, struct_as_record=False)
        native_domain = np.asarray(mat["wrapper_domain"], dtype=np.float64).reshape(-1)
        native_values = np.asarray(mat["wrapper_values"], dtype=np.float64)
        query = jnp.linspace(float(native_domain[0]), float(native_domain[-1]), 9)
        # The native capture passes a 1-by-9 MATLAB row. Rank-two queries
        # preserve its packed component-block layout in the Python API.
        actual_values = np.asarray(solution(query[None, :]))
        assert actual_values.shape == native_values.shape
        np.testing.assert_allclose(actual_values, native_values, rtol=0.0, atol=1e-5)
    else:
        assert observation["error_id"] == "MATLAB:deval:SolOutsideInterval"
        assert observation["phase"] == "solve"
        with pytest.raises(ValueError, match="outside integration interval"):
            chebfun_ode113(_rhs, span, jnp.array([1.0, 2.0]), options)
        assert observation["construction_success"] is False

    assert len(calls) == len(descriptive) == len(native_entries)
    for actual, json_entry, mat_entry in zip(calls, descriptive, native_entries):
        _assert_callback_payload(actual, json_entry, mat_entry)
    assert [v["flag"] for v in calls] == observation["callback_flags"]
    assert len(calls) == observation["callback_count"]

    if name == "wrapper_restart_stop":
        assert observation["callback_flags"].count("init") == 2
        assert observation["callback_flags"].count("done") == 2

"""Captured native N=3 initialization and public PDE replay.

Provenance: Chebfun7574c77, MATLAB R2025b; pde_general_native_prepare_v4_20261017.
NumPy/SciPy are used only to decode native fixture files in these tests.
"""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.chebfun1d._pde_ndf import dae_init, pde_driver
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.pde15s import pde15s
from chebfunjax.utils._matlab_twister import MatlabTwisterState
from chebfunjax.utils._matlab_uniform_stream import MatlabUniformStream

FIXTURE = Path(__file__).resolve().parent / "references" / "pde_general_native"


def _unwrap(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.reshape(-1, order="F")[0]
    return value


def _field(value, name):
    return _unwrap(getattr(_unwrap(value), name))


def _array(value):
    return jnp.asarray(_unwrap(value))


def _vector(value):
    return _array(value).reshape(-1, order="F")


def _source_stream(rng_struct):
    # Existing qualified MATLAB uniform fixture mapping: words first, index last.
    raw = np.asarray(_field(rng_struct, "State")).reshape(-1, order="F")
    assert raw.shape == (625,) and raw.dtype == np.uint32
    assert 0 <= int(raw[624]) <= 624
    state = MatlabTwisterState(
        jnp.asarray(raw[:624], dtype=jnp.uint32), jnp.asarray(raw[624], dtype=jnp.int32)
    )
    return MatlabUniformStream(state)


@pytest.fixture(scope="module")
def native():
    manifest = json.loads((FIXTURE / "FIXTURE_HASHES.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    assert json.loads((FIXTURE / "metadata.json").read_text())["complete"]
    summary = json.loads((FIXTURE / "solver_summary.json").read_text())
    assert all(
        summary[name]
        for name in (
            "replay_mesh_equal",
            "replay_state_equal",
            "replay_callbacks_equal",
            "replay_rng_equal",
        )
    )
    assert summary["J_size"] == [6, 6]
    # mat_dtype is essential: compact MAT integer storage may represent MATLAB doubles.
    record = _unwrap(
        loadmat(
            FIXTURE / "solver_capture.mat", mat_dtype=True, squeeze_me=False, struct_as_record=False
        )["record"]
    )
    public = loadmat(
        FIXTURE / "public_output.mat", mat_dtype=True, squeeze_me=False, struct_as_record=False
    )
    return record, public


def _assert_source_close(actual, expected):
    actual, expected = jnp.asarray(actual), jnp.asarray(expected)
    assert actual.shape == expected.shape
    bound = 100 * jnp.finfo(jnp.float64).eps * jnp.maximum(1.0, jnp.max(jnp.abs(expected)))
    assert float(jnp.max(jnp.abs(actual - expected))) <= float(bound)


def test_captured_initializer_after_native_jacobian(monkeypatch, native):
    """Isolate post-numjac initializer on native M/y/f0/DfDy, not numjac itself."""
    record, _ = native
    algebraic = _field(record, "algebraic")
    expected = _field(record, "initialized")
    mass = _array(_field(algebraic, "M0"))
    initial = _vector(_field(algebraic, "y"))
    f0 = _vector(_field(algebraic, "f"))
    jacobian = _array(_field(algebraic, "DfDy"))
    fac = _vector(_field(algebraic, "fac"))
    alg = _vector(_field(algebraic, "AlgVar")).astype(jnp.int32) - 1
    transformed = _vector(_field(algebraic, "F"))
    assert alg.size == 6
    # This fixture must take the native consistent-IC branch; no surrogate RHS.
    assert float(jnp.linalg.norm(transformed[alg])) <= float(
        1000 * jnp.finfo(jnp.float64).eps * jnp.linalg.norm(transformed)
    )
    calls = []

    def captured_initial_rhs(y):
        assert bool(jnp.array_equal(y, initial)), "Unexpected Newton evaluation in isolated fixture"
        calls.append("f0")
        return f0

    def captured_numjac(fun, y, f, threshold, *args, **kwargs):
        assert bool(jnp.array_equal(y, initial))
        assert bool(jnp.array_equal(f, f0))
        calls.append("jacobian")
        return jacobian, fac, []

    monkeypatch.setattr(dae_init, "dense_numjac", captured_numjac)
    options = _field(record, "options")
    result = dae_init.initialize_full_mass(
        captured_initial_rhs,
        mass,
        initial,
        reltol=float(_vector(_field(options, "RelTol"))[0]),
        abstol=float(_vector(_field(options, "AbsTol"))[0]),
        uniform_stream=_source_stream(_field(_field(record, "condest"), "rng_before")),
    )
    assert calls == ["f0", "jacobian"]
    assert result["reason"] == "initial_consistent"
    assert result["algebraic"].size == 6
    for key in ("y", "yp", "f"):
        _assert_source_close(result[key], _vector(_field(expected, key)))
    _assert_source_close(result["jac"], _array(_field(expected, "jac")))
    # SVD coordinates/signs and estimator RNG output are not invariant across
    # equivalent null-space bases; identical-J condest fixtures cover them separately.


def test_public_three_component_native_samples(monkeypatch, native):
    """Full public assembly, numerical Jacobian, initializer and NDF evolution."""
    record, public = native
    initial_values = _array(_field(record, "initial"))
    assert initial_values.shape == (9, 3)
    assert bool(jnp.all(initial_values == initial_values[0:1, :]))
    u0 = Chebfun.from_coeffs(initial_values[0:1, :], domain=(-1.0, 1.0))
    times = _vector(_field(record, "span"))
    assert bool(jnp.array_equal(times, _vector(_field(record, "primary_t"))))
    # Native nonAdaptiveEvent omits tOut updates: public tout is just [0].
    # Compare species/time cells using actual requested/solver times, not tout.
    assert np.asarray(public["tout"]).shape == (1, 1)
    assert float(np.asarray(public["tout"])[0, 0]) == float(times[0])
    assert np.asarray(public["samples"]).shape == (3, times.size)
    stream = _source_stream(_field(record, "rng_before_primary"))
    original = pde_driver.initialize_full_mass
    init_calls = []

    def initialize_with_native_stream(
        fun, mass, y, *, reltol=1e-6, abstol=1e-6, uniform_stream=None
    ):
        assert uniform_stream is None
        init_calls.append(mass.shape)
        return original(fun, mass, y, reltol=reltol, abstol=abstol, uniform_stream=stream)

    monkeypatch.setattr(pde_driver, "initialize_full_mass", initialize_with_native_stream)

    def rhs(u, v, w):
        return (0.1 * u.diff(2) - u + v, 0.2 * v.diff(2) + u - 2 * v + w, 0.05 * w.diff(2) + v - w)

    def boundary(u, v, w):
        return (u.diff(), v.diff(), w.diff())

    options = _field(record, "options")
    outputs = pde15s(
        rhs,
        times,
        u0,
        lbc=boundary,
        rbc=boundary,
        n=9,
        rtol=float(_vector(_field(options, "RelTol"))[0]),
        atol=float(_vector(_field(options, "AbsTol"))[0]),
    )
    assert init_calls == [(27, 27)]
    assert len(outputs) == times.size == 3
    probes = jnp.asarray([-0.75, 0.0, 0.625])
    for time_index, output in enumerate(outputs):
        expected = jnp.column_stack(
            [_vector(public["samples"][species, time_index]) for species in range(3)]
        )
        actual = output(probes)
        assert actual.shape == expected.shape == (3, 3)
        # Existing manufactured multicomponent forward bound; frozen before replay.
        assert float(jnp.max(jnp.abs(actual - expected))) < 1e-7

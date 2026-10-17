"""Native fixed-N prefix replay for published Chebfun chemical PDE example.

The full adaptive 31-time published example remains unqualified.
"""

import hashlib
import inspect
import json
from pathlib import Path

# uses-numpy: decode native MATLAB fixtures and make independent assertions.
import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.chebfun1d._pde_ndf import pde_driver
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebfun1d.pde15s import pde15s
from chebfunjax.utils._matlab_twister import MatlabTwisterState
from chebfunjax.utils._matlab_uniform_stream import MatlabUniformStream

FIXTURE = Path(__file__).resolve().parent / "references" / "pde_chemical_prefix_native"
SOURCE_ARRAY_FACTOR = 100.0
FORWARD_BOUND = 1e-7


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


def _source_close(actual, expected):
    """Existing native-array policy: 100 eps times max(1, source infinity scale)."""
    actual, expected = jnp.asarray(actual), jnp.asarray(expected)
    assert actual.shape == expected.shape
    bound = (
        SOURCE_ARRAY_FACTOR
        * jnp.finfo(jnp.float64).eps
        * jnp.maximum(1.0, jnp.max(jnp.abs(expected)))
    )
    assert float(jnp.max(jnp.abs(actual - expected))) <= float(bound)


def _source_stream(rng_struct):
    """Use captured MATLAB Twister state; do not reseed per solve or estimate."""
    raw = np.asarray(_field(rng_struct, "State")).reshape(-1, order="F")
    assert raw.shape == (625,) and raw.dtype == np.uint32
    assert 0 <= int(raw[624]) <= 624
    state = MatlabTwisterState(
        jnp.asarray(raw[:624], dtype=jnp.uint32), jnp.asarray(raw[624], dtype=jnp.int32)
    )
    return MatlabUniformStream(state)


def _assembly(initializer_fun):
    """Inspect existing solver closures at the full-mass initializer boundary."""
    full = inspect.getclosurevars(initializer_fun).nonlocals["fun"]
    scope = inspect.getclosurevars(full).nonlocals
    raw = scope["raw"]
    raw_scope = inspect.getclosurevars(raw).nonlocals
    projection = raw_scope["projection"]
    return full, raw, projection, scope["offset"], scope, raw_scope


@pytest.fixture(scope="module")
def native():
    hashes = json.loads((FIXTURE / "FIXTURE_HASHES.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    metadata = json.loads((FIXTURE / "metadata.json").read_text())
    assert (
        metadata["complete"]
        and metadata["reference_pin"] == "7574c77680d7e82b79626300bf255498271a72df"
    )
    assert metadata["diagnostic_N"] == 33 and metadata["diagnostic_times"] == [0, 0.1, 0.2]
    assert metadata["default_tolerance"] == 1e-6
    summary = json.loads((FIXTURE / "solver_summary.json").read_text())
    assert summary["fixed_N"] == 33 and summary["adaptive_grid"] is False
    record = _unwrap(
        loadmat(
            FIXTURE / "solver_capture.mat", mat_dtype=True, squeeze_me=False, struct_as_record=False
        )["record"]
    )
    public = loadmat(
        FIXTURE / "public_output.mat", mat_dtype=True, squeeze_me=False, struct_as_record=False
    )
    action_metadata = json.loads((FIXTURE / "boundary_actions_metadata.json").read_text())
    assert action_metadata["complete"]
    assert action_metadata["matlab_version"].startswith("25.2.0.3055257 (R2025b)")
    assert action_metadata["scope"] == "same-input native dense boundary actions; no PDE replay"
    actions = loadmat(
        FIXTURE / "boundary_actions.mat", mat_dtype=True, squeeze_me=False, struct_as_record=False
    )
    return record, public, actions


def _reaction(u, v, w):
    uv = u * v
    return (
        0.1 * u.diff(2) - 100 * uv,
        0.2 * v.diff(2) - 100 * uv,
        0.001 * w.diff(2) + 200 * uv,
    )


def _chemical_initial_from_jax_erf():
    """Construct the published profiles with the actual JAX erf primitive."""
    initial = chebfun(
        lambda x: jnp.stack(
            (
                1 - jsp.erf(10 * (x + 0.7)),
                1 + jsp.erf(10 * (x - 0.7)),
                jnp.zeros_like(x),
            ),
            axis=-1,
        )
    )
    return initial


def test_actual_jax_erf_constructor_analytic_control():
    """Check the actual constructor against the analytic published profiles."""
    probes = jnp.asarray([-0.75, 0.0, 0.625])
    actual = _chemical_initial_from_jax_erf()(probes)
    expected = jnp.stack(
        (
            1 - jsp.erf(10 * (probes + 0.7)),
            1 + jsp.erf(10 * (probes - 0.7)),
            jnp.zeros_like(probes),
        ),
        axis=-1,
    )
    assert actual.shape == expected.shape == (3, 3)
    assert float(jnp.max(jnp.abs(actual - expected))) < FORWARD_BOUND


def test_actual_erf_constructor_chemical_prefix_native_samples(monkeypatch, native):
    """Run the prefix from the actual JAX erf constructor, retaining native output bound."""
    record, public, _ = native
    times = _vector(public["requested_times"])
    options = _field(record, "options")
    reltol = float(_vector(_field(options, "RelTol"))[0])
    abstol = float(_vector(_field(options, "AbsTol"))[0])
    stream = _source_stream(_field(record, "rng_before_primary"))
    original = pde_driver.initialize_full_mass

    def initialize_with_native_stream(
        fun, mass, y, *, reltol=1e-6, abstol=1e-6, uniform_stream=None
    ):
        assert uniform_stream is None
        return original(fun, mass, y, reltol=reltol, abstol=abstol, uniform_stream=stream)

    monkeypatch.setattr(pde_driver, "initialize_full_mass", initialize_with_native_stream)
    outputs = pde15s(
        _reaction,
        times,
        _chemical_initial_from_jax_erf(),
        bc="neumann",
        n=33,
        rtol=reltol,
        atol=abstol,
    )
    assert len(outputs) == 3
    probes = _vector(public["sample_x"])
    samples = public["samples"]
    for time_index, output in enumerate(outputs):
        expected = jnp.column_stack([_vector(samples[species, time_index]) for species in range(3)])
        actual = output(probes)
        assert actual.shape == expected.shape == (3, 3)
        # Same predeclared native forward bound as captured-coefficient replay.
        assert float(jnp.max(jnp.abs(actual - expected))) < FORWARD_BOUND


def test_native_chemical_prefix_assembly_and_samples(monkeypatch, native):
    record, public, actions = native
    context = _field(record, "context")
    coefficients = _array(_field(record, "coefficients_before_prolongation"))
    assert coefficients.ndim == 2 and coefficients.shape == (101, 3)
    assert bool(jnp.all(jnp.isfinite(coefficients)))
    after_available = bool(
        np.asarray(_field(record, "coefficients_after_capture_available")).reshape(-1)[0]
    )
    after_coefficients = _array(_field(record, "coefficients_after_prolongation_if_exposed"))
    assert after_available and after_coefficients.shape == (33, 3)
    assert bool(jnp.all(jnp.isfinite(after_coefficients)))
    domain = _vector(_field(context, "DOMAIN"))
    assert domain.shape == (2,) and bool(jnp.array_equal(domain, jnp.asarray([-1.0, 1.0])))

    # Native coefficients isolate the PDE/assembly path from initial Chebfun fitting.
    u0 = Chebfun.from_coeffs(coefficients, domain=(-1.0, 1.0))
    # Native pdeSolve truncates only onefun and retains original pointValues.
    # Materialize the original input's own endpoints before changing its tech.
    original_point_values = u0.point_values
    assert original_point_values.shape == (2, 3)
    assert bool(jnp.all(jnp.isfinite(original_point_values)))
    u0 = u0.set_point_values(original_point_values)
    prolonged_coefficients = u0.funs[0].tech.prolong(33).coeffs
    _source_close(prolonged_coefficients, after_coefficients)
    initial = _vector(_field(record, "initial"))
    n = int(_vector(_field(context, "n"))[0])
    assert n == 33 and initial.shape == (n * 3,)
    initial_matrix = _array(_field(record, "initial_matrix"))
    assert initial_matrix.shape == (n, 3)
    _source_close(initial, initial_matrix.reshape(-1, order="F"))
    x_native = _vector(_field(context, "x"))
    _source_close(original_point_values, initial_matrix[jnp.asarray([0, -1]), :])
    u0_fixed = eqx.tree_at(lambda f: f.funs[0].tech, u0, u0.funs[0].tech.prolong(33))
    _source_close(u0_fixed(x_native), initial_matrix)

    times = _vector(_field(record, "span"))
    requested = _vector(public["requested_times"])
    assert bool(jnp.array_equal(times, requested))
    assert bool(jnp.array_equal(times, jnp.asarray([0.0, 0.1, 0.2])))
    public_samples = public["samples"]
    assert public_samples.shape == (3, 3)
    for species in range(3):
        for time_index in range(3):
            cell = np.asarray(_unwrap(public_samples[species, time_index]))
            assert cell.shape == (3, 1) and np.isfinite(cell).all()
    B = _array(_field(context, "B"))
    P = _array(_field(context, "P"))
    M = _array(_field(context, "M"))
    q = _vector(_field(context, "q"))
    offset_native = _vector(_field(context, "BCVALOFFSET"))
    native_initial_y = _vector(actions["initial_y"])
    native_initial_action = _vector(actions["native_initial_action"])
    probe_y = _vector(actions["probe_y"])
    native_probe_action = _vector(actions["native_probe_action"])
    native_rhs = _vector(_field(record, "initial_rhs_before_primary"))
    assert B.shape == (6, 99) and P.shape == M.shape == (99, 99)
    assert q.shape == offset_native.shape == (6,) and native_rhs.shape == (99,)
    assert native_initial_y.shape == probe_y.shape == (99,)
    assert native_initial_action.shape == native_probe_action.shape == (6,)
    assert bool(jnp.array_equal(native_initial_y, initial))
    assert bool(jnp.array_equal(native_initial_action, offset_native))
    assert int(_vector(_field(context, "numBCs"))[0]) == 6
    assert int(_vector(_field(context, "numLinBCs"))[0]) == 6
    assert int(_vector(_field(context, "numLBCs"))[0]) == 0
    assert int(_vector(_field(context, "numRBCs"))[0]) == 0
    assert bool(jnp.array_equal(_vector(_field(record, "difforder")), jnp.asarray([2, 2, 2])))
    assert bool(jnp.all(q == 0))
    assert bool(jnp.array_equal(P, M))

    options = _field(record, "options")
    reltol = float(_vector(_field(options, "RelTol"))[0])
    abstol = float(_vector(_field(options, "AbsTol"))[0])
    assert reltol == abstol == 1e-6
    stream = _source_stream(_field(record, "rng_before_primary"))
    original = pde_driver.initialize_full_mass
    initializer_calls = []

    def inspect_native_assembly(fun, mass, y, *, reltol=1e-6, abstol=1e-6, uniform_stream=None):
        full, raw, projection, offset, scope, raw_scope = _assembly(fun)
        initializer_calls.append((mass.shape, y.shape))
        assert mass.shape == projection.shape == (99, 99)
        _source_close(mass, M)
        _source_close(projection, P)
        _source_close(y, initial)
        # Source BCVALOFFSET is raw(t0, y0)[:6] - q. Compare the native
        # offset on the identical native y0, then check the actual production
        # offset against its own unmodified y0. Sub-eps initial evaluation
        # differences are amplified by B; cross-input offset bit parity is
        # deliberately not inferred from the initial-state 100eps bound.
        _source_close(raw(float(times[0]), native_initial_y)[:6] - q, offset_native)
        assert bool(jnp.array_equal(offset, raw(float(times[0]), y)[:6] - q))
        # Operator comparisons use the captured state AND captured offset context.
        # Initial function evaluation is checked separately above; no reference
        # state or offset is substituted into the production initializer below.
        reference_context_rhs = raw(float(times[0]), native_initial_y).at[:6].add(-offset_native)
        _source_close(reference_context_rhs, native_rhs)
        assert bool(jnp.all(mass[:6] == 0))

        # Compare source boundary actions on the exact captured states. The probe
        # vector is loaded from the MATLAB capture; do not regenerate its polynomial
        # or substitute a generic matrix product as the native oracle.
        _source_close(raw(float(times[0]), native_initial_y)[:6], native_initial_action)
        _source_close(
            raw(float(times[0]), native_initial_y)[:6] - offset_native,
            native_initial_action - offset_native,
        )
        _source_close(raw(float(times[0]), probe_y)[:6], native_probe_action)
        _source_close(
            raw(float(times[0]), probe_y)[:6] - offset_native, native_probe_action - offset_native
        )
        # W3 exposes the source-equivalent linear endpoint arrays in raw's closure.
        _source_close(raw_scope["boundary_matrix"], B)
        _source_close(raw_scope["linear_rhs"], q)
        # No source time or state is modified; continue the production initializer.
        assert uniform_stream is None
        return original(fun, mass, y, reltol=reltol, abstol=abstol, uniform_stream=stream)

    monkeypatch.setattr(pde_driver, "initialize_full_mass", inspect_native_assembly)
    outputs = pde15s(
        _reaction,
        times,
        u0,
        bc="neumann",
        n=33,
        rtol=reltol,
        atol=abstol,
    )
    assert initializer_calls == [((99, 99), (99,))]
    assert bool(jnp.array_equal(times, _vector(_field(record, "primary_t"))))
    assert len(outputs) == 3
    probes = _vector(public["sample_x"])
    for time_index, output in enumerate(outputs):
        expected = jnp.column_stack(
            [_vector(public_samples[species, time_index]) for species in range(3)]
        )
        actual = output(probes)
        assert actual.shape == expected.shape == (3, 3)
        # Predeclared existing PDE forward bound; diagnose failures without widening.
        assert float(jnp.max(jnp.abs(actual - expected))) < FORWARD_BOUND

"""Native mixed-order assembly and independent manufactured controls."""

import hashlib
import inspect
import json
from pathlib import Path

import jax.numpy as jnp

# uses-numpy: decode native MATLAB fixtures and compare captured arrays.
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.chebfun1d._pde_ndf import pde_driver
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.pde15s import pde15s
from chebfunjax.utils.quadrature import chebpts_ab

FIXTURE = Path(__file__).parent / "references" / "pde_mixed_native"


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
    actual, expected = jnp.asarray(actual), jnp.asarray(expected)
    assert actual.shape == expected.shape
    bound = 100 * jnp.finfo(jnp.float64).eps * jnp.maximum(1.0, jnp.max(jnp.abs(expected)))
    assert float(jnp.max(jnp.abs(actual - expected))) <= float(bound)


def _initial():
    # w=3+x^2=3.5*T0+0.5*T2: direct independent exact coefficients.
    return Chebfun.from_coeffs(
        jnp.asarray([[1.0, 2.0, 3.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]), domain=(-1.0, 1.0)
    )


def _rhs(u, v, w):
    return (0.1 * u.diff(2), 0.2 * v.diff(2), u * v - w)


def _boundary(u, v, w):
    return (u.diff(), v.diff())


def _assembly(initializer_fun):
    """Read existing closure at initializer boundary; no production debug API.

    The initializer receives lambda y: fun(current_time,y). These explicit
    source-local names make this an intentional assembly regression contract.
    """
    full = inspect.getclosurevars(initializer_fun).nonlocals["fun"]
    scope = inspect.getclosurevars(full).nonlocals
    raw = scope["raw"]
    projection = inspect.getclosurevars(raw).nonlocals["projection"]
    return full, raw, projection, scope["offset"]


@pytest.fixture(scope="module")
def native():
    # Parent seals this only after successful, stable native receipt.
    hashes = json.loads((FIXTURE / "FIXTURE_HASHES.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    metadata = json.loads((FIXTURE / "metadata.json").read_text())
    assert metadata["complete"] and metadata["reference_pin"].startswith("7574c77")
    summary = json.loads((FIXTURE / "solver_summary.json").read_text())
    assert summary["instrumented_replay"] is False
    assert summary["orders"] == [2, 2, 0] and summary["boundary_counts"] == [2, 2]
    record = _unwrap(
        loadmat(
            FIXTURE / "solver_capture.mat", mat_dtype=True, squeeze_me=False, struct_as_record=False
        )["record"]
    )
    public = loadmat(
        FIXTURE / "public_output.mat", mat_dtype=True, squeeze_me=False, struct_as_record=False
    )
    return record, public


def test_native_mixed_projection_offset_rhs_and_public_samples(monkeypatch, native):
    record, public = native
    context = _field(record, "context")
    probe = _field(record, "probe")
    original = pde_driver.initialize_full_mass
    seen = []

    def observed(fun, mass, y, *, reltol=1e-6, abstol=1e-6, uniform_stream=None):
        full, raw, projection, offset = _assembly(fun)
        seen.append(mass.shape)
        assert mass.shape == projection.shape == (27, 27) and offset.shape == (4,)
        _source_close(mass, _array(_field(record, "mass")))
        _source_close(projection, _array(_field(context, "P")))
        _source_close(mass, _array(_field(context, "M")))
        _source_close(offset, _vector(_field(context, "BCVALOFFSET")))
        _source_close(y, _vector(_field(record, "initial")))
        _source_close(fun(y), _vector(_field(record, "initial_rhs_before_probes")))
        assert bool(jnp.all(mass[:4] == 0))
        assert not bool(jnp.array_equal(projection[18:27, 18:27], jnp.eye(9)))
        _source_close(projection[18:27, 18:27], _array(_field(context, "P"))[18:27, 18:27])
        state = _vector(_field(probe, "state"))
        time = float(_vector(_field(probe, "time"))[0])
        _source_close(full(time, state), _vector(_field(probe, "rhs")))
        _source_close(raw(time, state)[:4], _vector(_field(probe, "endpoint_residuals")))
        # This call continues the actual initializer and full NDF path unchanged.
        result = original(fun, mass, y, reltol=reltol, abstol=abstol, uniform_stream=uniform_stream)
        assert result["algebraic"].size == 4
        return result

    monkeypatch.setattr(pde_driver, "initialize_full_mass", observed)
    options = _field(record, "options")
    times = _vector(_field(record, "span"))
    outputs = pde15s(
        _rhs,
        times,
        _initial(),
        lbc=_boundary,
        rbc=_boundary,
        n=9,
        rtol=float(_vector(_field(options, "RelTol"))[0]),
        atol=float(_vector(_field(options, "AbsTol"))[0]),
    )
    assert seen == [(27, 27)] and len(outputs) == 3
    assert bool(jnp.array_equal(times, _vector(_field(record, "primary_t"))))
    samples = public["samples"]
    assert samples.shape == (3, 3)
    for j, output in enumerate(outputs):
        expected = jnp.column_stack([_vector(samples[k, j]) for k in range(3)])
        actual = output(_vector(public["sample_x"]))
        assert actual.shape == expected.shape == (3, 3)
        assert float(jnp.max(jnp.abs(actual - expected))) < 1e-7
    # Record native return exactly; expected times above come from solver/span.
    assert float(np.asarray(public["tout"]).reshape(-1)[0]) == float(times[0])


def test_mixed_nonconstant_third_species_manufactured():
    times = [0.0, 0.005, 0.01]
    outputs = pde15s(
        _rhs, times, _initial(), lbc=_boundary, rbc=_boundary, n=9, rtol=1e-9, atol=1e-11
    )
    x = jnp.asarray([-0.75, 0.0, 0.625])
    assert len(outputs) == len(times)
    for t, output in zip(times, outputs):
        expected = jnp.column_stack(
            (jnp.ones_like(x), 2 * jnp.ones_like(x), 2 + (1 + x * x) * jnp.exp(-t))
        )
        assert float(jnp.max(jnp.abs(output(x) - expected))) < 1e-7


class _AssemblyCaptured(Exception):
    pass


def test_reordered_coupled_boundary_rows(monkeypatch):
    seen = []

    def left(u, v, w):
        return (v.diff(), u.diff() + v.diff())

    def right(u, v, w):
        return (u.diff() - 2 * v.diff(), v.diff())

    def inspect_only(fun, mass, y, **kwargs):
        full, raw, projection, offset = _assembly(fun)
        seen.append(True)
        x = chebpts_ab(9, -1.0, 1.0)
        probe = jnp.column_stack(
            (1 + x + 2 * x * x, 2 - 3 * x + 2 * x * x + x**3, 3 + x * x - 2 * x**3)
        )
        expected = jnp.asarray([-4.0, -7.0, -3.0, 4.0])
        # Independent polynomial endpoint derivatives, not a duplicated diffmat oracle.
        _source_close(raw(0.0, probe.reshape(-1, order="F"))[:4], expected)
        _source_close(full(0.0, probe.reshape(-1, order="F"))[:4], expected - offset)
        assert mass.shape == projection.shape == (27, 27)
        raise _AssemblyCaptured

    monkeypatch.setattr(pde_driver, "initialize_full_mass", inspect_only)
    with pytest.raises(_AssemblyCaptured):
        pde15s(_rhs, [0.0, 0.005, 0.01], _initial(), lbc=left, rbc=right, n=9)
    assert seen == [True]


@pytest.mark.parametrize("left_count", [1, 3])
def test_wrong_total_boundary_count_rejected_before_initialization(monkeypatch, left_count):
    seen = []

    def left(u, v, w):
        return (u.diff(), v.diff(), w)[:left_count]

    def forbidden(*args, **kwargs):
        seen.append(True)
        raise AssertionError("Invalid shape reached initialization")

    monkeypatch.setattr(pde_driver, "initialize_full_mass", forbidden)
    with pytest.raises(ValueError):
        pde15s(_rhs, [0.0, 0.005, 0.01], _initial(), lbc=left, rbc=_boundary, n=9)
    assert not seen


def test_boundary_residual_count_change_is_rejected(monkeypatch):
    state = {"change": False, "reached_initializer": False}

    def left(u, v, w):
        return (u.diff(),) if state["change"] else (u.diff(), v.diff())

    def inspect_only(fun, mass, y, **kwargs):
        state["reached_initializer"] = True
        state["change"] = True
        fun(y)  # Must reject one residual where two were frozen at assembly.
        raise AssertionError("Changed residual count was accepted")

    monkeypatch.setattr(pde_driver, "initialize_full_mass", inspect_only)
    with pytest.raises(ValueError):
        pde15s(_rhs, [0.0, 0.005, 0.01], _initial(), lbc=left, rbc=_boundary, n=9)
    assert state["reached_initializer"]

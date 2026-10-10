"""Exact order/dtype controls for native ODE113 history assembly."""

import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

native = importlib.import_module("chebfunjax.utils.native_ode113")


def original_assembly(history):
    """The five original native_ode113.py assignments, without a solver copy."""
    mesh = jnp.stack([s.t for s in history])
    values = jnp.stack([s.y for s in history])
    orders = jnp.stack([s.klast for s in history])
    phis = jnp.stack([s.phi for s in history])
    psis = jnp.stack([s.psi for s in history])
    return mesh, values, orders, phis, psis


def assert_bits_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.weak_type == b.weak_type
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def history_fixture(count, complex_values=False, nonfinite=False):
    history = []
    for i in range(count):
        y = np.array([float(i), -float(i)])
        phi = np.arange(28, dtype=float).reshape(2, 14) + i
        if complex_values:
            y = y + 1j * y[::-1]
            phi = phi + 1j * (phi + 0.5)
        if nonfinite and i % 3 == 0:
            y[0] = np.nan
            phi[0, 0] = np.inf
            phi[1, 0] = -np.inf
        history.append(
            SimpleNamespace(
                t=jnp.asarray(float(i), dtype=jnp.float64),
                y=jnp.asarray(y),
                klast=jnp.asarray(i % 13, dtype=jnp.int32),
                phi=jnp.asarray(phi),
                psi=jnp.asarray(np.arange(12, dtype=float) - i),
            )
        )
    return history


@pytest.mark.parametrize("count", [1, 255, 256, 257, 513])
def test_actual_field_shapes_and_partial_blocks(count):
    history = history_fixture(count)
    actual = native._assemble_history(history)
    expected = original_assembly(history)
    assert [a.shape for a in actual] == [
        (count,),
        (count, 2),
        (count,),
        (count, 2, 14),
        (count, 12),
    ]
    for a, b in zip(actual, expected):
        assert_bits_equal(a, b)


def test_complex_nonfinite_and_signed_zero_entries():
    history = history_fixture(257, complex_values=True, nonfinite=True)
    for a, b in zip(native._assemble_history(history), original_assembly(history)):
        assert_bits_equal(a, b)
    zeros = [jnp.asarray(v, dtype=jnp.float64) for v in [-0.0, 0.0, -0.0, 0.0, -0.0]]
    assert_bits_equal(native._stack_history_entries(zeros, block_size=2), jnp.stack(zeros))


def test_empty_matches_original_stack_error():
    with pytest.raises(ValueError, match="Need at least one array to stack"):
        original_assembly([])
    with pytest.raises(ValueError, match="Need at least one array to stack"):
        native._assemble_history([])


def test_recursive_concatenation_has_bounded_fanin(monkeypatch):
    values = [jnp.asarray([i, -i], dtype=jnp.int32) for i in range(65)]
    expected = jnp.stack(values)
    stack, concat = native.jnp.stack, native.jnp.concatenate
    counts = []

    def observed_stack(entries, *args, **kwargs):
        counts.append(("stack", len(entries)))
        return stack(entries, *args, **kwargs)

    def observed_concat(entries, *args, **kwargs):
        counts.append(("concat", len(entries)))
        return concat(entries, *args, **kwargs)

    monkeypatch.setattr(native.jnp, "stack", observed_stack)
    monkeypatch.setattr(native.jnp, "concatenate", observed_concat)
    actual = native._stack_history_entries(values, block_size=4)
    assert_bits_equal(actual, expected)
    assert counts and all(n <= 4 for _, n in counts)
    assert sum(kind == "concat" for kind, _ in counts) > 1


def test_original_promotion_and_weak_type():
    mixed = [
        jnp.asarray(2**50 + 1, dtype=jnp.int64),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float64),
    ]
    assert_bits_equal(native._stack_history_entries(mixed, block_size=2), jnp.stack(mixed))
    weak = [jnp.asarray(float(i)) for i in range(5)]
    assert_bits_equal(native._stack_history_entries(weak, block_size=2), jnp.stack(weak))


@pytest.mark.parametrize("reverse,complex_values", [(False, False), (True, False), (False, True)])
def test_complete_provider_and_dense_output_exact(monkeypatch, reverse, complex_values):
    def rhs(t, y):
        return 0.25 * y + jnp.asarray([jnp.sin(t), jnp.cos(t)])

    span = (1.0, 0.0) if reverse else (0.0, 1.0)
    initial = (
        jnp.asarray([1.0 + 0.5j, -0.25 + 0.75j]) if complex_values else jnp.asarray([1.0, -0.25])
    )
    options = {"RelTol": 1e-9, "AbsTol": 1e-11}
    candidate = native.native_ode113(rhs, span, initial, options)
    monkeypatch.setattr(native, "_assemble_history", original_assembly)
    original = native.native_ode113(rhs, span, initial, options)
    assert candidate["stats"] == original["stats"]
    for key in ("x", "y", "xe", "ie"):
        assert_bits_equal(candidate[key], original[key])
    for key in candidate["idata"]:
        assert_bits_equal(candidate["idata"][key], original["idata"][key])
    for queries in (jnp.linspace(*span, 17), candidate["x"], jnp.empty(0)):
        got = candidate["sol"](queries)
        expected = original["sol"](queries)
        assert got.shape == (2, queries.size)
        assert_bits_equal(got, expected)
        for a, b in zip(
            candidate["sol"](queries, return_derivative=True),
            original["sol"](queries, return_derivative=True),
        ):
            assert a.shape == (2, queries.size)
            assert_bits_equal(a, b)

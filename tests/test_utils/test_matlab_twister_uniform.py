"""Exact JAX tests against captured MATLAB R2025b uniform streams."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils._matlab_twister import (
    init_matlab_twister,
    matlab_rand_uniform,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "matlab_twister_uniform_2025b" / "native_uniform_v3.npz"
NATIVE = np.load(FIXTURE)
SEEDS = (0, 6178, 16051821, 42)


def _assert_state(state, prefix):
    np.testing.assert_array_equal(np.asarray(state.words), NATIVE[f"{prefix}_words"])
    np.testing.assert_array_equal(np.asarray(state.index), NATIVE[f"{prefix}_index"])


def _draw(state, shape, expected_key):
    values, next_state = matlab_rand_uniform(state, shape)
    np.testing.assert_array_equal(np.asarray(values), NATIVE[expected_key])
    return next_state


def test_native_initial_seed_states_and_zero_mapping():
    for seed in SEEDS:
        state = init_matlab_twister(seed)
        _assert_state(state, f"seed{seed}_initial")


def test_native_multicycle_chunk_and_monolithic_streams_and_states():
    for seed in SEEDS:
        prefix = f"seed{seed}_"
        state = init_matlab_twister(seed)
        _assert_state(state, prefix + "initial")
        chunks = []
        for shape, name in (
            ((312, 1), "chunk_312_a"),
            ((1, 1), "chunk_1"),
            ((312, 1), "chunk_312_b"),
            ((312, 1), "chunk_312_c"),
        ):
            values, state = matlab_rand_uniform(state, shape)
            np.testing.assert_array_equal(np.asarray(values).reshape(-1), NATIVE[prefix + name])
            chunks.append(np.asarray(values).reshape(-1))
            native_state_name = {
                "chunk_312_a": "state_after_312_a",
                "chunk_1": "state_after_1",
                "chunk_312_b": "state_after_312_b",
                "chunk_312_c": "state_after_312_c",
            }[name]
            _assert_state(state, prefix + native_state_name)
        np.testing.assert_array_equal(np.concatenate(chunks), NATIVE[prefix + "chunked"])
        np.testing.assert_array_equal(np.concatenate(chunks), NATIVE[prefix + "monolithic_937"])
        _assert_state(state, prefix + "chunked_final")

        monolithic, mono_state = matlab_rand_uniform(init_matlab_twister(seed), (937, 1))
        np.testing.assert_array_equal(np.asarray(monolithic).reshape(-1), NATIVE[prefix + "monolithic_937"])
        _assert_state(mono_state, prefix + "monolithic_final")


def test_native_boundary_crossing_shapes_empty_and_following_scalar():
    for seed in SEEDS:
        prefix = f"seed{seed}_mixed_"
        state = init_matlab_twister(seed)
        _assert_state(state, prefix + "initial")
        state = _draw(state, (311, 1), prefix + "before_boundary")
        _assert_state(state, prefix + "state_after_311")
        state = _draw(state, (2, 2), prefix + "crossing_matrix")
        _assert_state(state, prefix + "state_after_crossing_matrix")
        empty, state_after_empty = matlab_rand_uniform(state, (0, 3))
        assert empty.shape == (0, 3)
        assert empty.dtype == jnp.float64
        _assert_state(state_after_empty, prefix + "state_after_empty")
        # The empty call returns the same immutable state value.
        np.testing.assert_array_equal(np.asarray(state.words), np.asarray(state_after_empty.words))
        state = _draw(state_after_empty, (1, 1), prefix + "scalar_after_empty")
        _assert_state(state, prefix + "state_after_scalar")


def test_native_seed42_feval_draw_order_and_f_order_tensors():
    state = init_matlab_twister(42)
    _assert_state(state, "feval42_initial")

    uniforms, state = matlab_rand_uniform(state, (3, 1))
    # Native test_feval uses pts = 2*rand(3,1)-1; compare this actual input too.
    np.testing.assert_array_equal(2.0 * np.asarray(uniforms) - 1.0, NATIVE["feval42_pts"])
    _assert_state(state, "feval42_state_after_pts")

    for name in ("r10", "s10", "t10"):
        state = _draw(state, (10, 1), "feval42_" + name)
    _assert_state(state, "feval42_state_after_rst")

    for group in (1, 2):
        for item in (1, 2, 3):
            state = _draw(
                state,
                (100, 1),
                f"feval42_vec100_group{group}_{item}",
            )
        _assert_state(state, f"feval42_state_after_vec100_group{group}")

    for group in (1, 2):
        for item in (1, 2, 3):
            state = _draw(
                state,
                (10, 20, 30),
                f"feval42_tensor_group{group}_{item}",
            )
        _assert_state(state, f"feval42_state_after_tensor_group{group}")
    _assert_state(state, "feval42_final")


def test_jit_matches_native_words_values_and_empty_state():
    compiled = jax.jit(matlab_rand_uniform, static_argnames=("shape",))
    for seed in SEEDS:
        prefix = f"seed{seed}_"
        state = init_matlab_twister(seed)
        values, next_state = compiled(state, (937, 1))
        np.testing.assert_array_equal(np.asarray(values).reshape(-1), NATIVE[prefix + "monolithic_937"])
        _assert_state(next_state, prefix + "monolithic_final")

        matrix, matrix_state = compiled(state, (2, 3))
        expected_stream, _ = matlab_rand_uniform(state, (6, 1))
        np.testing.assert_array_equal(np.asarray(matrix).reshape(-1, order="F"), np.asarray(expected_stream).reshape(-1))
        empty, empty_state = compiled(matrix_state, (0, 3))
        assert empty.shape == (0, 3)
        np.testing.assert_array_equal(np.asarray(empty_state.words), np.asarray(matrix_state.words))
        np.testing.assert_array_equal(np.asarray(empty_state.index), np.asarray(matrix_state.index))


def test_shape_validation_is_static_and_seed_range_is_checked():
    for bad in (-1, 2**32, True, 1.0):
        try:
            init_matlab_twister(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid seed {bad!r}")
    state = init_matlab_twister(7)
    for bad_shape in ((-1, 2), [2, 3], (2, 3.0)):
        try:
            matlab_rand_uniform(state, bad_shape)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid shape {bad_shape!r}")

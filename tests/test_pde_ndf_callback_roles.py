"""Native replay for role-specific single independent callback inputs."""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp

# uses-numpy: decode native MATLAB fixtures and compare callback observations.
import numpy as np
from scipy.io import loadmat

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.pde15s import pde15s

FIXTURE = Path(__file__).parent / "references" / "pde_callback_arity_native"
BOUND = 1e-7  # Existing three-component manufactured/native PDE comparison policy.


def _unwrap(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.reshape(-1, order="F")[0]
    return value


def _native_output():
    hashes = json.loads((FIXTURE / "FIXTURE_HASHES.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    metadata = json.loads((FIXTURE / "metadata.json").read_text())
    assert metadata["complete"]
    assert metadata["fixed_N"] == 9
    assert len(metadata["pde_coordinate_values"]) == 9
    assert np.asarray(metadata["pde_coordinate_domain"]).shape == (2,)
    assert np.allclose(metadata["pde_coordinate_domain"], [-1.0, 1.0], rtol=0.0, atol=0.0)
    observed_times = np.asarray(metadata["boundary_time_values"], dtype=float)
    assert observed_times.size and np.all(np.isfinite(observed_times))
    assert np.min(observed_times) == 0.0
    assert np.max(observed_times) <= metadata["times"][-1]
    saved = loadmat(
        FIXTURE / "public_output.mat", squeeze_me=False, struct_as_record=False, mat_dtype=True
    )
    assert saved["samples"].shape == (3, 3)
    for sample in saved["samples"].flat:
        values = np.asarray(_unwrap(sample), dtype=np.float64)
        assert values.shape == (3, 1) and np.all(np.isfinite(values))
    np.testing.assert_array_equal(saved["tIn"], np.asarray(metadata["times"])[None, :])
    return metadata, saved


def test_pde_and_boundary_N_plus_1_signatures_use_their_source_roles():
    metadata, saved = _native_output()
    times = jnp.asarray(metadata["times"], dtype=jnp.float64)
    probe = jnp.asarray(metadata["probe"], dtype=jnp.float64)
    initial = Chebfun.from_coeffs(jnp.asarray([[1.0, 2.0, 4.0]]), domain=(-1.0, 1.0))
    pde_roles = []
    pde_coordinate_observations = []
    boundary_roles = []

    # MATLAB pdeSolve parseFun(pdeFun) maps the one independent input to x.
    def pde_x_u_v_w(x, u, v, w):
        has_coordinate_value = hasattr(x, "vals") and hasattr(x, "context")
        pde_roles.append(has_coordinate_value)
        if has_coordinate_value and len(pde_coordinate_observations) == 0:
            pde_coordinate_observations.append(np.asarray(x.vals, dtype=np.float64).reshape(-1))
        return (
            0.1 * u.diff(2) - u + v + x,
            0.2 * v.diff(2) + u - 2 * v + w + 2 * x,
            0.05 * w.diff(2) + v - w - x,
        )

    # MATLAB parseFun(bc,'lbc'/'rbc') maps one independent input to t.
    def bc_t_u_v_w(t, u, v, w):
        rank = jnp.ndim(t)
        boundary_roles.append(rank)
        return (u.diff() - t, v.diff() - 2 * t, w.diff() + t)

    outputs = pde15s(
        pde_x_u_v_w,
        times,
        initial,
        lbc=bc_t_u_v_w,
        rbc=bc_t_u_v_w,
        n=9,
        rtol=float(metadata["options"]["RelTol"]),
        atol=float(metadata["options"]["AbsTol"]),
    )
    assert len(outputs) == times.size
    assert pde_roles and all(pde_roles)
    assert boundary_roles and all(rank == 0 for rank in boundary_roles)
    assert len(pde_coordinate_observations) == 1
    native_x = np.asarray(metadata["pde_coordinate_values"], dtype=np.float64).reshape(-1)
    actual_x = pde_coordinate_observations[0]
    assert actual_x.shape == native_x.shape == (9,)
    coord_bound = 100 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(native_x))))
    assert float(np.max(np.abs(actual_x - native_x))) <= coord_bound

    native_samples = np.asarray(saved["samples"], dtype=object)
    assert native_samples.shape == (3, times.size)
    for time_index, output in enumerate(outputs):
        expected = np.column_stack(
            [
                np.asarray(_unwrap(native_samples[species, time_index]), dtype=np.float64).reshape(
                    -1
                )
                for species in range(3)
            ]
        )
        actual = np.asarray(output(probe), dtype=np.float64)
        assert actual.shape == expected.shape == (3, 3)
        assert float(np.max(np.abs(actual - expected))) < BOUND

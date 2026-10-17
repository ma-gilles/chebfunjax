"""Captured native numjac perturbations and actual quotient schedule.

Provenance
----------
MATLAB R2025b private/odenumjac.m; Chebfun pdeSolve.m.
Chebfun commit: 7574c77
"""

import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d._pde_ndf.dae_init import dense_numjac

FIXTURE = Path(__file__).parent / "references/pde_numjac_native/chemical_first_numjac.npz"
FIXTURE_SHA = "b57ddbb1d82a8c3aea3cf15c3cdccda806ba9e233baeb413485d558c26ab4b10"


def _bits(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint64), np.asarray(expected).view(np.uint64)
    )


def test_native_first_numjac_perturbation_and_quotient():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    with np.load(FIXTURE, allow_pickle=False) as data:
        y, f, thresh = (jnp.asarray(data[key]) for key in ("y", "f", "thresh"))
        states = data["perturbed_states"]
        fdel = data["fdel"]
        index = 0

        def callback(z):
            nonlocal index
            assert index < states.shape[1], "Unexpected numjac refinement"
            _bits(z, states[:, index])
            value = jnp.asarray(fdel[:, index])
            index += 1
            return value

        jac, fac, calls = dense_numjac(callback, y, f, thresh)
        assert index == len(calls) == 303
        _bits(jac, data["jac"])
        _bits(fac, data["fac"])
        for column, state in enumerate(calls):
            _bits(state, states[:, column])


@pytest.mark.parametrize("case", [0, 1], ids=["scalar", "three_components"])
def test_native_refined_numjac_callback_tape(case):
    fixture = FIXTURE.with_name("refinements.npz")
    assert (
        hashlib.sha256(fixture.read_bytes()).hexdigest()
        == "e951679c41ae5ec2c1f48290895300bf58505d0bff0baf8939c700fdf2acf64e"
    )
    with np.load(fixture, allow_pickle=False) as data:
        prefix = f"case{case}_"
        y = jnp.asarray(data[prefix + "y"])
        f = jnp.asarray(data[prefix + "f"])
        states = data[prefix + "states"]
        outputs = data[prefix + "outputs"]
        index = 0

        def callback(z):
            nonlocal index
            assert index < states.shape[1]
            _bits(z, states[:, index])
            value = jnp.asarray(outputs[:, index])
            index += 1
            return value

        jac, fac, calls = dense_numjac(callback, y, f, jnp.asarray(1e-6))
        assert index == len(calls) == states.shape[1] > y.size
        _bits(jac, data[prefix + "jac"])
        _bits(fac, data[prefix + "fac"])
        for column, state in enumerate(calls):
            _bits(state, states[:, column])

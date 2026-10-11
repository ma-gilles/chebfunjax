"""Independent native delta integration piece captures, Chebfun7574c77."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun


@pytest.mark.parametrize("case", [0, 1, 2])
def test_native_shared_boundary_and_complex_carry(case):
    data = json.loads(
        (Path(__file__).parents[2] / "references/delta_cumsum_native.json").read_text()
    )["cases"][case]
    ops = [lambda x: 0 * x, lambda x: 1j * x, jnp.exp]
    mags = [[[2.0], [6.0]], [[2.0], [6.0]], [[1.0, 2.0], [4.0, 8.0]]]
    locs = [[0.0], [0.0], [-0.25, 0.5]]
    f = Bndfun.from_function(ops[case], Domain((-1.0, 1.0)))
    result = Deltafun(f, jnp.asarray(locs[case]), jnp.asarray(mags[case])).cumsum()
    assert isinstance(result, list) and len(result) == len(data["pieces"])
    for actual, expected in zip(result, data["pieces"]):
        assert isinstance(actual, Deltafun)
        np.testing.assert_array_equal(
            np.asarray(actual.delta_locs).reshape(-1), np.asarray(expected["loc"]).reshape(-1)
        )
        np.testing.assert_array_equal(
            np.asarray(actual.delta_mags).reshape(-1), np.asarray(expected["mag"]).reshape(-1)
        )
        truth = np.asarray(expected["real"]) + 1j * np.asarray(expected["imag"])
        error = np.max(np.abs(np.asarray(actual.funPart(jnp.asarray(expected["points"]))) - truth))
        assert error < 1e-9


def zero():
    return Bndfun.from_function(lambda x: 0 * x, Domain((-1.0, 1.0)))


def test_residual_columns_use_native_threshold():
    d = Deltafun(zero(), jnp.asarray([0.0, 0.5]), jnp.asarray([[2.0, 3.0], [1e-12, 4.0]]))
    pieces = d.cumsum()
    assert len(pieces) == 3
    assert isinstance(pieces[0], Bndfun)
    for part in pieces[1:]:
        np.testing.assert_array_equal(part.delta_locs, jnp.asarray([0.5]))
        np.testing.assert_array_equal(part.delta_mags, jnp.asarray([[2.0]]))
    np.testing.assert_array_equal(
        jnp.asarray(
            [
                pieces[0](jnp.asarray(-0.5)),
                pieces[1].funPart(jnp.asarray(0.25)),
                pieces[2].funPart(jnp.asarray(0.75)),
            ]
        ),
        jnp.asarray([0.0, 2.0, 5.0]),
    )


def test_derivative_only_remains_deltafun():
    result = Deltafun(zero(), jnp.asarray([0.0]), jnp.asarray([[0.0], [6.0]])).cumsum()
    assert isinstance(result, Deltafun)
    np.testing.assert_array_equal(result.delta_mags, jnp.asarray([[6.0]]))


def test_endpoint_only_uses_left_jump_and_keeps_derivatives():
    result = Deltafun(
        zero(), jnp.asarray([-1.0, 1.0]), jnp.asarray([[2.0, 3.0], [4.0, 6.0]])
    ).cumsum()
    assert isinstance(result, Deltafun)
    np.testing.assert_array_equal(result.delta_mags, jnp.asarray([[4.0, 6.0]]))
    np.testing.assert_array_equal(
        result.funPart(jnp.asarray([-1.0, 0.0, 1.0])), jnp.asarray([2.0, 2.0, 2.0])
    )

"""Source exact grid recognition, including invalid and complex coordinates.

MATLAB provenance: @chebfun3/feval.m, tensorCase recognition.
Chebfun commit: 7574c77.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d._power import _grid_vector


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("case", ["grid", "perturbed", "nan", "inf"])
def test_exact_grid_predicate_and_coordinate_preservation(axis, complex_values, case):
    shape = (3, 4, 5)
    vector = np.linspace(-1., 1., shape[axis])
    if complex_values:
        vector = vector + 1j * vector[::-1]
    view = tuple(shape[axis] if k == axis else 1 for k in range(3))
    values = np.broadcast_to(vector.reshape(view), shape).copy()
    # This corner is outside the extracted coordinate line on every axis.
    if case == "perturbed":
        values[-1, -1, -1] += .125
    elif case == "nan":
        values[-1, -1, -1] = np.nan
    elif case == "inf":
        values[-1, -1, -1] = np.inf
    actual, exact = _grid_vector(jnp.asarray(values), axis)
    assert bool(exact) == (case == "grid")
    assert np.asarray(actual).tobytes() == vector.tobytes()


def test_constant_signed_zero_grid_preserves_coordinate_bits():
    values = np.full((1, 3, 1), -0.)
    actual, exact = _grid_vector(jnp.asarray(values), 1)
    assert bool(exact)
    assert np.signbit(np.asarray(actual)).all()

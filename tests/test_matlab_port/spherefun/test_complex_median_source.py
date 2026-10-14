"""Native complex median controls; Chebfun commit 7574c77, MATLAB R2025b.

These are supplemental source controls, not original MATLAB test predicates.
Complex scalar multiplication creates a complex-valued representation even
though the callable constructor discards imaginary input values.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._median import _odd_grid_median
from chebfunjax.spherefun.spherefun import Spherefun


def _fixture():
    path = Path(__file__).parents[2] / "fixtures/spherefun_complex_median_native_20261014.json"
    return json.loads(path.read_text())


def _complex(data):
    return np.asarray(data["real"]) + 1j*np.asarray(data["imag"])


@pytest.mark.parametrize("axis", [0, 1])
def test_captured_numeric_complex_ordering(axis):
    data = _fixture()
    actual = _odd_grid_median(jnp.asarray(_complex(data["numeric_input"])), axis)
    np.testing.assert_array_equal(actual, _complex(data[f"numeric_dim{axis+1}"]))


@pytest.mark.parametrize("dim", [1, 2])
def test_native_arithmetic_created_complex_spherefun(dim):
    case = next(c for c in _fixture()["cases"] if c["dim"] == dim)
    f = (1+2j)*Spherefun.from_function(lambda lam, th: jnp.cos(th)+0.3)
    g = f.median(dim)
    expected = _complex(case["values"])
    bound = 1000*np.finfo(float).eps*max(1., np.max(np.abs(expected)))
    np.testing.assert_array_equal(g.domain.breakpoints, case["domain"])
    assert len(g) == case["length"]
    assert g.n_columns == 1 and not g.is_transposed
    actual = np.asarray(g(jnp.asarray(case["points"])))
    assert np.max(np.abs(actual-expected)) < bound

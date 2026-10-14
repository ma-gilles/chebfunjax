"""Supplemental native controls for @separableApprox/median.m (7574c77).

The source has no dedicated Spherefun median test. These six R2025b cases
check the fixed sample grid, numeric-value constructor, domains and lengths.
The bound was selected before comparing implementations; it is not an
original MATLAB assertion. See docs/sphere_median_source_cpu_20261014.json.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun


@pytest.mark.parametrize("name", ["constant", "z", "x_plus_2z"])
def test_native_fixed_grid_median(name):
    callbacks = {
        "constant": lambda l, t: 1. + 0*l + 0*t,
        "z": lambda l, t: jnp.cos(t),
        "x_plus_2z": lambda l, t: jnp.cos(l)*jnp.sin(t) + 2*jnp.cos(t),
    }
    fixture = Path(__file__).parents[2] / "fixtures/spherefun_median_native_20261014.json"
    cases = json.loads(fixture.read_text())["cases"]
    f = Spherefun.from_function(callbacks[name])
    for case in (c for c in cases if c["name"] == name):
        g = f.median(int(case["dim"]))
        expected = np.asarray(case["values"])
        bound = 1000*np.finfo(float).eps*max(1., np.max(np.abs(expected)))
        np.testing.assert_array_equal(g.domain.breakpoints, case["domain"])
        assert len(g) == case["length"]
        assert g.n_columns == 1
        assert not g.is_transposed
        assert np.max(np.abs(np.asarray(g(jnp.asarray(case["points"]))) - expected)) < bound
    del f, g
    jax.clear_caches()


def test_empty_output_arity():
    f = Spherefun.empty()
    assert f.median(nargout=0) is None
    with pytest.raises(ValueError, match="unassignedOutputs"):
        f.median()
    with pytest.raises(ValueError, match="TooManyOutputs"):
        f.median(nargout=2)

"""Native empty GE remains empty and the public strike access raises.

Pinned @diskfun/constructor.m 7574c77. No captured tolerance is injected.
"""
import json
import math
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun.diskfun import Diskfun, _get_tol, _phase_one_disk, diskfun

RAW = json.loads((Path(__file__).parents[1] / "fixtures/disk_remaining_source_native_20261018.json").read_text())


def array(record):
    if math.prod(record["shape"]) == 0:
        return np.empty(record["shape"])
    return np.asarray([struct.unpack(">d", bytes.fromhex(w))[0]
                       for w in record["re_hex"]]).reshape(record["shape"], order="F")


def operation(name):
    if name == "numeric_scalar_eps1":
        return jnp.asarray([[1.]])
    if name.startswith("odd"):
        return lambda theta, radius: radius*jnp.cos(theta)
    if name.startswith("even"):
        return lambda theta, radius: radius**2
    if name.startswith("constant"):
        return lambda theta, radius: 1+0*theta
    return lambda theta, radius: 0*theta


@pytest.mark.parametrize("case", RAW["public"], ids=lambda c: c["name"])
def test_native_public_empty_pivot_and_unaffected_routes(case):
    op = operation(case["name"])
    level = float(array(case["level"]).reshape(()))
    if "error" in case:
        assert math.prod(case["phase"]["pivot_array"]["shape"]) == 0
        with pytest.raises(IndexError) as caught:
            diskfun(op, "polar", "eps", level)
        assert caught.value.identifier == case["error"]["identifier"]
        assert str(caught.value).removeprefix(caught.value.identifier + ": ") == case["error"]["message"]
    else:
        g = diskfun(op, "polar", "eps", level)
        native = case["actual"]
        assert len(g) == native["rank"]
        assert tuple(g.idx_plus) == tuple(array(native["idx_plus"]).astype(int).reshape(-1)-1)
        assert tuple(g.idx_minus) == tuple(array(native["idx_minus"]).astype(int).reshape(-1)-1)
        assert g.nonzero_poles == native["nonzero_poles"]
        actual = g(jnp.asarray([-math.pi, -.7, 0., math.pi]),
                   jnp.asarray([0., .25, .5, 1.]))
        # Existing pinned original constructor predicates use 2e3*eps for
        # basic polar/cartesian construction; no original bound is changed.
        np.testing.assert_allclose(np.asarray(actual).reshape(-1),
                                   array(native["values"]).reshape(-1),
                                   rtol=0, atol=2e3*np.finfo(float).eps)


@pytest.mark.parametrize("case", [c for c in RAW["public"] if "error" in c],
                         ids=lambda c: c["name"])
def test_no_pivot_uses_actual_production_tolerance(case):
    values = array(case["initial_values"])
    # Literal native adaptive tolerance call. No native tol injection.
    tol = _get_tol(values, 2*np.pi/8, 1/8,
                   float(array(case["level"]).reshape(())))[0]
    indices, pivots, pole, happy = _phase_one_disk(values, tol, 100., 8.)
    assert indices.shape == pivots.shape == (0, 2)
    assert not pole and happy


def test_source_exact_zero_placeholder_remains_distinct():
    indices, pivots, pole, happy = _phase_one_disk(np.zeros((3, 4)), 10., 100., 8.)
    np.testing.assert_array_equal(indices, [[0, 0]])
    np.testing.assert_array_equal(pivots, [[0., 0.]])
    assert not pole and happy


def test_source_nonzero_below_tolerance_has_no_invented_zero():
    values = np.asarray([[0., 0., 0., 0.], [1., 2., 3., 4.]])
    indices, pivots, pole, happy = _phase_one_disk(values, 10., 100., 8.)
    assert indices.size == pivots.size == 0
    assert not pole and happy


def test_source_empty_strike_still_errors_after_rank_warning():
    # Native sets failure at MAXRANK, then still indexes pivotArray(1,:).
    with pytest.warns(RuntimeWarning, match="high rank"):
        with pytest.raises(IndexError, match="MATLAB:badsubscript"):
            Diskfun.from_function(lambda theta, radius: radius**2,
                                  tol=1., max_rank=1)

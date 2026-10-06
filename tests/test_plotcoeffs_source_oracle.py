"""Independent literal-polynomial and captured MATLAB plotcoeffs controls.

Provenance
----------
MATLAB source : @chebfun/plotcoeffs.m, @chebtech/plotcoeffs.m,
    @trigtech/plotcoeffs.m
Chebfun commit: 7574c77

Inputs below are explicit original coefficient operands or analytically known
coefficients of x and x**2. No oracle output supplies function coefficients.
Only line data/source labels/markers/grid/scales are compared. Backend auto
limits and pixel equivalence are separate qualification.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np  # uses-numpy: assertion-only comparison of renderer host data
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain

_FIXTURE = Path(__file__).with_name("plotcoeffs_source_oracle_2025b.json")
if not _FIXTURE.exists():
    _FIXTURE = Path(__file__).parent / "fixtures" / _FIXTURE.name
_ORACLE = json.loads(_FIXTURE.read_text())


def _operand(name):
    if name.startswith("trig_"):
        odd = "odd" in name
        coeffs = jnp.asarray([1., 2., 3., 4., 5.] if odd else [1., 2., 3., 4.])
        f = cj.chebfun(coeffs, coeffs=True, trig=True,
                       domain=(2., 10.) if odd else (0., 2. * jnp.pi))
        return f, {"loglog": "loglog" in name}
    if name == "zero_cheb":
        return Chebfun.from_coeffs(jnp.asarray([0.])), {}
    if name in ("zero_trig", "constant_trig"):
        c = 0. if name == "zero_trig" else 2.
        return cj.chebfun(jnp.asarray([c]), coeffs=True, trig=True,
                          domain=(-3., 5.)), {}
    if name == "piecewise":
        p = Chebfun.from_coeffs(jnp.asarray([-.5, .5]), (-1., 0.))
        q = Chebfun.from_coeffs(jnp.asarray([.375, .5, .125]), (0., 1.))
        return Chebfun(funs=p.funs + q.funs, domain=Domain((-1., 0., 1.))), {}
    if name in ("columns", "quasimatrix"):
        # x = T1, x**2 = (T0+T2)/2. MATLAB hcat uses common degree padding.
        coeffs = jnp.asarray([[0., .5], [1., 0.], [0., .5]])
        if name == "columns":
            return Chebfun.from_coeffs(coeffs), {}
        cols = [Chebfun.from_coeffs(coeffs[:, j]) for j in range(2)]
        return Quasimatrix(cols, Domain((-1., 1.))), {}
    f = Chebfun.from_coeffs(jnp.asarray([1., .5, .125]),
                          (-2., 3.) if name == "cheb_coeffs" else (-1., 1.))
    if name == "style_override":
        return f, {"fmt": ".--", "markersize": 7, "color": (.1, .2, .3)}
    assert name == "cheb_coeffs"
    return f, {}


def _numeric(values):
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


@pytest.mark.parametrize("case", _ORACLE["cases"], ids=lambda c: c["name"])
def test_fresh_matlab_source_plotcoeffs_line_properties(case):
    f, kw = _operand(case["name"])
    fig, ax = cj.plotcoeffs(f, source=True, **kw)
    try:
        assert len(ax.lines) == len(case["lines"])
        for line, expected in zip(ax.lines, case["lines"], strict=True):
            # Source wave-number scaling and source marker formula use at most
            # four binary64 operations. 8eps bounds rounding across JAX/XLA
            # and MATLAB; exact coefficient magnitudes are required below.
            np.testing.assert_allclose(line.get_xdata(), _numeric(expected["x"]),
                                       rtol=8 * np.finfo(float).eps, atol=0.,
                                       equal_nan=True)
            np.testing.assert_array_equal(line.get_ydata(), _numeric(expected["y"]))
            assert line.get_marker() == expected["marker"]
            assert line.get_markersize() == pytest.approx(
                expected["markersize"], rel=8 * np.finfo(float).eps, abs=0.)
            assert line.get_linestyle().lower() == expected["linestyle"].lower()
            if case["name"] == "style_override":
                np.testing.assert_array_equal(mcolors.to_rgb(line.get_color()),
                                              expected["color"])
        assert ax.get_title() == case["title"]
        assert ax.get_xlabel() == case["xlabel"]
        assert ax.get_ylabel() == case["ylabel"]
        assert ax.get_xscale() == case["xscale"]
        assert ax.get_yscale() == case["yscale"]
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert any(line.get_visible() for line in ax.get_ygridlines())
    finally:
        plt.close(fig)
    # The source explicit limits and MATLAB tight-x adapter are independently
    # captured for these literal cases. Y autoscaling and pixels are excluded.
    np.testing.assert_allclose(ax.get_xlim(), case["xlim"],
                               rtol=8 * np.finfo(float).eps, atol=0.0)

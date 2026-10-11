"""All20 predicates from MATLAB tests/misc/test_coeffs2vals.m, plus2 supplements.

Source bound1000eps/vector infinity norm, adaptive exp length and all five
coefficient/value representations are retained. Existing random roundtrip and
single-mode controls remain supplemental. Python kind keyword arguments and
1D vectors adapt the native positional kind and scalar column conventions.

Provenance
----------
MATLAB source : tests/misc/test_coeffs2vals.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.chebtech import _coeffs_to_values, _values_to_coeffs
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)


class TestCoeffs2vals:
    def test_roundtrip(self):
        rng = np.random.default_rng(1)
        c = jnp.asarray(rng.standard_normal(17))
        back = _values_to_coeffs(_coeffs_to_values(c))
        assert float(jnp.max(jnp.abs(back - c))) < 100 * EPS

    def test_single_mode_is_chebyshev_poly(self):
        n = 12
        x = np.asarray(chebpts(n, kind=2))
        for k in [0, 3, 7]:
            c = jnp.zeros(n, dtype=jnp.float64).at[k].set(1.0)
            v = np.asarray(_coeffs_to_values(c))
            exact = np.cos(k * np.arccos(np.clip(x, -1, 1)))
            assert float(np.max(np.abs(v - exact))) < 100 * EPS


# Original source predicates use exp, its adaptive Chebfun length, and the
# coefficients/grid values of that same f, rather than random roundtrips.
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils import transforms as tr
from chebfunjax.utils.quadrature import legpts


@pytest.fixture(scope="module")
def source_exp_data():
    f = chebfun(jnp.exp)
    n = len(f)
    return (
        n,
        f.legcoeffs(),
        f.chebcoeffs(),
        f(legpts(n)[0]),
        f(chebpts(n, kind=1)),
        f(chebpts(n, kind=2)),
    )


@pytest.mark.parametrize("clause", range(1, 21))
def test_source_exp_conversion(clause, source_exp_data):
    n, cl, cc, vl, v1, v2 = source_exp_data
    if clause == 1:
        expected, got = cl, tr.chebcoeffs2legcoeffs(cc)
    elif clause == 2:
        expected, got = cl, tr.legvals2legcoeffs(vl)
    elif clause == 3:
        expected, got = cl, tr.chebvals2legcoeffs(v1, kind=1)
    elif clause == 4:
        expected, got = cl, tr.chebvals2legcoeffs(v2, kind=2)
    elif clause == 5:
        expected, got = cc, tr.legcoeffs2chebcoeffs(cl)
    elif clause == 6:
        expected, got = cc, tr.legvals2chebcoeffs(vl)
    elif clause == 7:
        expected, got = cc, tr.chebvals2chebcoeffs(v1, kind=1)
    elif clause == 8:
        expected, got = cc, tr.chebvals2chebcoeffs(v2, kind=2)
    elif clause == 9:
        expected, got = vl, tr.legcoeffs2legvals(cl)
    elif clause == 10:
        expected, got = vl, tr.chebcoeffs2legvals(cc)
    elif clause == 11:
        expected, got = vl, tr.chebvals2legvals(v1, kind=1)
    elif clause == 12:
        expected, got = vl, tr.chebvals2legvals(v2, kind=2)
    elif clause == 13:
        expected, got = v1, tr.legcoeffs2chebvals(cl, kind=1)
    elif clause == 14:
        expected, got = v1, tr.chebcoeffs2chebvals(cc, kind=1)
    elif clause == 15:
        expected, got = v1, tr.legvals2chebvals(vl, kind=1)
    elif clause == 16:
        expected, got = v1, tr.chebvals2chebvals(v2, 2, 1)
    elif clause == 17:
        expected, got = v2, tr.legcoeffs2chebvals(cl, kind=2)
    elif clause == 18:
        expected, got = v2, tr.chebcoeffs2chebvals(cc, kind=2)
    elif clause == 19:
        expected, got = v2, tr.legvals2chebvals(vl, kind=2)
    else:
        expected, got = v2, tr.chebvals2chebvals(v1, 1, 2)
    # MATLAB scalar data columns are Python vectors; explicit shape prevents
    # broadcasting from silently comparing a matrix to a source column.
    assert expected.shape == got.shape == (n,)
    assert np.linalg.norm(np.asarray(expected - got), ord=np.inf) < 1000 * EPS

"""Explicit lower-order RH/RHW with unchanged inherited RH envelopes.

Reference: scalar interpretation of pinned lagpts.m/besselroots.m with SciPy
special functions; NOT fresh MATLAB output. Source failures remain failures.
Node8e-10, weight4e-9 and moment2e-9 envelopes are inherited from the already
qualified general-alpha RH tests. No missing source terms or GW substitution.
"""

import json
import math
from pathlib import Path

import jax
import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_genlaguerre, gammaln

from chebfunjax.utils.laguerre_rh_general import (
    _laguerre_rh_general,
    _rh_initial_guesses_general,
)
from chebfunjax.utils.quadrature import lagpts

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/laguerre_rh_small_source_reference.json').read_text())


@pytest.mark.parametrize('alpha', [0.0, -0.5, 0.5, 0.3])
@pytest.mark.parametrize('n', [42, 100, 300, 1000])
def test_explicit_source_rule_or_source_convergence_failure(n, alpha):
    outputs = {}
    for method in ['RH', 'RHW']:
        reference = next(row for row in FIXTURE['cases'] if
                         (row['n'], row['alpha'], row['method']) == (n, alpha, method))
        if reference['status'] == 'source_convergence_error':
            assert reference['iterations'] == 9 and reference['index'] == n - 1
            with pytest.raises(jax.errors.JaxRuntimeError, match='Newton convergence guard'):
                lagpts(n, alpha, method=method)[0].block_until_ready()
            continue
        x, w, v = map(np.asarray, lagpts(n, alpha, method=method, bary=True))
        outputs[method] = x, w
        assert x.size == w.size == v.size == reference['length']
        assert np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(v).all()
        assert np.all(x > 0) and np.all(np.diff(x) > 0) and np.all(w >= 0)
        npt.assert_allclose(x, reference['x'], rtol=0, atol=8e-10)
        source_weights = np.asarray(reference['w'])
        normal = source_weights >= np.finfo(float).tiny
        npt.assert_allclose(w[normal], source_weights[normal], rtol=4e-9, atol=0)
        # Different libm exp implementations do not promise identical tiny
        # mantissas. Stopping/nonzero patterns remain exact at this grid.
        npt.assert_array_equal(w.view(np.uint64) != 0, source_weights.view(np.uint64) != 0)
        npt.assert_allclose([w @ x**k for k in range(5)],
                            [math.gamma(alpha+k+1) for k in range(5)], rtol=2e-9, atol=0)
        k = np.arange(1, n+1, dtype=float)
        eigenvalues = eigh_tridiagonal(2*k-1+alpha, np.sqrt(k[:-1]*(k[:-1]+alpha)),
                                      eigvals_only=True, select='i', select_range=(0, 15),
                                      lapack_driver='stebz')
        npt.assert_allclose(x[:16], eigenvalues, rtol=0, atol=8e-10)
        derivative = -eval_genlaguerre(n-1, alpha+1, x[:16])
        expected_weights = np.exp(gammaln(n+1+alpha)-gammaln(n+1))/(x[:16]*derivative**2)
        npt.assert_allclose(w[:16], expected_weights, rtol=4e-9, atol=0)
        assert np.max(np.abs(v)) == 1
        nonzero = v != 0
        npt.assert_array_equal(np.sign(v[nonzero]), ((-1.)**np.arange(x.size))[nonzero])
    if outputs:
        full_x, full_w = outputs['RH']
        short_x, short_w = outputs['RHW']
        npt.assert_allclose(short_x, full_x[:short_x.size], rtol=0, atol=8e-10)
        npt.assert_allclose(short_w, full_w[:short_w.size], rtol=4e-9, atol=0)


@pytest.mark.parametrize('n', [42, 100, 300, 1000])
@pytest.mark.parametrize('compressed', [False, True])
def test_source_initial_guess_geometry_and_capacity(n, compressed):
    x, itric, igatt = _rh_initial_guesses_general(n, .3, compressed)
    capacity = min(n, math.ceil(17*math.sqrt(n))) if compressed else n
    assert x.size == capacity
    assert itric == math.floor(3.6*n**.188 + .5)
    assert igatt == math.floor(capacity+1.31*n**.4-n+.5)
    assert itric >= 7
    assert np.all(np.asarray(x[:itric]) > 0)
    assert np.all(np.asarray(x[itric:capacity-max(igatt, 0)]) == 0)


@pytest.mark.parametrize('n', [1, 2, 23])
def test_invalid_source_seed_geometry_is_not_replaced_by_gw(n):
    with pytest.raises(ValueError, match='requires n>=2|initial-guess geometry'):
        _laguerre_rh_general(n, .3)

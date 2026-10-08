"""Direct EXP/EXPW source controls, not an RH-accuracy claim for small n.

Pinned lagpts.m source-expression fixture uses independent host SciPy special
functions, not fresh MATLAB. Source has O(n^-4) Airy node accuracy and only
O(n^-2/3) relative Airy weight accuracy. n42 source moment residual is ~1.4e-8.
Source-comparison envelopes: nodes8e-10 absolute, normal weights4e-9 relative;
mathematical moment envelope2e-9 applies at n>=100, where source meets it.
"""
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.laguerre_exp import _exp_last_nonzero, _laguerre_exp
from chebfunjax.utils.quadrature import lagpts

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/laguerre_exp_source_reference.json').read_text())


@pytest.mark.parametrize('alpha', [0.0, -0.5, 0.5, 0.3])
@pytest.mark.parametrize('n', [42, 100, 300, 1000, 3000])
def test_explicit_expansions_match_literal_source(n, alpha):
    outputs = {}
    for method in ['EXP', 'EXPW']:
        ref = next(r for r in FIXTURE['cases'] if (r['n'], r['alpha'], r['method']) == (n, alpha, method))
        assert ref['status'] == 'ok'
        x, w, v = map(np.asarray, lagpts(n, alpha, method=method, bary=True))
        assert x.size == w.size == v.size == ref['length']
        assert np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(v).all()
        assert np.all(x > 0) and np.all(np.diff(x) > 0) and np.all(w >= 0)
        npt.assert_allclose(x, ref['x'], rtol=0, atol=8e-10)
        ew = np.asarray(ref['w'])
        normal = ew >= np.finfo(float).tiny
        npt.assert_allclose(w[normal], ew[normal], rtol=4e-9, atol=0)
        npt.assert_array_equal(w.view(np.uint64) != 0, ew.view(np.uint64) != 0)
        moments = np.asarray([w @ x**k for k in range(5)])
        if n >= 100:
            npt.assert_allclose(moments, [math.gamma(alpha+k+1) for k in range(5)], rtol=2e-9, atol=0)
        else:
            source_moments = [ew @ np.asarray(ref['x'])**k for k in range(5)]
            npt.assert_allclose(moments, source_moments, rtol=2e-9, atol=0)
            # Preserve the source's actual lower accuracy; do not promise RH precision.
            assert ref['moment_maxrel'] > 1e-8
        assert np.max(np.abs(v)) == 1
        nz = v != 0
        npt.assert_array_equal(np.sign(v[nz]), ((-1.)**np.arange(x.size))[nz])
        raw = _laguerre_exp(n, alpha, comp_repr=(method == 'EXPW'))
        rw = np.asarray(raw[1])
        if method == 'EXPW':
            assert int(raw[2]) == ref['length']
            assert rw.view(np.uint64)[int(raw[2])-1] != 0
            assert np.all(rw.view(np.uint64)[int(raw[2]):] == 0)
        refraw = np.asarray(ref['raw_w'])
        normal = refraw >= np.finfo(float).tiny
        npt.assert_allclose(rw[:refraw.size][normal], refraw[normal], rtol=4e-9, atol=0)
        npt.assert_array_equal(rw[:refraw.size].view(np.uint64) != 0, refraw.view(np.uint64) != 0)
        outputs[method] = x, w
    x, w = outputs['EXPW']
    npt.assert_allclose(x, outputs['EXP'][0][:x.size], rtol=0, atol=8e-10)
    npt.assert_allclose(w, outputs['EXP'][1][:x.size], rtol=4e-9, atol=0)


@pytest.mark.parametrize('disabled', [True, False])
def test_last_nonzero_retains_subnormal_holes_and_nan(disabled):
    with jax.disable_jit(disabled):
        fn = jax.jit(_exp_last_nonzero)
        for bits, expected in [([0, 0], 0), ([1, 0], 1), ([1, 0, 2, 0], 3),
                               ([0, 0x8000000000000000], 0), ([0, 0x7ff8000000000000, 0], 2)]:
            values = np.asarray(bits, dtype=np.uint64).view(np.float64)
            assert int(fn(jnp.asarray(values))) == expected


@pytest.mark.parametrize('method', ['EXP', 'EXPW'])
@pytest.mark.parametrize('interval', [(1.0, np.inf), (-np.inf, -1.0)])
def test_interval_mapping_follows_barycentric_construction(method, interval):
    x, w, v = map(np.asarray, lagpts(1000, .3, method=method, bary=True))
    mx, mw, mv = map(np.asarray, lagpts(1000, .3, interval, method=method, bary=True))
    npt.assert_array_equal(mv, v)
    npt.assert_array_equal(mx, x+1 if np.isinf(interval[1]) else -x-1)
    normal = w >= np.finfo(float).tiny * math.e
    npt.assert_allclose(mw[normal], w[normal] / math.e, rtol=100*np.finfo(float).eps, atol=0)


@pytest.mark.parametrize('n', [1, 2, 8])
def test_invalid_region_geometry_is_not_replaced_by_another_algorithm(n):
    with pytest.raises(ValueError, match='requires n>=2|region geometry'):
        lagpts(n, method='EXP')


def test_expw_variable_length_requires_eager_adapter():
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(lambda: lagpts(300, .3, method='EXPW'))()


def test_explicit_methods_require_static_alpha():
    with pytest.raises(NotImplementedError, match='concrete alpha'):
        jax.jit(lambda alpha: lagpts(100, alpha, method='EXP'))(.3)

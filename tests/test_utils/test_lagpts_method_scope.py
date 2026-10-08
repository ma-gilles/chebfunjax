"""Source scope controls for Laguerre REC/GW/GLR and bounded RH integration.

Provenance: MATLAB ``lagpts.m`` and ``hermpts.m``, Chebfun commit
7574c77680d7e82b79626300bf255498271a72df. These tests distinguish the
ported algorithms from deliberately unsupported variants (including small RH).
"""
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.quadrature import lagpts


@pytest.mark.parametrize('n', [1, 42, 299])
def test_default_laguerre_uses_source_rec_range(n):
    for got, want in zip(lagpts(n, 0.5), lagpts(n, 0.5, method='REC'), strict=True):
        npt.assert_array_equal(got, want)


def test_default_laguerre_uses_gw_above_rec_cutoff():
    for got, want in zip(lagpts(300, 0.5), lagpts(300, 0.5, method='GW'), strict=True):
        npt.assert_array_equal(got, want)


@pytest.mark.parametrize('method', ['EXP'])
def test_unported_source_methods_fail_explicitly(method):
    with pytest.raises(NotImplementedError, match='not yet supported'):
        lagpts(42, method=method)


def test_lagpts_n_zero_precedes_invalid_options():
    x, w, v = lagpts(0, alpha=0.5j, interval=(0.0, 1.0), bary=True, method='bad')
    assert np.asarray(x).shape == np.asarray(w).shape == np.asarray(v).shape == (0,)


def test_recw_is_supported_at_small_order():
    for got, want in zip(lagpts(42, method='RECW'), lagpts(42, method='REC'), strict=True):
        npt.assert_array_equal(got, want)


@pytest.mark.parametrize('method', ['RH', 'RHW'])
def test_small_rh_preserves_source_convergence_failure(method):
    import jax
    with pytest.raises(jax.errors.JaxRuntimeError, match="Newton convergence guard"):
        lagpts(42, method=method)[0].block_until_ready()

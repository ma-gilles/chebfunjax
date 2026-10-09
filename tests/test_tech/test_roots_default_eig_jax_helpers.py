"""Source matrix controls and explicitly labelled adapter failure controls."""
import json
import os
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from numpy.linalg import LinAlgError

from chebfunjax.tech import chebtech as module


def record(name, data):
    directory = os.environ.get('CHEBFUN_RUNTIME_REPORT')
    if directory:
        (Path(directory).parent/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')


def pairs(a):
    return [[float(jnp.real(x)), float(jnp.imag(x))] for x in jnp.ravel(a)]


def component_bits(a):
    a = jnp.asarray(a)
    fmt = '>f' if jnp.real(a).dtype.itemsize == 4 else '>d'
    return [[struct.pack(fmt, float(jnp.real(x))).hex(),
             struct.pack(fmt, float(jnp.imag(x))).hex()] for x in jnp.ravel(a)]


@pytest.mark.parametrize('n', [3, 4, 50])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
@pytest.mark.parametrize('scale', [1., -3.])
def test_source_colleague_matrix(n, dtype, scale):
    c = jnp.linspace(.25, 1., n).astype(dtype)*scale
    if jnp.issubdtype(dtype, jnp.complexfloating):
        c = c*(1.+.5j)
    # Literal eager source operations provide an evaluation-context comparator.
    adjusted = -.5*c[:-1]/c[-1]
    adjusted = adjusted.at[-2].set(adjusted[-2]+.5)
    oh = .5*jnp.ones(n-2, dtype=jnp.float64)
    source = jnp.diag(oh, 1)+jnp.diag(oh, -1)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        source = source.astype(jnp.complex128)
    source = source.at[-2, -1].set(1.)
    source = source.at[:, 0].set(adjusted[::-1])
    actual, finite = module._roots_matrix_and_finite_jax(c)
    assert bool(finite)
    eps = jnp.finfo(jnp.real(c).dtype).eps
    bound = 32*eps*max(1., float(jnp.max(jnp.abs(adjusted))))
    indices = jnp.argwhere(actual != source)
    differences = [{'index': [int(x) for x in idx],
                    'actual': pairs(actual[tuple(idx)]),
                    'source': pairs(source[tuple(idx)])} for idx in indices]
    record(f'matrix_{n}_{jnp.dtype(dtype).name}_{scale}',
           {'dtype': str(actual.dtype), 'bound': float(bound), 'differences': differences,
            'coefficient_dtype': str(c.dtype), 'coefficient_bits': component_bits(c),
            'adjusted_dtype': str(adjusted.dtype), 'adjusted_bits': component_bits(adjusted),
            'actual_bits': component_bits(actual), 'source_bits': component_bits(source),
            'actual': pairs(actual), 'source': pairs(source)})
    assert actual.dtype == source.dtype
    assert bool(jnp.array_equal(actual[:, 1:], source[:, 1:]))
    assert bool(jnp.array_equal(actual == 0, source == 0))
    assert float(jnp.max(jnp.abs(actual-source))) <= bound


@pytest.mark.parametrize('case', ['real_real', 'real_complex', 'complex_real', 'complex_complex'])
def test_actual_eig_interface_and_analytic_roots(case):
    # T2=2x^2-1: T2 gives +/-sqrt(.5); T2+2 gives +/-i*sqrt(.5).
    complex_coeffs = case.startswith('complex')
    complex_roots = case.endswith('complex')
    c = jnp.asarray([2. if complex_roots else 0., 0., 1.],
                    dtype=jnp.complex128 if complex_coeffs else jnp.float64)
    if complex_coeffs:
        c = c*(1.+2j)
    result = module._roots_default_eigenvalues(c)
    expected = jnp.asarray([-1., 1.])*jnp.sqrt(.5)*(1j if complex_roots else 1.)
    record('interface_'+case, {'roots': pairs(result), 'dtype': str(result.dtype),
                              'expected_unordered': pairs(expected)})
    assert result.shape == (2,)
    assert result.flags.writeable and result.flags.owndata
    assert str(result.dtype) == ('complex128' if complex_coeffs or complex_roots else 'float64')
    # Explicitly unordered analytic correctness; no library sorting is added.
    error = jnp.min(jnp.abs(jnp.asarray(result)[:, None]-expected[None, :]), axis=1)
    assert float(jnp.max(error)) < 100*jnp.finfo(jnp.float64).eps
    matched = jnp.argmin(jnp.abs(jnp.asarray(result)[:, None]-expected), axis=1)
    assert int(matched[0]) != int(matched[1])


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_adapter_nonfinite_input_prevents_solver(monkeypatch, invalid):
    def forbidden(_):
        raise AssertionError('solver must not run for nonfinite matrix')
    monkeypatch.setattr(module, '_roots_eigvals_jax', forbidden)
    monkeypatch.setattr(module, '_roots_eig_and_metadata_jax', forbidden)
    with pytest.raises(LinAlgError, match='infs or NaNs'):
        module._roots_default_eigenvalues(jnp.asarray([invalid, 1., 1.]))


@pytest.mark.parametrize('invalid', [complex(float('nan'), 0), complex(float('inf'), 0),
                                    complex(0, float('nan')), complex(0, float('inf'))])
def test_adapter_nonfinite_output_prevents_native_filter(monkeypatch, invalid, isolated_eig_metadata):
    monkeypatch.setattr(module, '_roots_eigvals_jax',
                        lambda _: jnp.asarray([invalid, .25], dtype=jnp.complex128))
    with pytest.raises(LinAlgError, match='nonfinite JAX eig output'):
        module._roots_main(jax.device_get(jnp.asarray([1., 1., 1.])), 100*jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize('complex_input', [False, True])
def test_adapter_preserves_injected_order_and_dtype(monkeypatch, complex_input, isolated_eig_metadata):
    given = jnp.asarray([.75+0j, -.25+0j])
    monkeypatch.setattr(module, '_roots_eigvals_jax', lambda _: given)
    result = module._roots_default_eigenvalues(jnp.asarray([1., 1., 1.],
              dtype=jnp.complex128 if complex_input else jnp.float64))
    assert bool(jnp.array_equal(jnp.asarray(result), given))
    assert str(result.dtype) == ('complex128' if complex_input else 'float64')
    assert result.flags.writeable and result.flags.owndata


@pytest.fixture
def isolated_eig_metadata():
    """Injection must retrace phase2; do not reuse a provider from another case."""
    module._roots_eig_and_metadata_jax.clear_cache()
    try:
        yield
    finally:
        module._roots_eig_and_metadata_jax.clear_cache()

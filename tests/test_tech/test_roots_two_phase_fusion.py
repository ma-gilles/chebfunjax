"""Nonempty provider-order controls and retained IR for two-phase fusion.

The eigenvalue comparisons are same-provider compatibility, not MATLAB bit
parity. IR artifacts require independent operation review alongside the exact
represented-input source tests; symbol counts alone are not arithmetic proof.
"""
import hashlib
import json
import os
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech import chebtech as module


def _bytes(value):
    return jax.device_get(value).tobytes()


def _capture(value):
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'bytes_hex': _bytes(value).hex()}


def _report(name, data):
    directory = os.environ.get('CHEBFUN_RUNTIME_REPORT')
    if directory:
        (Path(directory).parent / name).write_text(json.dumps(data, indent=2)+'\n')


@pytest.mark.parametrize('case', ['real_real', 'real_complex', 'complex_real', 'complex_complex'])
def test_fused_actual_unsorted_provider_output(case):
    complex_input = case.startswith('complex')
    complex_spectrum = case.endswith('complex')
    c = jnp.asarray([2. if complex_spectrum else 0., 0., 1.],
                    dtype=jnp.complex128 if complex_input else jnp.float64)
    if complex_input:
        c = c*(1.+2j)
    matrix, matrix_finite = module._roots_matrix_and_finite_jax(c)
    roots, roots_finite, has_imaginary, real_roots = module._roots_eig_and_metadata_jax(matrix)
    provider_roots = module._roots_eigvals_jax(matrix)
    expected_has_imaginary = bool(jnp.any(jnp.imag(provider_roots) != 0))
    expected = (jnp.real(provider_roots)
                if not complex_input and not expected_has_imaginary else provider_roots)
    actual = module._roots_default_eigenvalues(c)
    _report('unsorted_'+case+'.json', {
        'matrix': _capture(matrix), 'raw': _capture(roots),
        'provider_raw': _capture(provider_roots), 'adapted': _capture(actual),
        'expected_adapted': _capture(expected),
        'scope': 'Unsorted same-provider compatibility; no MATLAB bit claim',
    })
    assert roots.shape == provider_roots.shape == (2,)
    assert bool(matrix_finite) == bool(jnp.all(jnp.isfinite(matrix)))
    assert bool(roots_finite) == bool(jnp.all(jnp.isfinite(provider_roots)))
    assert bool(has_imaginary) == expected_has_imaginary
    assert roots.dtype == provider_roots.dtype
    assert _bytes(roots) == _bytes(provider_roots)
    assert _bytes(real_roots) == _bytes(jnp.real(provider_roots))
    assert actual.dtype == expected.dtype
    assert _bytes(actual) == _bytes(expected)
    assert actual.flags.writeable and actual.flags.owndata


@pytest.mark.parametrize('dtype', [jnp.float64, jnp.complex128])
def test_fused_ir_contract(dtype):
    c = jnp.linspace(.25, 1., 50).astype(dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        c = c*(1.+.5j)
    matrix = module._roots_colleague_matrix_jax(c)
    objects = {
        'original_matrix': module._roots_colleague_matrix_jax.lower(c),
        'phase1': module._roots_matrix_and_finite_jax.lower(c),
        'original_eig': module._roots_eigvals_jax.lower(matrix),
        'phase2': module._roots_eig_and_metadata_jax.lower(matrix),
    }
    captures = {}
    directory = os.environ.get('CHEBFUN_RUNTIME_REPORT')
    for name, lowered in objects.items():
        stablehlo = str(lowered.compiler_ir(dialect='stablehlo'))
        optimized = lowered.compile().as_text()
        targets = re.findall(r'custom_call_target="([^"]+)"', optimized)
        captures[name] = {'stablehlo': stablehlo, 'optimized_hlo': optimized,
                          'custom_call_targets': targets,
                          'stablehlo_sha256': hashlib.sha256(stablehlo.encode()).hexdigest(),
                          'optimized_sha256': hashlib.sha256(optimized.encode()).hexdigest()}
    if directory:
        _report('ir_'+jnp.dtype(dtype).name+'.json', captures)
    # Structural prerequisites only. Independent review must inspect retained
    # barrier/operation ordering; exact rational source controls check results.
    assert 'optimization_barrier' in captures['original_matrix']['stablehlo']
    assert 'optimization_barrier' in captures['phase1']['stablehlo']
    original_targets = captures['original_eig']['custom_call_targets']
    phase2_targets = captures['phase2']['custom_call_targets']
    assert any('geev' in target for target in original_targets)
    assert phase2_targets == original_targets
    assert not any('geev' in target for target in captures['phase1']['custom_call_targets'])
    assert 'is_finite' in captures['phase1']['stablehlo']
    assert 'is_finite' in captures['phase2']['stablehlo']

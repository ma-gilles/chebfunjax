"""Native happinessCheck derives DATA.vscale from the coefficient grid.

Source: @chebtech/happinessCheck.m, vscale.m, standardCheck.m, strictCheck.m,
sampleTest.m and populate.m at7574c77680d7e82b79626300bf255498271a72df.
These authored controls distinguish the documented strictCheck values-ignored
contract, transient DATA scaling and supplied-value sampleTest denominator.
The retained native stage oracle is distinct from actual JAX FFT replay.
"""
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax.tech.chebtech as module
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

TECHS = [Chebtech1, Chebtech2]


@pytest.mark.parametrize('tech', TECHS)
@pytest.mark.parametrize('columns', [1, 2])
def test_strict_ignores_supplied_values_scale(tech, columns):
    # strictCheck explicitly ignores VALUES. The center grid point attains
    # 4+.5*T32=4.5 (and9 in column2), giving a threshold above each tail.
    coefficients = jnp.zeros((33, columns))
    coefficients = coefficients.at[0].set(jnp.arange(1, columns+1)*4.)
    coefficients = coefficients.at[32].set(jnp.arange(1, columns+1)*.5)
    values = jnp.ones_like(coefficients)
    if columns == 1:
        coefficients, values = coefficients[:, 0], values[:, 0]
    assert tech.happiness_check(
        coefficients, values, tol=.125, check='strict', sample_test=False
    ) == (True, 1)


@pytest.mark.parametrize('tech', TECHS)
def test_explicit_data_column_scales_and_input_copy(tech):
    coefficients = jnp.zeros((33, 2)).at[0].set(jnp.array([4., 8.]))
    coefficients = coefficients.at[32].set(jnp.array([.5, 1.]))
    scales = np.array([16., 32.])
    before = scales.copy()
    assert tech.happiness_check(
        coefficients, jnp.ones_like(coefficients), vscale=scales,
        tol=1/32, check='strict', sample_test=False
    ) == (True, 1)
    np.testing.assert_array_equal(scales, before)


@pytest.mark.parametrize('tech', TECHS)
@pytest.mark.parametrize('hscale,delta,expected', [(.25, .375, True), (8., .75, True), (8., 1.25, False)])
def test_sample_test_uses_representation_and_horizontal_scales(tech, hscale, delta, expected):
    coefficients = jnp.zeros(33).at[0].set(4.)
    # Source sampleTest budget sqrt(1/64)*max(hscale*1, vscale(f)=4).
    def operator(x):
        return jnp.full_like(x, 4.+delta)
    happy, cutoff = tech.happiness_check(
        coefficients, jnp.ones(33), op=operator, tol=1/64, hscale=hscale
    )
    assert happy is expected
    assert cutoff == (1 if expected else 33)


@pytest.mark.parametrize('tech', TECHS)
def test_supplied_values_do_not_replace_data_scale(tech):
    coefficients = jnp.zeros(33).at[0].set(1.)
    # Native budget=.125*max(.25*4, vscale(f)=1)=.125; error.25 rejects.
    # Unconditionally adding max(abs(values)) to DATA would accept this.
    def operator(x):
        return jnp.full_like(x, 1.25)
    assert tech.happiness_check(
        coefficients, jnp.full(33, 4.), op=operator, tol=1/64, hscale=.25
    ) == (False, 33)


@pytest.mark.parametrize('tech', TECHS)
def test_same_shape_new_coefficients_recompute_representation_scale(tech):
    first = jnp.zeros(33).at[0].set(1.).at[32].set(.5)
    second = first.at[0].set(4.)
    assert tech.happiness_check(first, jnp.ones(33), tol=.125,
                                check='strict', sample_test=False) == (False, 33)
    assert tech.happiness_check(second, jnp.ones(33), tol=.125,
                                check='strict', sample_test=False) == (True, 1)


@pytest.mark.parametrize('tech', TECHS)
def test_array_representation_grid_is_evaluated_once(tech, monkeypatch):
    original = tech.coeffs2vals
    calls = []
    def observed(coefficients):
        calls.append(coefficients.shape)
        return original(coefficients)
    monkeypatch.setattr(tech, 'coeffs2vals', staticmethod(observed))
    coefficients = jnp.zeros((33, 2)).at[0].set(jnp.array([4., 8.]))
    assert tech.happiness_check(coefficients, jnp.ones((33, 2)),
                                sample_test=False) == (True, 1)
    # Native vscale(f) precedes the standardCheck column loop. Repeating a
    # scalar-column transform could change its roundoff and global scale.
    assert calls == [(33, 2)]


def _native_stage():
    path = Path(__file__).with_name('fixtures')/'happiness_native_initial_scale_7574c77.json'
    data = json.loads(path.read_text())
    arrays = {name: jnp.asarray(np.frombuffer(bytes.fromhex(data[key]), dtype='<f8'))
              for name, key in [('coefficients', 'coeffs_hex'), ('values', 'values_hex'),
                                ('representation', 'representation_values_hex')]}
    return data, arrays


def _watch_tolerance(monkeypatch):
    original = module.standard_chop
    observed = []
    def traced(coefficients, tolerance):
        observed.append(struct.pack('>d', float(tolerance)).hex())
        return original(coefficients, tolerance)
    monkeypatch.setattr(module, 'standard_chop', traced)
    return observed


def test_actual_native_one_ulp_scale_with_native_kernel_oracle(monkeypatch):
    data, arrays = _native_stage()
    def native_kernel(coefficients):
        np.testing.assert_array_equal(coefficients, arrays['coefficients'])
        return arrays['representation']
    monkeypatch.setattr(Chebtech2, 'coeffs2vals', staticmethod(native_kernel))
    observed = _watch_tolerance(monkeypatch)
    assert Chebtech2.happiness_check(arrays['coefficients'], arrays['values'],
                                    sample_test=False) == (False, 4097)
    assert observed == [data['metadata']['scaled_tol_hex']]


def test_actual_native_stage_direct_jax_scale_replay(monkeypatch):
    data, arrays = _native_stage()
    representation = Chebtech2.coeffs2vals(arrays['coefficients'])
    represented_scale = float(jnp.max(jnp.abs(representation)))
    sampled_scale = float(jnp.max(jnp.abs(arrays['values'])))
    expected = np.finfo(np.float64).eps*max(1., represented_scale/sampled_scale)
    observed = _watch_tolerance(monkeypatch)
    Chebtech2.happiness_check(arrays['coefficients'], arrays['values'], sample_test=False)
    assert observed == [struct.pack('>d', expected).hex()]
    print(json.dumps({'scope': 'DirectJAXreplay, not nativeFFTbitparity',
                      'native_tolerance_hex': data['metadata']['scaled_tol_hex'],
                      'actual_tolerance_hex': observed[0],
                      'representation_scale_hex': struct.pack('>d', represented_scale).hex(),
                      'sampled_scale_hex': struct.pack('>d', sampled_scale).hex()}))

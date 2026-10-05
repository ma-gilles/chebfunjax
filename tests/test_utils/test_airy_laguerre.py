"""Independent complex Airy values and defining ODE checks for RH rays.

Provenance: lagpts.m (asyAiry, Chebfun7574c77), DLMF9.2/9.7.
Local oracle: mpmath Ai/Ai-prime at 100 decimal digits, evaluated at the
exact binary64 complex inputs. SciPy supplies the phase-conditioned far
and source soft-edge oracles. At some local inputs SciPy differs from
100-digit values by about 1e-13; the original qualification bounds remain.
These are helper qualification bounds, not new MATLAB large-n assertions.
"""
import jax
import mpmath as mp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import airy

from chebfunjax.utils.airy_laguerre import _airy_laguerre_rays

EPS = np.finfo(float).eps
RAY = complex(0.5, -np.sqrt(3) / 2)


@pytest.mark.parametrize('direction', [1, -1])
def test_local_ray_against_independent_complex_airy(direction):
    z = direction * RAY * np.linspace(0, 16, 1025)
    ai, aip = map(np.asarray, _airy_laguerre_rays(z))
    with mp.workdps(100):
        arguments = [mp.mpc(float(value.real), float(value.imag)) for value in z]
        expected_ai = np.array([complex(mp.airyai(value)) for value in arguments])
        expected_aip = np.array([complex(mp.airyai(value, derivative=1))
                                 for value in arguments])
    assert np.isfinite(expected_ai).all() and np.isfinite(expected_aip).all()
    npt.assert_allclose(ai, expected_ai, rtol=256 * EPS, atol=16 * EPS)
    npt.assert_allclose(aip, expected_aip, rtol=256 * EPS, atol=16 * EPS)


@pytest.mark.parametrize('direction,maximum', [(1, 1000), (-1, 80)])
def test_far_ray_phase_conditioned_independent_complex_airy(direction, maximum):
    z = direction * RAY * np.geomspace(16 + 1e-10, maximum, 513)
    ai, aip = map(np.asarray, _airy_laguerre_rays(z))
    expected_ai, expected_aip, _, _ = airy(z)
    assert np.isfinite(expected_ai).all() and np.isfinite(expected_aip).all()
    # The principal-power/exponential phase is conditioned by |z|^(3/2).
    # Use a componentwise relative bound that accounts for that phase
    # rounding, rather than a relative constant or a zero-sensitive norm.
    bound = 32 * EPS * (1 + np.abs(z)**1.5)
    assert np.all(np.abs(ai - expected_ai) <= bound * np.abs(expected_ai))
    assert np.all(np.abs(aip - expected_aip) <= bound * np.abs(expected_aip))


@pytest.mark.parametrize('n', [3000, 10000])
def test_source_soft_edge_arguments_against_independent_airy(n):
    for order in (n - 1, n, n + 1):
        z = np.array([.9251, .95, .975, .999])
        argument = (order * 3j * (np.sqrt(z) * np.sqrt(1 - z)
                                  - np.arccos(np.sqrt(z))))**(2 / 3)
        ai, aip = _airy_laguerre_rays(argument)
        expected_ai, expected_aip, _, _ = airy(argument)
        npt.assert_allclose(ai, expected_ai, rtol=512 * EPS, atol=32 * EPS)
        npt.assert_allclose(aip, expected_aip, rtol=512 * EPS, atol=32 * EPS)


@pytest.mark.parametrize('radial', [-16, -4, -.1, 0, .1, 4, 15.75, 16, 32, 100])
def test_analytic_derivatives_and_airy_ode(radial):
    z = radial * RAY
    ai, aip = _airy_laguerre_rays(z)
    first = jax.jvp(lambda value: _airy_laguerre_rays(value)[0], (z,), (1 + 0j,))[1]
    second = jax.jvp(lambda value: _airy_laguerre_rays(value)[1], (z,), (1 + 0j,))[1]
    npt.assert_allclose(first, aip, rtol=256 * EPS, atol=32 * EPS)
    npt.assert_allclose(second, z * ai, rtol=256 * EPS, atol=32 * EPS)


def test_invalid_arguments_do_not_present_a_general_airy_api():
    ai, aip = _airy_laguerre_rays(np.array([1 + 0j, -1 + 0j, 1j, np.nan + 0j]))
    assert np.isnan(ai).all() and np.isnan(aip).all()


def test_vmap_and_array_shape_agree():
    z = np.array([0, .1, 4, 32]) * RAY
    direct = _airy_laguerre_rays(z)
    mapped = jax.jit(jax.vmap(_airy_laguerre_rays))(z)
    for got, expected in zip(direct, mapped, strict=True):
        assert got.shape == (4,)
        npt.assert_allclose(got, expected, rtol=16 * EPS, atol=16 * EPS)

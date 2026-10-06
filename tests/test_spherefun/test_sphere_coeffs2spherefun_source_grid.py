"""Independent Fourier polynomial controls for source coeffs2spherefun."""

import cmath
import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np  # uses-numpy: independent expected grids and assertions
import pytest

from chebfunjax.spherefun.spherefun import Spherefun


@pytest.mark.parametrize("m,n", [(3, 5), (4, 6), (3, 6), (4, 5), (1, 1)])
def test_source_numeric_constructor_receives_direct_fourier_grid(monkeypatch, m, n):
    # A direct finite Fourier sum is independent of FFT shifts, padding and
    # the library transform. Include genuinely complex, non-Hermitian data.
    coeffs = np.array([[complex((i + 1) / 8, (j - 2) / 16)
                        for j in range(n)] for i in range(m)])
    me, ne = m + m % 2, n + n % 2
    expected = np.zeros((me // 2 + 1, ne))
    for r in range(me // 2 + 1):
        theta = 2 * math.pi * r / me
        for c in range(ne):
            lam = -math.pi + 2 * math.pi * c / ne
            expected[r, c] = sum(
                coeffs[i, j] * cmath.exp(1j * ((i - m // 2) * theta
                                              + (j - n // 2) * lam))
                for i in range(m) for j in range(n)).real
    observed = []
    token = object()

    def adaptive_constructor(cls, callback, **kwargs):
        raise AssertionError("MATLAB coeffs2spherefun uses a fixed numeric grid")

    monkeypatch.setattr(Spherefun, "from_function", classmethod(adaptive_constructor))

    def numeric_constructor(cls, values):
        observed.append(np.asarray(values))
        return token

    monkeypatch.setattr(Spherefun, "from_values", classmethod(numeric_constructor))
    assert Spherefun.coeffs2spherefun(jnp.asarray(coeffs)) is token
    assert len(observed) == 1
    assert observed[0].shape == expected.shape
    assert not np.iscomplexobj(observed[0])
    # Explicit roundoff allowance for a small direct sum and two FFTs.
    atol = 128 * np.finfo(float).eps * max(1., np.sum(np.abs(coeffs)))
    np.testing.assert_allclose(observed[0], expected, rtol=0, atol=atol)


@pytest.mark.parametrize("m,n", [(3, 3), (4, 4), (3, 4), (4, 3)])
def test_actual_constructor_retains_low_modes_across_odd_even_padding(m, n):
    # Fourier coefficients derived from2+cos(theta); no source-fitted data.
    coeffs = jnp.zeros((m, n), dtype=jnp.complex128)
    coeffs = coeffs.at[m // 2, n // 2].set(2.)
    coeffs = coeffs.at[m // 2 - 1, n // 2].set(.5)
    coeffs = coeffs.at[m // 2 + 1, n // 2].set(.5)
    f = Spherefun.coeffs2spherefun(coeffs)
    theta = jnp.array([0., .17, .63, 1.2, 2.4, jnp.pi])
    lam = jnp.array([-2.8, -.8, 0., .3, 1.6, 2.9])
    np.testing.assert_allclose(f(lam, theta), 2 + jnp.cos(theta), rtol=0,
                               atol=128 * 3 * np.finfo(float).eps)


def test_actual_constructor_real_projection_removes_imaginary_dc():
    coeffs = jnp.zeros((3, 5), dtype=jnp.complex128).at[1, 2].set(2 + 7j)
    f = Spherefun.coeffs2spherefun(coeffs)
    np.testing.assert_allclose(f(jnp.array([-.3, 1.7]), jnp.array([.2, 2.3])),
                               2., rtol=0, atol=128 * np.finfo(float).eps)


def test_actual_theta_nyquist_uses_real_grid_interpolant():
    # Source samples erase imaginary Nyquist amplitude. Reconstructing the
    # continuous real Fourier callback would incorrectly retain2*sin(2th).
    coeffs = jnp.zeros((4, 4), dtype=jnp.complex128).at[0, 2].set(1 + 2j)
    f = Spherefun.coeffs2spherefun(coeffs)
    theta = jnp.array([.13, .41, .83, 1.7, 2.3])
    lam = jnp.array([-2., -.5, 0., .8, 2.1])
    np.testing.assert_allclose(f(lam, theta), jnp.cos(2 * theta), rtol=0,
                               atol=256 * np.finfo(float).eps)


def test_actual_longitude_nyquist_with_regular_poles():
    # sin(theta)^2 times a complex unpaired lambda-Nyquist mode. This is
    # regular at both poles; source real grid reconstructs cos(2lambda).
    coeffs = jnp.zeros((5, 4), dtype=jnp.complex128)
    coeffs = coeffs.at[2, 0].set(.5 * (1 + 2j))
    coeffs = coeffs.at[0, 0].set(-.25 * (1 + 2j))
    coeffs = coeffs.at[4, 0].set(-.25 * (1 + 2j))
    f = Spherefun.coeffs2spherefun(coeffs)
    theta = jnp.array([0., .21, .73, 1.4, 2.3, jnp.pi])
    lam = jnp.array([-2.7, -1.2, -.2, .4, 1.7, 2.8])
    expected = jnp.sin(theta) ** 2 * jnp.cos(2 * lam)
    np.testing.assert_allclose(f(lam, theta), expected, rtol=0,
                               atol=512 * np.finfo(float).eps)


@pytest.mark.parametrize("case", range(5))
def test_fresh_matlab_public_matrix_inputs(case):
    # Same explicit mathematical matrix inputs as the independent MATLAB
    # API capture. Saved fitted coefficients never enter construction.
    reference = json.loads(Path(__file__).with_name(
        "sphere_coeffs2spherefun_matlab_reference.json").read_text())
    assert reference["source_head"] == "7574c77680d7e82b79626300bf255498271a72df"
    if case == 1:
        coeffs = jnp.zeros((3, 5), dtype=jnp.complex128)
        for row, value in [(0, .25j), (2, -.25j)]:
            coeffs = coeffs.at[row, 1].set(value).at[row, 3].set(value)
    else:
        coeffs = jnp.zeros((4, 6), dtype=jnp.complex128)
        if case == 0:
            coeffs = coeffs.at[2, 3].set(3).at[1, 3].set(1).at[3, 3].set(1)
        elif case == 2:
            coeffs = coeffs.at[1, 3].set(1-.5j).at[3, 3].set(1-.5j)
        else:
            coeffs = coeffs.at[0, 3].set(1 if case == 3 else 1j)
    f = Spherefun.coeffs2spherefun(coeffs)
    values = f(jnp.asarray(reference["lam"]), jnp.asarray(reference["theta"]))
    np.testing.assert_allclose(values, reference["values"][case], rtol=0,
                               atol=100 * 5 * np.finfo(float).eps)

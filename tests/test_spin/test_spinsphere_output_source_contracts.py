"""Independent sphere output ownership, Fourier grid and symmetry contracts.

Provenance
----------
MATLAB source : @spinoperator/solvepde.m; @spinopsphere/reshapeData.m;
    @spinopsphere/getCoeffs2ValsTransform.m; @trigtech/coeffs2vals.m
Chebfun commit: 7574c77
"""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_source_output_extracts_real_northern_grid(monkeypatch):
    mod = importlib.import_module("chebfunjax.operators.spinopsphere")
    n = 8
    coeffs = object()
    grid = np.arange(n * n).reshape(n, n) + 3j
    seen = {}
    sentinel = object()

    def transform(c):
        assert c is coeffs
        return grid

    def from_values(values):
        seen["values"] = np.asarray(values)
        return sentinel

    def no_resampling(*args, **kwargs):
        raise AssertionError("Solver output is numeric matrix input in MATLAB")

    monkeypatch.setattr(mod, "_sphere_output_coeffs2vals2", transform)
    monkeypatch.setattr(mod.Spherefun, "from_values", from_values)
    monkeypatch.setattr(mod.Spherefun, "from_function", no_resampling)
    assert mod._make_output_spherefun(coeffs, n) is sentinel
    np.testing.assert_array_equal(seen["values"], grid[[4, 5, 6, 7, 0], :].real)


def test_source_output_fft_grid_order():
    # A single theta-frequency and a single longitude-frequency have
    # independent analytic grid values; catches shift, sign and transpose.
    mod = importlib.import_module("chebfunjax.operators.spinopsphere")
    n = 8
    c = jnp.zeros((n, n), dtype=jnp.complex128)
    c = c.at[5, 6].set(2.0 + 3.0j)
    theta = -np.pi + 2 * np.pi * np.arange(n) / n
    lam = theta.copy()
    expected = (2.0 + 3.0j) * np.exp(1j * (theta[:, None] + 2 * lam[None, :]))
    np.testing.assert_allclose(np.asarray(mod._sphere_output_coeffs2vals2(c)),
                               expected, rtol=0, atol=2e-14)


@pytest.mark.parametrize("n", [5, 6])
@pytest.mark.parametrize("compiled", [False, True])
def test_axis_transform_mixed_column_symmetry(n, compiled):
    mod = importlib.import_module("chebfunjax.operators.spinopsphere")
    real = np.arange(1, n + 1, dtype=float)
    coeffs = np.column_stack((real, 1j * real, real + 2j * real[::-1],
                              np.zeros(n)))
    fn = mod._sphere_output_coeffs2vals_axis0
    if compiled:
        fn = jax.jit(fn)
    got = np.asarray(fn(jnp.asarray(coeffs)))
    modes = np.arange(-(n // 2), (n + 1) // 2)
    grid = -np.pi + 2 * np.pi * np.arange(n) / n
    expected = np.exp(1j * grid[:, None] * modes) @ coeffs
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)
    reflected = np.conj(got[np.mod(-np.arange(n), n)])
    np.testing.assert_array_equal(got[:, 0], reflected[:, 0])
    np.testing.assert_array_equal(got[:, 1], -reflected[:, 1])
    np.testing.assert_array_equal(got[:, 3], np.zeros(n))


@pytest.mark.parametrize("n", [0, 1])
def test_axis_transform_trivial_lengths(n):
    mod = importlib.import_module("chebfunjax.operators.spinopsphere")
    coeffs = jnp.full((n, 3), 2.0 + 3.0j)
    for fn in (mod._sphere_output_coeffs2vals_axis0,
               jax.jit(mod._sphere_output_coeffs2vals_axis0)):
        np.testing.assert_array_equal(np.asarray(fn(coeffs)), np.asarray(coeffs))

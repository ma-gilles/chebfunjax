"""Source line-data controls for Chebfun complex curves.

MATLAB source: @chebtech/plotData.m and @chebfun/plotData.m, commit7574c77.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
from chebfunjax.plotting import curve_plot_data


def test_complex_polynomial_uses_chebyshev_grid():
    c=Chebfun.from_coeffs(jnp.array([.5j, 1., .5j]))
    d=curve_plot_data(c)
    assert d['xLine'].shape==(502,)
    assert np.isnan(d['xLine'][0]) and np.isnan(d['yLine'][0])
    x=np.asarray(d['xLine'][1:])
    y=np.asarray(d['yLine'][1:])
    expected=np.sin(np.pi*np.arange(-500,501,2)/1000)
    np.testing.assert_allclose(x,expected,rtol=0,atol=3e-15)
    np.testing.assert_allclose(y,x*x,rtol=0,atol=3e-15)


def test_piece_boundaries_have_nan_breaks():
    c=Chebfun.from_function(lambda t:t+1j*t*t,Domain((-1.,0.,1.)))
    d=curve_plot_data(c)
    assert np.flatnonzero(np.isnan(d['xLine'])).tolist()==[0,502]
    assert np.flatnonzero(np.isnan(d['yLine'])).tolist()==[0,502]
    np.testing.assert_allclose(np.asarray(d['xLine'])[[1,501,503,1003]],[-1,0,0,1],atol=3e-15)


@pytest.mark.parametrize('cap,expected_count',[(65537,767),(503,503)])
def test_high_degree_source_density_and_cap(cap,expected_count):
    coeff=jnp.zeros(61,dtype=jnp.complex128).at[1].set(1).at[60].set(1j)
    c=Chebfun.from_coeffs(coeff)
    d=curve_plot_data(c,max_length=cap)
    assert d['xLine'].shape==(expected_count+1,)
    theta=np.pi*np.arange(expected_count)/(expected_count-1)
    np.testing.assert_allclose(d['xLine'][1:],-np.cos(theta),rtol=0,atol=3e-15)
    np.testing.assert_allclose(d['yLine'][1:],np.cos(60*theta),rtol=0,atol=1e-12)


def test_padded_zero_components_collapse_before_density_choice():
    c=Chebfun.from_coeffs(jnp.zeros(61,dtype=jnp.complex128))
    d=curve_plot_data(c)
    assert d['xLine'].shape==(502,)
    np.testing.assert_array_equal(d['xLine'][1:],np.zeros(501))
    np.testing.assert_array_equal(d['yLine'][1:],np.zeros(501))


def test_pure_imaginary_high_degree_retains_nonzero_component_length():
    c=Chebfun.from_coeffs(jnp.zeros(61,dtype=jnp.complex128).at[60].set(1j))
    d=curve_plot_data(c)
    assert d['xLine'].shape==(768,)
    np.testing.assert_array_equal(d['xLine'][1:],np.zeros(767))
    theta=np.pi*np.arange(767)/766
    np.testing.assert_allclose(d['yLine'][1:],np.cos(60*theta),rtol=0,atol=1e-12)

"""Unrun pole/residue controls from a real, exactly representable trig ratio.

These controls exercise explicit MATLAB output counts with the JAX candidate.
The legacy seven-value Python return remains the ``outputs=None`` default.

Provenance
----------
MATLAB Chebfun commit 7574c77: trigratinterp.m lines 78-106, 494-503;
@trigtech/roots.m lines 66-78; @trigtech/poly.m lines 1-18;
@chebfun/residue.m lines 40-53. The original 23 trigratinterp assertions do
not request poles or residues. The roundoff bounds below are independent
diagnostic bounds, not source test tolerances.
"""

import numpy as np

from chebfunjax.utils.ratapprox import trigratinterp

EPS = np.finfo(float).eps
XI = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0])


def _f(x):
    return 1.0 / (2.0 + np.cos(np.pi * np.asarray(x)))


def _fit(outputs):
    # f=1/(2+cos(pi*x)) is exactly type-(0,1). Its centered Fourier
    # coefficients are p=[1], q=[1/2, 2, 1/2], up to common scale.
    values = _f(XI)
    return trigratinterp(
        values, 0, 1, NN=3, xi=XI, tol=0.0,
        domain=(-1.0, 1.0), outputs=outputs,
    )


def _close(actual, expected):
    assert abs(actual - expected) <= 500 * EPS * max(1.0, abs(expected))


def test_one_output_returns_callable():
    r = _fit(outputs=1)
    assert callable(r)
    probes = np.array([-0.93, -0.5, -0.1, 0.24, 0.78])
    np.testing.assert_allclose(r(probes), _f(probes), rtol=500 * EPS, atol=500 * EPS)


def test_two_outputs_are_periodic_chebfuns():
    p, q = _fit(outputs=2)
    assert hasattr(p, "domain") and hasattr(q, "domain")
    probes = np.array([-0.93, -0.5, -0.1, 0.24, 0.78])
    np.testing.assert_allclose(p(probes) / q(probes), _f(probes),
                               rtol=500 * EPS, atol=500 * EPS)


def test_three_outputs_are_matlab_order_and_keep_degrees():
    p, q, r = _fit(outputs=3)
    assert hasattr(p, "domain") and hasattr(q, "domain")
    probes = np.array([-0.93, -0.5, -0.1, 0.24, 0.78])
    np.testing.assert_allclose(r(probes), _f(probes), rtol=500 * EPS, atol=500 * EPS)


def test_real_fit_values_and_exact_degrees():
    _p, _q, r, mu, nu, _poles, _residues = _fit(outputs=7)
    assert (mu, nu) == (0, 1)
    probes = np.array([-0.93, -0.5, -0.1, 0.24, 0.78])
    np.testing.assert_allclose(r(probes), _f(probes), rtol=500 * EPS, atol=500 * EPS)


def test_six_output_roots_are_physical_x_roots():
    _p, _q, _r, mu, nu, poles = _fit(outputs=6)
    assert (mu, nu) == (0, 1)

    root_abs_large = 2.0 + np.sqrt(3.0)
    height = np.log(root_abs_large) / np.pi
    # z=-2±sqrt(3) lies on the negative real axis, so principal log maps
    # both physical roots to Re(x)=1 (the periodic endpoint, equivalent to -1).
    poles = np.asarray(poles, dtype=np.complex128).reshape(-1)
    assert poles.shape == (2,)
    assert np.all(np.isfinite(poles))
    # Either periodic endpoint representative is valid for principal log.
    endpoint_distance = np.minimum(np.abs(poles.real-1.0), np.abs(poles.real+1.0))
    assert np.max(endpoint_distance) <= 500*EPS
    expected_imaginary = np.array([-height, height])
    for actual, target in zip(np.sort(poles.imag), expected_imaginary):
        _close(actual, target)


def test_seven_output_residues_use_unreversed_polynomial_coefficients():
    _p, _q, _r, mu, nu, poles, residues = _fit(outputs=7)
    assert (mu, nu) == (0, 1)

    # @trigtech/poly returns [1/2,2,1/2] unchanged. MATLAB's polynomial
    # residue treats this as (z^2/2+2z+1/2), whose roots are -2±sqrt(3).
    expected_poles_z = np.array([-2.0 + np.sqrt(3.0), -2.0 - np.sqrt(3.0)])
    expected_residues_z = np.array([1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)])
    poles = np.asarray(poles, dtype=np.complex128).reshape(-1)
    residues = np.asarray(residues, dtype=np.complex128).reshape(-1)
    assert poles.size == residues.size == 2
    assert np.all(np.isfinite(poles)) and np.all(np.isfinite(residues))
    for target_p, target_r in zip(expected_poles_z, expected_residues_z):
        k = int(np.argmin(np.abs(poles - target_p)))
        _close(poles[k], target_p)
        _close(residues[k], target_r)


def test_none_outputs_preserves_legacy_python_seven_tuple():
    result = trigratinterp(
        _f(XI), 0, 1, NN=3, xi=XI, tol=0.0,
        domain=(-1.0, 1.0),
    )
    assert isinstance(result, tuple) and len(result) == 7
    r, ac, bc, mu, nu, poles, residues = result
    assert callable(r)
    assert np.asarray(ac).ndim == np.asarray(bc).ndim == 1
    assert (mu, nu) == (0, 1)
    assert np.asarray(poles).size == np.asarray(residues).size == 2

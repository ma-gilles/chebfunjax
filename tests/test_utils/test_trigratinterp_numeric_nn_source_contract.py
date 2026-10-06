"""Numeric sample-count defaults at the MATLAB/Python API boundary.

Provenance: MATLAB trigratinterp.m parseInputs, Chebfun commit7574c77.
Explicit outputs use minimum NN; legacy seven-tuple preserves Python's
preexisting numeric-vector length inference. Analytic bounds are independent
floating-point controls, not source fixture tolerances.
"""
import numpy as np
import pytest

from chebfunjax.utils.ratapprox import trigratinterp


@pytest.mark.parametrize("outputs", [1, 3, 7])
def test_explicit_matlab_outputs_keep_source_minimum_numeric_nn(outputs):
    nodes = np.linspace(-1.0, 1.0, 11, endpoint=False)
    values = np.cos(np.pi * nodes)
    with pytest.raises(ValueError, match="f has 11 values but NN=3"):
        trigratinterp(values, 1, 0, outputs=outputs)


def test_explicit_sample_count_fits_numeric_low_degree_data():
    nodes = np.linspace(-1.0, 1.0, 11, endpoint=False)
    values = np.cos(np.pi * nodes)
    p, q, r = trigratinterp(values, 1, 0, NN=11, outputs=3)
    probes = np.asarray([-0.93, -0.2, 0.13, 0.76])
    expected = np.cos(np.pi * probes)
    eps = np.finfo(float).eps
    np.testing.assert_allclose(r(probes), expected, rtol=0.0, atol=512*eps)
    np.testing.assert_allclose(p(probes)/q(probes), expected, rtol=0.0, atol=512*eps)

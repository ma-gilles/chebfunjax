"""Independent large-order RH hard-edge weight controls.

Provenance: MATLAB lagpts.m, Chebfun commit7574c77680d7e82b79626300bf255498271a72df.
The4e-9 relative bound is the predeclared independent RH weight envelope, not
an original MATLAB large-n assertion. It exposed failures before the phase fix.
Stored first-root weights were computed with80-digit Laguerre three-term
recurrence, four Newton steps and x/((n+1)^2*L_(n+1)(x)^2); driver/evidence:
laguerre_rh_p33_mp_weight_diagnostic_driver_20261004.py and its frozen report.
Inputs used exact binary doubles; final Newton step magnitudes below7e-79.
These are independent numerical oracles, not printed MATLAB output.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import eigh_tridiagonal

from chebfunjax.utils.quadrature import lagpts


@pytest.fixture(scope='module')
def rule10000():
    return tuple(np.asarray(a) for a in lagpts(10000))


def test_first_six_hundred_weights_against_independent_tridiagonal(rule10000):
    _, w = rule10000
    k = np.arange(1, 10001, dtype=float)
    _, vectors = eigh_tridiagonal(2*k-1, k[:-1], select='i',
                                 select_range=(0, 599), lapack_driver='stemr')
    expected = vectors[0]**2
    mask = expected > 1e-25
    assert mask.sum() > 400
    npt.assert_allclose(w[:600][mask], expected[mask], rtol=4e-9, atol=0)


@pytest.mark.parametrize(('index', 'expected'), [
    (0, 0.0003709658830217780747820523447318673187149),
    (1, 0.0008630051852172339499656756123502964232141),
    (2, 0.001354497516841036948056715410848514164776),
])
def test_first_weights_against_eighty_digit_recurrence(rule10000, index, expected):
    _, w = rule10000
    npt.assert_allclose(w[index], expected, rtol=4e-9, atol=0)

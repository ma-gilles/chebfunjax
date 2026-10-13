"""Independent assertions against the saved MATLAB Chebfun trigpoly capture."""
from pathlib import Path

import numpy as np
import numpy.testing as npt
import scipy.io

from chebfunjax.utils.trigutils import trigpoly

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "trigpoly_native" / "capture.mat"


def _records():
    return scipy.io.loadmat(_FIXTURE, squeeze_me=True, struct_as_record=False)["records"]


def _check(record, degrees, domain, column=False):
    f = trigpoly(np.asarray(degrees).reshape((-1, 1) if column else (1, -1)),
                 domain=domain)
    npt.assert_array_equal(np.asarray(f.domain.breakpoints), domain)
    assert f.is_transposed is column
    actual_coeffs = np.asarray(f.coeffs)
    expected_coeffs = np.asarray(record.coeffs)
    if expected_coeffs.ndim == 1:
        actual_coeffs = actual_coeffs.reshape(-1, order="F")
    npt.assert_array_equal(actual_coeffs, expected_coeffs)
    values = np.stack([np.asarray(f(float(x))).reshape(-1)
                       for x in np.asarray(record.query)], axis=0)
    values = values.reshape(-1, order="F")
    native_values = np.asarray(record.values).reshape(-1)
    max_error = float(np.max(np.abs(values - native_values)))
    # A 100-epsilon envelope accommodates independent FFT/evaluation paths;
    # this is a supplemental native-sample comparison, not a MATLAB predicate.
    assert max_error <= 100 * np.finfo(np.float64).eps, max_error
    print(f"native sample max abs deviation: {max_error:.17g}")
    npt.assert_allclose(np.asarray(f.point_values).reshape(-1, order="F"),
                        np.asarray(record.pointValues).reshape(-1, order="F"),
                        rtol=2e-14, atol=2e-14)


def test_scalar_native_samples_and_coefficients():
    records = _records()
    _check(records[0], [1], [-1, 1])
    _check(records[1], [2], [-1, 1])


def test_row_column_and_domain_native_contracts():
    records = _records()
    _check(records[2], [-1, 0, 2], [-1, 1])
    _check(records[3], [-1, 0, 2], [-1, 1], column=True)
    _check(records[4], [1], [2, 5])


def test_empty_native_contract():
    record = _records()[5]
    f = trigpoly(np.array([]))
    assert f.isempty() == bool(record.isempty)

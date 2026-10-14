import json
from pathlib import Path

import numpy as np
import numpy.testing as npt

from chebfunjax.utils.va import va_eval, va_orthog

CAPTURE = Path(__file__).parent / "data" / "va_native_source.json"


def decode(record):
    shape = tuple(int(i) for i in record["shape"])
    re = np.array([int(x, 16) for x in np.atleast_1d(record["real_hex"])], dtype=np.uint64).view(np.float64)
    im = np.array([int(x, 16) for x in np.atleast_1d(record["imag_hex"])], dtype=np.uint64).view(np.float64)
    out = re + 1j * im if not record["isreal"] else re
    return out.reshape(shape, order="F")


def _assert_native_storage(actual, expected):
    actual_np = np.asarray(actual)
    assert tuple(actual_np.shape) == tuple(expected["shape"])
    assert expected["class"] == "double"
    assert np.isrealobj(actual_np) == bool(expected["isreal"])
    assert actual_np.dtype == (np.float64 if expected["isreal"] else np.complex128)


def test_native_capture_cases_match_shapes_and_values():
    data = json.loads(CAPTURE.read_text())
    tol = 100 * np.finfo(np.float64).eps
    assert data["matlab_head"] == "7574c77680d7e82b79626300bf255498271a72df"
    for case in data["cases"]:
        z = decode(case["Z"])
        eval_z = decode(case["evalZ"])
        pole_records = case["Pol"]
        if isinstance(pole_records, dict):
            pole_records = [pole_records]
        groups = [decode(x).reshape(-1, order="F") for x in pole_records]
        if groups:
            hes, r = va_orthog(z, int(case["n"]), Pol=groups)
            r0, r1 = va_eval(eval_z, hes, Pol=groups, return_derivative=True)
        else:
            h, r = va_orthog(z, int(case["n"]))
            hes = (h,)
            r0, r1 = va_eval(eval_z, h, return_derivative=True)
        native_hes = case["Hes"]
        if isinstance(native_hes, dict):
            native_hes = [native_hes]
        assert [tuple(x.shape) for x in hes] == [tuple(x["shape"]) for x in native_hes]
        assert tuple(r.shape) == tuple(case["R"]["shape"])
        assert tuple(r0.shape) == tuple(case["R0"]["shape"])
        assert tuple(r1.shape) == tuple(case["R1"]["shape"])
        for actual, expected in zip(hes, native_hes):
            _assert_native_storage(actual, expected)
            npt.assert_allclose(actual, decode(expected), rtol=tol, atol=tol)
        for actual, name in ((r, "R"), (r0, "R0"), (r1, "R1")):
            _assert_native_storage(actual, case[name])
            npt.assert_allclose(actual, decode(case[name]), rtol=tol, atol=tol)


def test_polynomial_and_rational_derivative_match_independent_jax_jvp():
    import jax
    import jax.numpy as jnp

    z = jnp.linspace(-1.0, 1.0, 9)
    eval_z = jnp.linspace(-0.8, 0.8, 7)
    poles = (jnp.asarray([2.0 + 1.0j, 3.0 - 1.0j]), jnp.asarray([-2.0j]))
    hes, _ = va_orthog(z, 3, Pol=poles)
    r0, r1 = va_eval(eval_z, hes, Pol=poles, return_derivative=True)
    def scalar_eval(x):
        return va_eval(x, hes, Pol=poles)[0]
    independent = jax.vmap(jax.jacfwd(scalar_eval))(eval_z)
    tol = 100 * np.finfo(np.float64).eps
    npt.assert_allclose(r1, independent, rtol=tol, atol=tol)
    compiled = jax.jit(lambda x: va_eval(x, hes, Pol=poles, return_derivative=True))(eval_z)
    npt.assert_allclose(compiled[0], r0, rtol=tol, atol=tol)
    npt.assert_allclose(compiled[1], r1, rtol=tol, atol=tol)


def _analytic_polynomial_block(h, x):
    """Power-basis oracle, independent of the VAeval value/derivative loops."""
    from numpy.polynomial import polynomial as poly

    columns = [np.asarray([1.0])]
    for k in range(h.shape[1]):
        q = poly.polymul(columns[k], [0.0, 1.0])
        for j in range(k + 1):
            q = poly.polyadd(q, -h[j, k] * columns[j])
        columns.append(q / h[k + 1, k])
    values = np.column_stack([poly.polyval(x, c) for c in columns])
    derivatives = np.column_stack([poly.polyval(x, poly.polyder(c)) for c in columns])
    return values, derivatives


def _analytic_rational_block(h, poles, x):
    """Common-denominator power-basis oracle for a pole recurrence block."""
    from numpy.polynomial import polynomial as poly

    numerators = [np.asarray([1.0 + 0j])]
    denominator = np.asarray([1.0 + 0j])
    for k, pole in enumerate(poles):
        linear = np.asarray([-pole, 1.0], dtype=np.complex128)
        numerator = numerators[k].copy()
        for j in range(k + 1):
            numerator = poly.polyadd(numerator, -h[j, k] * poly.polymul(numerators[j], linear))
        numerator = numerator / h[k + 1, k]
        numerators = [poly.polymul(c, linear) for c in numerators]
        numerators.append(numerator)
        denominator = poly.polymul(denominator, linear)
    den = poly.polyval(x, denominator)
    den_derivative = poly.polyval(x, poly.polyder(denominator))
    values, derivatives = [], []
    for numerator in numerators[1:]:
        num = poly.polyval(x, numerator)
        num_derivative = poly.polyval(x, poly.polyder(numerator))
        values.append(num / den)
        derivatives.append((num_derivative * den - num * den_derivative) / den**2)
    return np.column_stack(values), np.column_stack(derivatives)


def test_polynomial_and_rational_derivatives_match_power_basis_oracle():
    z = np.linspace(-1.0, 1.0, 9)
    eval_z = np.asarray([-0.65, -0.1 + 0.2j, 0.35, 0.7 - 0.1j])
    poles = (np.asarray([2.0 + 1.0j, 3.0 - 1.0j]), np.asarray([-2.0j]))
    hes, _ = va_orthog(z, 3, Pol=poles)
    r0, r1 = va_eval(eval_z, hes, Pol=poles, return_derivative=True)
    poly0, poly1 = _analytic_polynomial_block(np.asarray(hes[0]), eval_z)
    value_blocks, derivative_blocks = [poly0], [poly1]
    for h, pole_group in zip(hes[1:], poles):
        v, d = _analytic_rational_block(np.asarray(h), pole_group, eval_z)
        value_blocks.append(v)
        derivative_blocks.append(d)
    tol = 100 * np.finfo(np.float64).eps
    npt.assert_allclose(r0, np.concatenate(value_blocks, axis=1), rtol=tol, atol=tol)
    npt.assert_allclose(r1, np.concatenate(derivative_blocks, axis=1), rtol=tol, atol=tol)

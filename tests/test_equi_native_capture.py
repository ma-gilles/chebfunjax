"""Small captured R2025b controls for numeric Chebfun3 and factor `equi`.

These tests use literal native outputs from the guarded captures recorded in
`fixtures/`. They are not analytic polynomial oracles: FUNQUI can choose a
rational approximant that differs visibly from the generating expression.

Provenance
----------
MATLAB source: @chebfun3/chebfun3double.m, @chebfun/chebfun.m,
    @smoothfun/smoothfun.m, tests/chebfun3/test_equiFlag.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Native release: R2025b; capture drivers and input hashes are retained under
    /scratch/gpfs/GILLES/mg6942/tmp/chebfunjax_goal_shared_20261005/
    chebfun3_equi_native_capture_20261014 and chebfun1_equi_native_capture_20261014.
"""

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "equi_native"
_SOURCE_BOUND = 100.0 * np.finfo(np.float64).eps


def _native_record(path):
    with np.load(path, allow_pickle=False) as loaded:
        values = {key: loaded[key].copy() for key in loaded.files}
    record = SimpleNamespace(**{
        key: value.item() if value.ndim == 0 else value
        for key, value in values.items() if key != "input"
    })
    return record, values["input"]


def _tech(name):
    return {"chebtech1": Chebtech1, "chebtech2": Chebtech2,
            "trigtech": Trigtech}[str(name).lower()]


def _assert_source_query(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=_SOURCE_BOUND)


@pytest.mark.parametrize(
    "filename",
    [
        "case_01_real_poly_C2.npz",
        "case_02_complex_poly_C2.npz",
        "case_03_real_poly_C1.npz",
        "case_05_scalar_equi_C2.npz",
    ],
)
def test_numeric_3d_equi_matches_literal_native_capture(filename):
    record, values = _native_record(_FIXTURES / "native3d" / filename)
    domain = tuple(np.asarray(record.domain, dtype=float).reshape(-1))
    preference = ChebfunPref()
    preference.tech = _tech(record.cols_tech)

    result = chebfun3(values, domain=domain, equi=True, pref=preference)

    assert tuple(result.rank) == tuple(np.asarray(record.rank, dtype=int))
    if filename != "case_03_real_poly_C1.npz":
        assert tuple(result.length()) == tuple(np.asarray(record.length, dtype=int))
    points = np.asarray(record.query_xyz, dtype=float)
    actual = np.asarray([result(*points[:, k]) for k in range(points.shape[1])])
    expected = np.asarray(record.query_values).reshape(-1)
    _assert_source_query(actual, expected)
    if filename == "case_05_scalar_equi_C2.npz":
        assert result.iszero() is False
        axes = [jnp.linspace(domain[2*k], domain[2*k+1], 10)
                for k in range(3)]
        mesh = jnp.meshgrid(*axes, indexing="ij")
        tensor_values = np.asarray(result(*mesh))
        assert tensor_values.shape == (10, 10, 10)
        _assert_source_query(tensor_values, np.full((10, 10, 10), 2.75))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Captured native C1 factor lengths are (2,3,54), while this candidate "
        "returns (2,3,56); native query values still match within 3.8e-15."
    ),
)
def test_numeric_3d_c1_factor_lengths_match_literal_native_capture():
    record, values = _native_record(
        _FIXTURES / "native3d" / "case_03_real_poly_C1.npz")
    domain = tuple(np.asarray(record.domain, dtype=float).reshape(-1))
    preference = ChebfunPref()
    preference.tech = _tech(record.cols_tech)
    result = chebfun3(values, domain=domain, equi=True, pref=preference)
    assert tuple(result.length()) == tuple(
        np.asarray(record.length, dtype=int))


def test_equi_validation_follows_fixed_length_sampling_and_object_dispatch():
    domain = (-1.0, 1.0, -2.0, 2.0, -3.0, 3.0)
    existing = Chebfun3.from_coeffs(
        np.ones((1, 1, 1)), domain=domain)
    assert chebfun3(existing) is existing
    with pytest.raises(ValueError, match="EQUI flag is valid only"):
        chebfun3(existing, equi=True)
    with pytest.raises(ValueError, match="EQUI flag is valid only"):
        chebfun3(lambda x, y, z: x + y + z, domain=domain, equi=True)

    sampled = chebfun3(
        lambda x, y, z: x + 2.0*y - z,
        domain=domain,
        n=(3, 4, 5),
        equi=True,
    )
    assert isinstance(sampled, Chebfun3)
    assert tuple(sampled.domain) == domain


@pytest.mark.parametrize("option", ["trig", "periodic", "pref"])
def test_numeric_equi_periodic_options_select_trig_factors(option):
    from chebfunjax.chebpref import ChebfunPref

    record, values = _native_record(
        _FIXTURES / "native3d" / "case_06_scalar_equi_trig.npz")
    domain = tuple(np.asarray(record.domain, dtype=float).reshape(-1))
    kwargs = {option: True} if option != "pref" else {}
    if option == "pref":
        pref = ChebfunPref()
        pref.tech = Trigtech
        kwargs["pref"] = pref
    result = chebfun3(values, domain=domain, equi=True, **kwargs)
    assert tuple(result.rank) == tuple(np.asarray(record.rank, dtype=int))
    assert tuple(result.length()) == tuple(np.asarray(record.length, dtype=int))
    assert all(isinstance(tech, Trigtech)
               for factors in (result.cols, result.rows, result.tubes)
               for tech in factors)


def test_equi_coeffs_flag_converts_coefficients_before_numeric_constructor():
    coefficients = jnp.arange(3*4*2, dtype=jnp.float64).reshape((3, 4, 2)) / 10
    domain = (-1.0, 1.0, -2.0, 2.0, 0.0, 3.0)
    values = Chebfun3.coeffs2vals(coefficients)
    from_coefficients = chebfun3(
        coefficients, domain=domain, equi=True, coeffs=True)
    from_values = chebfun3(values, domain=domain, equi=True)
    assert tuple(from_coefficients.rank) == tuple(from_values.rank)
    assert tuple(from_coefficients.length()) == tuple(from_values.length())
    points = ((-0.4, 0.1, 2.2), (0.3, 1.4, 0.7), (0.9, -0.6, 1.1))
    left = np.asarray([from_coefficients(*point) for point in points])
    right = np.asarray([from_values(*point) for point in points])
    _assert_source_query(left, right)

    complex_coefficients = coefficients + 1j * coefficients[::-1, :, :]
    complex_values = (
        Chebfun3.coeffs2vals(jnp.real(complex_coefficients))
        + 1j * Chebfun3.coeffs2vals(jnp.imag(complex_coefficients))
    )
    complex_values_direct = Chebfun3.coeffs2vals(complex_coefficients)
    assert np.iscomplexobj(np.asarray(complex_values_direct))
    _assert_source_query(complex_values_direct, complex_values)
    complex_roundtrip = Chebfun3.vals2coeffs(complex_values_direct)
    np.testing.assert_allclose(
        complex_roundtrip,
        complex_coefficients,
        rtol=0.0,
        atol=1000.0 * float(ChebfunPref().cheb3Prefs.chebfun3eps),
    )
    complex_from_coefficients = chebfun3(
        complex_coefficients, domain=domain, equi=True, coeffs=True)
    complex_from_values = chebfun3(
        complex_values, domain=domain, equi=True)
    complex_left = np.asarray(
        [complex_from_coefficients(*point) for point in points])
    complex_right = np.asarray(
        [complex_from_values(*point) for point in points])
    assert np.iscomplexobj(complex_left)
    _assert_source_query(complex_left, complex_right)


def test_from_equidata_uses_same_native_numeric_path():
    record, values = _native_record(
        _FIXTURES / "native3d" / "case_01_real_poly_C2.npz")
    domain = tuple(np.asarray(record.domain, dtype=float).reshape(-1))
    result = Chebfun3.from_equidata(values, domain=domain)
    points = np.asarray(record.query_xyz, dtype=float)
    actual = np.asarray([result(*points[:, k]) for k in range(points.shape[1])])
    _assert_source_query(actual, np.asarray(record.query_values).reshape(-1))


@pytest.mark.parametrize(
    "filename",
    [
        "case_01_real_matrix_poly_C2.npz",
        "case_02_complex_matrix_poly_C2.npz",
        "case_03_real_matrix_poly_C1.npz",
        "case_04_periodic_matrix_samples_C2.npz",
    ],
)
def test_public_1d_equi_pref_matches_literal_native_capture(filename):
    record, values = _native_record(_FIXTURES / "native1d" / filename)
    preference = ChebfunPref()
    preference.tech = _tech(record.pref_tech)
    preference.enableFunqui = True
    preference.chebfuneps = float(record.pref_chebfuneps)
    domain = tuple(np.asarray(record.domain, dtype=float).reshape(-1))

    result = chebfun(values, domain=domain, pref=preference)

    assert result.n_columns == values.shape[1]
    assert result.funs[0].tech.__class__ is _tech(record.tech_class)
    assert result.funs[0].tech.n == int(record.length)
    query = np.asarray(record.query_points, dtype=float).reshape(-1)
    actual = np.asarray(result(query))
    expected = np.asarray(record.query_values).reshape(actual.shape)
    _assert_source_query(actual, expected)


@pytest.mark.slow
def test_numeric_3d_trig_equi_matches_literal_native_capture():
    record, values = _native_record(
        _FIXTURES / "native3d" / "case_04_periodic_trig_equi.npz")
    domain = tuple(np.asarray(record.domain, dtype=float).reshape(-1))
    preference = ChebfunPref()
    preference.tech = Trigtech
    result = chebfun3(values, domain=domain, equi=True, pref=preference)
    assert tuple(result.rank) == tuple(np.asarray(record.rank, dtype=int))
    assert tuple(result.length()) == tuple(np.asarray(record.length, dtype=int))
    points = np.asarray(record.query_xyz, dtype=float)
    actual = np.asarray([result(*points[:, k]) for k in range(points.shape[1])])
    _assert_source_query(actual, np.asarray(record.query_values).reshape(-1))


@pytest.mark.slow
@pytest.mark.parametrize(
    "filename",
    [
        "case_05_periodic_matrix_samples_trig.npz",
        "case_06_complex_periodic_matrix_samples_trig.npz",
    ],
)
def test_public_1d_trig_equi_matches_literal_native_capture(filename):
    record, values = _native_record(_FIXTURES / "native1d" / filename)
    preference = ChebfunPref()
    preference.tech = Trigtech
    preference.enableFunqui = True
    preference.chebfuneps = float(record.pref_chebfuneps)
    domain = tuple(np.asarray(record.domain, dtype=float).reshape(-1))
    result = chebfun(values, domain=domain, pref=preference)
    assert result.n_columns == values.shape[1]
    assert result.funs[0].tech.__class__ is Trigtech
    assert result.funs[0].tech.n == int(record.length)
    query = np.asarray(record.query_points, dtype=float).reshape(-1)
    actual = np.asarray(result(query))
    expected = np.asarray(record.query_values).reshape(actual.shape)
    _assert_source_query(actual, expected)

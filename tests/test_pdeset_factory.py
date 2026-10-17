from __future__ import annotations

import io

import pytest

from chebfunjax.chebfun1d.pdeset import (
    ODE_FIELDS,
    PDE_FIELDS,
    PdeSetError,
    pdeset,
    show_pdeset_options,
)


def test_zero_input_returns_all_native_fields_empty_in_struct_order():
    options = pdeset()
    assert tuple(options) == ODE_FIELDS + PDE_FIELDS
    assert len(options) == 32
    assert all(value is None for value in options.values())


def test_names_are_case_insensitive_but_full_names_are_required():
    options = pdeset("eps", 1.25, "relTol", 0.125, "plot", True)
    assert options["Eps"] == 1.25
    assert options["RelTol"] == 0.125
    assert options["Plot"] == "on"
    with pytest.raises(PdeSetError) as pde_prefix:
        pdeset("Ep", 1)
    assert pde_prefix.value.identifier == "CHEBFUN:pdeset:unknownOption"
    with pytest.raises(PdeSetError) as ode_prefix:
        pdeset("Rel", 0.1)
    assert ode_prefix.value.identifier == "CHEBFUN:pdeset:unknownOption"


def test_values_are_deferred_and_duplicate_pairs_use_last_value():
    options = pdeset("Eps", -4, "BDF", "not-validated", "Eps", 2.0)
    assert options["Eps"] == 2.0
    assert options["BDF"] == "not-validated"


def test_old_mapping_uses_exact_case_discards_custom_fields_and_merges():
    old = pdeset()
    old.update(Eps=0.75, RelTol=0.25, Plot=True, reltol=0.5, custom=17)
    options = pdeset(old, "N", 17, "RelTol", None, "Plot", False)
    assert options["Eps"] == 0.75
    assert options["N"] == 17
    assert options["RelTol"] is None
    assert options["Plot"] == "off"
    assert "reltol" not in options
    assert "custom" not in options


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "on"),
        (False, "off"),
        (1, "on"),
        (0, "off"),
        ([True, True], "on"),
        ([True, False], "off"),
        ([], "off"),
    ],
)
def test_plot_logical_values_follow_matlab_if_conversion(value, expected):
    assert pdeset("Plot", value)["Plot"] == expected
    assert pdeset("HoldPlot", value)["HoldPlot"] == expected


def test_plot_character_values_are_retained_verbatim():
    options = pdeset("Plot", "ON", "HoldPlot", "true")
    assert options["Plot"] == "ON"
    assert options["HoldPlot"] == "true"


def test_plot_logical_conversion_matches_native_nonfinite_and_complex_cases():
    assert pdeset("Plot", float("inf"))["Plot"] == "on"
    for value in (float("nan"), [1.0, float("nan")], [0.0, float("nan")]):
        with pytest.raises(PdeSetError) as error:
            pdeset("Plot", value)
        assert error.value.identifier == "MATLAB:nologicalnan"
    with pytest.raises(PdeSetError) as error:
        pdeset("Plot", 1 + 1j)
    assert error.value.identifier == "MATLAB:nologicalcomplex"
    # Conversion covers the whole native numeric array before MATLAB if.
    with pytest.raises(PdeSetError) as error:
        pdeset("Plot", [0.0, 1 + 1j])
    assert error.value.identifier == "MATLAB:nologicalcomplex"
    # Native onoff conversion precedes odeset odd-pair/unknown-field errors.
    with pytest.raises(PdeSetError) as error:
        pdeset("Plot", float("nan"), "Eps")
    assert error.value.identifier == "MATLAB:nologicalnan"
    with pytest.raises(PdeSetError) as error:
        pdeset("Unknown", 1, "Plot", float("nan"))
    assert error.value.identifier == "MATLAB:nologicalnan"


def test_public_factory_functions_have_individual_native_provenance():
    assert "Chebfun commit: 7574c77" in pdeset.__doc__
    assert "Chebfun commit: 7574c77" in show_pdeset_options.__doc__
    assert "The MathWorks, Inc." in pdeset.__doc__
    assert "University of Oxford" in show_pdeset_options.__doc__


def test_native_error_identifiers_and_odd_pair_error():
    with pytest.raises(PdeSetError) as unknown:
        pdeset("NoSuchOption", 2)
    assert unknown.value.identifier == "CHEBFUN:pdeset:unknownOption"
    with pytest.raises(PdeSetError) as odd:
        pdeset("Eps")
    assert odd.value.identifier == "MATLAB:odeset:ArgNameValueMismatch"


def test_display_is_separate_and_preserves_ode_then_pde_order():
    stream = io.StringIO()
    show_pdeset_options(stream)
    lines = stream.getvalue().splitlines()
    labels = [line.split(":", 1)[0].strip() for line in lines if ":" in line]
    assert labels == [
        "AbsTol", "RelTol", "NormControl", "NonNegative", "OutputFcn",
        "OutputSel", "Refine", "Stats", "InitialStep", "MaxStep", "MinStep",
        "BDF", "MaxOrder", "Jacobian", "JPattern", "Vectorized", "Mass",
        "MStateDependence", "MvPattern", "MassSingular", "InitialSlope", "Events",
        "Eps", "N", "Plot", "HoldPlot", "YLim", "PlotStyle", "PDEflag",
        "AdjustBCs", "ODESolver",
    ]


def test_top_level_public_factory_aliases():
    import chebfunjax as cj

    assert cj.pdeset()["Eps"] is None
    assert callable(cj.show_pdeset_options)



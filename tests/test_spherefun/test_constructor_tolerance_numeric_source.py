"""Source tolerance propagation and numeric constructor boundary controls.

Provenance
----------
MATLAB source : @spherefun/constructor.m (parseInputs/getTol/PhaseOne and
    final simplify), tests/spherefun/test_constructor.m clauses16,25,26,30
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.

The Gaussian clause25 uses the existing spherical-callable API and explicitly
adapts source Cartesian coordinates; it is not coverage of Cartesian dispatch.
"""
import importlib

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.spherefun import Spherefun


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("shape", [(3, 2), (4, 4), (5, 4), (6, 8), (7, 6)],
                         ids=["3x2", "4x4", "5x4", "6x8", "7x6"])
def test_zero_sample_dimensions_and_metadata(shape, disabled):
    with jax.disable_jit(disabled):
        f = Spherefun.from_values(jnp.zeros(shape))
        assert f.length() == (shape[1], shape[0])
        assert f.cols[0].coeffs.shape == (shape[0],)
        assert f.rows[0].coeffs.shape == (shape[1],)
        assert bool(jnp.all(f.cols[0].coeffs == 0))
        assert bool(jnp.all(f.rows[0].coeffs == 0))
        assert bool(jnp.all(jnp.isposinf(f.pivots)))
        assert f.idx_plus == (0,) and f.idx_minus == ()
        assert not f.nonzero_poles
        assert f.pivot_locations == ((-float(jnp.pi), 0.0),)
        lam = jnp.asarray([-.73, .1, 2.9])
        theta = jnp.asarray([0., .47, jnp.pi])
        assert bool(jnp.all(f(lam, theta) == 0))


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("columns", [2, 4, 8])
def test_missing_pole_samples_exact_identifier(columns, disabled):
    with jax.disable_jit(disabled), pytest.raises(ValueError) as caught:
        Spherefun.from_values(jnp.ones((1, columns)))
    assert str(caught.value) == "CHEBFUN:SPHEREFUN:constructor:poleSamples"


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("requested", [0.0, 2.220446049250313e-16, 1e-5],
                         ids=["zero", "eps", "loose"])
def test_public_tolerance_reaches_grid_and_simplify(monkeypatch, requested, disabled):
    module = importlib.import_module("chebfunjax.spherefun.spherefun")
    get_tol = module._get_tol_sphere
    simplify = Spherefun.simplify
    records, simplifications = [], []

    def observed_tol(values, hx, hy, pseudo_level):
        result = get_tol(values, hx, hy, pseudo_level)
        records.append((values.shape, pseudo_level, result[0]))
        return result

    def observed_simplify(self, tol=None):
        simplifications.append(tol)
        return simplify(self, tol)

    monkeypatch.setattr(module, "_get_tol_sphere", observed_tol)
    monkeypatch.setattr(Spherefun, "simplify", observed_simplify)
    with jax.disable_jit(disabled):
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th) + 0 * lam,
                                    tol=requested)
        # Independently analytic: |cos(theta)| <=1 and its divided differences
        # <=1, while endpoints attain1. Thus getTol's scale is exactly1.
        effective = max(float(jnp.finfo(jnp.float64).eps), requested)
        assert records
        for shape, pseudo, absolute in records:
            assert pseudo == effective
            expected = max(shape)**(2.0 / 3.0) * float(jnp.pi) * effective
            assert abs(absolute - expected) <= 4 * float(jnp.finfo(jnp.float64).eps) * expected
        assert records[-1][2] in simplifications
        theta = jnp.linspace(0, jnp.pi, 23)
        error = jnp.max(jnp.abs(f(jnp.full_like(theta, .37), theta) - jnp.cos(theta)))
        assert error < 1e-10


@pytest.mark.parametrize("disabled", [False, True])
def test_source_clause25_eps_factor_lengths_existing_callable_api(disabled):
    def gaussian(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return jnp.exp(-10 * ((x - 1 / jnp.sqrt(2.0))**2
                             + (z - 1 / jnp.sqrt(2.0))**2 + y**2))

    with jax.disable_jit(disabled):
        f = Spherefun.from_function(gaussian)
        g = Spherefun.from_function(gaussian, tol=1e-5)
        mf, nf = f.length()
        mg, ng = g.length()
        assert (mg < mf) and (ng < nf)


@pytest.mark.parametrize("disabled", [False, True])
def test_scalar_numeric_input_precedes_pole_sample_guard(disabled):
    with jax.disable_jit(disabled):
        f = Spherefun.from_values(jnp.ones((1, 1)))
        assert (f - 1).norm(jnp.inf) == 0


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("word,expected", [
    (0x0000000000000000, True),
    (0x8000000000000000, True),
    (0x0000000000000001, False),
    (0x0000000000000002, False),
    (0x8000000000000001, False),
    (0x000FFFFFFFFFFFFF, False),
    (0x7FF0000000000000, False),
    (0x7FF8000000000001, False),
], ids=["positive-zero", "negative-zero", "min-subnormal", "two-min-subnormal",
        "negative-subnormal", "max-subnormal", "infinity", "nan"])
def test_exact_zero_classifier_binary64_words(word, expected, disabled):
    module = importlib.import_module("chebfunjax.spherefun.spherefun")
    with jax.disable_jit(disabled):
        bits = jnp.asarray([0, word, 0x8000000000000000], dtype=jnp.uint64)
        values = jax.lax.bitcast_convert_type(bits, jnp.float64)
        assert bool(jax.jit(module._all_binary64_zeros)(values)) is expected
        # No floating comparison/conversion may discard the fixture's payload.
        assert bool(jnp.all(jax.lax.bitcast_convert_type(values, jnp.uint64) == bits))


@pytest.mark.parametrize("disabled", [False, True])
def test_signed_zero_numeric_samples_preserve_dimensions(disabled):
    with jax.disable_jit(disabled):
        bits = jnp.asarray([[0, 0x8000000000000000]] * 5, dtype=jnp.uint64)
        values = jax.lax.bitcast_convert_type(bits, jnp.float64)
        f = Spherefun.from_values(values)
        assert f.length() == (2, 5)
        assert bool(jnp.all(jnp.isposinf(f.pivots)))
        assert f.idx_plus == (0,) and not f.nonzero_poles


@pytest.mark.parametrize("disabled", [False, True])
def test_two_pole_rows_precede_general_zero_branch(monkeypatch, disabled):
    module = importlib.import_module("chebfunjax.spherefun.spherefun")

    def forbidden_classifier(values):
        raise AssertionError("two-row source pole branch must precede zero branch")

    monkeypatch.setattr(module, "_all_binary64_zeros", forbidden_classifier)
    with jax.disable_jit(disabled):
        f = Spherefun.from_values(jnp.zeros((2, 4)))
        assert f.nonzero_poles
        assert f.idx_plus == (0,) and f.idx_minus == ()
        assert bool(jnp.all(f.pivots == 1))
        assert bool(jnp.all(f.cols[0].coeffs == 0))
        # Shape simplification of this legacy non-general-zero path is a
        # separately documented gap; this asserts its source routing/metadata.

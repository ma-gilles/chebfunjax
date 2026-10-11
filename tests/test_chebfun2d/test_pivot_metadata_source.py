"""Retained raw pivots and literal source operation order; unrun draft.

Provenance: @chebfun2/constructor.m and @separableApprox/{cdr,iszero,
minandmax2,mtimes,uminus,plus,diff,cumsum,transpose,pivots}.m; Chebfun7574c77.
Mocked decomposition/constructor controls isolate assembly, not solver parity.
"""
# uses-numpy: inject singular values into the inherited NumPy SVD adapter
import hashlib
import json
import os
import re
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d import _extrema_source as extrema
from chebfunjax.chebfun2d import _numeric_constructor as numeric
from chebfunjax.chebfun2d._pivot_metadata import _cdr_weights, _retained_pivots
from chebfunjax.chebfun2d.chebfun2 import (
    Chebfun2,
    _chebfun2_fixed,
    _chebfun2_from_equi,
    _chebfun2_from_matrix,
)
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v, _diff_separable, _mul_separable
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech2


@pytest.fixture(autouse=True)
def prefs():
    saved = ChebfunPref()
    ChebfunPref.setDefaults('factory')
    try:
        yield
    finally:
        ChebfunPref.setDefaults(saved)


def poly(*coeffs):
    return Chebtech2.from_coeffs(jnp.asarray(coeffs, dtype=jnp.float64))


def represented(raw, cols=None, rows=None):
    raw = jnp.atleast_1d(jnp.asarray(raw, dtype=jnp.float64))
    return Chebfun2(approx=SeparableApprox(
        cols=cols if cols is not None else [poly(1)]*len(raw),
        rows=rows if rows is not None else [poly(1)]*len(raw),
        pivots=_cdr_weights(raw), pivot_values=raw,
        domain=(-1., 1., -1., 1.)))



def real_scalar_source_reference(raw, scalar):
    """Independent binary64 rounding of exact represented division, then 1/q."""
    divisor = Fraction.from_float(float(scalar))
    quotient = [float(Fraction.from_float(float(p))/divisor)
                for p in jax.device_get(raw).tolist()]
    weights = [float(Fraction(1)/Fraction.from_float(p)) for p in quotient]
    return jnp.asarray(quotient), jnp.asarray(weights)


def assert_known_division_ir(stable, optimized):
    """Trace stable SSA dependencies; optimized text still needs manual review."""
    main = stable.split('  }', 1)[0]
    definitions = {}
    for name, opcode, operands in re.findall(
            r'(%[\w]+) = stablehlo\.(\w+) ([^\n]+)', main):
        definitions[name] = (opcode, re.findall(r'%[\w]+', operands.split(' : ')[0]))

    def strip_convert(name):
        while definitions[name][0] == 'convert':
            name = definitions[name][1][0]
        return name

    division = next(name for name, (op, args) in definitions.items()
                    if op == 'divide' and args[0] == '%arg0')
    divisor = strip_convert(definitions[division][1][1])
    assert definitions[divisor][0] == 'optimization_barrier'
    assert definitions[definitions[divisor][1][0]][0] == 'broadcast_in_dim'
    result = next(name for name, (op, args) in definitions.items()
                  if op == 'optimization_barrier' and args == [division])
    assert any(op == 'divide' and args[1] == result
               for op, args in definitions.values())
    # This pattern binds a raw parameter / broadcast divisor in optimized HLO.
    # Full recorded dependency/constants review remains a required gate.
    assert re.search(r'ROOT %[^\n]+ divide\(%param_[^,]+, %broadcast[^)]+\)', optimized)

def test_raw_getter_retains_nonrecoverable_bits():
    p = float.fromhex('0x1.8000000000001p+0')
    f = represented([p])
    assert float(f.pivot_values[0]) == p
    assert float(1/f.pivots[0]) != p
    assert bool(jnp.array_equal(f.pivots, f.approx.pivots))  # compatibility API


def test_literal_extrema_scaling_uses_raw_values():
    raw = 1.5 + jnp.arange(32, dtype=jnp.float64)*2.**-52
    class Capture:
        def __matmul__(self, other):
            return other
    rows, cols = extrema._scale_factors(Capture(), Capture(), _cdr_weights(raw), raw)
    expected = 1/jnp.sqrt(raw)
    recovered = 1/jnp.sqrt(1/_cdr_weights(raw))
    assert bool(jnp.any(expected != recovered))
    assert bool(jnp.array_equal(rows, jnp.diag(expected)))
    assert bool(jnp.array_equal(cols, jnp.diag(expected)))


def test_numeric_aca_retains_original_selected_pivots(monkeypatch):
    raw = jnp.asarray([float.fromhex('0x1.8000000000001p+0'), -3.])
    rows = jnp.eye(2)
    cols = jnp.eye(2)
    monkeypatch.setattr(numeric, 'numeric_aca', lambda *args: (
        raw, jnp.asarray([[0, 0], [1, 1]]), rows, cols, False))
    data = numeric.numeric_cdr(jnp.eye(2), (-1., 1., -1., 1.), 2.**-52,
                               ('cheb', 'cheb'))
    assert data['pivot_values'] is raw
    assert bool(jnp.array_equal(data['pivots'], _cdr_weights(raw)))


def test_numeric_public_constructor_records_negative_raw_pivot():
    value = float.fromhex('-0x1.8000000000001p+0')
    f = Chebfun2.from_values(jnp.asarray([[value, 0.], [0., 0.]]))
    assert f.rank == 1
    assert float(f.pivot_values[0]) == value
    assert bool(jnp.array_equal(f.pivots, _cdr_weights(f.pivot_values)))


@pytest.mark.parametrize('value', [3., -3., 2.+3.j])
def test_scalar_raw_pivot(value):
    f = Chebfun2.from_values(jnp.asarray(value))
    assert bool(jnp.array_equal(f.pivot_values, jnp.asarray([value])))
    assert bool(jnp.array_equal(f.pivots, _cdr_weights(f.pivot_values)))


def test_scalar_zero_and_matrix_zero_have_distinct_native_pivots():
    scalar = Chebfun2.from_values(jnp.asarray(0.))
    matrix = Chebfun2.from_values(jnp.zeros((2, 2)))
    assert bool(jnp.array_equal(scalar.pivot_values, jnp.asarray([jnp.inf])))
    assert bool(jnp.array_equal(matrix.pivot_values, jnp.asarray([0.])))
    assert bool(jnp.array_equal(scalar.pivots, matrix.pivots))
    assert extrema._iszero(scalar.approx) and extrema._iszero(matrix.approx)
    # Native magnitude infinity dominates a NaN component of a complex
    # reciprocal; reuse the already-qualified numeric constructor predicate.
    assert bool(jnp.array_equal(_cdr_weights(jnp.asarray([0j])), jnp.asarray([0j])))


def test_callable_constructor_retains_raw_pivot():
    value = float.fromhex('0x1.8000000000001p+0')
    f = Chebfun2.from_function(lambda x, y: value*jnp.ones_like(x+y))
    assert f.rank == 1 and float(f.pivot_values[0]) == value


def test_callable_zero_uses_native_infinite_pivot():
    f = Chebfun2.from_function(lambda x, y: jnp.zeros_like(x+y))
    assert bool(jnp.array_equal(f.pivot_values, jnp.asarray([jnp.inf])))
    assert bool(jnp.array_equal(f.pivots, jnp.asarray([0.])))


def test_source_zero_predicate_does_not_confuse_sanitized_reciprocal(monkeypatch):
    f = represented([0.])  # unusual nonzero factors intentionally expose ordering
    calls = []
    def mesh(*args):
        calls.append('mesh')
        return jnp.zeros((10, 10))
    monkeypatch.setattr(extrema, '_mesh_values', mesh)
    assert not extrema._iszero(f.approx)
    assert calls == ['mesh']  # raw1/p isInf, despite sanitized CDR weight0


def test_unavailable_and_stale_metadata_rejected():
    f = Chebfun2(approx=SeparableApprox(cols=[poly(1)], rows=[poly(1)],
        pivots=jnp.asarray([2.]), domain=(-1., 1., -1., 1.)))
    with pytest.raises(NotImplementedError, match='unavailable'):
        _ = f.pivot_values
    stale = SeparableApprox(cols=[poly(1)], rows=[poly(1)], pivots=jnp.asarray([2.]),
        pivot_values=jnp.asarray([2.]), domain=(-1., 1., -1., 1.))
    with pytest.raises(ValueError, match='disagrees'):
        _retained_pivots(stale)


def test_source_preserving_operations_keep_exact_pivots():
    f = represented([float.fromhex('0x1.8000000000001p+0'), -3.],
                    [poly(1), poly(0, 1)], [poly(0, 1), poly(1)])
    for g in (f.transpose(), f.diff(), f.cumsum()):
        assert bool(jnp.array_equal(g.pivot_values, f.pivot_values))
    assert bool(jnp.array_equal(f.approx.diff().pivot_values, f.pivot_values))


def test_native_negation_and_scalar_operation_order():
    raw = 1.5 + jnp.arange(32, dtype=jnp.float64)*2.**-52
    f = represented(raw)
    g = f*7.
    expected_raw, expected_weights = real_scalar_source_reference(raw, 7.)
    assert bool(jnp.array_equal(g.pivot_values, expected_raw))
    assert bool(jnp.array_equal(g.pivots, expected_weights))
    assert bool(jnp.any(g.pivots != f.pivots*7.))
    assert bool(jnp.array_equal((-f).pivot_values, -raw))


def test_zero_scalar_native_rank_reduction():
    f = represented([2., 3.], [poly(1), poly(0, 1)], [poly(1), poly(0, 1)])
    g = f*0.
    assert g.rank == 1
    assert bool(jnp.array_equal(g.pivot_values, jnp.asarray([jnp.inf])))
    assert bool(jnp.all(g.approx.cols[0].coeffs == 0))
    assert bool(jnp.all(g.approx.rows[0].coeffs == 0))


def test_unknown_scalar_adapter_does_not_fabricate_raw_pivots():
    f = Chebfun2(approx=SeparableApprox(cols=[poly(1)], rows=[poly(1)],
        pivots=jnp.asarray([2.]), domain=(-1., 1., -1., 1.)))
    assert (f*3.).approx.pivot_values is None
    assert (-f).approx.pivot_values is None
    assert (f*0.).pivot_values[0] == jnp.inf  # native zero defines a new p


def test_rank_one_product_keeps_other_native_pivots():
    f = represented([2.], [poly(1, 1)], [poly(1)])
    g = represented([3., 4.], [poly(1), poly(0, 1)], [poly(0, 1), poly(1)])
    product = f*g
    assert bool(jnp.array_equal(product.pivot_values, g.pivot_values))


def test_rank_truncation_preserves_raw_prefix(monkeypatch):
    f = represented([2., 3.])
    monkeypatch.setattr(Chebfun2, 'from_function', classmethod(lambda cls, *a, **k: f))
    out = _chebfun2_fixed(lambda x, y: x+y, f.domain, None, 1, None, False)
    assert bool(jnp.array_equal(out.pivot_values, jnp.asarray([2.])))


def test_compression_preserves_source_double_reciprocal(monkeypatch):
    f = represented([2., 3.], [poly(1), poly(0, 1)], [poly(1), poly(0, 1)])
    singular = np.asarray([float.fromhex('0x1.8000000000001p+0'), 1.])
    monkeypatch.setattr(np.linalg, 'svd', lambda *a, **k: (np.eye(2), singular, np.eye(2)))
    out = f._compress()
    expected = 1/jnp.asarray(singular)
    assert bool(jnp.array_equal(out.pivot_values, expected))
    assert bool(jnp.array_equal(out.pivots, _cdr_weights(expected)))
    assert bool(jnp.any(out.pivots != jnp.asarray(singular)))


def test_raw_assembly_and_legacy_cdr_getter_contract():
    c = chebfun(jnp.asarray([1.]))
    raw = jnp.asarray([float.fromhex('0x1.8000000000001p+0')])
    f = Chebfun2.from_pivot_values([c], raw, [c])
    assert bool(jnp.array_equal(f.pivot_values, raw))
    legacy = Chebfun2.from_cdr([c], _cdr_weights(raw), [c])
    assert legacy.approx.pivot_values is None
    assert bool(jnp.array_equal(legacy.pivots, f.pivots))


def test_unsupported_vector_adapters_explicitly_drop_metadata():
    f = represented([2.], [poly(1, 1)], [poly(1, 1)])
    assert _diff_separable(f.approx).pivot_values is None
    assert _mul_separable(f.approx, f.approx).pivot_values is None
    assert hasattr(f.sum(1), "funs")
    assert hasattr(f.approx.sum(1), "funs")


def test_pytree_jit_and_coefficient_parameter_gradient():
    def evaluate(p):
        f = SeparableApprox(cols=[poly(1)], rows=[poly(1)], pivots=_cdr_weights(p),
            pivot_values=p, domain=(-1., 1., -1., 1.))
        return f(jnp.asarray(.25), jnp.asarray(-.5))
    def evaluate_coefficient(a):
        f = SeparableApprox(cols=[Chebtech2.from_coeffs(jnp.reshape(a, (1,)))],
            rows=[poly(1)], pivots=jnp.asarray([.5]),
            pivot_values=jnp.asarray([2.]), domain=(-1., 1., -1., 1.))
        return f(jnp.asarray(.25), jnp.asarray(-.5))
    assert float(jax.grad(evaluate_coefficient)(jnp.asarray(1.))) == .5
    p = jnp.asarray([2.])
    assert float(jax.jit(evaluate)(p)) == .5
    assert bool(jnp.array_equal(jax.grad(evaluate)(p), jnp.asarray([-.25])))
    f = represented(p).approx
    leaves, structure = jax.tree_util.tree_flatten(f)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    assert bool(jnp.array_equal(restored.pivot_values, p))
    assert bool(jnp.array_equal(restored.pivots, f.pivots))


def test_empty_source_getter():
    assert Chebfun2.empty().pivot_values.shape == (0,)


@pytest.mark.parametrize('kind', ['mixed', 'equi'])
def test_legacy_ge_adapters_forward_existing_raw_pivots(kind):
    p = float.fromhex('-0x1.8000000000001p+0')
    values = jnp.asarray([[p, 0.], [0., 0.]])
    dom = (-1., 1., -1., 1.)
    f = (_chebfun2_from_matrix(values, dom, 'cheb', 'cheb') if kind == 'mixed'
         else _chebfun2_from_equi(values, dom))
    assert f.rank == 1 and float(f.pivot_values[0]) == p


def test_fixed_sizes_rank_truncation_keeps_selected_pivot():
    f = _chebfun2_fixed(lambda x, y: x+y, (-1., 1., -1., 1.),
                       (3, 3), 1, None, False)
    assert f.rank == 1
    assert bool(jnp.array_equal(f.pivot_values, jnp.asarray([-2.])))


def test_public_raw_rank_one_extrema_and_scaling_dispatch(monkeypatch):
    p = float.fromhex('0x1.8000000000001p+0')
    f = Chebfun2.from_values(jnp.asarray([[-p, p], [-p, p]]))
    seen = []
    scale = extrema._scale_factors
    def observe(rows, cols, weights, pivot_values=None):
        seen.append(pivot_values)
        return scale(rows, cols, weights, pivot_values)
    monkeypatch.setattr(extrema, '_scale_factors', observe)
    values, locations = f.minandmax2()
    assert len(seen) == 1 and bool(jnp.array_equal(seen[0], f.pivot_values))
    # Independent exact polynomial is p*x; same ordinary polynomial bound as
    # qualified extrema controls, with no turbo or native rounding claim.
    assert bool(jnp.all(jnp.abs(values-jnp.asarray([-p, p])) <= 256*2.**-52*p))
    assert bool(jnp.all(jnp.abs(locations-jnp.asarray([[-1., 0.], [1., 0.]])) <= 256*2.**-52))


def test_public_raw_rank_two_fallback():
    f = represented([1., 1.], [poly(1), poly(33/64, 1/4, 1/2)],
                    [poly(9/16, -1/2, 1/2), poly(1)])
    values, locations = f.minandmax2()
    bound = 256*2.**-52*(181/64)
    assert bool(jnp.all(jnp.abs(values-jnp.asarray([0., 181/64])) <= bound))
    assert bool(jnp.linalg.norm(locations[0]-jnp.asarray([1/4, -1/8])) <= bound**.5)
    assert bool(jnp.all(jnp.abs(locations[1]-jnp.asarray([-1., 1.])) <= bound**.5))


def test_vector_all_empty_components_native_early_return():
    # Native vector isempty is ALL(component isempty), including vacuous all.
    empty = SeparableApprox(cols=[], rows=[], pivots=jnp.asarray([]),
                            domain=(-1., 1., -1., 1.))
    vector = Chebfun2v([empty, empty])
    assert vector.isempty()
    assert Chebfun2v.empty().isempty()
    # A mixed vector is not empty; no claim about mixed-component arithmetic.
    assert not Chebfun2v([empty, represented([2.]).approx]).isempty()
    for scalar in (0., 3.):
        for result in (vector*scalar, scalar*vector):
            assert result.isempty()
            assert len(result.components) == 0
            assert type(result) is type(Chebfun2v.empty())
    full = Chebfun2v([represented([2.]).approx]*2)
    assert len((full*vector).components) == 0
    assert len((vector*full).components) == 0


@pytest.mark.parametrize('known', [False, True])
@pytest.mark.parametrize('complex_pivot', [False, True])
def test_compiled_static_scalar_and_negation_with_coefficient_ad(known, complex_pivot):
    # Frozen nonrecoverable fixture; compile dynamic p to prevent constant fold.
    raw = 1.5 + jnp.arange(32, dtype=jnp.float64)*2.**-52
    scalar = 7.
    if complex_pivot:
        raw = raw*(1.+2j)
        scalar = 7.+3j

    def construct(p, coefficient):
        return Chebfun2(approx=SeparableApprox(
            cols=[Chebtech2.from_coeffs(jnp.atleast_1d(coefficient))]*len(p),
            rows=[poly(1)]*len(p), pivots=_cdr_weights(p),
            pivot_values=p if known else None, domain=(-1., 1., -1., 1.)))

    def operation(p, coefficient):
        f = construct(p, coefficient)
        return f*scalar, -f

    compiled_operation = jax.jit(operation)
    scaled, negated = compiled_operation(raw, jnp.asarray(2.))
    if known:
        # Capture the identical callable/output graph; no observer outputs added.
        lowered = compiled_operation.lower(raw, jnp.asarray(2.))
        stable = str(lowered.compiler_ir(dialect='stablehlo'))
        optimized = lowered.compile().as_text()
        assert_known_division_ir(stable, optimized)
        directory = os.environ.get('CHEBFUN_RUNTIME_REPORT')
        if directory:
            capture = {
                'scope': 'Exact original operation outputs; IR observation only',
                'stablehlo': stable, 'optimized_hlo': optimized,
                'stablehlo_sha256': hashlib.sha256(stable.encode()).hexdigest(),
                'optimized_sha256': hashlib.sha256(optimized.encode()).hexdigest(),
                'raw_input_bytes': jax.device_get(raw).tobytes().hex(),
                'raw_output_bytes': jax.device_get(scaled.approx.pivot_values).tobytes().hex(),
                'weight_output_bytes': jax.device_get(scaled.approx.pivots).tobytes().hex(),
                'dtype': str(raw.dtype),
            }
            (Path(directory)/('pivot_known_complex_'+str(complex_pivot)+'.json')).write_text(
                json.dumps(capture, indent=2)+'\n')
    if known and not complex_pivot:
        expected_raw, expected_weights = real_scalar_source_reference(raw, scalar)
        _, expected_negated = real_scalar_source_reference(raw, -1.)
    elif known:
        # Separate native cases check complex arithmetic at their original bounds.
        # They do not establish generic7+3j scalar accuracy or MATLAB bit parity.
        # Here test only the represented raw -> reciprocal dependency, using
        # a separate dynamic primitive graph (never the production helper).
        expected_raw = scaled.approx.pivot_values
        reciprocal = jax.jit(lambda q: jax.lax.div(jnp.ones_like(q), q))
        expected_weights = reciprocal(expected_raw)
        expected_negated = reciprocal(-raw)
    else:
        # Inherited compiled-expression compatibility, not eager bit parity.
        # The v2 eager assertion failed and remains preserved in its gate.
        def legacy_expression(p):
            weights = _cdr_weights(p)
            return weights*scalar, -weights
        expected_weights, expected_negated = jax.jit(legacy_expression)(raw)
    if known and not complex_pivot:
        assert bool(jnp.any(expected_weights != _cdr_weights(raw)*scalar))
    assert bool(jnp.array_equal(scaled.approx.pivots, expected_weights))
    assert bool(jnp.array_equal(negated.approx.pivots, expected_negated))
    if known:
        if not complex_pivot:
            assert bool(jnp.array_equal(scaled.approx.pivot_values, expected_raw))
        else:
            assert scaled.approx.pivot_values.shape == raw.shape
            assert bool(jnp.all(jnp.isfinite(scaled.approx.pivot_values)))
        assert bool(jnp.array_equal(negated.approx.pivot_values, -raw))
    else:
        assert scaled.approx.pivot_values is None
        assert negated.approx.pivot_values is None

    def value(coefficient):
        scaled, negated = operation(raw, coefficient)
        return jnp.real(scaled(0., 0.) + negated(0., 0.))

    expected_derivative = jnp.real(jnp.sum(expected_weights) - jnp.sum(_cdr_weights(raw)))
    derivative = jax.jit(jax.grad(value))(jnp.asarray(2.))
    assert abs(float(derivative-expected_derivative)) <= 32*jnp.finfo(jnp.float64).eps*max(1., abs(float(expected_derivative)))

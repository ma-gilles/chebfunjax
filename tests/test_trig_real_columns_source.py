"""Per-column isReal consumers from @trigtech, pinned Chebfun7574c77.

Metadata and stored-array assignments are exact predicates. Analytic finite
Fourier evaluations use 100eps at degree <=4; no constructor-engine claim.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.trigtech import Trigtech, trig_vals2coeffs, trigpts

EPS = jnp.finfo(jnp.float64).eps


def mixed():
    x = trigpts(5)
    values = jnp.stack([jnp.cos(jnp.pi*x), jnp.exp(1j*jnp.pi*x)], axis=1)
    return Trigtech.from_values(values)


def public(f):
    return Chebfun(funs=[_Piece(tech=f, interval=(-1., 1.))], domain=Domain((-1., 1.)))


def assert_mask(f, expected):
    assert f.real_columns == expected
    assert f.is_real is all(expected)
    assert jnp.array_equal(f.isReal, jnp.asarray(expected, dtype=jnp.bool_))


@pytest.mark.parametrize('imag,expected', [(2., True), (3., True), (4., False)])
def test_populate_three_eps_boundary(imag, expected):
    # populate numeric branch: coefficients precede real-value projection.
    values = jnp.full((5, 1), 1 + 1j*imag*EPS)
    f = Trigtech.from_values(values)
    assert_mask(f, (expected,))
    assert jnp.array_equal(f.coeffs, trig_vals2coeffs(values))
    assert jnp.array_equal(f.values, jnp.real(values) if expected else values)


@pytest.mark.parametrize('n', [4, 5])
def test_fixed_mixed_columns_and_endpoint_values(n):
    f = Trigtech.from_function(
        lambda x: jnp.stack([2+x, 1j*(3+x)], axis=1), n=n)
    values = jnp.stack([2+trigpts(n), 1j*(3+trigpts(n))], axis=1)
    values = values.at[0].set(jnp.asarray([2., 3j]))
    assert_mask(f, (True, False))
    assert jnp.array_equal(f.values, values)


@pytest.mark.parametrize('as_complex', [False, True])
def test_adaptive_probe_storage_not_zero_imaginary(as_complex):
    # JAX homogeneous storage adapter; no scalar MATLAB indexing capture.
    f = Trigtech.from_function(lambda x: jnp.ones_like(x)*(1+0j if as_complex else 1))
    assert_mask(f, (not as_complex,))
    assert jnp.max(jnp.abs(f(jnp.asarray([.13, .42]))-1)) <= 10*EPS


def test_raw_compatibility_mask_priority_and_shape_error():
    c = jnp.ones((3, 2), dtype=jnp.complex128)
    assert_mask(Trigtech(coeffs=c, is_real=False), (False, False))
    assert_mask(Trigtech(coeffs=c, real_columns=(True, False)), (True, False))
    with pytest.raises(ValueError, match='coefficient columns'):
        Trigtech(coeffs=c, real_columns=(True,))
    assert_mask(Trigtech.empty(), ())


def test_public_aggregate_and_direct_horner_distinction():
    c = jnp.asarray([[1+1e-5j, 2j]])
    f = Trigtech(coeffs=c, real_columns=(True, False))
    x = jnp.asarray([-.4, .2])
    public_values = f(x)
    direct = Trigtech.horner(x, c, (True, False))
    assert jnp.array_equal(public_values, jnp.broadcast_to(c, (2, 2)))
    assert jnp.array_equal(direct, jnp.asarray([[1., 2j], [1., 2j]]))


def test_mixed_pytree_jit_vmap_and_ad():
    f = mixed()
    leaves, definition = jax.tree_util.tree_flatten(f)
    restored = jax.tree_util.tree_unflatten(definition, leaves)
    assert_mask(restored, (True, False))
    x = jnp.asarray([-.31, .13, .61])
    evaluated = eqx.filter_jit(lambda tech, points: tech(points))(restored, x)
    expected = jnp.stack([jnp.cos(jnp.pi*x), jnp.exp(1j*jnp.pi*x)], axis=1)
    assert jnp.max(jnp.abs(evaluated-expected)) <= 100*EPS
    mapped = jax.vmap(lambda z: restored(z))(x)
    assert jnp.max(jnp.abs(mapped-expected)) <= 100*EPS
    derivative = jax.grad(lambda z: jnp.real(restored(z)[0]))(.13)
    assert jnp.abs(derivative + jnp.pi*jnp.sin(jnp.pi*.13)) <= 100*EPS
    rebuilt = eqx.filter_jit(lambda tech: -tech)(restored)
    assert_mask(rebuilt, (True, False))


@pytest.mark.parametrize('operation', ['prolong', 'simplify', 'diff', 'neg', 'flipud'])
def test_shape_preserving_operations_keep_mixed_mask(operation):
    f = mixed()
    g = {'prolong': lambda: f.prolong(7), 'simplify': f.simplify,
         'diff': f.diff, 'neg': lambda: -f, 'flipud': f.flipud}[operation]()
    assert_mask(g, (True, False))


def test_column_flip_and_cumsum_keep_native_mask_and_literal_cache():
    f = mixed()
    flipped = f.fliplr()
    assert_mask(flipped, (True, False))
    assert jnp.array_equal(flipped.values, f.values[:, ::-1])
    summed = flipped.cumsum(dim=2)
    assert_mask(summed, (True, False))
    assert jnp.array_equal(summed.values, jnp.cumsum(flipped.values, axis=1))
    # Extraction must not reproject the intentionally unchanged source mask.
    assert jnp.array_equal(flipped.extract_column(0).values, f.values[:, 1])


@pytest.mark.parametrize('k,expected', [(1, (True, False, True)), (2, (False, False))])
def test_dim2_difference_literal_adjacent_equality(k, expected):
    values = jnp.asarray([[1j, 2j, 3., 4.], [2j, 3j, 4., 5.]])
    f = Trigtech.from_values(values)
    assert_mask(f, (False, False, True, True))
    g = f.diff(k, dim=2)
    assert_mask(g, expected)
    assert jnp.array_equal(g.values, jnp.diff(f.values, n=k, axis=1))


@pytest.mark.parametrize('n', [4, 5])
def test_conjugation_exact_native_column_assignment(n):
    c = jnp.arange(n*2).reshape(n, 2)*(1+2j)
    v = jnp.arange(n*2).reshape(n, 2)*(3+4j)
    f = Trigtech(coeffs=c, real_columns=(True, False), _values=v)
    g = f.conj()
    assert_mask(g, (True, False))
    assert jnp.array_equal(g.coeffs[:, 0], c[:, 0])
    assert jnp.array_equal(g.coeffs[:, 1], jnp.conj(c[::-1, 1]))
    assert jnp.array_equal(g.values[:, 0], v[:, 0])
    assert jnp.array_equal(g.values[:, 1], jnp.conj(v[:, 1]))


@pytest.mark.parametrize('op', ['add', 'multiply', 'divide'])
def test_numeric_storage_predicate_and_broadcast(op):
    f = mixed()
    apply = {'add': lambda z: f+z, 'multiply': lambda z: f*z,
             'divide': lambda z: f/z}[op]
    assert_mask(apply(2.), (True, False))
    assert_mask(apply(2+0j), (False, False))
    assert_mask(apply(jnp.asarray([2., 3.])), (True, False))


def test_singleton_column_broadcast_and_binary_masks():
    f = mixed()
    real = f.extract_column(0)
    assert_mask(real+jnp.asarray([1., 2.]), (True, True))
    assert_mask(real*jnp.asarray([1., 2.]), (True, True))
    assert_mask(f+real, (True, False))
    assert_mask(f*real, (True, False))
    assert_mask(f*f.conj(), (True, True))


def test_reconstructing_powers_use_probe_storage():
    f = mixed()
    g = f**2
    # Public feval passes all(isReal), so a mixed complex probe stays complex.
    assert_mask(g, (False, False))
    x = jnp.asarray([-.3, .2])
    assert jnp.max(jnp.abs(g(x)-f(x)**2)) <= 100*EPS


@pytest.mark.parametrize('operation', ['real', 'imag', 'abs'])
def test_real_outputs_have_all_real_mask(operation):
    f = mixed()
    g = {'real': f.real, 'imag': f.imag, 'abs': lambda: abs(f)}[operation]()
    assert_mask(g, (True, True))


def test_integral_pairwise_inner_and_matrix_aggregate():
    f = Trigtech(coeffs=jnp.asarray([[1+1e-5j, 2j]]), real_columns=(True, False))
    assert jnp.array_equal(f.sum(), jnp.asarray([2., 4j]))
    assert_mask(f.sum(dim=2), (False,))
    g = Trigtech(coeffs=jnp.asarray([[3+2e-5j, 4j]]), real_columns=(True, False))
    ip = f.innerProduct(g)
    raw = 2*jnp.conj(f.coeffs).T@g.coeffs
    assert jnp.array_equal(ip, raw.at[0, 0].set(jnp.real(raw[0, 0])))
    assert_mask(f@jnp.eye(2), (False, False))


def test_columns_cat_assignment_and_deletion():
    f = mixed().fliplr()
    blocks = f.mat2cell([1, 1])
    assert_mask(blocks[0], (True,))
    assert_mask(blocks[1], (False,))
    joined = Trigtech.cell2mat(blocks)
    assert_mask(joined, (True, False))
    assert jnp.array_equal(joined.values, f.values)
    assigned = f.assign_columns([0], blocks[1])
    assert_mask(assigned, (False, False))
    assert jnp.array_equal(assigned.values[:, 0], blocks[1].values)
    assert_mask(f.assign_columns([0], None), (False,))


def test_public_factories_preserve_masks_and_cached_columns():
    f = public(mixed().fliplr())
    sub = f.extract_columns([1, 0])
    assert_mask(sub.funs[0].tech, (False, True))
    assert jnp.array_equal(sub.funs[0].tech.values, f.funs[0].tech.values[:, ::-1])
    repeated = f.repmat(1, 2)
    assert_mask(repeated.funs[0].tech, (True, False, True, False))
    assert jnp.array_equal(repeated.funs[0].tech.values, jnp.tile(f.funs[0].tech.values, (1, 2)))
    transposed = f.T
    assert_mask(transposed.funs[0].tech, (True, False))
    joined = Chebfun.horzcat(f, f)
    assert_mask(joined.funs[0].tech, (True, False, True, False))


def test_public_growth_zero_columns_and_assignment():
    f = public(mixed())
    grown = f.assign_columns([3], f.extract_columns(1))
    assert_mask(grown.funs[0].tech, (True, False, True, False))
    assert jnp.array_equal(grown.funs[0].tech.values[:, 2], jnp.zeros(grown.funs[0].tech.n))


def test_from_coefficients_inference_and_explicit_traced_mask():
    f = mixed()
    assert_mask(Trigtech.from_coeffs(f.coeffs), (True, False))
    rebuilt = eqx.filter_jit(lambda c: Trigtech.from_coeffs(
        c, real_columns=(True, False)))(f.coeffs)
    assert_mask(rebuilt, (True, False))
    assert jnp.array_equal(rebuilt.coeffs, f.coeffs)
    unknown = eqx.filter_jit(lambda c: Trigtech.from_coeffs(c))(f.coeffs)
    assert_mask(unknown, (False, False))


def test_nan_numeric_zero_division_classification():
    f = mixed()
    assert_mask(f/0., (False, False))
    assert_mask(Trigtech.mrdivide(f, 0.), (False, False))


def test_scalar_convolution_and_array_rejection():
    f = mixed()
    assert_mask(f.extract_column(0).circconv(f.extract_column(0)), (True,))
    with pytest.raises(ValueError, match='array-valued'):
        f.circconv(f)


def test_public_derivative_antiderivative_and_conjugate_factories():
    f = public(mixed())
    assert_mask(f.diff().funs[0].tech, (True, False))
    assert_mask(f.cumsum().funs[0].tech, (True, False))
    conjugate = f.conj()
    assert_mask(conjugate.funs[0].tech, (True, False))
    assert jnp.array_equal(conjugate.funs[0].tech.coeffs, f.funs[0].tech.conj().coeffs)
    assert_mask(f.ctranspose().funs[0].tech, (True, False))


def test_qr_repeats_source_aggregate_flag():
    f = mixed()
    q, r = f.qr()
    assert_mask(q, (False, False))
    x = jnp.asarray([-.23, .41])
    assert jnp.max(jnp.abs(q(x)@r-f(x))) <= 100*EPS


def test_scalar_extracted_range_uses_own_realness():
    f = mixed()
    (low, _), (high, _) = f.extract_column(0).minandmax()
    assert jnp.abs(low+1) <= 100*EPS
    assert jnp.abs(high-1) <= 100*EPS



def test_direct_horner_accepts_exposed_eager_logical_mask():
    c = jnp.asarray([[1+1e-5j, 2j]])
    f = Trigtech(coeffs=c, real_columns=(True, False))
    x = jnp.asarray([-.4, .2])
    expected = jnp.asarray([[1., 2j], [1., 2j]])
    assert jnp.array_equal(Trigtech.horner(x, c, f.isReal), expected)
    assert jnp.array_equal(Trigtech.horner(x, c, jnp.asarray([False])),
                           jnp.broadcast_to(c, (2, 2)))
    with pytest.raises(ValueError, match="one-dimensional logical array"):
        Trigtech.horner(x, c, jnp.asarray([[True, False]]))


def test_direct_horner_traced_mask_contract_and_static_tuple():
    f = mixed()
    x = jnp.asarray([-.2, .4])
    static = jax.jit(lambda c, z: Trigtech.horner(z, c, f.real_columns))(f.coeffs, x)
    eager = Trigtech.horner(x, f.coeffs, f.isReal)
    assert jnp.max(jnp.abs(static-eager)) <= 100*EPS
    with pytest.raises(TypeError, match="requires a static realness mask"):
        jax.jit(lambda mask: Trigtech.horner(x, f.coeffs, mask))(f.isReal)


@pytest.mark.parametrize('shape', [(1,), (1, 1)])
@pytest.mark.parametrize('columns', [False, True])
def test_numeric_division_one_element_row_is_source_scalar(shape, columns):
    f = mixed() if columns else mixed().extract_column(0)
    divisor = jnp.asarray(2.).reshape(shape)
    result = f/divisor
    assert result.coeffs.shape == f.coeffs.shape
    assert result.values.shape == f.values.shape
    assert_mask(result, f.real_columns)
    assert jnp.array_equal(result.coeffs, f.coeffs/2.)
    assert jnp.array_equal(result.values, f.values/2.)



def test_direct_horner_compiled_default_and_python_scalar_flags():
    c = jnp.asarray([[1+1e-5j, 2j]])
    x = jnp.asarray([-.4, .2])
    expected = jnp.broadcast_to(c, (2, 2))
    default = jax.jit(lambda coeffs, points: Trigtech.horner(points, coeffs))(c, x)
    assert jnp.array_equal(default, expected)
    for flag in (False, True, 0, 1):
        result = jax.jit(lambda coeffs, points: Trigtech.horner(
            points, coeffs, flag))(c, x)
        assert jnp.array_equal(result, jnp.real(expected) if flag else expected)

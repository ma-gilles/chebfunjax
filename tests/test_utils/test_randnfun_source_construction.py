"""Independent source-construction and explicit JAX random API contracts.

Injected numbers are chosen analytical inputs, NOT claimed MATLAB randn outputs.

Provenance
----------
MATLAB source : randnfun.m, tests/misc/test_randnfun.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils import _randnfun as engine


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('cmplx,big', [(False, False), (True, False),
                                       (False, True), (True, True)])
def test_injected_modes_multicolumn_and_normalization(monkeypatch, disabled,
                                                    cmplx, big):
    # L=4, lambda2 -> modes -2..2. Independent scalar source-index expansion.
    draws = np.arange(1, 21, dtype=float).reshape(5, 4).T / 8
    calls = []

    def draw(key, rows, columns):
        calls.append((rows, columns))
        return jnp.asarray(draws)

    monkeypatch.setattr(engine, '_normal_draw', draw)
    with jax.disable_jit(disabled):
        f = engine.randnfun(2, 2, [3, 7], 'trig', big=big, cmplx=cmplx, seed=7)
    assert calls == [(4, 5)]
    expected = np.empty((5, 2), dtype=complex)
    order = [4, 2, 0, 1, 3]
    for k, source in enumerate(order):
        for col in range(2):
            expected[k, col] = (draws[col, source] + 1j*draws[col+2, source])/np.sqrt(2)
    if not cmplx:
        expected = (expected + expected[::-1].conj()) / np.sqrt(2)
    expected /= np.sqrt(4 if big else 5)
    np.testing.assert_allclose(f.funs[0].tech.coeffs, expected, rtol=3e-15, atol=0)
    assert f.n_columns == 2
    assert tuple(f.domain.breakpoints) == (3, 7)


@pytest.mark.parametrize('disabled', [False, True])
def test_nonperiodic_source_halfway_sampling_and_pointvalues(monkeypatch, disabled):
    seen = []

    def draw(key, rows, columns):
        seen.append((rows, columns))
        # A pure zero-frequency real coefficient: independent constant solution.
        values = jnp.zeros((rows, columns))
        return values.at[0, 0].set(3.0)

    monkeypatch.setattr(engine, '_normal_draw', draw)
    import chebfunjax.utils.quadrature as quadrature
    original = quadrature.chebpts_ab
    sample_sizes = []

    def points(n, *args, **kwargs):
        sample_sizes.append(n)
        return original(n, *args, **kwargs)

    monkeypatch.setattr(quadrature, 'chebpts_ab', points)
    with jax.disable_jit(disabled):
        f = engine.randnfun(0.8, [-1, 1], seed=5)  # L/lambda=2.5 -> round=3
    assert sample_sizes == [35]
    assert seen == [(2, 9)]  # extended L=3.6, floor(L/lambda)=4
    assert f.funs[0].tech.coeffs.dtype == jnp.float64
    np.testing.assert_allclose(f.funs[0].tech.coeffs[0], 1.0, rtol=3e-15, atol=0)
    np.testing.assert_allclose(f._point_values, np.ones((2, 1)), rtol=3e-15, atol=0)


@pytest.mark.parametrize('disabled', [False, True])
def test_infinite_wavelength_source_scalar_ignores_count(monkeypatch, disabled):
    values = iter([2.0, 4.0])
    monkeypatch.setattr(engine, '_normal_draw',
                        lambda key, rows, columns: jnp.full((rows, columns), next(values)))
    with jax.disable_jit(disabled):
        f = engine.randnfun(jnp.inf, 3, [2, 5], 'complex', seed=1)
    assert f.n_columns == 1  # literal early return, not independent three columns
    np.testing.assert_allclose(f.funs[0].tech.coeffs, [(2+4j)/np.sqrt(2)], rtol=1e-15)


@pytest.mark.parametrize('disabled', [False, True])
def test_zero_mode_periodic_no_forced_minimum_bandwidth(monkeypatch, disabled):
    monkeypatch.setattr(engine, '_normal_draw',
                        lambda key, rows, columns: jnp.ones((rows, columns)))
    with jax.disable_jit(disabled):
        f = engine.randnfun(100, 'trig', seed=0)
    assert f.funs[0].tech.coeffs.shape == (1,)
    np.testing.assert_allclose(f.funs[0].tech.coeffs, [1], rtol=3e-15)


@pytest.mark.parametrize('disabled', [False, True])
def test_public_routes_keys_seeds_and_default_advance(monkeypatch, disabled):
    import chebfunjax as cj
    from chebfunjax.chebfun1d.randfuns import randnfun as source_route
    from chebfunjax.utils.randnfun import randnfun as periodic_route

    monkeypatch.setattr(engine, '_DEFAULT_KEY', jax.random.key(19))
    with jax.disable_jit(disabled):
        f = cj.randnfun(1, 'trig', seed=5)
        g = source_route(1, 'trig', seed=5)
        h = periodic_route(1, seed=5)
        explicit = cj.randnfun(1, 'trig', key=jax.random.key(5))
        np.testing.assert_array_equal(jax.random.key_data(engine._DEFAULT_KEY),
                                      jax.random.key_data(jax.random.key(19)))
        first = cj.randnfun(1, 'trig')
        second = cj.randnfun(1, 'trig')
        default = cj.randnfun(seed=5)
    for other in (g, h, explicit):
        np.testing.assert_array_equal(f.funs[0].tech.coeffs, other.funs[0].tech.coeffs)
    assert not np.array_equal(first.funs[0].tech.coeffs, second.funs[0].tech.coeffs)
    assert default.funs[0].tech.__class__.__name__ == 'Chebtech2'
    assert tuple(default.domain.breakpoints) == (-1, 1)
    with pytest.raises(ValueError, match='key or seed'):
        cj.randnfun(key=jax.random.key(1), seed=1)


@pytest.mark.parametrize('disabled', [False, True])
def test_construction_avoids_legacy_numpy_transform_and_rng(monkeypatch, disabled):
    import chebfunjax.tech.trigtech as trigtech
    import chebfunjax.utils.transforms as transforms

    def forbidden(*args, **kwargs):
        raise AssertionError('legacy numerical delegate reached')

    monkeypatch.setattr(np.random, 'randn', forbidden)
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    for name in ('vals2coeffs', 'coeffs2vals', '_vals2coeffs_np', '_coeffs2vals_np'):
        monkeypatch.setattr(transforms, name, forbidden)
    monkeypatch.setattr(trigtech, '_trig_eval_np', forbidden)
    with jax.disable_jit(disabled):
        a = engine.randnfun(1, 'trig', seed=17)
        b = engine.randnfun(1, seed=17)
    assert a.funs[0].tech.coeffs.size > 0
    assert b.funs[0].tech.coeffs.size > 0


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('cmplx', [False, True])
def test_nonperiodic_single_fourier_mode_accuracy(monkeypatch, disabled, cmplx):
    def draw(key, rows, columns):
        assert (rows, columns) == (2, 7)
        raw = jnp.zeros((rows, columns))
        return raw.at[0, 1:3].set(jnp.sqrt(7.0)/2)

    monkeypatch.setattr(engine, '_normal_draw', draw)
    with jax.disable_jit(disabled):
        f = engine.randnfun(2, seed=9, cmplx=cmplx)
    x = jnp.asarray([-.83, -.22, .17, .79])
    # Enlarged interval [-1,5], Fourier mode1 gives cos(pi*(x-2)/3).
    expected = jnp.cos(jnp.pi*(x-2)/3) / (jnp.sqrt(2.0) if cmplx else 1.0)
    np.testing.assert_allclose(f(x), expected,
                               rtol=0, atol=2e-13)


@pytest.mark.parametrize('disabled', [False, True])
def test_nonperiodic_array_columns_remain_joint(monkeypatch, disabled):
    def draw(key, rows, columns):
        assert (rows, columns) == (4, 9)
        return jnp.zeros((rows, columns)).at[:2, 0].set(jnp.asarray([3., 6.]))

    monkeypatch.setattr(engine, '_normal_draw', draw)
    with jax.disable_jit(disabled):
        f = engine.randnfun(.8, 2, [-1, 1], seed=3)
    assert f.n_columns == 2
    np.testing.assert_allclose(f.funs[0].tech.coeffs[0], [1., 2.], rtol=3e-15, atol=0)
    np.testing.assert_allclose(f._point_values, [[1., 2.], [1., 2.]], rtol=3e-15, atol=0)


def test_infinite_wavelength_big_source_zero_and_draw_count(monkeypatch):
    calls = []

    def draw(key, rows, columns):
        calls.append((rows, columns))
        return jnp.asarray([[-2.]])

    monkeypatch.setattr(engine, '_normal_draw', draw)
    f = engine.randnfun(jnp.inf, 'big', seed=1)
    assert calls == [(1, 1)]
    assert float(f.funs[0].tech.coeffs[0]) == 0.
    assert bool(jnp.signbit(f.funs[0].tech.coeffs[0]))


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('periodic', [False, True])
def test_mixed_real_complex_zero_columns_use_joint_source_horner(monkeypatch, disabled, periodic):
    tiny = np.finfo(float).eps

    def draw(key, rows, columns):
        assert rows == 6
        amplitude = jnp.sqrt(2.0 * columns)
        raw = jnp.zeros((rows, columns))
        return (raw.at[0, 0].set(amplitude)
                .at[3, 0].set(amplitude * tiny)
                .at[4, 0].set(2 * amplitude))

    monkeypatch.setattr(engine, '_normal_draw', draw)
    with jax.disable_jit(disabled):
        f = engine.randnfun(100., 3, trig=periodic, cmplx=True, seed=7)
        actual = f(jnp.asarray([-.7, .2, .8]))
    expected = np.tile([1 + 1j*tiny, 2j, 0], (3, 1))
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)
    assert np.all(np.imag(np.asarray(actual)[:, 0]) > 0)
    # Source trigtech.feval passes all(isReal), so the real-classified first
    # column is NOT independently real-Horner projected in this mixed matrix.
    if periodic:
        assert not f.funs[0].tech.is_real
    assert f.n_columns == 3


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('kind', ['all-zero', 'zero-and-constant', 'two-nonzero'])
def test_nonperiodic_column_tolerances_and_zero_envelopes(monkeypatch, disabled, kind):
    from chebfunjax.tech import chebtech

    observed = []
    original = chebtech._chop_columns

    def capture(coefficients, tol):
        observed.append(np.asarray(tol))
        return original(coefficients, tol)

    monkeypatch.setattr(chebtech, '_chop_columns', capture)
    constants = {'all-zero': [0., 0.], 'zero-and-constant': [0., 2.],
                 'two-nonzero': [1., 2.]}[kind]
    with jax.disable_jit(disabled):
        values = jnp.tile(jnp.asarray(constants), (25, 1))
        f = engine._from_nonperiodic_values(values, (-1., 1.))
    assert f.funs[0].tech.coeffs.shape == (1, 2)
    np.testing.assert_allclose(f.funs[0].tech.coeffs[0], constants, rtol=3e-15, atol=0)
    assert len(observed) == 1
    for value, tol in zip(constants, observed[0], strict=True):
        if value == 0:
            assert np.isnan(tol)
        else:
            assert tol == 1e-13


def test_source_nan_parser_defaults_and_overwrite():
    assert engine._parse((float('nan'), .8))[:3] == (.8, 1, (-1., 1.))
    assert engine._parse((.8, float('nan')))[:3] == (.8, 1, (-1., 1.))
    assert engine._parse(([float('nan'), float('nan')],))[:3] == (1., 1, (-1., 1.))


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('periodic,wavelength,empty', [
    (False, 1., True), (True, 1., True), (False, float('inf'), False),
], ids=['nonperiodic-zero-count', 'periodic-zero-count', 'infinite-ignores-zero-count'])
def test_zero_count_source_dispatch_and_draw_consumption(monkeypatch, disabled, periodic, wavelength, empty):
    monkeypatch.setattr(engine, '_DEFAULT_KEY', jax.random.key(31))
    before = np.asarray(jax.random.key_data(engine._DEFAULT_KEY)).copy()
    with jax.disable_jit(disabled):
        f = engine.randnfun(wavelength, 0, trig=periodic)
    assert f.isempty() is empty
    after = np.asarray(jax.random.key_data(engine._DEFAULT_KEY))
    assert np.array_equal(before, after) is empty


@pytest.mark.parametrize('disabled', [False, True])
def test_nonperiodic_endpoint_metadata_uses_full_coefficient_sums(monkeypatch, disabled):
    from chebfunjax.utils import transforms

    captured = []
    original = transforms._vals2coeffs_jax

    def capture(values):
        coefficients = original(values)
        captured.append(np.asarray(coefficients))
        return coefficients

    monkeypatch.setattr(transforms, '_vals2coeffs_jax', capture)
    with jax.disable_jit(disabled):
        values = jnp.reshape(jnp.sin(jnp.arange(66, dtype=jnp.float64) / 7), (33, 2))
        f = engine._from_nonperiodic_values(values, (-2., 3.))
    coefficients = captured[0]
    # Source lval/rval act on the full numeric constructor before simplify.
    left = jnp.sum(jnp.asarray(coefficients) * jnp.asarray([(-1)**k for k in range(33)])[:, None], axis=0)
    right = jnp.sum(jnp.asarray(coefficients), axis=0)
    np.testing.assert_array_equal(f._point_values, jnp.stack((left, right)))

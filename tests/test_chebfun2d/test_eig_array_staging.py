"""Native array representation and independent continuous arithmetic controls.

Sources: @separableApprox/svd.m, @chebfun2/eig.m,
@chebfun/{mtimes,innerProduct}.m; Chebfun commit 7574c77.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebfun2d._svd import _as_fun, source_svd
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def panel(coeffs, interval=(-1., 1.)):
    return _as_fun(Chebtech2(coeffs=jnp.asarray(coeffs)), interval)


@pytest.mark.parametrize('interval', [(-1., 1.), (2., 5.)])
def test_array_gram_matches_exact_monomial_integrals(interval):
    # Columns [1,t,t^2] and [1+2it,t^3] on the canonical coordinate t.
    f = panel([[1., 0., .5], [0., 1., 0.], [0., 0., .5]], interval)
    g = panel([[1., 0.], [2j, .75], [0., 0.], [0., .25]], interval)
    expected = jnp.array([[2., 0.], [4j/3, 2/5], [2/3, 0.]])
    expected *= (interval[1]-interval[0])/2
    np.testing.assert_allclose(f.T @ g, expected, atol=3e-15, rtol=0)
    np.testing.assert_allclose(g.H @ f, jnp.conj(expected.T), atol=3e-15, rtol=0)
    np.testing.assert_allclose(g.T @ f, expected.T, atol=3e-15, rtol=0)


def test_piecewise_array_gram_overlaps_breakpoints():
    # Polynomial functions stored on distinct breakpoint sets.
    f = chebfun(lambda x: jnp.stack([1+0*x, x], axis=-1), domain=(-1., 0., 1.))
    g = chebfun(lambda x: jnp.stack([x, x*x], axis=-1), domain=(-1., .25, 1.))
    np.testing.assert_allclose(f.T @ g, [[0., 2/3], [2/3, 0.]],
                               atol=3e-15, rtol=0)


def test_singleton_constant_array_shape():
    f = panel([[2.]])
    g = panel([[3., 4.]])
    product = f.T @ g
    assert product.shape == (1, 2)
    np.testing.assert_array_equal(product, [[12., 16.]])


def kernel():
    c = [chebfun(lambda x: 1+0*x), chebfun(lambda x: x)]
    r = [chebfun(lambda x: 1+x), chebfun(lambda x: 2+x)]
    return Chebfun2.from_cdr(c, jnp.array([2., 3.]), r)


def test_svd_array_option_preserves_public_factors():
    f = kernel()
    left, s, right = source_svd(f.approx, full=True)
    u, sa, v = source_svd(f.approx, full=True, as_array=True)
    assert isinstance(left, list) and isinstance(right, list)
    assert isinstance(u, Chebfun) and isinstance(v, Chebfun)
    np.testing.assert_array_equal(s, sa)
    x = jnp.array([-.7, .1, .8])
    np.testing.assert_array_equal(u(x), jnp.column_stack([c(x) for c in left]))
    np.testing.assert_array_equal(v(x), jnp.column_stack([c(x) for c in right]))


def test_eig_retains_panels_until_complete_mandatory_lifting(monkeypatch):
    from chebfunjax.chebfun2d import _eig

    original_svd = _eig.source_svd
    original = Chebfun.mat2cell
    splits = []

    def record_split(self, *args, **kwargs):
        splits.append(self)
        return original(self, *args, **kwargs)

    def svd_then_observe(*args, **kwargs):
        result = original_svd(*args, **kwargs)
        # Existing QR adapts incoming columns internally. The contract being
        # changed here begins at the completed SVD's array-valued outputs.
        splits.clear()
        return result

    monkeypatch.setattr(Chebfun, 'mat2cell', record_split)
    monkeypatch.setattr(_eig, 'source_svd', svd_then_observe)
    f = kernel()
    splits.clear()
    values, lifted = _eig.source_eig(f.approx, normalize=False)
    # Native performs U*S*V even for eigenvalues-only output. All splits are
    # forbidden before the final public-output adaptation of lifted columns.
    assert len(splits) == 1
    assert splits[0].n_columns == 2
    expected = jnp.array([3-jnp.sqrt(17.), 3+jnp.sqrt(17.)])
    np.testing.assert_allclose(jnp.sort(jnp.real(values)), expected,
                               atol=2e-13, rtol=0)
    assert float((f @ lifted-lifted @ jnp.diag(values)).norm()) < 2e-12
    assert all(float(c.norm()) > 0 for c in lifted.cols)


@pytest.mark.parametrize('pad', [0, 3])
def test_complex_self_gram_and_post_prolong_equality(pad):
    import jax

    # [1+it, (2-i)+(3+2i)t] has this exact continuous Gram matrix.
    coeffs = jnp.array([[1., 2-1j], [1j, 3+2j]])
    left = Chebtech2(coeffs=coeffs)
    right = Chebtech2(coeffs=jnp.pad(coeffs, ((0, pad), (0, 0))))
    expected = jnp.array([[8/3, 16/3-4j], [16/3+4j, 56/3]])
    for actual in (left.inner(right), jax.jit(lambda a, b: a.inner(b))(left, right)):
        np.testing.assert_allclose(actual, expected, atol=8e-15, rtol=0)
        np.testing.assert_array_equal(jnp.imag(jnp.diag(actual)), jnp.zeros(2))
        assert bool(jnp.all(jnp.real(jnp.diag(actual)) >= 0))
    f, g = _as_fun(left, (-1., 1.)), _as_fun(right, (-1., 1.))
    np.testing.assert_allclose(f.H @ g, expected, atol=8e-15, rtol=0)


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('pad', [0, 3])
def test_complex_scalar_inner_retains_known_real_storage(cls, pad):
    import jax

    coeffs = jnp.array([1+2j, 3-1j])
    f = cls(coeffs=coeffs)
    g = cls(coeffs=jnp.pad(coeffs, (0, pad)))
    expected = 50/3  # integral |(1+2i)+(3-i)t|^2
    for actual in (f.inner(f), f.inner(g), jax.jit(lambda a: a.inner(a))(f)):
        assert not jnp.iscomplexobj(actual)
        np.testing.assert_allclose(actual, expected, atol=8e-15, rtol=0)
    # Distinct traced operands have a fixed complex dtype regardless of
    # runtime equality, but the native nonnegative scalar value is retained.
    dynamic = jax.jit(lambda a, b: a.inner(b))(f, g)
    np.testing.assert_allclose(dynamic, expected, atol=8e-15, rtol=0)
    assert jnp.imag(dynamic) == 0

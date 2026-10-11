import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("kind", ["clenshawlord", "maehly"])
def test_transpose_preserved_without_conjugation(kind):
    # Real coefficients in Maehly avoid its existing complex extraction adapter.
    coefficients = [1.0, 2.0, 3.0] if kind == "maehly" else [1 + 2j, 2 - 1j, 3 + 0.5j]
    f = cj.chebfun(jnp.asarray(coefficients), coeffs=True)
    p, q, r = cj.chebpade(f, 2, 0, kind)
    pt, qt, rt = cj.chebpade(f.transpose(), 2, 0, kind)
    assert not p.is_transposed and not q.is_transposed
    assert pt.is_transposed and qt.is_transposed
    np.testing.assert_array_equal(np.asarray(pt.chebcoeffs(3)), np.asarray(p.chebcoeffs(3)))
    np.testing.assert_array_equal(np.asarray(qt.chebcoeffs(1)), np.asarray(q.chebcoeffs(1)))
    np.testing.assert_array_equal(np.asarray(rt(0.25)), np.asarray(r(0.25)))


@pytest.mark.parametrize("kind", ["clenshawlord", "maehly"])
def test_native_singfun_rejection_before_arithmetic(kind):
    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
    from chebfunjax.fun.singfun import Singfun
    from chebfunjax.tech.chebtech import Chebtech2

    f = Chebfun(
        funs=[_Piece(Singfun(Chebtech2.from_coeffs(jnp.asarray([1.0])), (-0.5, 0.0)), (-1.0, 1.0))],
        domain=Domain((-1.0, 1.0)),
    )
    with pytest.raises(ValueError, match="support functions with singularities"):
        cj.chebpade(f, 1, 1, kind)


def test_native_omitted_denominator_degree():
    f = cj.chebfun(jnp.asarray([1.0, 2.0, 3.0]), coeffs=True)
    p, q, _ = cj.chebpade(f, 2)
    np.testing.assert_array_equal(np.asarray(p.chebcoeffs(3)), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.asarray(q.chebcoeffs(1)), [1.0])


def test_native_both_optional_argument_orders():
    f = cj.chebfun(jnp.asarray([1.0, 2.0, 3.0]), coeffs=True)
    for kind in ("clenshawlord", "maehly"):
        p, q, _ = cj.chebpade(f, 2, 0, kind, 4)
        pr, qr, _ = cj.chebpade(f, 2, 0, 4, kind)
        np.testing.assert_array_equal(np.asarray(pr.chebcoeffs(3)), np.asarray(p.chebcoeffs(3)))
        np.testing.assert_array_equal(np.asarray(qr.chebcoeffs(1)), np.asarray(q.chebcoeffs(1)))

"""Original threecomponents predicates1–15, native Chebfun7574c77.

Python * represents elementwise TIMES; @ represents native matrix MTIMES.
Scalar vector evaluations retain native vector2norm; no sample substitute for
continuous Chebfun2/2v norms.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

TOL = 1e5 * jnp.finfo(jnp.float64).eps


@pytest.fixture(scope="module")
def fields():
    f = chebfun2(lambda x, y: x, domain=[-1, 1, -1, 1])
    F = Chebfun2v([f.approx, f.approx, f.approx])
    G = Chebfun2v([f.approx, (2 * f).approx, (3 * f).approx])
    return f, F, G


@pytest.mark.parametrize("clause", range(1, 16))
def test_native_arithmetic(clause, fields):
    f, F, G = fields
    q = jnp.pi / 6
    row = jnp.array([[1.0, 2.0, 3.0]])
    col = row.T
    if clause in (1, 2):
        kwargs = {} if clause == 1 else {"domain": [-1, 1, -1, 1]}
        other = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: x, lambda x, y: x, **kwargs)
        assert (F - other).norm() < TOL
    elif clause == 3:
        assert jnp.linalg.norm((F + G)(1, 1).reshape(-1) - jnp.array([2.0, 3.0, 4.0])) < TOL
    elif clause in (4, 5, 6, 7):
        H = {4: lambda: G + 1, 5: lambda: G + row, 6: lambda: 1 + G, 7: lambda: row + G}[clause]()
        expected = q * jnp.array([1.0, 2.0, 3.0]) + (1 if clause in (4, 6) else row.reshape(-1))
        assert jnp.linalg.norm(H(q, 1).reshape(-1) - expected) < TOL
    elif clause in (8, 9, 10, 11):
        H = {8: lambda: G * 1, 9: lambda: G * col, 10: lambda: 1 * G, 11: lambda: col * G}[clause]()
        expected = G(q, 1).reshape(-1) if clause in (8, 10) else q * jnp.array([1.0, 4.0, 9.0])
        assert jnp.linalg.norm(H(q, 1).reshape(-1) - expected) < TOL
    elif clause in (12, 14):
        H = G @ 1 if clause == 12 else 1 @ G
        assert jnp.linalg.norm(H(q, 1).reshape(-1) - G(q, 1).reshape(-1)) < TOL
    elif clause == 13:
        H = G.ctranspose() @ col
        assert (H - (f + 4 * f + 9 * f)).norm() < TOL
    else:
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN2V:mtimes:double1"):
            col @ G


@pytest.mark.parametrize("clause", range(16, 21))
def test_native_calculus(clause, fields):
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2

    _, F, G = fields
    f = chebfun2(lambda x, y: jnp.sin(x * y))

    def scalar(a):
        return Chebfun2(approx=a)

    def extend(v):
        return Chebfun2v(v.components + [(0 * f).approx])

    if clause == 16:
        error = ((F + G).curl() - (F.curl() + G.curl())).norm()
        bound = 100 * TOL
    elif clause == 17:
        left = scalar((f * G).divergence())
        right = scalar(G.dot(extend(f.grad()))) + f * scalar(G.divergence())
        error = (left - right).norm()
        bound = 100 * TOL
    elif clause == 18:
        error = scalar(G.curl().divergence()).norm()
        bound = 10 * TOL
    elif clause == 19:
        error = (scalar(f.grad().divergence()) - f.laplacian()).norm()
        bound = 10 * TOL
    else:
        error = (G.curl().curl() - (extend(scalar(G.divergence()).grad()) - G.laplacian())).norm()
        bound = 10 * TOL
    assert error < bound


def test_native_torus_21_22():
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2

    dom = [0, 2 * jnp.pi, 0, 2 * jnp.pi]
    u = chebfun2(lambda u, v: u, domain=dom)
    v = chebfun2(lambda u, v: v, domain=dom)
    x = (3 + v.cos()) * u.cos()
    y = (3 + v.cos()) * u.sin()
    z = v.sin()
    r = Chebfun2v([x.approx, y.approx, z.approx])
    n = r.normal()
    r1, r2 = r.diff(1, 1), r.diff(1, 2)
    g = x.diff(1, 1) * r1 + x.diff(1, 2) * r2
    V = n.cross(g)
    f1 = Chebfun2(approx=V.components[1]).diff(1, 2)
    f2 = r.diff(1, 2)
    fx = Chebfun2(approx=(f1 * f2).components[0])
    assert fx(2 * jnp.pi, ":").norm() < 10 * TOL
    assert fx(0, ":").norm() < 10 * TOL


def test_numeric_orientation_and_source_guards(fields):
    import numpy as np

    _, F, G = fields
    row = G.T
    for out in [
        row + np.array([1.0, 2.0, 3.0]),
        row.times(jnp.array([[1.0, 2.0, 3.0]])),
        2.0 @ row,
    ]:
        assert out.is_transposed
    # Literal native mtimes first transpose branch returns a column here.
    assert not (row @ 2.0).is_transposed
    complex_added = G + (1 + 2j)
    np.testing.assert_allclose(
        complex_added(0.5, 0.2).reshape(-1), [1.5 + 2j, 2.0 + 2j, 2.5 + 2j], atol=1e-15, rtol=0
    )
    matrix = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    product = matrix @ G
    np.testing.assert_allclose(product(0.5, 0.2).reshape(-1), [0.5, 2.0], atol=1e-15, rtol=0)
    for call, identifier in [
        (lambda: F + jnp.ones(2), "plus:doubleSize"),
        (lambda: F + G.T, "plus:transposed"),
        (lambda: F.times(jnp.ones((1, 3))), "times:double"),
        (lambda: F @ G, "mtimes:sizes"),
        (lambda: F @ jnp.ones((3, 1)), "mtimes:double2"),
    ]:
        with pytest.raises(ValueError, match=identifier):
            call()


def test_empty_multiplication_precedes_legacy_dot(fields):
    _, _, G = fields
    empty = Chebfun2v.empty()
    assert (G.T * empty).isempty()
    assert (empty * G.T).isempty()
    assert G.times(jnp.array([])).isempty()
    assert (G + jnp.array([])).isempty()

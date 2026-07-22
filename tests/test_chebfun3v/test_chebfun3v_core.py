"""Core unit/mirror tests for Chebfun3v (no MATLAB golden ref required).

These exercise the Chebfun3v surface (construction, arithmetic, calculus,
complex parts, integration, composition, and accessors) with low-degree
polynomial fields so they run quickly, providing broad coverage of
src/chebfunjax/chebfun3d/chebfun3v.py independently of the golden-reference
fixtures.

Author: Claude Fable 5.
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

TOL = 1e-11
P = (0.3, -0.4, 0.5)


def _lin(name):
    return {"x": lambda x, y, z: x,
            "y": lambda x, y, z: y,
            "z": lambda x, y, z: z}[name]


def _pos():
    return Chebfun3v.from_functions(_lin("x"), _lin("y"), _lin("z"))


# ----------------------------------------------------------------------
# Construction / accessors
# ----------------------------------------------------------------------

class TestConstruction:
    def test_from_functions_counts(self):
        assert Chebfun3v.from_functions(_lin("x")).n_components == 1
        assert Chebfun3v.from_functions(_lin("x"), _lin("y")).n_components == 2
        assert _pos().n_components == 3

    def test_from_functions_bad_count(self):
        with pytest.raises(ValueError):
            Chebfun3v.from_functions()

    def test_list_and_varargs_equivalent(self):
        f = Chebfun3.from_function(_lin("x"))
        v1 = Chebfun3v([f, f])
        v2 = Chebfun3v(f, f)
        assert v1.n_components == v2.n_components == 2

    def test_too_many_components(self):
        f = Chebfun3.from_function(_lin("x"))
        with pytest.raises(ValueError):
            Chebfun3v([f, f, f, f])

    def test_domain_mismatch(self):
        f = Chebfun3.from_function(_lin("x"))
        g = Chebfun3.from_function(_lin("x"), domain=(-2, 2, -1, 1, -1, 1))
        with pytest.raises(ValueError):
            Chebfun3v([f, g])

    def test_empty(self):
        E = Chebfun3v()
        assert E.isempty()
        assert E.domain is None
        assert E.n_components == 0
        assert "empty" in repr(E)

    def test_getitem_and_repr(self):
        v = _pos()
        assert isinstance(v[0], Chebfun3)
        assert "Chebfun3v with 3 components" in repr(v)

    def test_size(self):
        inf = float("inf")
        assert _pos().size() == (3, inf, inf, inf)
        assert _pos().size(1) == 3
        assert _pos().transpose().size() == (inf, inf, inf, 3)
        assert Chebfun3v().size() == (None, None, None, None)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

class TestEval:
    def test_feval(self):
        np.testing.assert_allclose(np.asarray(_pos()(*P)),
                                   np.array([0.3, -0.4, 0.5]), atol=TOL)

    def test_feval_empty(self):
        assert np.asarray(Chebfun3v()(*P)).size == 0


# ----------------------------------------------------------------------
# Arithmetic
# ----------------------------------------------------------------------

class TestArithmetic:
    def test_add_variants(self):
        v = _pos()
        assert np.allclose(np.asarray((v + 1)(*P)),
                           np.array([1.3, 0.6, 1.5]), atol=TOL)
        assert np.allclose(np.asarray((v + [1, 2, 3])(*P)),
                           np.array([1.3, 1.6, 3.5]), atol=TOL)
        assert np.allclose(np.asarray((1 + v)(*P)),
                           np.array([1.3, 0.6, 1.5]), atol=TOL)
        assert np.allclose(np.asarray((v + v)(*P)),
                           2 * np.array([0.3, -0.4, 0.5]), atol=TOL)

    def test_add_chebfun3(self):
        v = _pos()
        f = Chebfun3.from_function(_lin("x"))
        assert np.allclose(np.asarray((v + f)(*P)),
                           np.array([0.6, -0.1, 0.8]), atol=TOL)

    def test_sub(self):
        v = _pos()
        assert float((v - v).norm()) < TOL
        assert np.allclose(np.asarray((v - 1)(*P)),
                           np.array([-0.7, -1.4, -0.5]), atol=TOL)
        assert np.allclose(np.asarray((1 - v)(*P)),
                           np.array([0.7, 1.4, 0.5]), atol=TOL)

    def test_pos_neg(self):
        v = _pos()
        assert (+v) is v
        assert np.allclose(np.asarray((-v)(*P)),
                           -np.array([0.3, -0.4, 0.5]), atol=TOL)

    def test_mul_variants(self):
        v = _pos()
        assert np.allclose(np.asarray((2 * v)(*P)),
                           2 * np.array([0.3, -0.4, 0.5]), atol=TOL)
        assert np.allclose(np.asarray((v * [1, 2, 3])(*P)),
                           np.array([0.3, -0.8, 1.5]), atol=TOL)
        assert float((v * v - v ** 2).norm()) < TOL

    def test_mul_chebfun3(self):
        v = _pos()
        f = Chebfun3.from_function(_lin("z"))
        assert np.allclose(np.asarray((v * f)(*P)),
                           0.5 * np.array([0.3, -0.4, 0.5]), atol=TOL)
        # scalar-field .* vector on the left too
        assert float((f * v - v * f).norm()) < TOL

    def test_div_and_pow(self):
        v = _pos()
        assert np.allclose(np.asarray((v / 2)(*P)),
                           0.5 * np.array([0.3, -0.4, 0.5]), atol=TOL)
        f = Chebfun3.from_function(lambda x, y, z: 2.0 + 0 * x)
        assert np.allclose(np.asarray((v / f)(*P)),
                           0.5 * np.array([0.3, -0.4, 0.5]), atol=TOL)
        assert np.allclose(np.asarray((v ** 2)(*P)),
                           np.array([0.09, 0.16, 0.25]), atol=TOL)

    def test_matmul_inner(self):
        v = _pos()
        inner = v.ctranspose() @ [1, 1, 1]
        # x + y + z at P
        assert abs(float(inner(*P)) - 0.4) < TOL
        inner2 = v.ctranspose() @ v
        assert abs(float(inner2(*P)) - 0.5) < TOL

    def test_rmatmul_raises(self):
        with pytest.raises(ValueError):
            [1, 2, 3] @ _pos()

    def test_transpose(self):
        v = _pos()
        assert v.transpose().is_transposed
        assert v.T.is_transposed
        assert not v.T.T.is_transposed


# ----------------------------------------------------------------------
# Complex parts
# ----------------------------------------------------------------------

class TestComplex:
    def test_real_imag_conj_isreal(self):
        v = _pos()
        assert v.isreal()
        cv = 1j * v
        assert not cv.isreal()
        assert float(cv.real().norm()) < TOL
        assert float((cv.imag() - v).norm()) < TOL
        assert float((cv.conj() + cv).norm()) < TOL

    def test_isPeriodicTech(self):
        assert not _pos().isPeriodicTech()


# ----------------------------------------------------------------------
# Vector ops / calculus
# ----------------------------------------------------------------------

class TestCalculus:
    def test_dot_cross_norm_magnitude(self):
        v = _pos()
        assert abs(float(v.dot(v)(*P)) - 0.5) < TOL
        # x_hat cross y_hat = z_hat
        ex = Chebfun3v.from_functions(lambda x, y, z: 1.0 + 0 * x,
                                      lambda x, y, z: 0 * x,
                                      lambda x, y, z: 0 * x)
        ey = Chebfun3v.from_functions(lambda x, y, z: 0 * x,
                                      lambda x, y, z: 1.0 + 0 * x,
                                      lambda x, y, z: 0 * x)
        assert np.allclose(np.asarray(ex.cross(ey)(*P)),
                           np.array([0, 0, 1]), atol=TOL)
        assert abs(float(v.norm()) - np.sqrt(8.0)) < 1e-8
        assert abs(float(v.magnitude()(*P)) - np.sqrt(0.5)) < TOL

    def test_diff_family(self):
        v = _pos()
        assert np.allclose(np.asarray(v.diffx()(*P)),
                           np.array([1, 0, 0]), atol=TOL)
        assert np.allclose(np.asarray(v.diffy()(*P)),
                           np.array([0, 1, 0]), atol=TOL)
        assert np.allclose(np.asarray(v.diffz()(*P)),
                           np.array([0, 0, 1]), atol=TOL)

    def test_divergence_curl_laplacian(self):
        v = _pos()
        assert abs(float(v.divergence()(*P)) - 3.0) < TOL
        assert abs(float(v.div()(*P)) - 3.0) < TOL
        assert float(v.curl().norm()) < TOL
        assert float(v.laplacian().norm()) < TOL
        assert float(v.lap().norm()) < TOL
        assert float(v.divgrad().norm()) < TOL

    def test_divergence_two_components(self):
        v = Chebfun3v.from_functions(_lin("x"), _lin("y"))
        assert abs(float(v.divergence()(*P)) - 2.0) < TOL

    def test_jacobian(self):
        v = _pos()  # identity map -> Jacobian determinant 1
        assert abs(float(v.jacobian()(*P)) - 1.0) < TOL


# ----------------------------------------------------------------------
# Range / roots / integration / composition
# ----------------------------------------------------------------------

class TestMisc:
    def test_minandmax3est(self):
        v = Chebfun3v.from_functions(_lin("x"), domain=(-2, 4, -1, 1, -1, 1))
        assert np.allclose(np.asarray(v.minandmax3est()),
                           np.array([-2, 4]), atol=1e-10)
        assert np.asarray(Chebfun3v().minandmax3est()).size == 0

    def test_roots_empty_and_reshape(self):
        assert np.asarray(Chebfun3v().roots()).size == 0
        f = Chebfun3.from_function(_lin("x"))
        g = Chebfun3.from_function(_lin("y"))
        h = Chebfun3.from_function(_lin("z"))
        r = Chebfun3v([f, g, h]).roots()
        assert r.shape == (1, 3)
        assert np.allclose(np.asarray(r), 0.0, atol=1e-10)

    def test_integral_line(self):
        F = Chebfun3v.from_functions(lambda x, y, z: 8 * x ** 2 * y * z,
                                     lambda x, y, z: 5 * z,
                                     lambda x, y, z: -4 * x * y)
        val = F.integral(lambda t: (t, t ** 2, t ** 3), domain=(0, 1))
        assert abs(float(val) - 1.0) < 1e-10

    def test_compose_scalar(self):
        F = _pos()
        g = Chebfun3.from_function(lambda x, y, z: x + y + z)
        assert abs(float(F.compose(g)(*P)) - 0.4) < TOL

    def test_quiver3(self):
        out = _pos().quiver3(n=5)
        assert len(out) == 6
        assert all(np.asarray(a).shape == (5, 5, 5) for a in out)
        out2 = _pos().quiver(n=4)
        assert np.asarray(out2[0]).shape == (4, 4, 4)

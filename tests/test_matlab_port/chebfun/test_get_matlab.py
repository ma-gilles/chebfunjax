"""Port of MATLAB Chebfun tests/chebfun/test_get.m (Fable 5).

MATLAB cells map to Python lists: at simplevel 0 the output is a list
over columns of lists over pieces; at simplevel 1 a 2-D list indexed
[piece][column]; at simplevel 2 a numeric array where MATLAB returns a
numeric matrix.  Quasimatrices are lists of Chebfun columns.

Provenance
----------
MATLAB source : tests/chebfun/test_get.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import get

jax.config.update("jax_enable_x64", True)


def _quasi(f):
    """cheb2quasi: split an array-valued chebfun into columns."""
    return [f.extract_columns(j) for j in range(f.n_columns)]


def _shape(cell):
    """(rows, cols) of a 2-D list-of-lists cell."""
    return (len(cell), len(cell[0]))


def _isnum(x):
    return hasattr(x, "shape")


class TestChebfunGet:
    def setup_method(self):
        self.fs1 = cj.chebfun(lambda x: jnp.sin(x), domain=(-1.0, 1.0))
        self.fs2 = cj.chebfun(lambda x: jnp.cos(x),
                              domain=(-1.0, 0.0, 1.0))
        self.fa1 = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1),
            domain=(-1.0, 1.0))
        self.fa2 = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1),
            domain=(-1.0, 0.0, 1.0))
        self.fq1 = _quasi(self.fa1)
        self.fq2 = _quasi(self.fa2)
        self.fq3 = [self.fs1, self.fs2]

    def test_simplevel_2(self):
        c = get(self.fs1, "coeffs")
        assert _isnum(c) and c.shape[1] == 1  # pass(1)
        c = get(self.fs2, "coeffs")
        assert isinstance(c, list) and _shape(c) == (2, 1)  # pass(2)
        c = get(self.fa1, "coeffs")
        assert _isnum(c) and c.shape[1] == 2  # pass(3)
        c = get(self.fa2, "coeffs")
        assert isinstance(c, list) and _shape(c) == (2, 2)  # pass(4)
        c = get(self.fq1, "coeffs")
        assert _isnum(c) and c.shape[1] == 2  # pass(5)
        c = get(self.fq2, "coeffs")
        assert isinstance(c, list) and _shape(c) == (2, 2)  # pass(6)
        c = get(self.fq3, "coeffs")
        # pass(7): ragged piece counts -> per-column cells.
        assert isinstance(c, list) and len(c) == 2
        assert len(c[0]) == 1 and len(c[1]) == 2

    def test_simplevel_1(self):
        c = get(self.fs1, "coeffs", 1)
        assert isinstance(c, list) and _shape(c) == (1, 1)  # pass(8)
        c = get(self.fs2, "coeffs", 1)
        assert isinstance(c, list) and _shape(c) == (2, 1)  # pass(9)
        c = get(self.fa1, "coeffs", 1)
        assert isinstance(c, list) and _shape(c) == (1, 2)  # pass(10)
        c = get(self.fa2, "coeffs", 1)
        assert isinstance(c, list) and _shape(c) == (2, 2)  # pass(11)
        c = get(self.fq1, "coeffs", 1)
        assert isinstance(c, list) and _shape(c) == (1, 2)  # pass(12)
        c = get(self.fq2, "coeffs", 1)
        assert isinstance(c, list) and _shape(c) == (2, 2)  # pass(13)
        c = get(self.fq3, "coeffs", 1)
        # pass(14): ragged -> per-column cells.
        assert isinstance(c, list) and len(c) == 2
        assert len(c[0]) == 1 and len(c[1]) == 2

    def test_simplevel_0(self):
        # Level 0 is a list over columns of lists over pieces.
        c = get(self.fs1, "coeffs", 0)
        assert len(c) == 1 and len(c[0]) == 1  # pass(15)
        c = get(self.fs2, "coeffs", 0)
        assert len(c) == 1 and len(c[0]) == 2  # pass(16)
        c = get(self.fa1, "coeffs", 0)
        assert len(c) == 2 and all(len(cc) == 1 for cc in c)  # 17
        c = get(self.fa2, "coeffs", 0)
        assert len(c) == 2 and all(len(cc) == 2 for cc in c)  # 18
        c = get(self.fq1, "coeffs", 0)
        assert len(c) == 2 and all(len(cc) == 1 for cc in c)  # 19
        c = get(self.fq2, "coeffs", 0)
        assert len(c) == 2 and all(len(cc) == 2 for cc in c)  # 20
        c = get(self.fq3, "coeffs", 0)
        assert len(c) == 2 and len(c[0]) == 1 and len(c[1]) == 2

    def test_row_chebfuns(self):
        c = get(self.fs1.transpose(), "coeffs")
        assert _isnum(c) and c.shape[0] == 1  # pass(22)
        c = get(self.fs2.transpose(), "coeffs")
        assert isinstance(c, list) and _shape(c) == (1, 2)  # pass(23)
        c = get(self.fa1.transpose(), "coeffs")
        assert _isnum(c) and c.shape[0] == 2  # pass(24)
        c = get(self.fa2.transpose(), "coeffs")
        assert isinstance(c, list) and _shape(c) == (2, 2)  # pass(25)

    def test_deltas(self):
        x = cj.chebfun(lambda t: t, domain=(-1.0, 1.0))
        fd1 = self.fs1 + x.dirac()
        fd2 = (self.fs2 + (x - 0.5).dirac() + (x + 0.5).dirac())
        c = get(fd1, "deltas")
        assert float(jnp.max(jnp.abs(
            c - jnp.asarray([[0.0], [1.0]])))) < 5e-15  # pass(29)
        c = get(fd2, "deltas")
        want = jnp.asarray([[-0.5, 0.5], [1.0, 1.0]])
        assert float(jnp.max(jnp.abs(c - want))) < 5e-15  # pass(30)

    def test_exponents(self):
        fse1 = cj.chebfun(lambda t: 1.0 / (t + 2), domain=(-2.0, 2.0),
                          exps=(-1.0, 0.0))
        exps = get(fse1, "exponents")
        assert np.allclose(np.asarray(exps), [[-1.0, 0.0]])  # 32
        exps = get(fse1, "exponents", 1)
        assert isinstance(exps, list) and _shape(exps) == (1, 1)  # 33
        exps = get(fse1, "exponents", 0)
        assert len(exps) == 1 and len(exps[0]) == 1  # pass(34)

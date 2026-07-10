"""Port of MATLAB Chebfun tests/bndfun/test_qr.m (Opus 4.8).

MATLAB ``qr(f)`` computes an (abstract) QR factorisation of an array-valued
Bndfun (a quasimatrix): Q is an array-valued Bndfun with orthonormal columns
in the L2 inner product and R is an upper-triangular numeric matrix.
chebfunjax has neither array-valued (quasimatrix) Bndfun nor a ``qr`` method,
so every assertion is xfail with that precise reason.  Each MATLAB helper
``test_one_qr`` / ``test_one_qr_with_perm`` contributes two assertions
(orthogonality of Q, accuracy of Q*R); the pass indices are noted per method.

Provenance
----------
MATLAB source : tests/bndfun/test_qr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

DOM = Domain((-2.0, 7.0))
_QR_MISSING = (
    "chebfunjax lacks array-valued (quasimatrix) Bndfun and has no qr() method"
)


def _bf(f, n=None):
    # array-valued builds never converge in chebfunjax; a small fixed n keeps
    # the xfail fast (the qr() call raises AttributeError regardless).
    return Bndfun.from_function(f, DOM, n=n)


class TestBndfunQR:
    @pytest.mark.xfail(reason=_QR_MISSING, raises=AttributeError)
    def test_qr_sin(self):  # pass(1:4): orthogonality + accuracy, with/without perm
        f = _bf(jnp.sin)
        f.qr()

    @pytest.mark.xfail(reason=_QR_MISSING, raises=AttributeError)
    def test_qr_cos_exp(self):  # pass(5:8)
        f = _bf(lambda x: jnp.stack([jnp.cos(x), jnp.exp(x)], axis=-1), n=17)
        f.qr()

    @pytest.mark.xfail(reason=_QR_MISSING, raises=AttributeError)
    def test_qr_monomials(self):  # pass(9:12)
        f = _bf(lambda x: jnp.stack([x ** k for k in range(8)], axis=-1), n=17)
        f.qr()

    @pytest.mark.xfail(reason=_QR_MISSING, raises=AttributeError)
    def test_qr_mixed_complex(self):  # pass(13:16)
        f = _bf(
            lambda x: jnp.stack(
                [1.0 / (1 + 1j * x ** 2), jnp.sinh((1 - 1j) * x), jnp.exp(x) - x ** 3],
                axis=-1,
            ),
            n=17,
        )
        f.qr()

    @pytest.mark.xfail(reason=_QR_MISSING, raises=AttributeError)
    def test_qr_vector_flag(self):  # pass(17): permutation 'vector' flag
        f = _bf(
            lambda x: jnp.stack(
                [1.0 / (1 + 1j * x ** 2), jnp.sinh((1 - 1j) * x), jnp.exp(x) - x ** 3],
                axis=-1,
            ),
            n=17,
        )
        f.qr(mode="vector")

    @pytest.mark.xfail(reason=_QR_MISSING, raises=AttributeError)
    def test_qr_rank_deficient_shapes(self):  # pass(18)
        f = _bf(lambda x: jnp.stack([x, x, x], axis=-1), n=17)
        Q, R = f.qr()
        assert Q.shape == (3, 3) and R.shape == (3, 3)

    @pytest.mark.xfail(reason=_QR_MISSING, raises=AttributeError)
    def test_qr_rank_deficient_orthogonality(self):  # pass(19)
        f = _bf(lambda x: jnp.stack([x, x, x], axis=-1), n=17)
        Q, R = f.qr()
        ip = np.asarray(Q.inner(Q))
        assert float(np.max(np.abs(ip - np.eye(3)))) < float(f.vscale) * np.finfo(
            np.float64
        ).eps

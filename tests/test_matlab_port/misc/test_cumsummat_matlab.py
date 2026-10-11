"""Literal six tests/misc/test_cumsummat.m predicates plus independent controls.

Provenance
----------
Chebfun pin 7574c77; n=5 and spectral matrix norm bound 1e-10 are native.
ChebColloc classes require n in Python: their instances select the collocation
family, while the public n and domain arguments determine size and scaling.
Independent full-basis oracle integrates NumPy Chebyshev polynomials, truncates
mode n before imposing the left endpoint constant (native colloc semantics).
"""

import numpy as np
import pytest

from chebfunjax.chebpref import ChebopPref
from chebfunjax.discretization.chebcolloc import ChebColloc1, ChebColloc2
from chebfunjax.utils.diffmat import cumsummat
from chebfunjax.utils.quadrature import chebpts

TOL = 1e-10


@pytest.fixture(autouse=True)
def restore_session():
    previous = ChebopPref._defaults
    ChebopPref.setDefaults("factory")
    try:
        yield
    finally:
        ChebopPref._defaults = previous


def full_basis_oracle(n, kind):
    # Independent nodes and coefficient integration; do not call cumsummat,
    # library transforms or the implementation's private recurrence.
    theta = (
        np.pi * (np.arange(n - 1, -1, -1) + 0.5) / n
        if kind == 1
        else np.pi * np.arange(n - 1, -1, -1) / (n - 1)
    )
    x = np.cos(theta)
    values = np.polynomial.chebyshev.chebvander(x, n - 1)
    integrals = []
    for k in range(n):
        coeff = np.zeros(n)
        coeff[k] = 1
        anti = np.polynomial.chebyshev.chebint(coeff)[:n]
        anti[0] -= np.polynomial.chebyshev.chebval(-1, anti)
        integrals.append(np.polynomial.chebyshev.chebval(x, anti))
    return np.array(integrals).T @ np.linalg.inv(values)


def native_compare(got, reference, kind, scale=1):
    assert np.linalg.norm(np.asarray(got) - np.asarray(reference), 2) < TOL
    assert np.linalg.norm(np.asarray(got) - scale * full_basis_oracle(5, kind), 2) < TOL


class TestCumsummat:
    def test_source_1_default(self):
        native_compare(cumsummat(5), ChebColloc2(5).cumsummat(), 2)

    def test_source_2_interval(self):
        native_compare(cumsummat(5, [-2, 2]), 2 * ChebColloc2(5).cumsummat(), 2, 2)

    def test_source_3_colloc_instance(self):
        native_compare(cumsummat(5, [-2, 2], ChebColloc2(5)), 2 * ChebColloc2(5).cumsummat(), 2, 2)

    def test_source_4_colloc_constructor(self):
        native_compare(cumsummat(5, ChebColloc1), ChebColloc1(5).cumsummat(), 1)

    def test_source_5_colloc_string(self):
        native_compare(
            cumsummat(5, [-0.5, 0.5], "chebcolloc1"), ChebColloc1(5).cumsummat() / 2, 1, 0.5
        )

    def test_source_6_default_preference(self):
        ChebopPref.setDefaults("discretization", ChebColloc2)
        native_compare(cumsummat(5), ChebColloc2(5).cumsummat(), 2)

    def test_interval_scaling(self):
        A = np.asarray(cumsummat(5))
        B = np.asarray(cumsummat(5, domain=(-2.0, 2.0)))
        assert float(np.max(np.abs(B - 2 * A))) < TOL

    def test_antiderivative_of_polynomial(self):
        n = 9
        x = np.asarray(chebpts(n, kind=2))
        got = np.asarray(cumsummat(n)) @ (3 * x**2)
        assert float(np.max(np.abs(got - (x**3 + 1)))) < TOL

    def test_session_domain_and_explicit_default_family(self):
        ChebopPref.setDefaults("domain", [-2, 2])
        ChebopPref.setDefaults("discretization", ChebColloc1)
        native_compare(cumsummat(5), 2 * ChebColloc2(5).cumsummat(), 2, 2)
        native_compare(cumsummat(5, ChebColloc1), 2 * ChebColloc1(5).cumsummat(), 1, 2)

    def test_factory_and_object_dispatch(self):
        native_compare(cumsummat(5, lambda: ChebColloc1(5)), ChebColloc1(5).cumsummat(), 1)

        class Custom:
            def cumsummat(self, n):
                return np.eye(n)

        assert np.array_equal(cumsummat(5, [-2, 2], lambda: Custom()), 2 * np.eye(5))
        assert np.array_equal(cumsummat(5, [-0.5, 0.5], Custom()), np.eye(5) / 2)

    def test_instance_selects_family_and_breakpoints_use_support(self):
        native_compare(
            cumsummat(5, [-2, 2], ChebColloc1(7, (-3, 3))), 2 * ChebColloc1(5).cumsummat(), 1, 2
        )
        with pytest.warns(UserWarning, match="breakpoints"):
            native_compare(cumsummat(5, [-2, 0, 2]), 2 * ChebColloc2(5).cumsummat(), 2, 2)

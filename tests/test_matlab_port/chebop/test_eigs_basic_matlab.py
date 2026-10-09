"""Literal four predicates from tests/chebop/test_eigs_basic.m.

MATLAB source: tests/chebop/test_eigs_basic.m, @linop/eigs.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.

The only output adapter packs Python's actual returned function columns and
lambda vector into native-shaped ChebMatrix containers. No sampled residuals.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D, eval_at
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop


def _diagonal(values, domain):
    # Numeric block packing only; native V*D is the same block multiplication.
    return ChebMatrix(
        [[complex(values[i]) if i == j else 0.0
          for j in range(len(values))] for i in range(len(values))],
        domain=domain,
    )


@pytest.fixture(scope="module")
def linop_source_result():
    domain = (0.0, float(jnp.pi))
    operator = linop(-D(domain, 2))
    operator = operator.add_constraint(eval_at(domain[0], domain=domain), 0.0)
    operator = operator.add_constraint(eval_at(domain[1], domain=domain), 0.0)
    pref = ChebopPref(discretization="chebcolloc2")
    values, columns = operator.eigs(k=10, pref=pref)
    functions = ChebMatrix(
        [[column.blocks[0][0] for column in columns]], domain=domain,
    )
    residual = operator @ functions - functions @ _diagonal(values, domain)
    return values, residual


@pytest.fixture(scope="module")
def chebop_source_result():
    domain = (0.0, float(jnp.pi))
    operator = Chebop(lambda x, u: -u.diff(2), domain=domain)
    operator.lbc = "dirichlet"
    operator.rbc = "dirichlet"
    pref = ChebopPref(discretization="values")
    values, columns = operator.eigs(k=10, pref=pref, return_eigenfunctions=True)
    functions = ChebMatrix([list(columns)], domain=domain)
    residual = operator(functions) - functions @ _diagonal(values, domain)
    return values, residual


def _source_spectrum_error(values):
    values = jnp.asarray(values)
    # Native real eigenvalues negate in real arithmetic before the complex
    # square root. Negating an already complex value can manufacture -0j and
    # choose the lower side of sqrt's negative-real branch cut. Do not discard
    # any genuine imaginary eigenvalue component.
    negative = (-jnp.real(values)).astype(jnp.complex128) if bool(
        jnp.all(jnp.imag(values) == 0)) else -values
    actual = jnp.sqrt(negative)
    expected = 1j * jnp.arange(1, 11)
    return jnp.max(jnp.abs(actual - expected))


def test_native_1_linop_sqrt_spectrum(linop_source_result):
    values, _ = linop_source_result
    assert _source_spectrum_error(values) < 1e-10


def test_native_2_chebop_sqrt_spectrum(chebop_source_result):
    values, _ = chebop_source_result
    assert _source_spectrum_error(values) < 1e-10


def test_native_3_linop_continuous_frobenius(linop_source_result):
    _, residual = linop_source_result
    assert residual.norm() < 1e-7


def test_native_4_chebop_continuous_frobenius(chebop_source_result):
    _, residual = chebop_source_result
    assert residual.norm() < 1e-7

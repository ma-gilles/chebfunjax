"""Original coefficient/value assertions on matching native random operands.

Provenance
----------
MATLAB source: tests/chebfun2v/test_coeffs_vals.m
Chebfun commit:7574c77
Native rng0 CDR fixtures preserve source input polynomials. They do not qualify
randnfun2's Python stream or constructor. Native slots8/9 report numeric norms
and then use all(pass(:)); that literal truthiness is tested separately from
retained historical Python accuracy controls.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d._pivot_metadata import _cdr_weights
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech2

TOL = 1000 * ChebfunPref().techPrefs.chebfuneps
REFERENCE = Path(__file__).resolve().parents[2] / 'references' / 'chebfun2v_coefficients_native.json'


def _native_operand(data):
    cols = jnp.asarray(data['column_coefficients']).reshape(data['column_shape'])
    rows = jnp.asarray(data['row_coefficients']).reshape(data['row_shape'])
    raw = jnp.asarray(data['raw']).reshape(-1)
    return Chebfun2(approx=SeparableApprox(
        cols=[Chebtech2.from_coeffs(cols[:, k]) for k in range(cols.shape[1])],
        rows=[Chebtech2.from_coeffs(rows[:, k]) for k in range(rows.shape[1])],
        pivots=_cdr_weights(raw), pivot_values=raw, domain=tuple(data['domain'])))


@pytest.fixture(scope='module')
def source_operands():
    reference = json.loads(REFERENCE.read_text())
    assert reference['native_original_test_result']
    return _native_operand(reference['u']), _native_operand(reference['v'])


def test_original_coefficients_and_values(source_operands):
    u, v = source_operands
    field = Chebfun2v([u.approx, v.approx])
    x, y = field.coeffs2()
    assert (Chebfun2.from_coeffs(x)-u).norm(jnp.inf) < TOL
    assert (Chebfun2.from_coeffs(y)-v).norm(jnp.inf) < TOL
    x50, y50 = field.coeffs2(50, 60)
    assert jnp.linalg.norm(x50-u.coeffs2(50, 60), ord=jnp.inf) < TOL
    assert jnp.linalg.norm(y50-v.coeffs2(50, 60), ord=jnp.inf) < TOL
    rebuilt = Chebfun2v.coeffs2chebfun2v(u.coeffs2(), v.coeffs2())
    assert (field-rebuilt).norm() < TOL
    x, y = rebuilt.coeffs2()
    a, b = Chebfun2v.coeffs2vals(x, y)
    assert jnp.linalg.norm(Chebfun2.coeffs2vals(x)-a, ord=jnp.inf) < TOL
    assert jnp.linalg.norm(Chebfun2.coeffs2vals(y)-b, ord=jnp.inf) < TOL
    c, d = Chebfun2v.vals2coeffs(a, b)
    # The original final two slots have no '<tol': all(pass(:)) interprets
    # their nonzero numeric norms as true. Preserve this source quirk.
    assert bool(jnp.linalg.norm(x-c, ord=jnp.inf))
    assert bool(jnp.linalg.norm(y-d, ord=jnp.inf))
    # Independent historical Python transform error controls retained.
    assert jnp.max(jnp.abs(x-c)) < TOL
    assert jnp.max(jnp.abs(y-d)) < TOL

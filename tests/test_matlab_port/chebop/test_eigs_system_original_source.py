"""Literal four default predicates of native test_eigs_system.m (7574c77).

The original system2 file repeats these operands using cell syntax and adds
C1/no-x clauses; those final two clauses remain outside this C2 test.
"""
import json
import math

import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop


def test_original_default_smooth_and_piecewise_four_predicates(record_property):
    expected = jnp.abs(jnp.asarray([
        0., -.5+jnp.sqrt(3.)/2*1j, -.5-jnp.sqrt(3.)/2*1j,
        -.5+jnp.sqrt(15.)/2*1j, -.5-jnp.sqrt(15.)/2*1j,
    ]))
    op = Chebop(lambda x, u, v: [-u+v.diff(), u.diff()], (0., math.pi))
    op.lbc = lambda u, v: u
    op.rbc = lambda u, v: u
    errors = []
    outputs = []
    for domain in [(0., math.pi), (0., math.pi/2, math.pi)]:
        op.domain = domain
        vectors, values = op.eigs(k=5)
        errors.append(float(jnp.max(jnp.abs(jnp.abs(values)-expected))))
        # @chebmatrix/norm.m default: Frobenius norm of block L2 norms,
        # column-major block order, exactly the native AV-V*D computation.
        norms = []
        for column, value in zip(vectors, values):
            applied = op(column)
            norms.extend((a-u*value).norm(2) for a, u in zip(applied, column))
        errors.append(float(jnp.sqrt(jnp.sum(jnp.asarray(norms)**2))))
        outputs.append({'domain':domain,
                        'eigenvalues_real_imag':[[float(jnp.real(v)), float(jnp.imag(v))] for v in values],
                        'component_lengths':[[len(f) for f in column] for column in vectors]})
    record_property('actual_four_errors', json.dumps(errors))
    record_property('actual_outputs', json.dumps(outputs))
    assert all(error < 1e-9 for error in errors)

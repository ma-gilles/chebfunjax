"""All six default original system2 clauses with actual cell-style callbacks.

MATLAB source: tests/chebop/test_eigs_system2.m, Chebfun7574c77.
Python u[0]/u[1] represents native u{1}/u{2}; the final callback omits x.
Boundary callbacks likewise take the single cell argument. Native block
Frobenius L2 residuals sum squared component L2 norms across all modes.
"""
import json
import math

import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop


def test_all_six_original_cell_and_autonomous_predicates():
    correct = jnp.abs(jnp.asarray([
        0., -.5+jnp.sqrt(3.)/2*1j, -.5-jnp.sqrt(3.)/2*1j,
        -.5+jnp.sqrt(15.)/2*1j, -.5-jnp.sqrt(15.)/2*1j]))
    errors = []

    def predicates(operator, pref=None):
        columns, values = operator.eigs(k=5, pref=pref)
        errors.append(float(jnp.max(jnp.abs(jnp.abs(values)-correct))))
        squared = []
        for column, value in zip(columns, values):
            applied = operator(column)
            squared.extend((a-u*value).norm(2)**2
                           for a, u in zip(applied, column))
        errors.append(float(jnp.sqrt(jnp.sum(jnp.asarray(squared)))))

    # Native clauses1/2: smooth domain, @(x,u) cell argument.
    operator = Chebop(lambda x, u: [-u[0]+u[1].diff(), u[0].diff()],
                     (0., math.pi))
    operator.lbc = lambda u: u[0]
    operator.rbc = lambda u: u[0]
    predicates(operator)
    # Native clauses3/4: mutate the same operator to a piecewise domain.
    operator.domain = (0., math.pi/2, math.pi)
    predicates(operator)
    # Native clauses5/6: autonomous cell callback and first-kind preference.
    operator = Chebop(lambda u: [-u[0]+u[1].diff(), u[0].diff()],
                     (0., math.pi))
    operator.lbc = lambda u: u[0]
    operator.rbc = lambda u: u[0]
    predicates(operator, {'discretization': 'chebcolloc1'})
    print(json.dumps({'source': 'tests/chebop/test_eigs_system2.m',
                      'errors': errors, 'bounds': [1e-9]*6}), flush=True)
    assert len(errors) == 6
    assert all(error < 1e-9 for error in errors)

"""First-kind eigs preferences, source adaptation and independent spectra.

Provenance: Chebfun7574c77 @linop/eigs.m, @chebcolloc1/toFunctionOut.m,
tests/chebop/test_eigs_system2.m final two numerical predicates. Native
cell notation is represented by explicit Python component arguments here.
The six original cell/autonomous clauses are separately verified in
test_eigs_system2_original_cells_source.py.
"""
import math

import jax.numpy as jnp
import pytest

from chebfunjax.operators import _linear_altdisc
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D, I, eval_at
from chebfunjax.operators.chebop import Chebop
from chebfunjax.tech.chebtech import Chebtech1


@pytest.mark.parametrize("domain", [(0., math.pi), (0., math.pi/2, math.pi)])
def test_original_c1_maxwell_spectrum_and_block_l2_residuals(domain):
    op=Chebop(lambda x,u,v: [-u+v.diff(),u.diff()],domain)
    op.lbc=lambda u,v:u
    op.rbc=lambda u,v:u
    vectors,values=op.eigs(k=5,pref={'discretization':'chebcolloc1'})
    expected=jnp.abs(jnp.asarray([0.,-.5+jnp.sqrt(3.)/2*1j,
                                  -.5-jnp.sqrt(3.)/2*1j,
                                  -.5+jnp.sqrt(15.)/2*1j,
                                  -.5-jnp.sqrt(15.)/2*1j]))
    assert float(jnp.max(jnp.abs(jnp.abs(values)-expected)))<1e-9
    norms=[]
    for column,value in zip(vectors,values):
        norms.extend((a-u*value).norm(2) for a,u in zip(op(column),column))
    assert float(jnp.sqrt(jnp.sum(jnp.asarray(norms)**2)))<1e-9
    assert all(isinstance(piece.tech,Chebtech1) for column in vectors for f in column for piece in f.funs)


def dirichlet(operator):
    return operator.addbc(eval_at(-1.),0.).addbc(eval_at(1.),0.)


def test_explicit_c1_preferences_use_source_minimum_dimension_and_output_tech(monkeypatch):
    seen=[]
    original=_linear_altdisc.LinearDiscretization

    class Observer(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            seen.append((self.backend,self.dimensions))

    monkeypatch.setattr(_linear_altdisc,'LinearDiscretization',Observer)
    operator=dirichlet(linop(D()**2))
    values,columns=operator.eigs(k=3,sigma=0,pref={'discretization':'chebcolloc1'})
    expected=-(jnp.pi*jnp.arange(1,4)/2)**2
    assert float(jnp.max(jnp.abs(jnp.sort(jnp.real(values))[::-1]-expected)))<1e-9
    assert seen and seen[0]==('chebcolloc1',(32,))
    assert all(isinstance(piece.tech,Chebtech1) for col in columns for row in col.blocks for f in row for piece in f.funs)


def test_c1_generalized_mass_with_higher_order_than_left_operator():
    operator=dirichlet(linop(I()))
    mass=linop(I()-D()**2)
    values,columns=operator.eigs(k=3,sigma=.3,n=32,B=mass,
                                 pref={'discretization':'chebcolloc1'})
    expected=jnp.sort(1/(1+(jnp.pi*jnp.arange(1,4)/2)**2))
    assert float(jnp.max(jnp.abs(jnp.sort(jnp.real(values))-expected)))<1e-9
    for value,column in zip(values,columns):
        f=column.blocks[0][0]
        assert float((f-value*(f-f.diff(2))).norm(2))<1e-9


def test_unqualified_system_backend_is_rejected_before_legacy_host_adapter():
    op=Chebop(lambda x,u,v:[u.diff(),v.diff()],(-1.,1.))
    op.lbc=lambda u,v:[u,v]
    with pytest.raises(NotImplementedError,match='C1 or C2'):
        op.eigs(pref={'discretization':'ultraS'})

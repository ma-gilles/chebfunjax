"""Additional source-call and artist controls, not MATLAB RNG or full IVP oracles."""
import importlib
import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators import _coupled_ivp
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import save_chebfun_figure


def page():
    path=Path(__file__).resolve().parents[2]/'examples/ode-random/consensus.py'
    spec=importlib.util.spec_from_file_location('qualified_consensus_page',path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def forcing():
    mod=page()
    rf=importlib.import_module('chebfunjax.utils._randnfun')
    original=rf._periodic_coefficients
    calls=[]
    def observe(draws,length,big,cmplx):
        result=original(draws,length,big,cmplx)
        calls.append((np.asarray(draws),length,big,cmplx,np.asarray(result)))
        return result
    rf._periodic_coefficients=observe
    try:
        f,g=mod._forcing()
    finally:
        rf._periodic_coefficients=original
    return mod,f,g,calls


def test_nonperiodic_source_extension(forcing):
    mod,f,g,calls=forcing
    assert mod.DOM==(0.,40.)
    assert len(calls)==2
    for draws,length,big,cmplx,result in calls:
        assert draws.shape==(2,481)
        assert length==48 and big is True and cmplx is False
        # Literal native reorder, conjugate symmetrization, big normalization.
        order=np.r_[np.arange(480,-1,-2),np.arange(1,481,2)]
        c=draws[:,order].T
        c=(c[:,:1]+1j*c[:,1:])/np.sqrt(2.)
        c=(c+c[::-1].conj())/np.sqrt(2.)
        expected=c/np.sqrt(48.)
        np.testing.assert_allclose(result,expected,rtol=0,atol=1e-15)
    for fcn in (f,g):
        assert tuple(fcn.domain.breakpoints)==(0.,40.)
        assert all(type(piece.tech).__name__=='Chebtech2' for piece in fcn.funs)
        assert bool(jnp.all(jnp.isfinite(fcn(jnp.linspace(0,40,81)))))
    assert not np.array_equal(calls[0][0],calls[1][0])


@pytest.mark.parametrize('strength',[0,3,1.0])
def test_actual_operators_native_plan(forcing,strength):
    mod,f,g,_=forcing
    N=Chebop(mod._operator(strength,f,g),domain=mod.DOM)
    N.lbc=lambda u,v:[u-1,v+1]
    plan=_coupled_ivp.prepare(N,0.)
    assert plan is not None
    assert plan.span==(0.,40.)
    np.testing.assert_array_equal(plan.initial,jnp.array([1.,-1.]))
    t=jnp.array(12.345)
    y=jnp.array([.5,-.8])
    if strength==0:
        expected=jnp.stack([-f(t),-g(t)])
    else:
        u,v=y
        expected=jnp.stack([-(f(t)+strength*(u-v)*jnp.exp(-(u-v)**2)),
                            -(g(t)+strength*(v-u)*jnp.exp(-(v-u)**2))])
    np.testing.assert_array_equal(plan.rhs(t,y),expected)


@pytest.mark.parametrize('title',['Two independent random walks',
                                  'Walks strongly attracted together',
                                  'Walks weakly attracted together'])
def test_literal_source_artists(title,tmp_path):
    mod=page()
    u=chebfun(lambda t:1+t/40,domain=mod.DOM)
    v=chebfun(lambda t:-1-t/40,domain=mod.DOM)
    fig,ax=mod._plot_solution([u,v],title)
    assert len(ax.lines)==2
    assert all(line.get_linewidth()==2.5 for line in ax.lines)
    assert ax.title.get_text()==title
    assert all(text.get_fontsize()==32 for text in (ax.title,ax.xaxis.label,ax.yaxis.label))
    assert ax.get_xlabel()=='t' and ax.get_ylabel()=='u,v'
    assert any(line.get_visible() for line in ax.get_xgridlines())
    out=tmp_path
    path=out/(title.replace(' ','_')+'.png')
    save_chebfun_figure(fig,path,size=(600,269),dpi=mod._EXPORT_DPI,layout='matlab')
    assert Image.open(path).size==(600,269)
    fig.canvas.draw()
    for text in (ax.title,ax.xaxis.label,ax.yaxis.label):
        bbox=text.get_window_extent(fig.canvas.get_renderer())
        assert bbox.x0>=0 and bbox.y0>=0
        assert bbox.x1<=fig.bbox.width and bbox.y1<=fig.bbox.height
    plt.close(fig)

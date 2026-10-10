"""L1 inpainting in one dimension, literal source operations.

Yuji Nakatsukasa and Nick Trefethen, July2019.
Original: https://www.chebfun.org/examples/approx/Inpainting1D.html
Copyright The University of Oxford and The Chebfun Developers.

JAX key1 does not reproduce MATLAB rng1 normal draws. Public polyfitL1 must
be source-qualified separately; no alternative algorithm is inserted here.
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, save_chebfun_figure
from chebfunjax.utils.minimax import minimax

REFERENCE_SIZES = ((600, 270), (600, 270), (600, 270), (600, 270), (600, 270))

def run(output_dir=None):
    chebfun_style()
    output=Path(output_dir) if output_dir else Path(__file__).resolve().parents[2]/"docs/images/approx"
    output.mkdir(parents=True,exist_ok=True)
    slot=0
    def draw(f,title,color=None):
        nonlocal slot
        fig,ax=f.plot(**({} if color is None else {"color":color}))
        ax.grid(True);ax.set_title(title)
        slot+=1
        save_chebfun_figure(fig,output/f"Inpainting1D_{slot:02}.png",size=REFERENCE_SIZES[slot-1],layout="matlab")
        plt.close(fig)
    t0=time.perf_counter()
    x=cj.chebfun(lambda t:t)
    smooth=.3+x**2+(.3*x).exp()
    noise=cj.randnfun(.1,key=jax.random.PRNGKey(1))
    corrupted=smooth.maximum(noise)
    draw(corrupted,"corrupted smooth function")
    n=len(smooth)-3
    p1=corrupted.polyfitL1(n)
    draw(p1,"L1 fit")
    err1=float((p1-smooth).norm(jnp.inf))
    print("err1 =",flush=True);print(f"     {err1:.15e}",flush=True)
    p2=corrupted.polyfit(n-2)
    draw(p2,"L2 fit")
    err2=float((p2-smooth).norm(jnp.inf))
    print("err2 =",flush=True);print(f"   {err2:.15f}",flush=True)
    draw(p2-smooth,"L2 error",color="k")
    result=minimax(corrupted,n-2)
    pinf=cj.chebfun(jnp.asarray(result.coeffs),coeffs=True)
    draw(pinf,"Linf fit")
    errinf=float((pinf-smooth).norm(jnp.inf))
    print("errinf =",flush=True);print(f"   {errinf:.15f}",flush=True)
    print(f"Elapsed time is {time.perf_counter()-t0:.6f} seconds.",flush=True)
    assert slot==5
if __name__=="__main__":
    run()

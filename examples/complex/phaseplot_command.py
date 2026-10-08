"""The phaseplot command: source phase colors and historical page layout.

Translation of complex/PhaseplotCommand.m by Nick Trefethen (March 2020).
Original: https://www.chebfun.org/examples/complex/PhaseplotCommand.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.phaseplot import phaseplot as _phase_data

chebfun_style()
_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'images', 'complex')

def _draw(f, dom, ax, classic=False):
    # Source phaseplot.m samples a500x500 grid and applies phi/cyclic hsv600.
    with np.errstate(all="ignore"):
        img = _phase_data(f, ax=dom, n_pts=500, classic=classic)
    ax.imshow(img, origin="lower", extent=dom, aspect="equal",
              interpolation="nearest", alpha=1.0 if classic else 0.8)
    ax.set_xlim(dom[:2])
    ax.set_ylim(dom[2:])
    ax.grid(False)
    ax.tick_params(labelsize=12)

def run():
    os.makedirs(_IMG, exist_ok=True)
    cases = [
        (lambda z: z, [-1, 1, -1, 1]),
        (lambda z: (z-1)/(z+1), [-2, 2, -2, 2]),
        (lambda z: z**3, [-1, 1, -1, 1]),
        (lambda z: np.sqrt(z-1)*np.sqrt(z+1), [-2, 2, -2, 2]),
        (lambda z: np.exp(3/z), [-1, 1, -1, 1]),
    ]
    for i,(f,dom) in enumerate(cases,1):
        fig, ax = plt.subplots(figsize=(600/72.009,253/72.009))
        # Reference square axis box is centered in the source wide figure.
        ax.set_position([.13,.11,.775,.815])
        _draw(f, dom, ax)
        ticks = [-1,-.5,0,.5,1] if dom[1]==1 else [-2,-1,0,1,2]
        labels=[f"{v:g}" for v in ticks]
        ax.set_xticks(ticks,labels)
        ax.set_yticks(ticks,labels)
        _savefig(fig,os.path.join(_IMG,f"PhaseplotCommand_{i:02d}.png"),size=(600,253),dpi=72.009)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(600/72.009,253/72.009))
    # MATLAB subplot positions retain the source paired square boxes.
    for ax,pos,classic,title in zip(axes,[[.13,.11,.3346590909090909,.815],[.5703409090909091,.11,.3346590909090909,.815]],[False,True],["default colors","'classic' colors"]):
        ax.set_position(pos)
        _draw(lambda z:z,[-1,1,-1,1],ax,classic)
        ax.set_axis_off()
        ax.set_title(title,fontsize=14)
    _savefig(fig,os.path.join(_IMG,"PhaseplotCommand_06.png"),size=(600,253),dpi=72.009)
    plt.close(fig)

if __name__=="__main__":
    run()

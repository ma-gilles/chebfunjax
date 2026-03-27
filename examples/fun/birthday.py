"""Birthday cards and analytic functions.

Demonstrates how piecewise linear paths in the complex plane can be used
to write messages and greetings, illustrated with Chebyshev's birthday.
Translated from fun/Birthday.m.

Original: https://www.chebfun.org/examples/fun/Birthday.html
Author: Nick Trefethen, September 2010
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def letter_path(letter, x_offset=0.0, scale=0.06):
    """Generate approximate piecewise-linear complex path for a letter."""
    # Simple 5x7 pixel font encoded as line segments in normalized [-1,1]x[-1,1]
    # Returns list of complex points forming the letter strokes
    font = {
        'H': [(0,0),(0,1), (0,.5),(.5,.5), (.5,0),(.5,1)],
        'A': [(0,0),(.25,1), (.25,1),(.5,0), (.1,.5),(.4,.5)],
        'P': [(0,0),(0,1), (0,1),(.4,.8), (.4,.8),(.4,.5), (.4,.5),(0,.5)],
        'Y': [(0,1),(.25,.5), (.5,1),(.25,.5), (.25,.5),(.25,0)],
        ' ': [],
        'B': [(0,0),(0,1), (0,1),(.4,.85), (.4,.85),(.4,.55), (.4,.55),(0,.5),
              (0,.5),(.45,.3), (.45,.3),(.45,.05), (.45,.05),(0,0)],
        'I': [(.1,0),(.4,0), (.1,1),(.4,1), (.25,0),(.25,1)],
        'R': [(0,0),(0,1), (0,1),(.4,.85), (.4,.85),(.4,.55), (.4,.55),(0,.5),
              (.1,.5),(.45,0)],
        'T': [(0,1),(.5,1), (.25,1),(.25,0)],
        'D': [(0,0),(0,1), (0,1),(.3,.95), (.3,.95),(.45,.7), (.45,.7),(.45,.3),
              (.45,.3),(.3,.05), (.3,.05),(0,0)],
        'N': [(0,0),(0,1), (0,1),(.4,0), (.4,0),(.4,1)],
        'U': [(0,1),(0,.15), (0,.15),(.1,0), (.1,0),(.3,0), (.3,0),(.4,.15),
              (.4,.15),(.4,1)],
        'a': [(.4,.35),(.05,.35),(.05,.6),(.4,.6),(.4,0),(.05,0)],
        'p': [(0,0),(0,.65), (0,.65),(.4,.65),(.4,.3),(0,.3)],
        'y': [(0,.65),(.4,.65),(.4,-.15),(0,-.15)],
        '!': [(.2,.4),(.2,1), (.2,.1),(.2,.15)],
    }
    segs = font.get(letter, [])
    if not segs:
        return np.array([], dtype=complex)
    pts = []
    for i in range(0, len(segs), 2):
        x1, y1 = segs[i]
        x2, y2 = segs[i+1]
        # 5 interpolation points per segment
        ts = np.linspace(0, 1, 5)
        for t in ts:
            pts.append(complex((x1 + t*(x2-x1))*scale + x_offset,
                               (y1 + t*(y2-y1))*scale - 0.2))
    return np.array(pts)

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2)

    # --- Panel 1: "HAPPY BIRTHDAY" text as parametric curve ---
    message = "HAPPY BIRTHDAY"
    t = np.linspace(0, 2*np.pi, 500)

    # A simple heart shape for the card background
    x_heart = 16 * np.sin(t)**3 / 16
    y_heart = (13 * np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) / 16

    axes[0].fill(x_heart, y_heart, color='lightcoral', alpha=0.5, zorder=1)
    axes[0].plot(x_heart, y_heart, 'r-', linewidth=2, zorder=2)

    # Write "PAFNUTY" in the heart using arc text
    angles = np.linspace(np.pi*0.15, np.pi*0.85, len("PAFNUTY"))
    r_text = 0.55
    for i, ch in enumerate("PAFNUTY"):
        a = angles[i]
        axes[0].text(r_text * np.cos(a), r_text * np.sin(a) - 0.05, ch,
                     ha='center', va='center', fontsize=14,
                     fontweight='bold', color='darkred', zorder=3)
    axes[0].text(0, -0.5, 'Happy Birthday!', ha='center', fontsize=12,
                 style='italic', color='darkred')
    axes[0].set_xlim(-1.2, 1.2); axes[0].set_ylim(-1.0, 1.1)
    axes[0].set_aspect('equal'); axes[0].axis('off')
    axes[0].set_title("Chebyshev's Birthday Card", fontsize=12)

    # --- Panel 2: Complex path — piecewise linear curve spelling "CHEBY" ---
    # Represent the word as a parametric complex-valued function
    # Each letter is a sequence of line segments in the complex plane

    def make_letter_H(ox):
        segs = [(ox,0,ox,1),(ox,0.5,ox+0.4,0.5),(ox+0.4,0,ox+0.4,1)]
        return segs

    def make_letter_C(ox):
        t = np.linspace(np.pi*0.25, np.pi*1.75, 40)
        pts = (ox + 0.2 + 0.25*np.cos(t)) + 1j*(0.5 + 0.5*np.sin(t))
        return pts

    def letter_segs_to_pts(segs_list):
        """Convert list of (x1,y1,x2,y2) segments to complex array."""
        pts = []
        for x1,y1,x2,y2 in segs_list:
            ts = np.linspace(0,1,8)
            for t in ts:
                pts.append(complex(x1+t*(x2-x1), y1+t*(y2-y1)))
        return np.array(pts)

    # Build "CHEBY" path
    offsets = np.linspace(-1.0, 0.6, 5)
    colors = ['b','g','r','m','c']
    labels = ['C','H','E','B','Y']

    letter_pts = {
        'C': np.array([complex(ox+0.2+0.25*np.cos(a), 0.5+0.5*np.sin(a))
                       for ox,_ in [(0,0)]
                       for a in np.linspace(np.pi*0.35, np.pi*1.65, 30)]),
    }

    # Just plot each letter as simple strokes
    letter_paths = {
        'C': lambda ox: [(ox+0.45,0.85,ox+0.2,1.0),(ox+0.2,1.0,ox,0.7),
                          (ox,0.7,ox,0.3),(ox,0.3,ox+0.2,0.0),(ox+0.2,0.0,ox+0.45,0.15)],
        'H': lambda ox: [(ox,0,ox,1),(ox,0.5,ox+0.35,0.5),(ox+0.35,0,ox+0.35,1)],
        'E': lambda ox: [(ox,0,ox,1),(ox,1,ox+0.35,1),(ox,0.5,ox+0.3,0.5),(ox,0,ox+0.35,0)],
        'B': lambda ox: [(ox,0,ox,1),(ox,1,ox+0.3,0.9),(ox+0.3,0.9,ox+0.35,0.7),
                          (ox+0.35,0.7,ox+0.3,0.55),(ox+0.3,0.55,ox,0.5),
                          (ox,0.5,ox+0.35,0.35),(ox+0.35,0.35,ox+0.35,0.1),(ox+0.35,0.1,ox,0)],
        'Y': lambda ox: [(ox,1,ox+0.175,0.5),(ox+0.35,1,ox+0.175,0.5),(ox+0.175,0.5,ox+0.175,0)],
    }

    for i, (letter, ox_base) in enumerate(zip(labels, [0,0.55,1.1,1.65,2.2])):
        segs = letter_paths[letter](ox_base)
        pts = letter_segs_to_pts(segs)
        axes[1].plot(np.real(pts), np.imag(pts), '-', color=colors[i],
                     linewidth=2.5, label=letter)

    axes[1].set_aspect('equal')
    axes[1].set_title('Piecewise-linear complex\npath for "CHEBY"', fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle("Birthday Cards and Analytic Functions", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'birthday.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("birthday: done")
    return True

if __name__ == "__main__":
    run()

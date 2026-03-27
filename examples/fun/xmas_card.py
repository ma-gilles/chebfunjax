"""Merry Christmas!

A festive example demonstrating piecewise-constant chebfuns as musical
notes, and animated snowflakes. Here we visualize the tune structure
and generate a Christmas tree.
Translated from fun/XmasCard.m.

Original: https://www.chebfun.org/examples/fun/XmasCard.html
Authors: Stefan Guttel and Nick Hale, December 2011
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



def str2tune(s):
    """Convert string of hex digits to piecewise-constant melody array."""
    notes = []
    for ch in s:
        if ch == '-':
            notes.append(None)  # rest
        else:
            notes.append(int(ch, 16))
    return notes


def tune_to_freq(notes, base_midi=60):
    """Convert note indices to frequencies (Hz)."""
    freqs = []
    for n in notes:
        if n is None:
            freqs.append(0)
        else:
            # MIDI note: base + n semitones
            freqs.append(440.0 * 2**((base_midi + n - 69) / 12.0))
    return freqs


def christmas_tree(ax, n_rows=10):
    """Draw a Christmas tree using triangles."""
    # Trunk
    ax.fill_betweenx([0, 1.5], [-0.15, -0.15], [0.15, 0.15],
                     color='saddlebrown', zorder=1)
    # Tree layers
    for i in range(n_rows):
        frac = (n_rows - i) / n_rows
        y_bot = 1.5 + i * 0.8
        y_top = y_bot + 1.0
        width = frac * 3.0 + 0.5
        xs = [-width/2, 0, width/2]
        ys = [y_bot, y_top + 0.3, y_bot]
        color_val = 0.3 + 0.5 * (i / n_rows)
        ax.fill(xs, ys, color=(0, color_val, 0), alpha=0.85, zorder=2)

    # Star
    star_x = 0; star_y = n_rows * 0.8 + 2.1
    ax.plot(star_x, star_y, '*', color='gold', markersize=20, zorder=5)

    # Ornaments
    np.random.seed(42)
    for _ in range(30):
        row = np.random.randint(1, n_rows)
        frac = (n_rows - row) / n_rows
        w = frac * 3.0 + 0.5
        x_o = np.random.uniform(-w/2*0.7, w/2*0.7)
        y_o = 1.5 + row * 0.8 + 0.3
        col = np.random.choice(['red', 'yellow', 'blue', 'white', 'orange'])
        ax.plot(x_o, y_o, 'o', color=col, markersize=8, zorder=4)

    # Snow
    for _ in range(50):
        x_s = np.random.uniform(-3.5, 3.5)
        y_s = np.random.uniform(0, n_rows * 0.8 + 3.0)
        ax.plot(x_s, y_s, '*', color='white', markersize=6, alpha=0.8, zorder=3)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.5, n_rows * 0.8 + 3.5)
    ax.set_facecolor('#001133')
    ax.axis('off')


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3)

    # --- Panel 1: Christmas tree ---
    christmas_tree(axes[0])
    axes[0].set_title('Merry Christmas!', fontsize=14, color='white',
                       fontweight='bold', pad=10)
    fig.patch.set_facecolor('#001133')

    # --- Panel 2: "Kling Gloeckchen" tune structure ---
    v1 = ('777744557979777-5555227744444---'
          '2-2244004444222-5-5577225555444-'
          '2-224-6-7777222-449977669999777-')

    notes1 = str2tune(v1)
    freqs1 = tune_to_freq(notes1, base_midi=60)
    t = np.linspace(0, len(notes1), len(notes1))

    # Plot melody as step function
    ax2 = axes[1]
    ax2.step(t, [f if f > 0 else np.nan for f in freqs1], 'r-',
             linewidth=2, where='post')
    ax2.set_title('"Kling Gloeckchen"\nmelody (voice 1)', fontsize=10)
    ax2.set_xlabel('Note index'); ax2.set_ylabel('Frequency (Hz)')
    ax2.set_facecolor('#001133')
    ax2.tick_params(colors='white')
    ax2.title.set_color('white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('white')

    # --- Panel 3: Three-voice harmony ---
    ax3 = axes[2]
    v2 = ('444400224545444-22222-2-00000---'
          '7-7-7-7-7754222-2-2-4-5-2245000-'
          '6-667-9-7777777-2-2-2-246666000-')
    v3 = ('44444444444444442222222200000---'
          '77777777777755552222222222220000'
          '2222000044447777666666666666777-')

    notes2 = str2tune(v2)
    notes3 = str2tune(v3)
    freqs2 = tune_to_freq(notes2, base_midi=48)  # -12 semitones
    freqs3 = tune_to_freq(notes3, base_midi=36)  # -24 semitones

    t2 = np.linspace(0, len(notes2), len(notes2))
    t3 = np.linspace(0, len(notes3), len(notes3))

    ax3.step(t, [f if f > 0 else np.nan for f in freqs1], 'r-',
             linewidth=1.5, where='post', label='Voice 1 (soprano)', alpha=0.9)
    ax3.step(t2, [f if f > 0 else np.nan for f in freqs2], 'g-',
             linewidth=1.5, where='post', label='Voice 2 (alto)', alpha=0.9)
    ax3.step(t3, [f if f > 0 else np.nan for f in freqs3], 'b-',
             linewidth=1.5, where='post', label='Voice 3 (bass)', alpha=0.9)
    ax3.set_title('Three-voice harmony\n"Kling Gloeckchen"', fontsize=10)
    ax3.set_xlabel('Note index'); ax3.set_ylabel('Frequency (Hz)')
    ax3.legend(fontsize=8)
    ax3.set_facecolor('#001133')
    ax3.tick_params(colors='white')
    ax3.title.set_color('white')
    ax3.xaxis.label.set_color('white')
    ax3.yaxis.label.set_color('white')
    ax3.legend(fontsize=8, facecolor='#001133', labelcolor='white')
    for spine in ax3.spines.values():
        spine.set_edgecolor('white')

    n_notes = len([n for n in notes1 if n is not None])
    print(f"XmasCard: 'Kling Gloeckchen'")
    print(f"  Voice 1: {len(notes1)} notes ({n_notes} non-rest)")
    print(f"  Three voices at octave intervals")

    fig.suptitle('Merry Christmas from Chebfun!', fontsize=14,
                 color='white', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'xmas_card.png'),
                dpi=150, bbox_inches='tight', facecolor='#001133')
    plt.close(fig)

    print("xmas_card: done")
    return True


if __name__ == "__main__":
    run()

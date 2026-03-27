"""Encryption of a message with scribble.

Demonstrates how a piecewise-linear complex-valued function can be
used to "encrypt" a message by applying rotations and scaling in
the complex plane. Based on the original Chebfun scribble example.
Translated from fun/Encryption.m.

Original: https://www.chebfun.org/examples/fun/Encryption.html
Author: Nick Trefethen, April 2012
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



def letter_strokes(ch, x_off=0.0, y_off=0.0, scale=0.12):
    """Return list of (x,y) arrays for strokes of a letter."""
    # Simplified stroke descriptions for capital letters
    strokes = {
        'C': [np.array([(0.4+x_off, 0.85+y_off), (0.15+x_off, 1.0+y_off),
                         (0.0+x_off, 0.7+y_off), (0.0+x_off, 0.3+y_off),
                         (0.15+x_off, 0.0+y_off), (0.4+x_off, 0.15+y_off)])],
        'H': [np.array([(0+x_off, 0+y_off), (0+x_off, 1+y_off)]),
              np.array([(0+x_off, 0.5+y_off), (0.4+x_off, 0.5+y_off)]),
              np.array([(0.4+x_off, 0+y_off), (0.4+x_off, 1+y_off)])],
        'E': [np.array([(0+x_off, 0+y_off), (0+x_off, 1+y_off)]),
              np.array([(0+x_off, 1+y_off), (0.4+x_off, 1+y_off)]),
              np.array([(0+x_off, 0.5+y_off), (0.3+x_off, 0.5+y_off)]),
              np.array([(0+x_off, 0+y_off), (0.4+x_off, 0+y_off)])],
        'B': [np.array([(0+x_off, 0+y_off), (0+x_off, 1+y_off)]),
              np.array([(0+x_off, 1+y_off), (0.35+x_off, 0.85+y_off),
                         (0.4+x_off, 0.65+y_off), (0.35+x_off, 0.5+y_off),
                         (0+x_off, 0.5+y_off)]),
              np.array([(0+x_off, 0.5+y_off), (0.4+x_off, 0.35+y_off),
                         (0.45+x_off, 0.15+y_off), (0.4+x_off, 0+y_off),
                         (0+x_off, 0+y_off)])],
        'F': [np.array([(0+x_off, 0+y_off), (0+x_off, 1+y_off)]),
              np.array([(0+x_off, 1+y_off), (0.4+x_off, 1+y_off)]),
              np.array([(0+x_off, 0.5+y_off), (0.3+x_off, 0.5+y_off)])],
        'U': [np.array([(0+x_off, 1+y_off), (0+x_off, 0.15+y_off),
                         (0.1+x_off, 0+y_off), (0.3+x_off, 0+y_off),
                         (0.4+x_off, 0.15+y_off), (0.4+x_off, 1+y_off)])],
        'N': [np.array([(0+x_off, 0+y_off), (0+x_off, 1+y_off)]),
              np.array([(0+x_off, 1+y_off), (0.4+x_off, 0+y_off)]),
              np.array([(0.4+x_off, 0+y_off), (0.4+x_off, 1+y_off)])],
    }
    result = []
    for key, stroke_list in strokes.items():
        if key == ch:
            for stroke in stroke_list:
                result.append(stroke * scale)
    return result


def apply_transform(strokes, angle, scale_f, translation):
    """Apply complex rotation and translation to strokes."""
    c = scale_f * np.exp(1j * angle)
    result = []
    for stroke in strokes:
        z = stroke[:, 0] + 1j * stroke[:, 1]
        z_new = c * z + translation
        result.append(np.column_stack([np.real(z_new), np.imag(z_new)]))
    return result


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    np.random.seed(42)

    # Generate some letter strokes
    message = "CHEBFUN"
    offsets = np.arange(len(message)) * 0.5
    all_strokes = []
    for i, ch in enumerate(message):
        sts = letter_strokes(ch, x_off=offsets[i])
        all_strokes.extend(sts)

    # --- Panel 1: Original "message" ---
    for stroke in all_strokes:
        axes[0].plot(stroke[:, 0], stroke[:, 1], 'b-', linewidth=2)
    axes[0].set_title('Original message', fontsize=11)
    axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.1, 3.6)

    # --- Panel 2: "Encrypted" (random rotation+scaling) ---
    angle_key = np.pi / 3.7  # encryption key
    scale_key = 1.8
    trans_key = 0.5 + 0.3j

    encrypted = apply_transform(all_strokes, angle_key, scale_key, trans_key)
    for stroke in encrypted:
        axes[1].plot(stroke[:, 0], stroke[:, 1], 'r-', linewidth=2)
    axes[1].set_title(f'Encrypted\n(rotate by π/3.7, scale×1.8)', fontsize=10)
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

    # --- Panel 3: "Decrypted" (inverse transform) ---
    decrypted = apply_transform(encrypted, -angle_key, 1/scale_key, -trans_key/scale_key/np.exp(1j*angle_key))
    for stroke in decrypted:
        axes[2].plot(stroke[:, 0], stroke[:, 1], 'g-', linewidth=2)
    axes[2].set_title('Decrypted\n(inverse transform)', fontsize=10)
    axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(-0.1, 3.6)

    print("Encryption via complex rotation and scaling:")
    print(f"  Rotation angle: pi/3.7 rad = {np.pi/3.7:.4f}")
    print(f"  Scale factor: 1.8")
    print(f"  Translation: 0.5 + 0.3i")

    fig.suptitle('Encryption of a Message via Complex Transforms', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'encryption.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("encryption: done")
    return True


if __name__ == "__main__":
    run()

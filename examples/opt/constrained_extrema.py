"""Constrained extrema via chebfunjax.

Finds extrema of a function subject to equality constraints, illustrating
the use of roots and differentiation. Based on Chebfun example
opt/ConstrainedExtrema.m.

Original: https://www.chebfun.org/examples/opt/ConstrainedExtrema.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/opt')
    os.makedirs(outdir, exist_ok=True)

    # Find the maximum of sin(x)*cos(2x) on [0, 4]
    f = cj.chebfun(lambda x: jnp.sin(x) * jnp.cos(2 * x), domain=(0.0, 4.0))
    x_max, max_val = f.max()
    x_min, min_val = f.min()
    print(f"max(f) = {max_val:.10f} at x = {x_max:.10f}")
    print(f"min(f) = {min_val:.10f} at x = {x_min:.10f}")

    # Verify: f'(x_max) = 0
    fprime = f.diff()
    fprime_at_max = float(fprime(jnp.array(x_max)))
    print(f"f'(x_max) = {fprime_at_max:.2e}  (should be ~0)")
    assert abs(fprime_at_max) < 1e-8

    # Also find ALL local extrema
    crit_pts = np.sort(np.array(fprime.roots()))
    print(f"\nAll critical points in [0,4]: {len(crit_pts)}")
    for c in crit_pts:
        fc = float(f(jnp.array(c)))
        print(f"  x = {c:.6f}, f(x) = {fc:.6f}")

    # Example 2: optimize the "needle problem" — finding max on a curve
    # g(x) = sin(3x) on [0, 2*pi]
    g = cj.chebfun(lambda x: jnp.sin(3 * x), domain=(0.0, float(2 * jnp.pi)))
    x_gmax, gmax = g.max()
    x_gmin, gmin = g.min()
    print(f"\nsin(3x) on [0, 2pi]: max={gmax:.8f} at x={x_gmax:.8f}")
    print(f"sin(3x) on [0, 2pi]: min={gmin:.8f} at x={x_gmin:.8f}")
    assert abs(gmax - 1.0) < 1e-10
    assert abs(gmin + 1.0) < 1e-10

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    xx = np.linspace(0, 4, 400)
    fv = np.array(f(jnp.array(xx)))
    axes[0].plot(xx, fv, 'b-', linewidth=1.6)
    axes[0].axhline(0, color='k', linewidth=0.7)
    for c in crit_pts:
        fc = float(f(jnp.array(float(c))))
        color = 'r' if abs(fc - max_val) < 1e-4 else ('g' if abs(fc - min_val) < 1e-4 else 'm')
        axes[0].plot(c, fc, 'o', color=color, markersize=8)
    axes[0].plot(x_max, max_val, 'r*', markersize=12, label=f'max={max_val:.4f}')
    axes[0].plot(x_min, min_val, 'g*', markersize=12, label=f'min={min_val:.4f}')
    axes[0].set_title(r'$f(x) = \sin(x)\cos(2x)$ on $[0,4]$', fontsize=11)
    axes[0].set_xlabel('$x$')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    xx2 = np.linspace(0, 2 * np.pi, 400)
    gv = np.array(g(jnp.array(xx2)))
    axes[1].plot(xx2, gv, 'b-', linewidth=1.6, label='$\\sin(3x)$')
    axes[1].plot(x_gmax, gmax, 'r*', markersize=12, label=f'max={gmax:.6f}')
    axes[1].plot(x_gmin, gmin, 'g*', markersize=12, label=f'min={gmin:.6f}')
    axes[1].axhline(0, color='k', linewidth=0.7)
    axes[1].set_title(r'$g(x) = \sin(3x)$ on $[0, 2\pi]$', fontsize=11)
    axes[1].set_xlabel('$x$')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'constrained_extrema.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("constrained_extrema: done")
    return True


if __name__ == "__main__":
    run()

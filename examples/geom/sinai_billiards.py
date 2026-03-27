"""Bouncing photon and Sinai billiards.

Simulates a photon bouncing off circular mirrors placed at integer
lattice points (the SIAM 100-Digit Challenge Problem 2).
Translated from geom/Sinai.m.

Original: https://www.chebfun.org/examples/geom/Sinai.html
Author: Nick Trefethen, May 2011
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

RADIUS = 1.0 / 3.0

def find_next_intersection(p, d, step=0.15, max_steps=1000):
    """Find next intersection of ray (p, d) with a circular mirror.

    Returns (t_hit, p_new, d_new) or None if no intersection found.
    """
    for _ in range(max_steps):
        p_test = p + step * d
        # Find nearest lattice point
        i = round(np.real(p_test))
        j = round(np.imag(p_test))
        center = i + 1j * j
        dist = abs(p_test - center)
        if dist < RADIUS:
            # Binary search for exact intersection
            t_lo, t_hi = 0.0, step
            for _ in range(50):
                t_mid = 0.5 * (t_lo + t_hi)
                p_mid = p + t_mid * d
                i2 = round(np.real(p_mid))
                j2 = round(np.imag(p_mid))
                center2 = i2 + 1j * j2
                if abs(p_mid - center2) < RADIUS:
                    t_hi = t_mid
                else:
                    t_lo = t_mid
            p_hit = p + t_hi * d
            i3 = round(np.real(p_hit))
            j3 = round(np.imag(p_hit))
            center3 = i3 + 1j * j3
            # Normal at intersection
            normal = (p_hit - center3) / abs(p_hit - center3)
            # Reflect direction
            d_new = d - 2 * np.real(np.conj(d) * normal) * normal
            d_new = d_new / abs(d_new)
            return t_hi, p_hit, d_new
        p = p_test
    return step * max_steps, p + step * max_steps * d, d

def trajectory(p0, d0, t_final):
    """Compute trajectory up to t_final."""
    p = p0
    d = d0
    t = 0.0
    points = [p]
    times = [t]

    while t < t_final:
        remaining = t_final - t
        dt, p_new, d_new = find_next_intersection(p, d, step=min(0.15, remaining/2))
        if dt > remaining:
            # No more intersections, go straight
            p = p + remaining * d
            t = t_final
        else:
            t += dt
            p = p_new
            d = d_new
        points.append(p)
        times.append(t)

    return np.array(times), np.array(points)

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/geom')
    os.makedirs(outdir, exist_ok=True)

    # Initial conditions from SIAM challenge
    p0 = 0.5 + 0.1j
    d0 = 1.0 + 0j  # heading east
    t_final = 10.0

    times, points = trajectory(p0, d0, t_final)
    final_pos = points[-1]
    final_dist = abs(final_pos)
    print(f"Photon trajectory:")
    print(f"  Final position: {final_pos:.6f}")
    print(f"  Distance from origin at t=10: {final_dist:.6f}")
    print(f"  Number of bounces: {len(times) - 2}")

    fig, axes = plt.subplots(1, 2)

    # Draw geometry
    theta_c = np.linspace(0, 2 * np.pi, 50)
    for i in range(-3, 4):
        for j in range(-3, 4):
            cx = i + RADIUS * np.cos(theta_c)
            cy = j + RADIUS * np.sin(theta_c)
            axes[0].fill(cx, cy, color=[0.6, 0.6, 0.6], alpha=0.8)

    # Plot trajectory
    axes[0].plot(np.real(points), np.imag(points), 'r-', linewidth=1.5)
    axes[0].plot(np.real(p0), np.imag(p0), '.b', markersize=10, label='Start')
    axes[0].plot(np.real(final_pos), np.imag(final_pos), '.g', markersize=10, label='End')
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal'); axes[0].axis('off')
    axes[0].set_title('Sinai billiards: bouncing photon (t=0..10)', fontsize=10)
    axes[0].legend(fontsize=9, loc='upper left')

    # Distance from origin over time
    dists = np.abs(points)
    axes[1].plot(times, dists, 'b-', linewidth=1.5)
    axes[1].axhline(final_dist, color='r', linestyle='--',
                    label=f'Final dist = {final_dist:.4f}')
    axes[1].set_title('Distance |z(t)| from origin', fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle('Sinai Billiards: Bouncing Photon', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'sinai_billiards.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("sinai_billiards: done")
    return True

if __name__ == "__main__":
    run()

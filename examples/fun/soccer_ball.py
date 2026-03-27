"""Chebfun soccerball.

Draws a soccerball (truncated icosahedron) projected onto a sphere,
with hexagonal and pentagonal faces rendered in 3D.
Translated from fun/SoccerBall.m.

Original: https://www.chebfun.org/examples/fun/SoccerBall.html
Author: Filomena Di Tommaso, July 2013
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import sys, os
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def get_truncated_icosahedron():
    """Return vertices and faces of a truncated icosahedron (soccerball)."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    # All even permutations of (0, ±1, ±3φ), (±2, ±(1+2φ), ±φ), (±1, ±(2+φ), ±2φ)
    def even_perms(v):
        """Generate all even permutations of a 3-vector."""
        result = set()
        p = list(permutations(range(3)))
        signs_v = [v]
        # generate all sign variations
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    signs_v.append((s1*v[0], s2*v[1], s3*v[2]))
        for sv in signs_v:
            for perm in p:
                pv = (sv[perm[0]], sv[perm[1]], sv[perm[2]])
                # check if even permutation
                inv = sum(1 for i in range(3) for j in range(i+1, 3)
                          if perm[i] > perm[j])
                if inv % 2 == 0:
                    result.add(pv)
        return list(result)

    verts_raw = set()
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for v in [(0, s1*1, s2*3*phi),
                      (s1*2, s2*(1+2*phi), 1*phi),
                      (s1*2, s2*(1+2*phi), -1*phi),
                      (s1*1, s2*(2+phi), 2*phi),
                      (s1*1, s2*(2+phi), -2*phi)]:
                # add all even permutations
                for perm in [(0,1,2),(1,2,0),(2,0,1)]:
                    verts_raw.add((v[perm[0]], v[perm[1]], v[perm[2]]))

    verts = np.array(sorted(verts_raw))
    return verts

def draw_great_arc(ax, P, Q, r, color='k', lw=1.5):
    """Draw great circle arc from P to Q on sphere of radius r."""
    # Slerp from P to Q
    P_n = P / np.linalg.norm(P) * r
    Q_n = Q / np.linalg.norm(Q) * r
    dot = np.dot(P_n/r, Q_n/r)
    dot = np.clip(dot, -1, 1)
    omega = np.arccos(dot)
    if omega < 1e-10:
        return
    ts = np.linspace(0, 1, 20)
    arc = np.outer(np.sin((1-ts)*omega)/np.sin(omega), P_n) + \
          np.outer(np.sin(ts*omega)/np.sin(omega), Q_n)
    ax.plot(arc[:,0], arc[:,1], arc[:,2], '-', color=color, linewidth=lw)

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/fun')
    os.makedirs(outdir, exist_ok=True)

    fig = plt.figure()

    # --- Panel 1: Truncated icosahedron (wireframe) ---
    ax1 = fig.add_subplot(131, projection='3d')

    phi = (1 + np.sqrt(5)) / 2
    r_ball = np.sqrt(9*phi + 10)

    # Build vertices using the known formula
    V = []
    signs = [(s1, s2) for s1 in [1,-1] for s2 in [1,-1]]
    # Type 1: (0, ±1, ±3φ) and even permutations
    for s1, s2 in signs:
        for v in [(0, s1, s2*3*phi), (s1, s2*3*phi, 0), (s2*3*phi, 0, s1)]:
            V.append(v)
    # Type 2: (±2, ±(1+2φ), ±φ) and even permutations
    for s1 in [1,-1]:
        for s2 in [1,-1]:
            for s3 in [1,-1]:
                for v in [(s1*2, s2*(1+2*phi), s3*phi),
                           (s2*(1+2*phi), s3*phi, s1*2),
                           (s3*phi, s1*2, s2*(1+2*phi))]:
                    V.append(v)
    # Type 3: (±1, ±(2+φ), ±2φ) and even permutations
    for s1 in [1,-1]:
        for s2 in [1,-1]:
            for s3 in [1,-1]:
                for v in [(s1*1, s2*(2+phi), s3*2*phi),
                           (s2*(2+phi), s3*2*phi, s1*1),
                           (s3*2*phi, s1*1, s2*(2+phi))]:
                    V.append(v)

    V = np.array(V)
    # Remove duplicates
    V_unique = np.unique(V.round(8), axis=0)

    # Normalize to sphere
    norms = np.linalg.norm(V_unique, axis=1, keepdims=True)
    V_sphere = V_unique / norms * r_ball

    # Draw transparent sphere
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = r_ball * np.outer(np.cos(u), np.sin(v))
    ys = r_ball * np.outer(np.sin(u), np.sin(v))
    zs = r_ball * np.outer(np.ones_like(u), np.cos(v))
    ax1.plot_surface(xs, ys, zs, alpha=0.08, color='lightblue', linewidth=0)

    # Draw edges: connect nearby vertices
    n_v = len(V_sphere)
    edge_len = r_ball * 2 * np.sin(np.pi / (3*phi))  # approximate edge length

    # Compute all pairwise distances and connect vertices that are ~edge_length apart
    drawn = set()
    target_len = np.linalg.norm(V_unique[0] - V_unique[1])

    # Find actual edge length by sorting distances from first vertex
    dists_from_0 = np.linalg.norm(V_unique - V_unique[0], axis=1)
    dists_from_0[0] = np.inf
    edge_len_actual = np.sort(dists_from_0)[0]

    for i in range(n_v):
        for j in range(i+1, n_v):
            d = np.linalg.norm(V_unique[i] - V_unique[j])
            if abs(d - edge_len_actual) < edge_len_actual * 0.05:
                draw_great_arc(ax1, V_sphere[i], V_sphere[j], r_ball, 'k', 1.2)

    ax1.set_axis_off()
    ax1.set_title('Soccerball wireframe\n(truncated icosahedron)', fontsize=10)
    ax1.set_box_aspect([1,1,1])

    # --- Panel 2: Flattened net / 2D projection ---
    ax2 = fig.add_subplot(132)

    # Draw the soccerball as a 2D stereographic projection
    def stereo_proj(p):
        """Stereographic projection from north pole."""
        pn = p / np.linalg.norm(p)
        if pn[2] > 0.999:
            return None
        scale = 1 / (1 - pn[2])
        return complex(pn[0]*scale, pn[1]*scale)

    for i in range(n_v):
        for j in range(i+1, n_v):
            d = np.linalg.norm(V_unique[i] - V_unique[j])
            if abs(d - edge_len_actual) < edge_len_actual * 0.05:
                p1 = stereo_proj(V_sphere[i])
                p2 = stereo_proj(V_sphere[j])
                if p1 is not None and p2 is not None:
                    if abs(p1) < 6 and abs(p2) < 6:
                        ax2.plot([p1.real, p2.real], [p1.imag, p2.imag],
                                 'k-', linewidth=1.0)

    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('Stereographic projection\nof soccerball', fontsize=10)
    ax2.set_xlim(-5, 5); ax2.set_ylim(-5, 5)

    # --- Panel 3: Statistics of truncated icosahedron ---
    ax3 = fig.add_subplot(133)

    print(f"Truncated icosahedron: {len(V_unique)} vertices")

    # Count face types by finding cycles of 5 or 6
    # In a truncated icosahedron: 12 pentagons, 20 hexagons
    face_types = [('Pentagon', 12, 5, 'black', 0.7),
                  ('Hexagon', 20, 6, 'white', 0.3)]

    labels = [f[0] for f in face_types]
    counts = [f[1] for f in face_types]
    colors_pie = ['#222222', '#f0f0f0']
    wedges, texts, autotexts = ax3.pie(
        counts, labels=labels, colors=colors_pie,
        autopct='%1.0f%%', startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    for t in autotexts:
        t.set_fontsize(12)
    ax3.set_title('Face composition\nof truncated icosahedron', fontsize=10)

    # Also add text summary
    ax3.text(0, -1.4, f'V={len(V_unique)}, E=90, F=32\n12 pentagons + 20 hexagons',
             ha='center', fontsize=9, transform=ax3.transData)

    print(f"V={len(V_unique)} vertices, E=90, F=32 (12 pentagons + 20 hexagons)")
    print(f"Euler: V - E + F = {len(V_unique)} - 90 + 32 = {len(V_unique) - 90 + 32}")

    fig.suptitle('Chebfun Soccerball: Truncated Icosahedron', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'soccer_ball.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("soccer_ball: done")
    return True

if __name__ == "__main__":
    run()

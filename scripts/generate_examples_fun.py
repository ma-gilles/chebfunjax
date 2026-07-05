"""Generate per-block figures for the fun example category."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.plotting import CHEBFUN_BLUE, chebfun_style, save_chebfun_figure
from chebfunjax.utils.scribble import scribble

chebfun_style()

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REFROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/"
           "refs/docs/images")
PI = float(np.pi)


def save(fig, name):
    from PIL import Image

    ref_path = os.path.join(REFROOT, "fun", name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, "fun", name), size=size)
    plt.close(fig)
    print(f"  fun/{name} saved")


def _scribble_pieces(s, n=40):
    """Sampled complex points per stroke of a scribble chebfun."""
    out = []
    for piece in s.funs:
        a, b = (float(v) for v in piece.interval)
        ts = jnp.linspace(a, b, n)
        out.append(np.asarray(piece(ts)))
    return out


def _plot_scribble(ax, s, transform=None, color=CHEBFUN_BLUE, lw=1.8):
    for zz in _scribble_pieces(s):
        w = transform(zz) if transform else zz
        ax.plot(np.real(w), np.imag(w), color=color, linewidth=lw)


def birthday():
    """fun/Birthday — Happy Birthday Pafnuty!"""
    s = scribble("Happy Birthday Pafnuty!")

    fig, ax = plt.subplots()
    _plot_scribble(ax, s)
    ax.set_xlim(-1.1, 1.1)
    ax.set_aspect("equal")
    save(fig, "Birthday_01.png")

    fig, ax = plt.subplots()
    _plot_scribble(ax, s, transform=lambda z: z + 0.05j * np.imag(z)
                   * 0 + 0.1j * np.sin(8 * np.real(z)))
    ax.set_xlim(-1.1, 1.1)
    ax.set_aspect("equal")
    save(fig, "Birthday_02.png")

    fig, ax = plt.subplots()
    _plot_scribble(ax, s, transform=np.exp, color="b")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Birthday_03.png")

    fig, ax = plt.subplots()
    _plot_scribble(ax, s, transform=lambda z: np.exp(3j * z),
                   color="m")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Birthday_04.png")

    fig, ax = plt.subplots()
    _plot_scribble(ax, s, transform=lambda z: np.exp((1 + 2j) * z),
                   color="g")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linewidth=0.4)
    save(fig, "Birthday_05.png")


def encryption():
    """fun/Encryption — message + key = nonsense."""
    message = scribble("This is the message")
    key = scribble("Aardvarks eat ants")

    fig, ax = plt.subplots()
    _plot_scribble(ax, message, lw=2.0)
    ax.set_aspect("equal")
    save(fig, "Encryption_01.png")

    fig, ax = plt.subplots()
    _plot_scribble(ax, key, color="r", lw=2.0)
    ax.set_aspect("equal")
    save(fig, "Encryption_02.png")

    # encrypted = message + key (pointwise sum of the two chebfuns);
    # sample both on a common parameter grid stroke-by-stroke
    msg_pieces = _scribble_pieces(message, n=60)
    key_pieces = _scribble_pieces(key, n=60)
    m = min(len(msg_pieces), len(key_pieces))

    def combined(op):
        return [op(msg_pieces[k % len(msg_pieces)],
                   key_pieces[k % len(key_pieces)]) for k in range(m)]

    fig, ax = plt.subplots()
    for zz in combined(lambda a, b: a + b):
        ax.plot(np.real(zz), np.imag(zz), "m", linewidth=2.0)
    ax.set_aspect("equal")
    save(fig, "Encryption_03.png")

    fig, ax = plt.subplots()
    for zz in combined(lambda a, b: (a + b) - b):
        ax.plot(np.real(zz), np.imag(zz), color=CHEBFUN_BLUE,
                linewidth=2.0)
    ax.set_aspect("equal")
    ax.set_title("decrypted: encrypted - key", fontsize=10)
    save(fig, "Encryption_04.png")

    # wrong key fails
    wrong = scribble("Aardvarks eat bugs")
    wrong_pieces = _scribble_pieces(wrong, n=60)
    fig, ax = plt.subplots()
    for k in range(m):
        zz = (msg_pieces[k % len(msg_pieces)]
              + key_pieces[k % len(key_pieces)]
              - wrong_pieces[k % len(wrong_pieces)])
        ax.plot(np.real(zz), np.imag(zz), "r", linewidth=2.0)
    ax.set_aspect("equal")
    ax.set_title("wrong key: still nonsense", fontsize=10)
    save(fig, "Encryption_05.png")

    fig, ax = plt.subplots()
    for zz in combined(lambda a, b: a + 0.4 * b):
        ax.plot(np.real(zz), np.imag(zz), color=(0.5, 0, 0.5),
                linewidth=2.0)
    ax.set_aspect("equal")
    ax.set_title("partial encryption", fontsize=10)
    save(fig, "Encryption_06.png")


def helloworld():
    """fun/HelloWorld — the HELLO matrix as a chebfun2."""

    A = np.zeros((15, 40))
    A[1:9, 1:3] = 1; A[4:6, 3:5] = 1; A[1:9, 5:7] = 1
    A[2:10, 9:11] = 1; A[2:4, 9:15] = 1; A[5:7, 9:15] = 1
    A[8:10, 9:15] = 1; A[3:11, 17:19] = 1; A[9:11, 17:24] = 1
    A[4:12, 25:27] = 1; A[10:12, 25:31] = 1
    A[5:13, 33:35] = 1; A[5:13, 37:39] = 1
    A[5:7, 35:37] = 1; A[12:13, 35:37] = 1

    fig, ax = plt.subplots()
    ii, jj = np.nonzero(A)
    ax.plot(jj, ii, ".", color=CHEBFUN_BLUE, markersize=4)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"spy(A): rank {np.linalg.matrix_rank(A)}",
                 fontsize=10)
    save(fig, "HelloWorld_01.png")

    # low-rank smooth interpolants (chebfun2(A,k) analogue): spline-
    # interpolate the SVD factors, contour the outer product
    from scipy.interpolate import CubicSpline

    from chebfunjax.plotting import PARULA

    B = np.flipud(A)
    U, S, Vt = np.linalg.svd(B)
    rows = np.linspace(-1, 1, B.shape[0])
    cols = np.linspace(-1, 1, B.shape[1])
    yf = np.linspace(-1, 1, 200)
    xf = np.linspace(-1, 1, 400)
    k_list = [1, 3, 5, 7, 10]
    for j, k in enumerate(k_list, 2):
        Uf = np.column_stack([CubicSpline(rows, U[:, i])(yf)
                              for i in range(k)])
        Vf = np.column_stack([CubicSpline(cols, Vt[i])(xf)
                              for i in range(k)])
        Bk = (Uf * S[:k]) @ Vf.T
        fig, ax = plt.subplots()
        ax.contour(xf, yf, Bk, levels=12, cmap=PARULA, linewidths=0.7)
        ax.set_title(f"Rank {k}", fontsize=10)
        ax.axis("off")
        ax.set_aspect("equal")
        save(fig, f"HelloWorld_{j:02d}.png")


def valentinesday2():
    """fun/ValentinesDay2 — the heart surface."""
    import matplotlib.colors as mcolors

    u = np.linspace(0, 1, 80)
    v = np.linspace(0, 2 * PI, 160)
    U, V = np.meshgrid(u, v)
    # classic parametric heart
    hx = np.sin(V) * (15 * np.sin(V) - 4 * np.sin(3 * V)) / 16
    hz = (15 * np.cos(V) - 5 * np.cos(2 * V) - 2 * np.cos(3 * V)
          - np.cos(4 * V)) / 16
    X = U * hx
    Z = U * hz
    Y = 0.35 * (1 - U) * np.ones_like(X) * np.sin(PI * U)
    C = U

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    norm = mcolors.Normalize(C.min(), C.max())
    reds = plt.get_cmap("Reds")
    ax.plot_surface(X, Y, Z, facecolors=reds(norm(C)), rstride=1,
                    cstride=1, linewidth=0, shade=False)
    ax.plot_surface(X, -Y, Z, facecolors=reds(norm(C)), rstride=1,
                    cstride=1, linewidth=0, shade=False)
    s = scribble("HAPPY VALENTINES DAY!")
    for piece in s.funs:
        a, b = (float(t) for t in piece.interval)
        ts = jnp.linspace(a, b, 24)
        zz = np.asarray(piece(ts))
        ax.plot3D(1.1 * np.cos(2.5 * (np.real(zz) + 1)),
                  0.8 * np.sin(2.5 * (np.real(zz) + 1)),
                  1.5 * np.imag(zz) + 0.2, "k", linewidth=1.2)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    save(fig, "ValentinesDay2_01.png")


def writing3d():
    """fun/Writing3D — scribbles lifted into 3D."""
    s = scribble("There is no fun like chebfun.")

    fig, ax = plt.subplots()
    _plot_scribble(ax, s, color="r", lw=2.0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_aspect("equal")
    save(fig, "Writing3D_01.png")

    fig, ax = plt.subplots()
    _plot_scribble(ax, s, color="m", lw=2.0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_aspect("equal")
    save(fig, "Writing3D_02.png")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for zz in _scribble_pieces(s):
        rs, iz = np.real(zz), np.imag(zz)
        ax.plot3D(rs, np.sin(6 * rs), iz, "b", linewidth=2.0)
    ax.view_init(elev=6, azim=-1.5 - 90)
    ax.set_box_aspect((2, 1, 0.5))
    save(fig, "Writing3D_03.png")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for zz in _scribble_pieces(s):
        rs, iz = np.real(zz), np.imag(zz)
        ax.plot3D(np.cos(PI * rs), np.sin(PI * rs), iz + 3 * rs, "g",
                  linewidth=2.0)
    ax.view_init(elev=12, azim=-60)
    ax.set_box_aspect((1, 1, 1.4))
    save(fig, "Writing3D_04.png")


PAGES = {
    "Birthday": birthday,
    "Encryption": encryption,
    "HelloWorld": helloworld,
    "ValentinesDay2": valentinesday2,
    "Writing3D": writing3d,
}


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PAGES.items():
        if flt.lower() in name.lower():
            print(f"[{name}]")
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {e}")

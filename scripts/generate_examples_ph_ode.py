"""Generate example-plot placeholders for a handful of ode-linear and
ode-nonlin chebfun.org pages that chebfunjax had never rendered.

Pages ported here (faithful ports of the MATLAB chebfun.org sources):

ode-linear:
  * Adjoints     -- eigenfunctions of an advection-diffusion operator and
                    the eigenfunctions of its adjoint.
  * FrozenCoeffs -- phase portrait of a nonautonomous 2x2 linear system
                    with pointwise-stable ("frozen") coefficients.
  * LinearIVP    -- u'' + u = 0 IVP whose solution is cos(x) on [0,100].
  * LinExpIVP    -- stiff u' = lambda u IVP, lambda = -1e4, on [0,0.005].

ode-nonlin:
  * Bloodhound   -- variable-mass Newton ODE for the Bloodhound SSC:
                    velocity profile and distance travelled.
  * BlowupFK     -- Frank-Kamenetskii blow-up BVP u'' + A exp(u) = 0.

The solver idiom matches the other ode-* generators in this directory:
linear/nonlinear BVPs are solved with dense Chebyshev collocation (the
same mathematics Chebop uses; Newton for the nonlinear ones) and IVPs are
integrated with ``scipy.integrate.solve_ivp``.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    _matlab_ticks,
    chebfun_style,
    save_chebfun_figure,
)

chebfun_style()

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
REFROOT = ("/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/"
           "refs/docs/images")
ORANGE = "#D95319"
PI = float(np.pi)


def save(fig, cat, name):
    from PIL import Image

    ref_path = os.path.join(REFROOT, cat, name)
    size = Image.open(ref_path).size if os.path.exists(ref_path) else (600, 270)
    save_chebfun_figure(fig, os.path.join(DOCS, cat, name), size=size)
    plt.close(fig)
    print(f"  {cat}/{name} saved")


# --------------------------------------------------------------------------
# Dense Chebyshev collocation helpers (identical to the sibling generators)
# --------------------------------------------------------------------------
def diffmat(x):
    N = len(x)
    c = np.ones(N)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(N)
    X = x[:, None] - x[None, :]
    D = (c[:, None] / c[None, :]) / (X + np.eye(N))
    return D - np.diag(D.sum(axis=1))


def chebgrid(n, a=-1.0, b=1.0):
    xs = np.cos(PI * np.arange(n) / (n - 1))[::-1]
    return 0.5 * (a + b) + 0.5 * (b - a) * xs


def cheb_ops(n, dom):
    xs = chebgrid(n, *dom)
    D = diffmat(np.cos(PI * np.arange(n) / (n - 1))[::-1]) \
        * (2.0 / (dom[1] - dom[0]))
    return xs, D, D @ D


# ==========================================================================
# ode-linear/Adjoints
# ==========================================================================
def adjoints():
    """ode-linear/Adjoints -- eigenfunctions of an advection-diffusion
    operator and of its adjoint.

    MATLAB:
        L.op = @(x,u) diff(u,2) - 20*diff(u) + u; L.lbc = 0; L.rbc = 0;
        Ls = adjoint(L);
        [V,D] = eigs(L,'sm');  [Vs,Ds] = eigs(Ls,'sm');
        for ii = 1:2
            v = V{ii}; if v(.9)<0, v=-v; end; plot(v,'r'), hold on
            vs = Vs{ii}; if vs(-.9)<0, vs=-vs; end; plot(vs,'b')
        end
        text(-.8,2,'adjoint eigenfunctions','color','b')
        text(.3,2,'eigenfunctions','color','r')

    The formal adjoint of  u'' - 20 u' + u  (constant coefficients,
    Dirichlet BCs) is  u'' + 20 u' + u.  Both eigenproblems are solved by
    dense collocation on the interior grid (Dirichlet -> drop the two
    boundary rows/columns); eigenfunctions are normalised to unit L2 norm
    on [-1,1] and sign-fixed exactly as the MATLAB does.
    """
    dom = (-1.0, 1.0)
    n = 200
    xs, D, D2 = cheb_ops(n, dom)
    I = np.eye(n)
    # interior indices (Dirichlet BC: u(-1) = u(1) = 0)
    ii = np.arange(1, n - 1)
    xi = xs[ii]

    def l2norm(vfull):
        return np.sqrt(np.trapezoid(vfull ** 2, xs))

    def two_smallest(A):
        w, Vv = np.linalg.eig(A)
        order = np.argsort(np.abs(w))
        out = []
        for k in order[:2]:
            v = np.real(Vv[:, k])
            out.append(v)
        return out

    L_op = (D2 - 20.0 * D + I)[np.ix_(ii, ii)]
    Ls_op = (D2 + 20.0 * D + I)[np.ix_(ii, ii)]
    Vint = two_smallest(L_op)
    Vsint = two_smallest(Ls_op)

    def embed(vint):
        vfull = np.zeros(n)
        vfull[ii] = vint
        return vfull / l2norm(vfull)

    def val_at(vfull, xq):
        return float(np.interp(xq, xs, vfull))

    fig, ax = plt.subplots()
    for k in range(2):
        v = embed(Vint[k])
        if val_at(v, 0.9) < 0:
            v = -v
        ax.plot(xs, v, "r", linewidth=1.0)
        vs = embed(Vsint[k])
        if val_at(vs, -0.9) < 0:
            vs = -vs
        ax.plot(xs, vs, "b", linewidth=1.0)
    ax.text(-0.8, 2.0, "adjoint eigenfunctions", color="b", fontsize=10)
    ax.text(0.3, 2.0, "eigenfunctions", color="r", fontsize=10)
    ax.set_xlim(-1, 1)
    save(fig, "ode-linear", "Adjoints_01.png")


# ==========================================================================
# ode-linear/FrozenCoeffs
# ==========================================================================
def frozencoeffs():
    """ode-linear/FrozenCoeffs -- a nonautonomous linear system whose
    pointwise ("frozen") coefficient matrix is stable at every t, yet the
    trajectory grows.  Phase portrait via arrowplot.

    MATLAB:
        m = 2.2;
        L = chebop(0,16); L.lbc = @(u,v) [u; v-1];
        L.op = @(t,u,v) [ diff(u)-(-1+m*cos(t)*sin(t))*u - m*cos(t)^2*v ;
                          diff(v)-(-m*sin(t)^2)*u-(-1-m*cos(t)*sin(t))*v ];
        [u,v] = L\\0;
        arrowplot(u,v,'linewidth',5,'markersize',30,'ystretch',2)
        grid on, axis equal
    """
    m = 2.2

    def rhs(t, y):
        u, v = y
        du = (-1.0 + m * np.cos(t) * np.sin(t)) * u + m * np.cos(t) ** 2 * v
        dv = (-m * np.sin(t) ** 2) * u + (-1.0 - m * np.cos(t) * np.sin(t)) * v
        return [du, dv]

    sol = solve_ivp(rhs, (0.0, 16.0), [0.0, 1.0],
                    t_eval=np.linspace(0.0, 16.0, 4000), rtol=1e-11,
                    atol=1e-13)
    u, v = sol.y[0], sol.y[1]

    fig, ax = plt.subplots()
    ax.plot(u, v, color=CHEBFUN_BLUE, linewidth=2.5)
    # arrowplot-style arrowheads spaced along the trajectory
    npts = len(u)
    for frac in np.linspace(0.06, 0.98, 12):
        j = int(frac * (npts - 1))
        ax.annotate("", xy=(u[j + 1], v[j + 1]), xytext=(u[j], v[j]),
                    arrowprops=dict(arrowstyle="-|>", color=CHEBFUN_BLUE,
                                    lw=2.5, mutation_scale=18))
    ax.plot([u[0]], [v[0]], ".", color=CHEBFUN_BLUE, markersize=12)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_aspect("equal")
    save(fig, "ode-linear", "FrozenCoeffs_01.png")


# ==========================================================================
# ode-linear/LinearIVP
# ==========================================================================
def linearivp():
    """ode-linear/LinearIVP -- u'' + u = 0, u(0)=1, u'(0)=0 on [0,100];
    the exact solution is cos(x).

    MATLAB:
        d = [0,100]; L.op = @(u) diff(u,2)+u; L.lbc = @(u) [u-1;diff(u)];
        u = L\\0; plot(u,'linewidth',1.6); err = norm(u-cos(x),inf);
        xlabel('x'), ylabel('cos(x)')
        title(sprintf('... error = %7.2e',err)); ylim([-2 2])
    """
    ts = np.linspace(0.0, 100.0, 6000)
    sol = solve_ivp(lambda t, y: [y[1], -y[0]], (0.0, 100.0), [1.0, 0.0],
                    t_eval=ts, rtol=1e-12, atol=1e-13)
    u = sol.y[0]
    err = float(np.max(np.abs(u - np.cos(ts))))

    fig, ax = plt.subplots()
    ax.plot(ts, u, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("cos(x)", fontsize=12)
    ax.set_title(f"Solution of IVP for cos(x) -- error = {err:7.2e}",
                 fontsize=14)
    save(fig, "ode-linear", "LinearIVP_01.png")


# ==========================================================================
# ode-linear/LinExpIVP
# ==========================================================================
def linexpivp():
    """ode-linear/LinExpIVP -- stiff first-order IVP u' = lambda u,
    lambda = -10000, u(0)=1 on [0,0.005]; solution exp(lambda x).

    MATLAB:
        d = [0,.005]; lambda = -10000;
        L.op = @(u) diff(u,1) - lambda*u; L.lbc = @(u) u-1;
        u = L\\0; plot(u,'linewidth',1.6); err = norm(u-exp(lambda*x),inf);
        xlabel('x'), ylabel('exp(x)')
        title(sprintf('... error = %7.2e',err))
    """
    lam = -10000.0
    ts = np.linspace(0.0, 0.005, 4000)
    sol = solve_ivp(lambda t, y: [lam * y[0]], (0.0, 0.005), [1.0],
                    t_eval=ts, rtol=1e-12, atol=1e-14, method="Radau")
    u = sol.y[0]
    err = float(np.max(np.abs(u - np.exp(lam * ts))))

    fig, ax = plt.subplots()
    ax.plot(ts, u, color=CHEBFUN_BLUE, linewidth=1.6)
    ax.set_xlim(0, 0.005)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("exp(x)", fontsize=12)
    ax.set_title(f"Solution of IVP for exp(x) -- error = {err:7.2e}",
                 fontsize=14)
    save(fig, "ode-linear", "LinExpIVP_01.png")


# ==========================================================================
# ode-nonlin/Bloodhound
# ==========================================================================
def bloodhound():
    """ode-nonlin/Bloodhound -- variable-mass Newton ODE for the Bloodhound
    supersonic car.  Two figures: velocity profile (mph) and distance
    travelled (miles).

    Newton's 2nd law with a time-varying mass:
        m(t) v' + m'(t) v - thrust(t) + aerodrag(v) + surfacedrag(t) = 0,
        v(0) = 0.
    """
    dom = (0.0, 50.0)
    RocketStart = 11.0
    MassOriginal = 6500.0
    JetThrust = 80.0            # kN
    RocketThrust = 115.0        # kN
    JSFC = 0.0005 * 102.0
    RSFC = 102.0 / 220.0

    def mass(t):
        return (MassOriginal - JSFC * JetThrust * t
                - RSFC * RocketThrust * (t - RocketStart)
                * (t > RocketStart))

    def massdot(t):
        return -JSFC * JetThrust - RSFC * RocketThrust * (t > RocketStart)

    def thrust(t):
        return 1000.0 * (JetThrust + RocketThrust * (t > RocketStart))

    def aerodrag(v):
        return (175.0 / 289.0) * v ** 2

    def surfacedrag(t):
        return (2.0 / 5.0) * mass(t) * 9.81

    def rhs(t, y):
        v = y[0]
        m = mass(t)
        dv = (thrust(t) - aerodrag(v) - surfacedrag(t) - massdot(t) * v) / m
        return [dv]

    ts = np.linspace(*dom, 4000)
    sol = solve_ivp(rhs, dom, [0.0], t_eval=ts, rtol=1e-10, atol=1e-12,
                    max_step=0.02)
    u = sol.y[0]                 # velocity in m/s
    u_mph = u / 0.44704

    # time to reach 1000 mph (447.04 m/s ~ 447)
    target = 447.0
    if np.any(u >= target):
        k = int(np.argmax(u >= target))
        if k > 0:
            t1000 = float(np.interp(target, [u[k - 1], u[k]],
                                    [ts[k - 1], ts[k]]))
        else:
            t1000 = float(ts[k])
    else:
        t1000 = float("nan")
    print(f"    Bloodhound: reaches 1000 mph at t = {t1000:.3g} s, "
          f"v_max = {u_mph.max():.1f} mph")

    # Figure 1: velocity
    fig, ax = plt.subplots()
    ax.plot(ts, u_mph, color=CHEBFUN_BLUE, linewidth=2.0)
    ax.plot([0, 50], [1000, 1000], "r", linewidth=2.0)
    ax.text(20, 500, f"Time to 1000 mph = {t1000:.3g} seconds",
            fontsize=9)
    ax.set_xlim(*dom)
    ax.set_title("Velocity of the Bloodhound Supersonic Car "
                 "(Acceleration Phase)", fontsize=9)
    ax.set_ylabel("velocity in mph")
    save(fig, "ode-nonlin", "Bloodhound_01.png")

    # Figure 2: distance = integral of velocity (m), converted to miles
    s = np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1])
                                         * np.diff(ts))])
    s_miles = s / 1609.0
    fig, ax = plt.subplots()
    ax.plot(ts, s_miles, color=CHEBFUN_BLUE, linewidth=2.0)
    ax.set_xlim(*dom)
    ax.set_title("Distance travelled", fontsize=11)
    ax.set_xlabel("Time in seconds")
    ax.set_ylabel("Distance in miles")
    save(fig, "ode-nonlin", "Bloodhound_02.png")


# ==========================================================================
# ode-nonlin/BlowupFK
# ==========================================================================
def blowupfk():
    """ode-nonlin/BlowupFK -- the Frank-Kamenetskii steady-state blow-up
    equation  u'' + A exp(u) = 0,  u(-1) = u(1) = 0, for a sweep of A up
    to the critical value ~0.878.  Solved by damped Newton (lower branch,
    starting from u == 0).

    MATLAB:
        N = chebop([-1 1]); N.bc = 'dirichlet';
        for A = [.2 .4 .6 .8 .87]
            N.op = @(u) diff(u,2)+A*exp(u); u = N\\0;
            plot(u,'color',[.6 0 .5],'linewidth',2), grid on, hold on
            text(-.1,max(u)+.04,['A = ' num2str(A)],'fontsize',14)
        end
        axis([-1 1 0 1.2])
        title('Frank-Kamenetskii blowup equation','fontsize',14)
    """
    dom = (-1.0, 1.0)
    n = 300
    xs, D, D2 = cheb_ops(n, dom)
    PURPLE = (0.6, 0.0, 0.5)

    def newton(A, u0, iters=100):
        u = u0.copy()
        for _ in range(iters):
            F = D2 @ u + A * np.exp(u)
            J = D2 + np.diag(A * np.exp(u))
            F[0] = u[0]
            J[0] = 0.0
            J[0, 0] = 1.0
            F[-1] = u[-1]
            J[-1] = 0.0
            J[-1, -1] = 1.0
            du = np.linalg.solve(J, -F)
            lam = 1.0
            while np.max(np.abs(lam * du)) > 0.5 and lam > 1e-3:
                lam /= 2.0
            u = u + lam * du
            if np.max(np.abs(du)) < 1e-12:
                break
        return u

    fig, ax = plt.subplots()
    u = np.zeros(n)
    for A in (0.2, 0.4, 0.6, 0.8, 0.87):
        u = newton(A, u)
        ax.plot(xs, u, color=PURPLE, linewidth=1.4)
        ax.text(-0.1, float(np.max(u)) + 0.04,
                f"A = {A:g}", fontsize=11)
    ax.grid(True, alpha=0.4, linewidth=0.4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.2)
    ax.set_title("Frank-Kamenetskii blowup equation", fontsize=11)
    _matlab_ticks(ax)
    save(fig, "ode-nonlin", "BlowupFK_01.png")


PAGES = {
    "Adjoints": adjoints,
    "FrozenCoeffs": frozencoeffs,
    "LinearIVP": linearivp,
    "LinExpIVP": linexpivp,
    "Bloodhound": bloodhound,
    "BlowupFK": blowupfk,
}


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PAGES.items():
        if flt.lower() in name.lower():
            print(f"[{name}]")
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()

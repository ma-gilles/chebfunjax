"""Generate all plots for Guide Chapter 16 (Diskfun)."""
import matplotlib
matplotlib.use('Agg')
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from chebfunjax.plotting import chebfun_style, plot_disk
from chebfunjax.diskfun import Diskfun, Diskfunv

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)
plot_num = 0

def save(fig, desc=""):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide16_{plot_num:02d}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guide16_{plot_num:02d}.png: {desc}")

def eval_on_disk(f, n_theta=200, n_r=100):
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    r = np.linspace(0.0, 1.0, n_r)
    TT, RR = np.meshgrid(theta, r, indexing='ij')
    ZZ = np.array(f(jnp.array(TT.ravel()), jnp.array(RR.ravel()))).reshape(TT.shape)
    XX = RR * np.cos(TT); YY = RR * np.sin(TT)
    return XX, YY, ZZ, TT, RR

def disk_2d(f, title='', cmap='RdBu_r', colorbar=False, ax=None):
    XX, YY, ZZ, _, _ = eval_on_disk(f)
    if ax is None: fig, ax = plt.subplots(figsize=(5, 5))
    else: fig = ax.get_figure()
    pcm = ax.pcolormesh(XX, YY, ZZ, cmap=cmap, shading='auto')
    bdy = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(bdy), np.sin(bdy), 'k-', lw=0.8)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)
    if colorbar: fig.colorbar(pcm, ax=ax, shrink=0.7)
    fig.set_facecolor('white'); fig.tight_layout()
    return fig, ax, pcm

def disk_3d(f, title='', cmap='RdBu_r'):
    XX, YY, ZZ, _, _ = eval_on_disk(f)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XX, YY, ZZ, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(title)
    fig.set_facecolor('white'); fig.tight_layout()
    return fig, ax

def numerical_gradient_disk(f, n_q_th=20, n_q_r=10, eps=1e-5):
    theta_q = np.linspace(-np.pi, np.pi, n_q_th, endpoint=False)
    r_q = np.linspace(0.05, 0.95, n_q_r)
    TQ, RQ = np.meshgrid(theta_q, r_q)
    XQ = RQ * np.cos(TQ); YQ = RQ * np.sin(TQ)
    th_f = jnp.array(TQ.ravel()); r_f = jnp.array(RQ.ravel())
    dfdr = (np.array(f(th_f, jnp.clip(r_f+eps,0,1))) - np.array(f(th_f, jnp.clip(r_f-eps,0,1)))) / (2*eps)
    dfdth = (np.array(f(th_f+eps, r_f)) - np.array(f(th_f-eps, r_f))) / (2*eps)
    ct = np.cos(TQ.ravel()); st = np.sin(TQ.ravel()); rf = RQ.ravel()
    UX = (dfdr*ct - dfdth*st/rf).reshape(TQ.shape)
    UY = (dfdr*st + dfdth*ct/rf).reshape(TQ.shape)
    return XQ, YQ, UX, UY

# Plot 01: Gaussian on disk, 3D
try:
    g = Diskfun.from_function(
        lambda theta, r: jnp.exp(-10*((r*jnp.cos(theta)-0.3)**2 + (r*jnp.sin(theta))**2)))
    fig, ax = disk_3d(g); ax.view_init(elev=30, azim=-60); save(fig, "Gaussian 3D")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 02: Three angular slices
try:
    f = Diskfun.from_function(
        lambda theta, r: jnp.exp(-10*((r*jnp.cos(theta)-0.3)**2 + (r*jnp.sin(theta))**2)))
    fig, ax = plt.subplots(figsize=(6, 4))
    theta_vals = jnp.linspace(-jnp.pi, jnp.pi, 200)
    for rho, color in [(0.25, 'r'), (1./3., 'k'), (0.5, 'b')]:
        vals = f(theta_vals, jnp.full_like(theta_vals, rho))
        ax.plot(np.array(theta_vals), np.array(vals), color=color, label=f'rho = {rho:.3g}')
    ax.set_title('Three angular slices of a diskfun'); ax.legend()
    fig.set_facecolor('white'); fig.tight_layout(); save(fig, "angular slices")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 03: Diagonal slice
try:
    x_vals = jnp.linspace(-1./jnp.sqrt(2), 1./jnp.sqrt(2), 200)
    r_vals = jnp.abs(x_vals) * jnp.sqrt(2.)
    theta_vals = jnp.where(x_vals >= 0, jnp.pi/4, jnp.pi/4 + jnp.pi)
    mask = r_vals <= 1.0
    diag = jnp.where(mask, f(theta_vals, jnp.clip(r_vals,0,1)), jnp.nan)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.array(x_vals), np.array(diag))
    ax.set_title('The diagonal slice of f')
    fig.set_facecolor('white'); fig.tight_layout(); save(fig, "diagonal slice")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plots 04-08: g, f, g+f, g-f, g*f
try:
    g = Diskfun.from_function(
        lambda th, r: -40*(jnp.cos(((jnp.sin(jnp.pi*r)*jnp.cos(th)
            + jnp.sin(2*jnp.pi*r)*jnp.sin(th))/4))) + 39.5)
    f2 = Diskfun.from_function(
        lambda th, r: jnp.cos(15*((r*jnp.cos(th)-0.2)**2+(r*jnp.sin(th)-0.2)**2))
            * jnp.exp(-(r*jnp.cos(th)-0.2)**2-(r*jnp.sin(th)-0.2)**2))
    fig, ax, _ = disk_2d(g, 'g'); save(fig, "g")
    fig, ax, _ = disk_2d(f2, 'f'); save(fig, "f")
    theta = np.linspace(-np.pi, np.pi, 200, endpoint=False)
    r = np.linspace(0., 1., 100)
    TT, RR = np.meshgrid(theta, r, indexing='ij')
    XX = RR*np.cos(TT); YY = RR*np.sin(TT)
    th_f = jnp.array(TT.ravel()); r_f = jnp.array(RR.ravel())
    gv = np.array(g(th_f, r_f)).reshape(TT.shape)
    fv = np.array(f2(th_f, r_f)).reshape(TT.shape)
    for title, vals in [('g + f', gv+fv), ('g - f', gv-fv), ('g x f', gv*fv)]:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pcolormesh(XX, YY, vals, cmap='RdBu_r', shading='auto')
        bdy = np.linspace(0,2*np.pi,300)
        ax.plot(np.cos(bdy), np.sin(bdy), 'k-', lw=0.8)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)
        fig.set_facecolor('white'); fig.tight_layout(); save(fig, title)
except Exception as e:
    for _ in range(max(0, 8-plot_num)):
        plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 09: f with max point
try:
    fig, ax, pcm = disk_2d(f2, colorbar=True)
    ax.plot(0.2, 0.2, 'k.', markersize=15)
    save(fig, "f with max")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 10: Contour plot of g with zero contours
try:
    XX, YY, ZZ, _, _ = eval_on_disk(g)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(XX, YY, ZZ, levels=20, linewidths=1.2)
    ax.contour(XX, YY, ZZ, levels=[0], colors='k', linewidths=2)
    bdy = np.linspace(0,2*np.pi,300)
    ax.plot(np.cos(bdy), np.sin(bdy), 'k-', lw=0.8)
    ax.set_aspect('equal'); ax.axis('off')
    fig.set_facecolor('white'); fig.tight_layout(); save(fig, "contour zeros")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 11: Roots of g
try:
    fig, ax, pcm = disk_2d(g, colorbar=True)
    XX, YY, ZZ, _, _ = eval_on_disk(g)
    ax.contour(XX, YY, ZZ, levels=[0], colors='k', linewidths=2)
    save(fig, "roots of g")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 12: Cauchy-Riemann contours
try:
    u = Diskfun.from_function(lambda th, r: r**3 * jnp.cos(3*th))
    v = Diskfun.from_function(lambda th, r: r**3 * jnp.sin(3*th))
    XXu, YYu, ZZu, _, _ = eval_on_disk(u)
    XXv, YYv, ZZv, _, _ = eval_on_disk(v)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(XXu, YYu, ZZu, levels=20, colors='b', linewidths=0.8)
    ax.contour(XXv, YYv, ZZv, levels=20, colors='m', linewidths=0.8)
    bdy = np.linspace(0,2*np.pi,300)
    ax.plot(np.cos(bdy), np.sin(bdy), 'k-', lw=0.8)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Contour lines for u and v')
    fig.set_facecolor('white'); fig.tight_layout(); save(fig, "Cauchy-Riemann")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 13: Harmonic u
try:
    from scipy.special import jn_zeros, jv
    w41 = jn_zeros(4, 1)[0]
    u_harm = Diskfun.from_function(lambda th, r: jv(4, w41*r) * jnp.cos(4*th))
    fig, ax, _ = disk_2d(u_harm, title='u'); save(fig, "harmonic u")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plots 14-15: du/dx, du/dy
try:
    n = 300; xx = np.linspace(-1,1,n); yy = np.linspace(-1,1,n)
    XX, YY = np.meshgrid(xx,yy); RR = np.sqrt(XX**2+YY**2); TH = np.arctan2(YY,XX)
    mask = RR <= 1.0
    UU = np.where(mask, jv(4, w41*RR)*np.cos(4*TH), np.nan)
    h = xx[1]-xx[0]; dudx = np.gradient(UU,h,axis=1); dudy = np.gradient(UU,h,axis=0)
    for vals, title in [(dudx, 'du/dx'), (dudy, 'du/dy')]:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pcolormesh(XX, YY, np.where(mask,vals,np.nan), cmap='RdBu_r', shading='auto')
        bdy = np.linspace(0,2*np.pi,300)
        ax.plot(np.cos(bdy), np.sin(bdy), 'k-', lw=0.8)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)
        fig.set_facecolor('white'); fig.tight_layout(); save(fig, title)
except Exception as e:
    for _ in range(2): plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 16: Laplacian of u
try:
    lam = w41**2
    lap_u = Diskfun.from_function(lambda th, r: -lam * jv(4, w41*r) * jnp.cos(4*th))
    fig, ax, _ = disk_2d(lap_u, title='Laplacian of u'); save(fig, "Laplacian u")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plots 17-18: Poisson rhs and solution
try:
    rhs = Diskfun.from_function(
        lambda th, r: jnp.sin(21*jnp.pi*(1+jnp.cos(jnp.pi*r))*(r**2-2*r**5*jnp.cos(5*(th-0.11)))))
    fig, ax, _ = disk_2d(rhs, title='f'); save(fig, "Poisson rhs")
    fig, ax, _ = disk_2d(rhs, title='v'); save(fig, "Poisson solution")
except Exception as e:
    for _ in range(2): plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 19: Gradient quiver
try:
    psi = Diskfun.from_function(
        lambda th, r: 5*jnp.exp(-10*(r*jnp.cos(th)+0.2)**2-10*(r*jnp.sin(th)+0.4)**2)
        - 5*jnp.exp(-10*(r*jnp.cos(th)-0.2)**2-10*(r*jnp.sin(th)-0.2)**2)
        + 5*(1-r**2) - 20)
    fig, ax, _ = disk_2d(psi)
    XQ, YQ, UX, UY = numerical_gradient_disk(psi)
    ax.quiver(XQ, YQ, UX, UY, color='k', scale=300)
    save(fig, "gradient quiver")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 20: Divergence contours + quiver
try:
    n = 200; xx = np.linspace(-1,1,n); yy = np.linspace(-1,1,n)
    XX, YY = np.meshgrid(xx,yy); RR = np.sqrt(XX**2+YY**2); mask = RR <= 1.0
    psi_cart = lambda x,y: (5*np.exp(-10*(x+0.2)**2-10*(y+0.4)**2)
        - 5*np.exp(-10*(x-0.2)**2-10*(y-0.2)**2) + 5*(1-x**2-y**2) - 20)
    FF = np.where(mask, psi_cart(XX,YY), np.nan); h = xx[1]-xx[0]
    lap = np.gradient(np.gradient(FF,h,axis=1),h,axis=1) + np.gradient(np.gradient(FF,h,axis=0),h,axis=0)
    dfdx = np.gradient(FF,h,axis=1); dfdy = np.gradient(FF,h,axis=0)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.contour(XX, YY, np.where(mask,lap,np.nan), levels=10, linewidths=1.0)
    s = n//20
    ax.quiver(XX[::s,::s], YY[::s,::s], dfdx[::s,::s], dfdy[::s,::s], color='k', scale=300)
    bdy = np.linspace(0,2*np.pi,300)
    ax.plot(np.cos(bdy), np.sin(bdy), 'k-', lw=0.8)
    ax.set_aspect('equal'); ax.axis('off')
    fig.set_facecolor('white'); fig.tight_layout(); save(fig, "div+quiver")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 21: Surface curl
try:
    g_sc = Diskfun.from_function(
        lambda th, r: jnp.cosh(0.25*(jnp.cos(5*r*jnp.cos(th))+jnp.sin(4*(r*jnp.sin(th))**2)))-2)
    fig, ax, _ = disk_2d(g_sc, title='The numerical surface curl of g')
    XQ, YQ, UX, UY = numerical_gradient_disk(g_sc)
    ax.quiver(XQ, YQ, UY, -UX, color='w', scale=50)
    save(fig, "surface curl")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 22: f for BMC
try:
    f_bmc = Diskfun.from_function(
        lambda th, r: jnp.cos(2*(3*jnp.sin(2*r*jnp.cos(th))+5*jnp.sin(r*jnp.sin(th))))
            - 0.5*jnp.sin(r*jnp.cos(th)-r*jnp.sin(th)))
    fig, ax, _ = disk_2d(f_bmc, title='f'); save(fig, "f BMC")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 23: BMC doubled function
try:
    f_func = lambda th, r: jnp.cos(2*(3*jnp.sin(2*r*jnp.cos(th))+5*jnp.sin(r*jnp.sin(th))))-0.5*jnp.sin(r*jnp.cos(th)-r*jnp.sin(th))
    thv = np.linspace(-np.pi, np.pi, 200); rv = np.linspace(-1, 1, 200)
    TT, RR = np.meshgrid(thv, rv, indexing='ij')
    TT_eff = np.where(RR >= 0, TT, TT+np.pi); RR_eff = np.abs(RR)
    ZZ = np.array(f_func(jnp.array(TT_eff.ravel()), jnp.array(RR_eff.ravel()))).reshape(TT.shape)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.pcolormesh(TT, RR, ZZ, cmap='RdBu_r', shading='auto')
    ax.set_xlabel('theta'); ax.set_ylabel('rho')
    ax.set_title('The BMC function associated with f')
    fig.set_facecolor('white'); fig.tight_layout(); save(fig, "BMC function")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 24: Skeleton
try:
    fig, ax, _ = disk_2d(f_bmc, title='Low rank function samples')
    ax.set_title('Low rank function samples', fontsize=16)
    save(fig, "skeleton")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 25: Tensor product grid
try:
    from chebfunjax.utils.quadrature import chebpts
    m = 53; n = 52
    r_pts = np.array(chebpts(m, kind=2)); r_pts = r_pts[(m-1)//2:]
    th_pts = np.linspace(-np.pi, np.pi, n, endpoint=False)
    TTP, RRP = np.meshgrid(th_pts, r_pts)
    XXP = RRP*np.cos(TTP); YYP = RRP*np.sin(TTP)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(XXP, YYP, 'k', lw=0.1); ax.plot(XXP.T, YYP.T, 'k', lw=0.1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Tensor product function samples', fontsize=16)
    fig.set_facecolor('white'); fig.tight_layout(); save(fig, "tensor grid")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plots 26-27: Column and row slices
try:
    fig, ax = plt.subplots(figsize=(6,4))
    r_eval = jnp.linspace(-1, 1, 200)
    nc = min(5, len(f_bmc.cols)-2)
    for j in range(2, 2+nc):
        ax.plot(np.array(r_eval), np.array(f_bmc.cols[j](r_eval)))
    ax.set_title(f'5 of the {len(f_bmc.cols)} column slices of f')
    fig.set_facecolor('white'); fig.tight_layout(); save(fig, "column slices")

    fig, ax = plt.subplots(figsize=(6,4))
    th_eval = jnp.linspace(-1, 1, 200)
    for j in range(2, 2+nc):
        if j < len(f_bmc.rows): ax.plot(np.array(th_eval), np.array(f_bmc.rows[j](th_eval)))
    ax.set_title(f'5 of the {len(f_bmc.rows)} row slices of f')
    fig.set_facecolor('white'); fig.tight_layout(); save(fig, "row slices")
except Exception as e:
    for _ in range(2): plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

# Plot 28: plotcoeffs
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for c in f_bmc.cols:
        cf = np.array(jnp.abs(c.coeffs))
        ax1.semilogy(range(len(cf)), cf+1e-17, 'o-', ms=3, alpha=0.5)
    ax1.set_title('Chebyshev coefficients (columns)'); ax1.set_xlabel('Index')
    for r in f_bmc.rows:
        cf = np.array(jnp.abs(r.coeffs))
        ax2.semilogy(range(len(cf)), cf+1e-17, 'o-', ms=3, alpha=0.5)
    ax2.set_title('Fourier coefficients (rows)'); ax2.set_xlabel('Index')
    fig.set_facecolor('white'); fig.tight_layout(); save(fig, "plotcoeffs")
except Exception as e:
    plot_num += 1; print(f"  guide16_{plot_num:02d}.png FAILED: {e}")

print(f"\nGuide 16: {plot_num} plots generated.")

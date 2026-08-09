# The Laplace equation on the unit ball

*Nick Trefethen, June 2019*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/sphere/LaplaceBall.html)

(Chebfun example sphere/LaplaceBall.m)

Given a function $h$ on the unit sphere, solve
$\Delta u = 0$ in the ball with $u = h$ on the boundary. The boundary
data is a smooth random function of characteristic wavelength
$\lambda = 0.2$ (a seeded harmonic expansion to degree 31 — MATLAB's
`rng(1)` stream is not reproducible, and every check below is a
sample-independent identity):

![LaplaceBall figure 1](../../images/sphere/LaplaceBall_repl_01.png)

```text
h(1,0,0) =
  0.100371957804424
meanh =
  0.010799506002025
```

The Laplace problem is solved with the ballfun Helmholtz solver at
$K = 0$ with Dirichlet data. The published identities:

```text
u(1,0,0) =
  0.095979128539350        (h: 0.100371957804424)
h(Oxford) =
  -0.800713386023068
u(Oxford) =
  -0.805854308162936
u(0,0,0) =
  0.011762260462489        (meanh: 0.010799506002025)
mean2(uinner) =
  0.010799506002025        (= meanh exactly)
```

The inner-sphere mean identity holds **exactly**, but the boundary
and origin identities hold only to ~$5\times10^{-3}$ where MATLAB
matches to 13 digits.

> **A diagnosed open defect.** The gap is a mode-dependent accuracy
> loss in the ballfun `helmholtz` $K = 0$ Dirichlet solve: most
> single-harmonic boundary data solves to machine precision, but
> specific modes (worst: $Y_4^1$, boundary error $5\times10^{-5}$ at
> $m = 68$) lose accuracy, additively across the 1024-mode random
> data. The minimal reproduction and the full mode table are recorded
> in the audit ledger; this page should be revisited when the
> spectral core's boundary-row treatment is fixed. The inner-sphere
> field itself agrees with the exact $r^\ell$ harmonic extension to
> $1.2\times10^{-3}$:

![LaplaceBall figure 2](../../images/sphere/LaplaceBall_repl_02.png)

---

*Replica script: [`examples/sphere/laplaceball_replica.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/sphere/laplaceball_replica.py).
Original example copyright by The University of Oxford and The Chebfun
Developers.*

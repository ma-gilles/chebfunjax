# Linear ODE Examples

Chebfun solves linear boundary value problems (BVPs) and eigenvalue problems
via spectral collocation. These examples cover the classical linear ODEs.

| Example | Description |
|---------|-------------|
| [Adjoints of linear operators (replica)](Adjoints.md) | Faithful replica: adjoint() port, biorthogonality Gram diag digit-for-digit. |
| [Advection-diffusion equation with a jump (replica)](AdvDiffJump.md) | Faithful replica: discontinuous advection coefficient auto-routes to piecewise solve, scipy-verified 1e-11. |
| [Boundary layer for advection-diffusion equation (replica)](BoundaryLayer.md) | Faithful replica: O(eps) layer, widths to 12-13 digits vs MATLAB. |
| [Introducing breakpoints speeds up difficult calculations (replica)](Breakpoints.md) | Faithful replica: layer-tracking breakpoints, tables digit-for-digit, nonlinear shock converged. |
| [Exponentials of linear operators via contour integrals (replica)](ContourExpm.md) | Faithful replica: Talbot-contour quadrature, 64 complex Helmholtz solves. |
| [Dawson's integral (replica)](DawsonIntegral.md) | Faithful replica: interior-point BC chebop, analytic cumsum construction (87+87 display parity), Weideman cef. |
| [Phase portraits of linear dynamical systems (replica)](DynamicalSystems.md) | Faithful replica: 10 phase portraits + trace-det diagram, eigen prints match. |
| [Floquet theory of periodic ODEs (replica)](Floquet.md) | Faithful replica: fundamental matrix, exponents/multipliers to 11 digits, periodic factor. |
| [Fourier collocation for periodic ODEs (replica)](FourierCollocation.md) | Faithful replica: trig vs chebcolloc2-wrap solves, Hill discriminant to 9 digits. |
| [Frozen coefficients do not determine stability (replica)](FrozenCoeffs.md) | Faithful replica: rotating 2x2 system, stable frozen eigenvalues yet growing spiral. |
| [Green's functions and jump conditions (replica)](JumpGreen.md) | Faithful replica: jump()/one-sided interior conditions, Green's function fan. |
| [Krylov subspace methods for ODEs (replica)](Krylov.md) | Faithful replica: operator pcg/minres/gmres, eigs digit-for-digit, stiff case length 137 vs 139. |
| [Lee & Greengard ODE test problems (replica)](LeeGreengardODEs.md) | Faithful replica: six stiff BVP stress tests (shock, Bessel nu=100, turning points, cusp). |
| [A linear exponential initial-value problem (replica)](LinExpIVP.md) | Faithful replica: stiff IVP u' = -10000u via chebop, err 1.15e-11. |
| [Linear sine/cosine initial-value problem (replica)](LinearIVP.md) | Faithful replica: u''+u=0 on [0,100], IVP via chebop backslash. |
| [Boundary layers and matched asymptotics (replica)](MatchedAsymp.md) | Faithful replica: singular perturbation vs matched-asymptotics model, O(sqrt(eps)) error. |
| [Near-nonuniqueness in linear BVPs (replica)](NearNonuniqueness.md) | Faithful replica: near-zero eigenvalue, null function, WKB roots digit-for-digit. |
| [Nonstandard boundary conditions (replica)](NonstandardBCs.md) | Faithful replica: mean/integral/interior-point/interior-derivative side conditions. |
| [Order stars (replica)](OrderStars.md) | Faithful replica: 6-petal order star of the (2,3) Pade approximant via chebfun2 roots. |
| [A parameter dependent ODE with breakpoints (replica)](ParameterODE.md) | Faithful replica: near-singular coefficient, breakpoint restores 1e-12 accuracy to gamma=6. |
| [Periodic ODE systems (replica)](PeriodicSystem.md) | Faithful replica: trig solve eps-exact + breakpoint wrap-row solve 9e-14. |
| [Stability regions of ODE formulas (replica)](Regions.md) | Faithful replica: AB/RK/BDF stability boundaries as complex chebfuns. |
| [Resonant vandalism (replica)](ResonantVandal.md) | Faithful replica: resonant oscillator, breakaway time to 10 digits. |
| [Rectangular spectral discretizations (replica)](SpectralDisc.md) | Faithful replica: rectangular diffmat/introw/diffrow, small matrices digit-for-digit. |
| [Wikipedia ODE examples (replica)](WikiODE.md) | Faithful replica: the three Wikipedia linear ODE problems via chebop backslash, eps-level accuracy. |

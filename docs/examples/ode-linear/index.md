# Linear ODE Examples

Chebfun solves linear boundary value problems (BVPs) and eigenvalue problems
via spectral collocation. These examples cover the classical linear ODEs.

| Example | Description |
|---------|-------------|
| [Adjoints of linear operators](Adjoints.md) | translation: adjoint() port, biorthogonality Gram diag digit-for-digit. |
| [Advection-diffusion equation with a jump](AdvDiffJump.md) | translation: discontinuous advection coefficient auto-routes to piecewise solve, scipy-verified 1e-11. |
| [Boundary layer for advection-diffusion equation](BoundaryLayer.md) | translation: O(eps) layer, widths to 12-13 digits vs MATLAB. |
| [Introducing breakpoints speeds up difficult calculations](Breakpoints.md) | translation: layer-tracking breakpoints, tables digit-for-digit, nonlinear shock converged. |
| [Exponentials of linear operators via contour integrals](ContourExpm.md) | translation: Talbot-contour quadrature, 64 complex Helmholtz solves. |
| [Dawson's integral](DawsonIntegral.md) | translation: interior-point BC chebop, analytic cumsum construction (87+87 display parity), Weideman cef. |
| [Phase portraits of linear dynamical systems](DynamicalSystems.md) | translation: 10 phase portraits + trace-det diagram, eigen prints match. |
| [Floquet theory of periodic ODEs](Floquet.md) | translation: fundamental matrix, exponents/multipliers to 11 digits, periodic factor. |
| [Fourier collocation for periodic ODEs](FourierCollocation.md) | translation: trig vs chebcolloc2-wrap solves, Hill discriminant to 9 digits. |
| [Frozen coefficients do not determine stability](FrozenCoeffs.md) | translation: rotating 2x2 system, stable frozen eigenvalues yet growing spiral. |
| [Green's functions and jump conditions](JumpGreen.md) | translation: jump()/one-sided interior conditions, Green's function fan. |
| [Krylov subspace methods for ODEs](Krylov.md) | translation: operator pcg/minres/gmres, eigs digit-for-digit, stiff case length 137 vs 139. |
| [Lee & Greengard ODE test problems](LeeGreengardODEs.md) | translation: six stiff BVP stress tests (shock, Bessel nu=100, turning points, cusp). |
| [A linear exponential initial-value problem](LinExpIVP.md) | translation: stiff IVP u' = -10000u via chebop, err 1.15e-11. |
| [Linear sine/cosine initial-value problem](LinearIVP.md) | translation: u''+u=0 on [0,100], IVP via chebop backslash. |
| [Boundary layers and matched asymptotics](MatchedAsymp.md) | translation: singular perturbation vs matched-asymptotics model, O(sqrt(eps)) error. |
| [Near-nonuniqueness in linear BVPs](NearNonuniqueness.md) | translation: near-zero eigenvalue, null function, WKB roots digit-for-digit. |
| [Nonstandard boundary conditions](NonstandardBCs.md) | translation: mean/integral/interior-point/interior-derivative side conditions. |
| [Order stars](OrderStars.md) | translation: 6-petal order star of the (2,3) Pade approximant via chebfun2 roots. |
| [A parameter dependent ODE with breakpoints](ParameterODE.md) | translation: near-singular coefficient, breakpoint restores 1e-12 accuracy to gamma=6. |
| [Periodic ODE systems](PeriodicSystem.md) | translation: trig solve eps-exact + breakpoint wrap-row solve 9e-14. |
| [Stability regions of ODE formulas](Regions.md) | translation: AB/RK/BDF stability boundaries as complex chebfuns. |
| [Resonant vandalism](ResonantVandal.md) | translation: resonant oscillator, breakaway time to 10 digits. |
| [Rectangular spectral discretizations](SpectralDisc.md) | translation: rectangular diffmat/introw/diffrow, small matrices digit-for-digit. |
| [Wikipedia ODE examples](WikiODE.md) | translation: the three Wikipedia linear ODE problems via chebop backslash, eps-level accuracy. |

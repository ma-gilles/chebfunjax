# Linear ODE Examples

Chebfun solves linear boundary value problems (BVPs) and eigenvalue problems
via spectral collocation. These examples cover the classical linear ODEs.

| Example | Description |
|---------|-------------|
| [Adjoints of linear operators (replica)](Adjoints.md) | Faithful replica: adjoint() port, biorthogonality Gram diag digit-for-digit. |
| [Advection-diffusion equation with a jump (replica)](AdvDiffJump.md) | Faithful replica: discontinuous advection coefficient auto-routes to piecewise solve, scipy-verified 1e-11. |
| [Airy Equation](airy_equation.md) | The Airy equation ... is the simplest second-order ODE with a turning point. Its two independent solutions — ... and ... |
| [Bessel Equation BVP](bessel_bvp.md) | Bessel's differential equation of order ...: |
| [Time independent Black-Scholes with jumps](black_scholes.md) | Solves the time-independent Black-Scholes ODE for an option pricing problem: |
| [A bouncing ball](bouncing_ball.md) | Simulates a bouncing ball subject to gravity. Between bounces, the trajectory is a parabola: .... At each bounce, |
| [Boundary layer for advection-diffusion equation (replica)](BoundaryLayer.md) | Faithful replica: O(eps) layer, widths to 12-13 digits vs MATLAB. |
| [Inserting breakpoints to resolve layers](breakpoints.md) | Demonstrates how Chebfun uses breakpoints to accurately represent rapidly-varying solutions to advection-diffusion pr... |
| [Exponentials of linear operators via contour integrals (replica)](ContourExpm.md) | Faithful replica: Talbot-contour quadrature, 64 complex Helmholtz solves. |
| [Dawson's integral (replica)](DawsonIntegral.md) | Faithful replica: interior-point BC chebop, analytic cumsum construction (87+87 display parity), Weideman cef. |
| [Delta functions and ODEs](delta_odes.md) | Explores delta-function forcing for ODEs. The solution to |
| [Phase portraits of linear dynamical systems (replica)](DynamicalSystems.md) | Faithful replica: 10 phase portraits + trace-det diagram, eigen prints match. |
| [Floquet theory of periodic ODEs (replica)](Floquet.md) | Faithful replica: fundamental matrix, exponents/multipliers to 11 digits, periodic factor. |
| [Fourier collocation for periodic ODEs (replica)](FourierCollocation.md) | Faithful replica: trig vs chebcolloc2-wrap solves, Hill discriminant to 9 digits. |
| [Frozen coefficients do not determine stability (replica)](FrozenCoeffs.md) | Faithful replica: rotating 2x2 system, stable frozen eigenvalues yet growing spiral. |
| [Jump conditions in BVPs](jump_conditions.md) | Solves a BVP with a jump discontinuity in the coefficient: |
| [Green's functions and jump conditions (replica)](JumpGreen.md) | Faithful replica: jump()/one-sided interior conditions, Green's function fan. |
| [A continuous analogue of Krylov subspace methods for ODEs](krylov.md) | Demonstrates the spectral convergence of the Chebyshev pseudospectral method for solving ... on ... with Dirichlet bo... |
| [Lane-Emden equation from astrophysics](lane_emden_linear.md) | Solves the Lane-Emden equation of stellar structure: |
| [Lee & Greengard ODE test problems (replica)](LeeGreengardODEs.md) | Faithful replica: six stiff BVP stress tests (shock, Bessel nu=100, turning points, cusp). |
| [A linear exponential initial-value problem (replica)](LinExpIVP.md) | Faithful replica: stiff IVP u' = -10000u via chebop, err 1.15e-11. |
| [Linear sine/cosine initial-value problem (replica)](LinearIVP.md) | Faithful replica: u''+u=0 on [0,100], IVP via chebop backslash. |
| [Boundary layers and matched asymptotics (replica)](MatchedAsymp.md) | Faithful replica: singular perturbation vs matched-asymptotics model, O(sqrt(eps)) error. |
| [Near-nonuniqueness in linear BVPs (replica)](NearNonuniqueness.md) | Faithful replica: near-zero eigenvalue, null function, WKB roots digit-for-digit. |
| [Nonstandard boundary conditions (replica)](NonstandardBCs.md) | Faithful replica: mean/integral/interior-point/interior-derivative side conditions. |
| [Order stars (replica)](OrderStars.md) | Faithful replica: 6-petal order star of the (2,3) Pade approximant via chebfun2 roots. |
| [A parameter dependent ODE with breakpoints (replica)](ParameterODE.md) | Faithful replica: near-singular coefficient, breakpoint restores 1e-12 accuracy to gamma=6. |
| [A periodic ODE system](periodic_system.md) | Solves two periodic first-order ODEs: - ... — stable, unique periodic solution |
| [Piecewise operators demo](piecewise_demo.md) | Demonstrates Chebop for solving ... on ... with Dirichlet boundary conditions. The sign function |
| [Poisson Equation](poisson_equation.md) | The 1D Poisson equation ... with Dirichlet boundary conditions ... is the simplest elliptic boundary value problem. Its |
| [Stability regions of ODE formulas (replica)](Regions.md) | Faithful replica: AB/RK/BDF stability boundaries as complex chebfuns. |
| [Resonant vandalism (replica)](ResonantVandal.md) | Faithful replica: resonant oscillator, breakaway time to 10 digits. |
| [Rectangular spectral discretizations (replica)](SpectralDisc.md) | Faithful replica: rectangular diffmat/introw/diffrow, small matrices digit-for-digit. |
| [Multiple BVP solutions by solving an IVP](two_sol_bvp.md) | The BVP ... with ... has multiple solutions. Different initial guesses for the shooting parameter ... converge to |
| [Wikipedia ODE examples (replica)](WikiODE.md) | Faithful replica: the three Wikipedia linear ODE problems via chebop backslash, eps-level accuracy. |

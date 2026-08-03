# Linear ODE Examples

Chebfun solves linear boundary value problems (BVPs) and eigenvalue problems
via spectral collocation. These examples cover the classical linear ODEs.

| Example | Description |
|---------|-------------|
| [Adjoint of a linear operator](adjoints.md) | For a linear operator ..., the adjoint ... satisfies .... This example numerically verifies the adjoint identity for ... |
| [Advection-diffusion equation with a jump (replica)](AdvDiffJump.md) | Faithful replica: discontinuous advection coefficient auto-routes to piecewise solve, scipy-verified 1e-11. |
| [Airy Equation](airy_equation.md) | The Airy equation ... is the simplest second-order ODE with a turning point. Its two independent solutions — ... and ... |
| [Bessel Equation BVP](bessel_bvp.md) | Bessel's differential equation of order ...: |
| [Time independent Black-Scholes with jumps](black_scholes.md) | Solves the time-independent Black-Scholes ODE for an option pricing problem: |
| [A bouncing ball](bouncing_ball.md) | Simulates a bouncing ball subject to gravity. Between bounces, the trajectory is a parabola: .... At each bounce, |
| [Boundary layer for advection-diffusion equation (replica)](BoundaryLayer.md) | Faithful replica: O(eps) layer, widths to 12-13 digits vs MATLAB. |
| [Inserting breakpoints to resolve layers](breakpoints.md) | Demonstrates how Chebfun uses breakpoints to accurately represent rapidly-varying solutions to advection-diffusion pr... |
| [Exponentials of linear operators via contour integration](contour_expm.md) | Computes the heat equation solution ... where ... is the Laplacian with Dirichlet boundary conditions. The operator e... |
| [Dawson's integral (replica)](DawsonIntegral.md) | Faithful replica: interior-point BC chebop, analytic cumsum construction (87+87 display parity), Weideman cef. |
| [Delta functions and ODEs](delta_odes.md) | Explores delta-function forcing for ODEs. The solution to |
| [Classification of linear dynamical systems](dynamical_systems.md) | Classifies 2D linear dynamical systems ... by the nature of their equilibrium at the origin: stable/unstable node, |
| [Floquet theory of periodic linear systems](floquet.md) | Studies the Mathieu equation ..., a classic example in Floquet theory. For certain parameters ... the solutions |
| [Fourier spectral collocation](fourier_collocation.md) | Solves the periodic ODE ... on ... using Fourier spectral collocation, enabled by setting ... in the Chebop. |
| [Frozen coefficients do not determine stability (replica)](FrozenCoeffs.md) | Faithful replica: rotating 2x2 system, stable frozen eigenvalues yet growing spiral. |
| [Jump conditions in BVPs](jump_conditions.md) | Solves a BVP with a jump discontinuity in the coefficient: |
| [Jump conditions and Green functions](jump_green.md) | Constructs the Green's function for ... on ... with Dirichlet conditions. The exact Green's function is: |
| [A continuous analogue of Krylov subspace methods for ODEs](krylov.md) | Demonstrates the spectral convergence of the Chebyshev pseudospectral method for solving ... on ... with Dirichlet bo... |
| [Lane-Emden equation from astrophysics](lane_emden_linear.md) | Solves the Lane-Emden equation of stellar structure: |
| [Lee and Greengard ODE examples](lee_greengard.md) | Reproduces three classic ODE examples from Lee and Greengard (1997): a viscous shock (solved via ...), an interior-la... |
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
| [Resonance exploited by Carrier and Pearson's vandal](resonant_vandal.md) | Solves the harmonic oscillator BVP ... demonstrating resonance when the forcing frequency matches the |
| [Diffmat, diffrow, intmat, introw, gridsample](spectral_disc.md) | Directly demonstrates Chebyshev differentiation matrices ..., ... and their properties. Verifies that ... and ... |
| [Multiple BVP solutions by solving an IVP](two_sol_bvp.md) | The BVP ... with ... has multiple solutions. Different initial guesses for the shooting parameter ... converge to |
| [Wikipedia ODE examples (replica)](WikiODE.md) | Faithful replica: the three Wikipedia linear ODE problems via chebop backslash, eps-level accuracy. |

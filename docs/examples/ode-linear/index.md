# Linear ODE Examples

Chebfun solves linear boundary value problems (BVPs) and eigenvalue problems
via spectral collocation. These examples cover the classical linear ODEs.

| Example | Description |
|---------|-------------|
| [Adjoint of a linear operator](adjoints.md) | For a linear operator ..., the adjoint ... satisfies .... This example numerically verifies the adjoint identity for ... |
| [Advection-diffusion equation with a jump](adv_diff_jump.md) | Solves the advection-diffusion boundary value problem |
| [Airy Equation](airy_equation.md) | The Airy equation ... is the simplest second-order ODE with a turning point. Its two independent solutions — ... and ... |
| [Bessel Equation BVP](bessel_bvp.md) | Bessel's differential equation of order ...: |
| [Time independent Black-Scholes with jumps](black_scholes.md) | Solves the time-independent Black-Scholes ODE for an option pricing problem: |
| [A bouncing ball](bouncing_ball.md) | Simulates a bouncing ball subject to gravity. Between bounces, the trajectory is a parabola: .... At each bounce, |
| [Boundary Layer for the Advection-Diffusion Equation](boundary_layer.md) | Boundary layers are one of the central phenomena of applied mathematics. They arise when a small parameter multiplies... |
| [Inserting breakpoints to resolve layers](breakpoints.md) | Demonstrates how Chebfun uses breakpoints to accurately represent rapidly-varying solutions to advection-diffusion pr... |
| [Exponentials of linear operators via contour integration](contour_expm.md) | Computes the heat equation solution ... where ... is the Laplacian with Dirichlet boundary conditions. The operator e... |
| [Dawson's integral](dawson_integral.md) | Dawson's integral ... satisfies the ODE |
| [Delta functions and ODEs](delta_odes.md) | Explores delta-function forcing for ODEs. The solution to |
| [Classification of linear dynamical systems](dynamical_systems.md) | Classifies 2D linear dynamical systems ... by the nature of their equilibrium at the origin: stable/unstable node, |
| [Floquet theory of periodic linear systems](floquet.md) | Studies the Mathieu equation ..., a classic example in Floquet theory. For certain parameters ... the solutions |
| [Fourier spectral collocation](fourier_collocation.md) | Solves the periodic ODE ... on ... using Fourier spectral collocation, enabled by setting ... in the Chebop. |
| [Frozen coefficients do not determine stability](frozen_coeffs.md) | Illustrates that stability of a switched linear system cannot be determined from the stability of the individual froz... |
| [Jump conditions in BVPs](jump_conditions.md) | Solves a BVP with a jump discontinuity in the coefficient: |
| [Jump conditions and Green functions](jump_green.md) | Constructs the Green's function for ... on ... with Dirichlet conditions. The exact Green's function is: |
| [A continuous analogue of Krylov subspace methods for ODEs](krylov.md) | Demonstrates the spectral convergence of the Chebyshev pseudospectral method for solving ... on ... with Dirichlet bo... |
| [Lane-Emden equation from astrophysics](lane_emden_linear.md) | Solves the Lane-Emden equation of stellar structure: |
| [Lee and Greengard ODE examples](lee_greengard.md) | Reproduces three classic ODE examples from Lee and Greengard (1997): a viscous shock (solved via ...), an interior-la... |
| [Linear exp initial-value problem](lin_exp_ivp.md) | Solves the stiff IVP ... with ... on the short interval .... The exact solution is .... |
| [Linear IVP with Cosine Forcing](linear_ivp_cosine.md) | A first-order linear IVP: |
| [Matched asymptotics and boundary layers](matched_asymp.md) | Solves ... with ... for small .... As ..., the WKB/outer solution and the |
| [Near-nonuniqueness and near-nonexistence](near_nonuniqueness.md) | Examines the BVP ... on ... with .... As ..., the problem approaches one |
| [Nonstandard boundary conditions](nonstandard_bcs.md) | Solves BVPs with nonstandard boundary conditions, including: - A mean-zero condition: ... |
| [Order stars](order_stars.md) | Visualizes order stars for Pade approximants to the exponential function .... An order star for a rational approximan... |
| [A parameter-dependent ODE with breakpoints](parameter_ode.md) | Solves ... with ..., where ... is a parameter-dependent piecewise function. The exact solution is |
| [A periodic ODE system](periodic_system.md) | Solves two periodic first-order ODEs: - ... — stable, unique periodic solution |
| [Piecewise operators demo](piecewise_demo.md) | Demonstrates Chebop for solving ... on ... with Dirichlet boundary conditions. The sign function |
| [Poisson Equation](poisson_equation.md) | The 1D Poisson equation ... with Dirichlet boundary conditions ... is the simplest elliptic boundary value problem. Its |
| [Stability regions of ODE formulas](regions.md) | Plots the stability regions of classical ODE time-stepping methods in the complex ... plane. Methods include Adams-Ba... |
| [Resonance exploited by Carrier and Pearson's vandal](resonant_vandal.md) | Solves the harmonic oscillator BVP ... demonstrating resonance when the forcing frequency matches the |
| [Diffmat, diffrow, intmat, introw, gridsample](spectral_disc.md) | Directly demonstrates Chebyshev differentiation matrices ..., ... and their properties. Verifies that ... and ... |
| [Multiple BVP solutions by solving an IVP](two_sol_bvp.md) | The BVP ... with ... has multiple solutions. Different initial guesses for the shooting parameter ... converge to |
| [Linear ODEs from Wikipedia](wiki_odes.md) | A collection of classical linear ODEs, each with known exact solutions, demonstrating the accuracy of Chebyshev spect... |

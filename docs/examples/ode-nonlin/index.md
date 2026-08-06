# Nonlinear ODE Examples

Chebfun solves nonlinear BVPs via Newton's method (the `solve` method
with automatic differentiation). These examples show problems from
physics and engineering.

| Example | Description |
|---------|-------------|
| [An Allen-Cahn equation with continuation](allen_cahn.md) | Solves the Allen-Cahn equation |
| [The Blasius function (replica)](Blasius.md) | Faithful replica: wall shear to 5e-11, displacement constant matches, singularity failure reproduced. |
| [Bloodhound supersonic car](bloodhound.md) | Models the acceleration of the Bloodhound SSC supersonic car: |
| [Blowup equation (Frank-Kamenetskii) (replica)](BlowupFK.md) | Faithful replica: five steady states matching the closed-form solution to 12 digits. |
| [System of two nonlinear BVPs (replica)](BVPSystem.md) | Faithful replica: Newton update history now reported for systems; 7-step quadratic convergence. |
| [The Carrier equation (replica)](Carrier.md) | Faithful replica: three solution branches selected by the initial guess, with Newton convergence histories. |
| [Phase portraits with chebop/quiver (replica)](ChebopQuiver.md) | Faithful replica: van der Pol, damped and undamped pendulum, Lotka-Volterra; quiver gains its system and slope-field cases. |
| [Delay differential equations in Chebfun](delay_odes.md) | Solves delay differential equations (DDEs) including the pantograph equation |
| [A droplet sitting on a surface (replica)](Droplets.md) | Faithful replica: volume to 12 digits; unknown contact radius solved as a scalar parameter. |
| [Exact solutions of nonlinear ODEs from Bender and Orszag (replica)](ExactSolns.md) | Faithful replica: four closed-form ODEs; BVP error 1.3e-15, default Newton guess now satisfies the BCs. |
| [Four bugs on a rectangle](four_bugs.md) | Four bugs start at the corners of a ... rectangle. Each bug always moves directly toward the next bug (clockwise). Th... |
| [Fourier collocation for nonlinear periodic ODEs (replica)](FourierCollocationNonLin.md) | Faithful replica: two Newton branches, second-solution length 81 exact. |
| [A nonlinear system of Guckenheimer and Holmes (replica)](GuckenheimerHolmes.md) | Faithful replica: heteroclinic cycle, crossing-time gaps growing geometrically at ~1.33-1.38 per cycle. |
| [A Gulf Stream model (replica)](GulfStream.md) | Faithful replica: 3rd-order nonlinear BVP with two left-end conditions; conserved quantity I = 1/2. |
| [IVP capabilities of chebop (replica)](IVPCapabilities.md) | Faithful replica: van der Pol marching (display parity), phase-plane direction field, forcing, collocation IVP solver. |
| [Lane-Emden equation from astrophysics (nonlinear)](lane_emden_nonlin.md) | Solves the nonlinear Lane-Emden equation for polytropic indices ...: |
| [Logistic map and chaos (replica)](Logistic.md) | Faithful replica: chebfun iterates in the parameter r; lengths vs MATLAB R2025b, exposing three length bugs. |
| [Logistic map and chaos (replica)](Logistic2.md) | Faithful replica: chebfun-composed logistic iterates, point values to 14 digits. |
| [Lorenz attractor](lorenz_attractor.md) | Numerically integrates the Lorenz system: |
| [Lyapunov exponents (replica)](LyapunovExponents.md) | Faithful replica: Lorenz separation over 10 decades, exponent 0.930 vs published 0.934. |
| [Modelling diseases (replica)](ModellingDiseases.md) | Faithful replica: SIR model, peak 240 exact, crossover time to 11 digits. |
| [Orbiting around fixed masses (replica)](Orbits.md) | Faithful replica: complex-plane orbits via ode113; arc length and closest approach to 9-10 digits. |
| [Parameter-dependent ODEs: three examples](param_odes.md) | Demonstrates three ODE problems with parameters: 1. An eigenvalue-type boundary condition with an interior constraint |
| [Picard iteration for ODE existence proof (replica)](Picard.md) | Faithful replica: iterate error orders t^1..t^4 confirmed by fit; k=4 floor traced to the solver's residual. |
| [Half-wave rectifier](rectifier.md) | Simulates a stiff half-wave rectifier circuit with a diode: |
| [Nonlinear ODE modeling solar magnetic fields](solar_fields.md) | Solves the nonlinear ODE arising in the modeling of force-free solar magnetic fields in a spherical geometry. The equ... |
| [A square limit cycle (replica)](SquareCycle.md) | Faithful replica: heteroclinic cycle through four saddles; switching times match to plotting accuracy. |
| [Three-body problem](three_body.md) | Integrates the Newtonian three-body problem for the famous figure-8 orbit discovered by Chenciner and Montgomery (200... |
| [Pythagorean planets (replica)](ThreePlanets.md) | Faithful replica: complex 3-body self-ionization at t~86; centre of mass conserved to 2.8e-13. |
| [Two electrons orbiting symmetrically about a nucleus (replica)](TwoElectrons.md) | Faithful replica: all 7 published values reproduce; refined z(T) agrees at 1e-13. |

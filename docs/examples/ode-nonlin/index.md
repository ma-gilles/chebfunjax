# Nonlinear ODE Examples

Chebfun solves nonlinear BVPs via Newton's method (the `solve` method
with automatic differentiation). These examples show problems from
physics and engineering.

| Example | Description |
|---------|-------------|
| [An Allen-Cahn equation with continuation](AllenCahn.md) | translation: continuation to eps = 0.003; nine-site arity-trap sweep. |
| [The Blasius function](Blasius.md) | translation: wall shear to 5e-11, displacement constant matches, singularity failure reproduced. |
| [Bloodhound supersonic car](Bloodhound.md) | translation: t1000 = 27.4 s matching the published figure; converged where MATLAB's run warns of Newton failure. |
| [Blowup equation (Frank-Kamenetskii)](BlowupFK.md) | translation: five steady states matching the closed-form solution to 12 digits. |
| [System of two nonlinear BVPs](BVPSystem.md) | translation: Newton update history now reported for systems; 7-step quadratic convergence. |
| [The Carrier equation](Carrier.md) | translation: three solution branches selected by the initial guess, with Newton convergence histories. |
| [Phase portraits with chebop/quiver](ChebopQuiver.md) | translation: van der Pol, damped and undamped pendulum, Lotka-Volterra; quiver gains its system and slope-field cases. |
| [Delay differential equations in Chebfun](DelayDifferentialEquations.md) | translation: 19 of 22 sections, hand-Newton matching MATLAB to 12 digits; 3 sections behind a ledgered performance wall. |
| [A droplet sitting on a surface](Droplets.md) | translation: volume to 12 digits; unknown contact radius solved as a scalar parameter. |
| [Exact solutions of nonlinear ODEs from Bender and Orszag](ExactSolns.md) | translation: four closed-form ODEs; BVP error 1.3e-15, default Newton guess now satisfies the BCs. |
| [Fourier collocation for nonlinear periodic ODEs](FourierCollocationNonLin.md) | translation: two Newton branches, second-solution length 81 exact. |
| [A nonlinear system of Guckenheimer and Holmes](GuckenheimerHolmes.md) | translation: heteroclinic cycle, crossing-time gaps growing geometrically at ~1.33-1.38 per cycle. |
| [A Gulf Stream model](GulfStream.md) | translation: 3rd-order nonlinear BVP with two left-end conditions; conserved quantity I = 1/2. |
| [IVP capabilities of chebop](IVPCapabilities.md) | translation: van der Pol marching (display parity), phase-plane direction field, forcing, collocation IVP solver. |
| [The Lane-Emden equation from astrophysics (partial translation)](LaneEmden.md) | Partial translation: n = 0, 1 exact to 6e-12; n >= 2 blocked on the ledgered singular-endpoint Jacobian defect. |
| [Logistic map and chaos](Logistic.md) | translation: chebfun iterates in the parameter r; lengths vs MATLAB R2025b, exposing three length bugs. |
| [Logistic map and chaos](Logistic2.md) | translation: chebfun-composed logistic iterates, point values to 14 digits. |
| [The fractal structure of the Lorenz attractor](LorenzAttractor.md) | translation: complex-time pole tables to 4 decimals; ratinterp complex-pole fix. |
| [Lyapunov exponents](LyapunovExponents.md) | translation: Lorenz separation over 10 decades, exponent 0.930 vs published 0.934. |
| [Modelling diseases](ModellingDiseases.md) | translation: SIR model, peak 240 exact, crossover time to 11 digits. |
| [Orbiting around fixed masses](Orbits.md) | translation: complex-plane orbits via ode113; arc length and closest approach to 9-10 digits. |
| [Picard iteration for ODE existence proof](Picard.md) | translation: iterate error orders t^1..t^4 confirmed by fit; k=4 floor traced to the solver's residual. |
| [A square limit cycle](SquareCycle.md) | translation: heteroclinic cycle through four saddles; switching times match to plotting accuracy. |
| [The three-body problem](ThreeBodyProblem.md) | translation: figure-eight orbit; ratinterp type (151,8) and error matching MATLAB to 4 digits. |
| [Pythagorean planets](ThreePlanets.md) | translation: complex 3-body self-ionization at t~86; centre of mass conserved to 2.8e-13. |
| [Two electrons orbiting symmetrically about a nucleus](TwoElectrons.md) | translation: all 7 published values reproduce; refined z(T) agrees at 1e-13. |

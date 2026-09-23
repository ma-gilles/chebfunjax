# Complex Analysis

Chebfun handles complex-valued functions naturally, enabling contour
integration, winding number computation, and argument principle calculations.

| Example | Description |
|---------|-------------|
| [A keyhole contour integral](KeyholeContour.md) | translation: log(x)tanh(x) around the keyhole to 2.8e-15 (beats the published 1.3e-14). |
| [Rouche's theorem](RoucheTheorem.md) | translation: winding-number illustrations; 3 roots inside as Rouche predicts. |
| [Integrals over closed contours](ClosedContours.md) | translation: residues 5/3 pi i (1e-15) and essential-singularity 2 pi i. |
| [Arc length of complex paths](ComplexArcLength.md) | translation: keyhole and flower arc lengths to 13-15 digits; 64 equal-arclength points. |
| [Phase and argument](Arguments.md) | translation: angle vs unwrapped argument on a spiral; smooth square-root branch. |
| [Phase portraits of singularities](Singularities.md) | translation: removable, pole-order, and essential singularities as phase portraits. |
| [Zeros of the Riemann zeta function](ZetaZeros.md) | translation: 8 critical-line zeros to ~10 digits via complex chebfun rootfinding. |
| [Analytic continuation](AnalyticContinuation.md) | translation: chebfun length 30 exact; ratinterp poles capture tanh's poles with the published accuracy cascade. |
| [Portraits with poles](PortraitsWithPoles.md) | translation: smash-trick phase portraits of pole-bearing functions. |
| [Complex minimax](ComplexMinimax.md) | translation: AAA-Lawson error circles with winding number 9; disk error to 8 digits. |
| [Phase portraits with chebfun2](PhasePortraits.md) | translation: sin(z), cos(z^2), near-roots-of-unity, sin(z)-sinh(z). |
| [Ablowitz-Fokas double keyhole](KeyholeAblowitzFokas.md) | translation: both contours give sqrt(2)/2 to the last digit. |
| [Rational harmonic zeros](RationalHarmonic.md) | translation: 10 lensing zeros (max 5n-5) and the 15-zero perturbed count exactly as published. |
| [The phaseplot command](PhaseplotCommand.md) | translation: handle-based phase portraits including branch cuts and an essential singularity. |
| [Hyperfunctions](Hyperfuns.md) | translation: delta and Heaviside as boundary values of analytic functions. |
| [Visualizing conformal maps](ConformalVis.md) | translation: half-strip to disk via sinh + Mobius, quasimatrix of joined squares, scribble text. |
| [Conformal map to a square](ConformalSquare.md) | translation: Schwarz-Christoffel disk-to-square map by integrating f' along rays and circles. |
| [Conformal mapping of an L-shaped region](ConformalL.md) | translation: lightning least-squares Laplace solve + AAA map with exponentially clustered poles. |
| [Conformal mapping in Chebfun](ConformalMapping.md) | translation: Kerzman-Stein conformal map of a random smooth region, AAA rational representations. |
| [Conformal maps to an annulus](ConformalMapping2.md) | translation: conformal2 doubly-connected maps, conformal modulus rho to 15 digits vs MATLAB. |

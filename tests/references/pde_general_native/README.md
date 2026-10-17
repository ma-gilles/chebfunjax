# Native real multicomponent PDE and condition-estimator fixtures

Reference Chebfun: 7574c77680d7e82b79626300bf255498271a72df.
Native MATLAB: R2025b Update 2. CPU binary64 only.

The PDE harness invokes ordinary `pdeSolve(f,t,u0,bc,opts)` method dispatch
with a vertically concatenated Chebfun initial state, fixed n=9, Neumann
handles, and three species/times. Installed ode15s supplies the primary solve.
An instrumented scratch replay must reproduce its returned mesh, states,
callbacks and post-solve RNG exactly; all four recorded gates pass.
The actual six-by-six algebraic Jacobian is captured before the source
condition check. No manufactured Jacobian replaces the native PDE assembly.
Native fixed-N tOut is [0] because the pinned callback does not update it;
physical samples are indexed using the actual requested/native solver times.
Complete native return-signature parity is not claimed.

The standalone estimator captures use the native LU factors and normest1
operator actions, including exact action inputs and final uniform RNG state.
The suite also compares candidate LU factors and full condition estimates,
threshold decisions, block widths, singular/zero routes and consecutive draws.
MAT readers must use mat_dtype=True to retain semantic double types despite
MATLAB's compact integer storage encoding.

Two native replay tests qualify the already-captured post-numjac initializer
and the actual public Python N=3 solve. Manufactured N=3/N=5 controls retain
existing forward-error bounds. These controls do not qualify mixed boundary
orders/counts, complex systems, variable mass matrices or every PDE example.
The uniform-stream owner is estimator-scoped and persistent; it is not a
library-wide MATLAB rng implementation or a concurrency-ordering guarantee.

The first 27-case gate had 18 passes/9 failures: six boolean assertion type
errors and three real scalar rounding discrepancies. Native-factor operator
traces passed, isolating the numerical gap to the default LU provider.
Using the existing source-derived real LU schedule fixes the discrepancies;
the second gate passed 36 tests, including nine native factor comparisons,
without changing acceptance bounds or fixtures.

Numeric captures, hashes and original observation harnesses may be retained.
Licensed ode15s/private source copies and INSTRUMENTATION.diff are excluded.
The retained observation harness requires the recorded scratch instrumentation;
it is not a standalone redistributable replacement for installed MATLAB.

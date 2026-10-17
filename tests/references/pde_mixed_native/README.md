# Native polynomial PDE boundary fixtures

Pinned Chebfun7574c77680d7e82b79626300bf255498271a72df; MATLAB R2025b Update2.
CPU binary64. Numerical MATs are authoritative; use mat_dtype=True when decoding.

Mixed case: fixedN9, three expressions with orders(2,2,0), two residuals per
endpoint. Actual native P/M and offset/RHS were observed at the installed
ode15s boundary. Block rows are7,7,9; the zero-order block is not identity.
All nine species/time sample cells are present. This is a primary-only capture,
not an instrumented-solver replay or initializer-coordinate comparison.
Its joint capture process exited1 because a separate arity harness failed
before solving; the mixed case itself completed and its guarded inputs stayed
unchanged. Subsequent CPU tests compare its captured arrays and public values.

Arity case: N+1 PDE callback receives space; endpoint callbacks receive time.
The first harness used named handles, which pinned backCompat cannot parse.
The second used anonymous wrappers but Chebmatrix numel yielded only one
sample cell. Its complete metadata was insufficient evidence. Version3 uses
explicit species/time size loops and requires all9 cells to have3 finite values
before marking completion. Version3 process exited0 with stable source guards
and no survivors. No equations, reference values or acceptance bounds were fit.

Original bounds:100eps scaled for source arrays;1e-7 for public/manufactured
solutions. Empty/absent-boundary and reordered/count-mismatch controls are
independent source/manufactured checks, not additional native solution captures.
For systems, native MassSingular=yes retains full-mass initialization/DAE
startup even for full-rank zero-boundary projection. Scalar no-BC compatibility
is separate and unchanged.

Observation scripts and source hashes are retained. Their manifests refer to
original scratch paths; regenerate path guards before a relocated recapture.
No licensed MATLAB solver/private-source copies are distributed.
Remaining scope includes shorthand/chemical full example, complex/variable
mass/Fourier paths, native return representation and legacy SciPy routes.

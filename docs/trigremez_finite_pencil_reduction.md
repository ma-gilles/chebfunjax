# Finite trigonometric Remez pencil reduction

Source: Chebfun7574c77680d7e82b79626300bf255498271a72df, @chebfun/trigremez.m computeTrialFunctionRational. The library new provider uses JAX QR/eig; the source uses generalized eig(A,B).

## Exact-arithmetic invertibility

Let N=2(m+n)+2 distinct reference points be in cyclic order, with alternating signs sigma_i=(-1)^i. P spans real trigonometric polynomials of degree m, Q those of degree n, and S=diag(sigma). A vector in the kernel of [P,SQ] corresponds (up to an irrelevant sign) to real polynomials u,v with u(theta_i)=sigma_i*v(theta_i). A real null vector suffices because the matrix is real.

The product H=u*v has degree at most m+n, and H(theta_i)=sigma_i*v(theta_i)^2. If v is nonzero at every node, these values alternate around the circle, forcing at least N zeros, more than the maximum2(m+n) of a nonzero trigonometric polynomial.

Zero nodes do not invalidate this count. At a zero of v on the reference, u also vanishes, so H has multiplicity at least2. Between consecutive nodes with nonzero v, suppose r intermediate nodes have zero v. They contribute at least2r root multiplicity. For r even, endpoint H signs differ, so total multiplicity in the gap must be odd and is at least2r+1; for r odd it is at least2r. If r=0 the sign change requires at least one zero. Every gap therefore contributes at least r+1, and the gap counts sum to N. If no nonzero node exists, N>2n already forces v=0. Otherwise H must vanish identically; analyticity then forces u=0 or v=0, and their node equations with N>2m,2n force both zero. Thus [P,SQ] is nonsingular. P has full column rank by the same root count.

Write a complete QR P=U R, and L=U[:,2m+1:] for its left null-space basis. Therefore B_reduced=L^T S Q is nonsingular: a null vector would give SQb in the range of P, contradicting the preceding argument. Left multiplication of the original pencil by U^T eliminates numerator columns and leaves (L^T F Q)b=h*(L^T S Q)b. Solve this finite eigenproblem, recover numerator coefficients with R, and apply the source pole-free q.roots test.

The original generalized B is singular by construction (its numerator block is zero); this does not make the full pencil singular. Under these reference assumptions the full pencil is regular, with finite eigenpairs represented by the reduced problem and remaining eigenvalues at infinity.

## Finite precision and invalid references

This proof concerns exact arithmetic and distinct points modulo a period, in cyclic order. A duplicate node, both endpoints of the same period, or a reordered alternating-sign assignment violates the hypotheses. Clustered distinct nodes can make QR/reduced solves ill-conditioned in floating point even though exact B_reduced is invertible. No fitted condition threshold or claim of arbitrary degenerate eigenspace identity is made. Native and JAX eigenvector common sign/scale, order in degenerate eigenspaces, and roundoff-level extrema can differ.

CONDITIONING.json records double-precision static QR/SVD diagnostics of retained source reference arrays (no approximation recomputation). Native first-trial condition2(B_reduced)=sqrt(2) in all four captured cases. Native final condition numbers are approximately2.420 (abs2,2),3.840 (exp6,6),3.089 (shifted exact1,1),24.380 (exact0,1). Qualified Python final conditions are2.420 and4.915 for the first two cases. These observations describe the qualified references; they are not universal conditioning bounds.

All four native initial trials have exactly one pole-free finite eigenpair, so backend eigenvalue order does not affect those comparisons. The corrected positive-iteration branch passes every original13predicate with original bounds; public abs/exp outputs match native to1.22e-15/6.22e-15 on501 captured query points, including native5/1 iterations and no convergence warnings. The exp case's rounding-scale extrema/status differ and are retained explicitly.

Invalid-reference diagnostics are outside the native valid-reference algorithm; this proof introduces no additional runtime rejection policy. No universal invalid-pencil equivalence is claimed.

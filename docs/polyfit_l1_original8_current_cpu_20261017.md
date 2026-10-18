# Fresh literal native invocation

The unmodified pinned `test_polyfitL1()` function now also passes all eight
default predicates in a clean bounded native run (13.3182s in the function).
This supersedes the invocation limitation of the historical harness below.
Python evidence remains explicitly bound to36e; the supplemental external
JAX cosine warning case and full library API remain unresolved.

# Eight original L1 fitting predicates: current CPU audit

Frozen 36e6e43f; native source 7574c77. Each unchanged predicate ran individually under 180 seconds / 4 GiB. **8 pass, 0 fail, 0 unverified among these eight.** No library or test edits.

| Native clause | Input/domain | Fit degree / basis degree | Current moment norm | Original bound |
|---|---|---:|---:|---:|
| 1 | exp(x)*sin(10*x) [-1.0, 1.0] | 5 / 5 | 6.28273e-15 | <1e-8 |
| 2 | exp(x)*sin(10*x) [-1.0, 1.0] | 10 / 5 | 2.11002e-13 | <1e-8 |
| 3 | exp(x)*sin(10*x) [-2.0, 2.0] | 5 / 5 | 2.66076e-14 | <1e-8 |
| 4 | abs(x) [-1.0, 1.0] | 7 / 7 | 2.16849e-14 | <1e-8 |
| 5 | abs(x) [-2.0, 2.0] | 7 / 7 | 6.94195e-14 | <1e-8 |
| 6 | sin(x-.1).*cos(3*x) [-3.0, 2.0] | 11 / 11 | 4.59948e-14 | <1e-8 |
| 7 | sin(x-.1)+cos(3*x) [0.0, 100.0] | 11 / 11 | 3.86757e-13 | <1e-8 |
| 8 | sin(x-.1)+cos(3*x) [0.0, 100.0] | 101 / 101 | 1.31989e-12 | <1e-8 |

Case 2 retains the original T0..5 basis at fit degree 10. Cases 4/5 retain splitting; cases 7/8 retain the full [0,100] domain. All receipts have stable inputs, no censor and no surviving processes.

Historical native source predicates and Python passes are distinguished from these fresh current gates. Exact test bytes and historical core native source hashes match; MATLAB was not rerun.

The supplemental cosine degree-four warning test remains unresolved and is not one of the original eight. Its sampling provider discrepancy remains open. No warnings, NaNs, roots, constructor routing or tolerances were changed.

These eight functional predicate passes do not close the full API, warnings, representation, preferences, pages, performance or complete CI. The JSON report and raw caseNN_run1 receipts / caseNN captures record hashes, norms and limits.

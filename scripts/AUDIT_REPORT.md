# jaxchebfun Examples Audit Report

**Date:** 2026-03-26
**Scope:** All 386 original MATLAB Chebfun examples vs jaxchebfun translations

---

## Executive Summary

| Metric | Count | % of 386 |
|--------|-------|----------|
| MATLAB originals | 386 | 100% |
| Translated (have Python script) | 267 | 69% |
| Documented (have doc page matched to MATLAB) | 281 | 73% |
| Fully done (script + doc) | 254 | 66% |
| Untranslated | 114 | 30% |
| Extra Python scripts (original work, not in MATLAB) | 97 | — |

**Total Python scripts:** 407 (267 translations + 97 original + 43 with broken doc links)
**Total doc pages:** 329

---

## Issue Categories

### 1. BROKEN CHEBFUN.ORG LINKS (140 of 303 unique URLs = 46%)

**Severity: HIGH** — Nearly half of all chebfun.org links return 404.

Root causes:
- **ODE categories** (ode-eig, ode-linear, ode-nonlin): Links use lowercased/concatenated names (e.g., `adjoints.html`) but chebfun.org uses CamelCase (`Adjoints.html`). **All 97 ODE doc page links are broken.**
- **Fabricated/wrong names**: Some doc pages link to MATLAB example names that never existed on chebfun.org (e.g., `ChebCoeffs`, `Interp`, `PiecewiseSmooth`, `ComplexFunctions`)
- **Cross-category links**: 10 pages link to different categories (e.g., `temp/frac_calc` → `integro/FracCalc`) where the URL path doesn't exist
- **Non-existent categories**: 3 pages link to `ode/` (not a real Chebfun category)

**Working links:** 163/303 (54%)

### 2. STUB DOC PAGES (68 of 329 = 21%)

**Severity: MEDIUM** — These pages contain only:
- Title + author + one-sentence summary
- `from examples.{cat}.{name} import run; run()` code block
- One output image

No mathematical narrative, no code snippets, no references. Categories affected:
- stats: 21 stubs
- fun: 12 stubs
- geom: 12 stubs
- sphere: 12 stubs
- temp: 11 stubs

### 3. APPROX2 DUPLICATE SCRIPTS (55 scripts)

**Severity: HIGH** — The `examples/approx2/` directory contains 55 CamelCase scripts (e.g., `AAAApprox.py`, `BestApprox.py`) that are **near-duplicates of `examples/approx/` scripts**. These are 1D approximation examples incorrectly placed in the 2D approximation category. They have:
- No doc pages
- No images in `docs/images/approx2/`
- Slightly different from originals (missing `chebfun_style()` call, extra env var setup)

The real approx2 translations (5 snake_case scripts like `chebfun2_basics.py`) are separate.

### 4. MISATTRIBUTED AUTHORS (confirmed in sample)

**Severity: HIGH** — At least one doc page (`stats/bayesian_gradebook.md`) attributes authorship to the wrong person (says "Nick Trefethen, September 2014" instead of correct "Toby Driscoll, November 2013"). A systematic author verification is needed.

### 5. MISREPRESENTED CONTENT (confirmed in sample)

**Severity: MEDIUM** — Some doc pages fundamentally mischaracterize the original example:
- `ode-nonlin/lorenz_attractor.md`: Original is about complex singularity detection via `ratinterp`; jaxchebfun reduces to trivial ODE solve
- `sphere/advection_diffusion.md`: Original is about the **unit ball** (3D); jaxchebfun says "sphere"
- `geom/ellipse.md`: Says "Poisson's formula" but original uses arc-length integration

### 6. DUPLICATE DOC PAGES (6 pairs)

**Severity: LOW** — Both CamelCase and snake_case doc pages exist for the same example:
- `calc/Integrals.md` AND `calc/integrals.md`
- `calc/MeanValueTheorem.md` AND `calc/mean_value_theorem.md`
- `calc/SnellsLaw.md` AND `calc/snells_law.md`
- `roots/BesselRoots.md` AND `roots/bessel_roots.md`
- `roots/NewtonRaphson.md` AND `roots/newton_raphson.md`
- `roots/RootsNearAxis.md` AND `roots/roots_near_axis.md`

### 7. DOC PAGES WITH NO CHEBFUN.ORG LINK (5 pages)

**Severity: LOW**
- `approx2/chebfun2_basics`
- `approx2/smooth_functions_2d`
- `calc/DeltaDerivs`
- `calc/SurfaceRevolution`
- `roots/FundamentalTheoremAlgebra`

### 8. DOC PAGES WITHOUT MATCHING SCRIPTS (27 pages)

**Severity: MEDIUM** — CamelCase doc pages exist but the actual Python script uses a different (snake_case) name. The doc pages have content but point to scripts that don't exist under the CamelCase name.

Mostly in: complex (10), quad (4), calc (4), roots (9)

---

## Per-Category Status

| Category | MATLAB | Translated | Documented | Fully Done | Untranslated | Extra |
|----------|--------|-----------|------------|------------|--------------|-------|
| applics | 8 | 7 | 3 | 3 | 1 | 1 |
| **approx** | **55** | **55** | **55** | **55** | **0** | **13** |
| approx2 | 16 | 3 | 3 | 3 | 16 | 57 |
| **approx3** | **11** | **11** | **11** | **11** | **0** | **2** |
| calc | 6 | 6 | 6 | 6 | 0 | 2 |
| cheb | 10 | 8 | 8 | 8 | 4 | 2 |
| complex | 22 | 3 | 3 | 3 | 12 | 9 |
| disk | 2 | 0 | 0 | 0 | 2 | 1 |
| fourier | 5 | 3 | 3 | 3 | 2 | 0 |
| fun | 16 | 12 | 12 | 12 | 4 | 1 |
| geom | 14 | 13 | 12 | 12 | 1 | 1 |
| integro | 6 | 5 | 5 | 5 | 1 | 2 |
| linalg | 18 | 5 | 5 | 5 | 16 | 0 |
| **ode-eig** | **17** | **17** | **17** | **17** | **1** | **3** |
| **ode-linear** | **32** | **32** | **32** | **32** | **3** | **3** |
| **ode-nonlin** | **30** | **28** | **28** | **28** | **4** | **4** |
| **ode-random** | **10** | **10** | **10** | **10** | **0** | **0** |
| opt | 11 | 5 | 5 | 5 | 7 | 0 |
| pde | 15 | 3 | 3 | 3 | 8 | 8 |
| quad | 12 | 5 | 5 | 5 | 6 | 4 |
| roots | 20 | 7 | 7 | 7 | 11 | 7 |
| sphere | 13 | 13 | 12 | 12 | 0 | 1 |
| stats | 22 | 21 | 21 | 21 | 2 | 0 |
| temp | 11 | 2 | 2 | 2 | 9 | 0 |
| veccalc | 4 | 0 | 0 | 0 | 4 | 1 |

Bold = categories at or near 100% translation.

### Best covered categories (100%):
- approx (55/55), approx3 (11/11), ode-random (10/10), calc (6/6)

### Worst covered categories (< 30%):
- linalg (5/18 = 28%), approx2 (3/16 = 19%), disk (0/2), veccalc (0/4)

---

## Image Coverage

- **352/407 scripts** have images in `docs/images/` (86%)
- **55 scripts missing images** — all 55 are the approx2 CamelCase duplicates
- **0 broken image references** in doc pages
- Images appear to render correctly (PNG format, reasonable sizes)

---

## Chebfun Style

- **407/407 scripts** (100%) use `chebfun_style()` from `chebfunjax.plotting`
- **407/407 scripts** (100%) have a `run()` function
- **0 scripts** are missing savefig calls (when they have images)

---

## Text Quality Assessment (8-example sample)

| Rating | Count | Examples |
|--------|-------|----------|
| MOSTLY_FAITHFUL | 3 (37%) | AAAApprox, AbsoluteValue, bessel_function_roots |
| DIVERGENT | 5 (63%) | adjoints, bayesian_gradebook, ellipse, lorenz_attractor, advection_diffusion |

Common issues in DIVERGENT pages:
- Reduced to 1-sentence summaries with no mathematical narrative
- Missing references and citations
- Mischaracterized mathematical content
- At least 1 wrong author attribution

---

## Priority Fixes

### P0 — Data Integrity
1. **Delete 55 approx2 CamelCase duplicates** — these are 1D approx copies in wrong directory
2. **Fix misattributed authors** — systematic verification needed across all 329 pages
3. **Fix 140 broken chebfun.org links** — ODE categories need CamelCase URLs; fabricated names need correction

### P1 — Coverage Gaps
4. **Remove 6 duplicate doc pages** (CamelCase + snake_case pairs)
5. **Fix 27 orphaned CamelCase doc pages** — rename to match snake_case scripts, or redirect
6. **Add 5 missing chebfun.org links**

### P2 — Content Quality
7. **Expand 68 stub doc pages** with actual narrative from Chebfun originals
8. **Review 5 mischaracterized examples** for factual accuracy

### P3 — Coverage Expansion
9. **Translate 114 untranslated MATLAB examples** (30% of originals)
10. **Add docs/images for 97 extra Python scripts**

---

## Files

- Audit script: `scripts/audit_examples_v2.py`
- Full CSV: `scripts/audit_results_v2.csv`
- URL validation results: `/tmp/chebfun_url_results.txt`

#!/usr/bin/env python3
"""Compare chebfunjax-generated figures against Chebfun reference renders.

For each (generated, reference) image pair this computes lightweight
perceptual metrics and (optionally) writes a stacked montage PNG
(reference on top, chebfunjax below) for visual review.

Reference images are the MATLAB renders published on chebfun.org. They are
NOT stored in the repo; pass their location via --refs (default: the audit
scratch area). Guide figures pair by filename (guideNN_MM.png). Example
figures pair by filename when names match, otherwise by document order
within a page (see --pages mode).

Usage:
    python scripts/compare_plots.py --kind guide                # all chapters
    python scripts/compare_plots.py --kind guide --only 03      # one chapter
    python scripts/compare_plots.py --pair gen.png ref.png      # single pair
    python scripts/compare_plots.py --kind examples --only approx

Output: CSV of metrics (ranked worst-first) + montages under --out.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_REFS = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/refs/docs/images"
)
PROJECT = Path(__file__).resolve().parent.parent
GENERATED = PROJECT / "docs" / "images"
DEFAULT_OUT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/chebfunjax_audit_20260702/out/compare"
)

THUMB = (128, 64)


def _load_gray(path: Path, size=THUMB) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size)
    return np.asarray(img, dtype=np.float64) / 255.0


def _load_rgb_hist(path: Path, bins: int = 32) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB").resize((256, 128)))
    hists = [
        np.histogram(img[..., c], bins=bins, range=(0, 255), density=True)[0]
        for c in range(3)
    ]
    return np.concatenate(hists)


def compare_pair(gen: Path, ref: Path) -> dict:
    """Return similarity metrics for one image pair (higher score = worse).

    A generated file byte-identical to the reference is flagged
    NOT_REGENERATED: it means the chebfun.org MATLAB render is still in
    place and chebfunjax never produced the figure — identical bytes
    would otherwise score a perfect 0 and mask exactly the failure this
    tool exists to catch.
    """
    if gen.read_bytes() == ref.read_bytes():
        gi = Image.open(gen)
        return {
            "generated": str(gen),
            "reference": str(ref),
            "gen_size": f"{gi.width}x{gi.height}",
            "ref_size": f"{gi.width}x{gi.height}",
            "aspect_err": 0.0,
            "gray_mae": 0.0,
            "hist_corr": 1.0,
            "badness": 999.0,
            "flag": "NOT_REGENERATED",
        }
    gi, ri = Image.open(gen), Image.open(ref)
    aspect_gen = gi.width / gi.height
    aspect_ref = ri.width / ri.height
    aspect_err = abs(aspect_gen - aspect_ref) / aspect_ref

    mae = float(np.abs(_load_gray(gen) - _load_gray(ref)).mean())

    hg, hr = _load_rgb_hist(gen), _load_rgb_hist(ref)
    denom = float(np.linalg.norm(hg) * np.linalg.norm(hr))
    hist_corr = float(np.dot(hg, hr) / denom) if denom > 0 else 0.0

    # Composite badness: pixel difference dominates, aspect and palette
    # mismatches add on top.
    badness = mae + 0.5 * aspect_err + 0.5 * (1.0 - hist_corr)
    return {
        "generated": str(gen),
        "reference": str(ref),
        "gen_size": f"{gi.width}x{gi.height}",
        "ref_size": f"{ri.width}x{ri.height}",
        "aspect_err": round(aspect_err, 4),
        "gray_mae": round(mae, 4),
        "hist_corr": round(hist_corr, 4),
        "badness": round(badness, 4),
        "flag": "",
    }


def montage(gen: Path, ref: Path, out: Path) -> None:
    """Write a stacked comparison image: reference on top, generated below."""
    ri, gi = Image.open(ref).convert("RGB"), Image.open(gen).convert("RGB")
    w = max(ri.width, gi.width)
    sep = 8
    m = Image.new("RGB", (w, ri.height + gi.height + sep), (255, 210, 210))
    m.paste(ri, (0, 0))
    m.paste(gi, (0, ri.height + sep))
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(out)


def collect_pairs(kind: str, refs: Path, only: str | None) -> list[tuple[Path, Path]]:
    """Pair generated and reference images by matching relative filename."""
    pairs = []
    if kind == "guide":
        ref_dir = refs / "guide"
        gen_dir = GENERATED / "guide"
        pattern = f"guide{only}_*.png" if only else "guide*.png"
        for ref in sorted(ref_dir.glob(pattern)):
            gen = gen_dir / ref.name
            pairs.append((gen, ref))
    else:  # examples: every non-guide category
        for ref_dir in sorted(p for p in refs.iterdir() if p.is_dir()):
            cat = ref_dir.name
            if cat == "guide" or (only and cat != only):
                continue
            for ref in sorted(ref_dir.glob("*.png")):
                pairs.append((GENERATED / cat / ref.name, ref))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["guide", "examples"], default="guide")
    ap.add_argument("--only", help="chapter number (guide) or category (examples)")
    ap.add_argument("--pair", nargs=2, metavar=("GEN", "REF"),
                    help="compare a single explicit pair")
    ap.add_argument("--refs", type=Path, default=DEFAULT_REFS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--montages", type=int, default=0,
                    help="write montages for the N worst pairs (0 = none)")
    args = ap.parse_args()

    if args.pair:
        gen, ref = Path(args.pair[0]), Path(args.pair[1])
        row = compare_pair(gen, ref)
        for k, v in row.items():
            print(f"{k}: {v}")
        out = args.out / "single" / f"{gen.stem}_vs_ref.png"
        montage(gen, ref, out)
        print(f"montage: {out}")
        return 0

    pairs = collect_pairs(args.kind, args.refs, args.only)
    rows, missing = [], []
    for gen, ref in pairs:
        if not gen.exists():
            missing.append(str(gen))
            continue
        rows.append(compare_pair(gen, ref))

    rows.sort(key=lambda r: -r["badness"])
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.kind}{'_' + args.only if args.only else ''}"
    csv_path = args.out / f"metrics_{tag}.csv"
    if rows:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    for row in rows[: args.montages]:
        gen, ref = Path(row["generated"]), Path(row["reference"])
        montage(gen, ref, args.out / "montages" / tag / f"{gen.stem}.png")

    not_regen = [r for r in rows if r.get("flag") == "NOT_REGENERATED"]
    print(f"compared: {len(rows)}  missing generated: {len(missing)}  "
          f"NOT_REGENERATED (byte-identical to chebfun.org ref): {len(not_regen)}")
    for m in missing[:20]:
        print(f"  MISSING {m}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    for r in not_regen[:20]:
        print(f"  NOT_REGENERATED {Path(r['generated']).name}")
    if len(not_regen) > 20:
        print(f"  ... and {len(not_regen) - 20} more")
    if rows:
        print(f"metrics: {csv_path}")
        print("worst 10:")
        for row in rows[:10]:
            print(f"  {row['badness']:>7} {Path(row['generated']).name} "
                  f"(mae={row['gray_mae']}, aspect={row['aspect_err']}, "
                  f"hist={row['hist_corr']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

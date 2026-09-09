#!/usr/bin/env python3
"""Render contact sheets of candidate rubber frames, one tile per cell.

Two uses. As a preview it answers "what does a good frame actually look like",
which is hard to judge from a score alone. As the input to park triage it puts
every park's frames side by side, so a park whose camera never shows a usable
rubber is obvious at a glance rather than after labeling a dozen cells.

Tiles are cropped and zoomed the same way rubber_04b and rubber_04c do it, so
what appears here is exactly what a labeler would be asked to click on. Each
tile is captioned with the park, the cell, and the frame score.

Usage:
    python pipeline/rubber_04e_contact_sheet.py --season 2025
    python pipeline/rubber_04e_contact_sheet.py --season 2025 \
        --park CLE --per-park 12
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from rubber_04b_label_pack import build_crop

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBBER_DIR = REPO_ROOT / "data" / "rubber"

# Tiles are downscaled from the labeler's 3x view: enough to judge whether the
# rubber and pivot foot are legible, small enough to fit a park per screen.
TILE_W, TILE_H = 440, 150
CAPTION_H = 22


def load_scores(season: int) -> dict[str, list[dict]]:
    path = RUBBER_DIR / f"frame_scores_{season}.csv"
    out: dict[str, list[dict]] = {}
    if not path.exists():
        return out
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["cell_id"], []).append(row)
    for v in out.values():
        v.sort(key=lambda r: -float(r["score"]))
    return out


def load_parks(season: int) -> dict[str, str]:
    path = RUBBER_DIR / f"clip_manifest_{season}.csv"
    parks: dict[str, str] = {}
    if not path.exists():
        return parks
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            parks[f"{r['game_pk']}_{r['pitcher']}_{r['stand']}"] = r.get("park", "?")
    return parks


def tile_for(frames_dir: Path, rec: dict) -> np.ndarray | None:
    img = cv2.imread(str(frames_dir / rec["frame_file"]))
    if img is None:
        return None
    canvas, _, _ = build_crop(img, int(rec["crop_cx"]), int(rec["crop_cy"]))
    return cv2.resize(canvas, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)


def caption(tile: np.ndarray, text: str, score: float) -> np.ndarray:
    out = np.zeros((TILE_H + CAPTION_H, TILE_W, 3), np.uint8)
    out[CAPTION_H:] = tile
    # Same thresholds the labeler colours by, so a tile that looks red here is
    # the same tile the labeler would tell you not to bother with.
    col = (132, 220, 61) if score >= 0.60 else (84, 180, 255) if score >= 0.45 \
        else (107, 107, 255)
    cv2.putText(out, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1,
                cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--park", default=None, help="limit to one park")
    ap.add_argument("--per-park", type=int, default=6)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    frames_dir = RUBBER_DIR / "frames" / str(args.season)
    out_dir = Path(args.out_dir) if args.out_dir else RUBBER_DIR / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(args.season)
    parks = load_parks(args.season)

    # Best candidate per cell, grouped by park and ordered best-first so the top
    # of each sheet shows the park at its most favourable. A park that looks bad
    # even in its own best frames is a park to exclude.
    by_park: dict[str, list[tuple[float, str, dict]]] = {}
    for cell_id, cands in scores.items():
        park = parks.get(cell_id, "?")
        if args.park and park != args.park:
            continue
        by_park.setdefault(park, []).append(
            (float(cands[0]["score"]), cell_id, cands[0]))
    for v in by_park.values():
        v.sort(key=lambda t: -t[0])

    written = []
    for park, items in sorted(by_park.items()):
        picks = items[: args.per_park]
        tiles = []
        for score, cell_id, rec in picks:
            t = tile_for(frames_dir, rec)
            if t is None:
                continue
            tiles.append(caption(t, f"{park}  {cell_id}  score {score:.2f}", score))
        if not tiles:
            continue
        cols = min(args.cols, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((rows * (TILE_H + CAPTION_H), cols * TILE_W, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * (TILE_H + CAPTION_H):(r + 1) * (TILE_H + CAPTION_H),
                  c * TILE_W:(c + 1) * TILE_W] = t
        path = out_dir / f"{park}_{args.season}.png"
        cv2.imwrite(str(path), sheet)
        written.append((park, len(tiles), path))

    for park, n, path in written:
        print(f"{park:>4}  {n:2d} tiles  -> {path}")
    print(f"\n{len(written)} sheets in {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score every candidate frame so unusable ones can be filtered before labeling.

Two separate problems make the raw label pack painful, and they need different
fixes:

  1. Wrong frame. rubber_03 saves up to ~4 candidates per cell, and some are
     broadcast transitions, replay wipes, or home-plate camera angles rather
     than the centre-field mound view. Graphics are easy to catch because
     baseball frames live almost entirely in turf-green and dirt-tan hues, while
     a transition wipe is saturated magenta/cyan -- so the fraction of pixels in
     natural hues separates them cleanly.

  2. Wrong crop on a usable frame. The label pack centres each crop on the
     bottom edge of rubber_03's motion bounding box, which is the pivot foot
     when the motion blob really was the pitcher. When it latched onto a batter,
     an umpire, or a score bug instead, the crop lands on empty grass even
     though the frame itself was fine. That cell is recoverable by re-centring,
     not by skipping.

So this scores each candidate frame twice: once for whether the FRAME is a
centre-field field view at all, and once for whether a given crop centre shows
mound dirt with a plausible rubber bar in it. For the crop centre it tries both
the motion prior and a dirt-blob estimate and keeps whichever scores better,
which rescues the mis-centred cells.

The rubber bar is a confidence signal rather than a gate, deliberately. At parks
where the rubber is buried in mound dirt the bar is genuinely hard to see, and
those cells are still labelable by eye -- gating on bar detection would throw
away exactly the parks the Phase 0 spike already showed are hardest.

Output: data/rubber/frame_scores_<season>.csv, one row per candidate frame:
    cell_id, cand, frame_file, frame_class, frame_score, crop_score,
    crop_cx, crop_cy, crop_source, natural_frac, dirt_frac, grass_frac,
    white_frac, bar_w_px, bar_conf, score

`score` is the overall 0-1 usability estimate the labeler sorts on.

Usage:
    python pipeline/rubber_04d_frame_score.py --season 2025
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBBER_DIR = REPO_ROOT / "data" / "rubber"

CROP_W, CROP_H = 440, 150          # must match rubber_04b_label_pack
NATURAL_MIN = 0.45                 # below this the frame is graphics, not field
DIRT_MIN = 0.08                    # a mound crop needs some dirt
WHITE_MAX = 0.14                   # above this we are looking at chalk lines
BAR_W_RANGE = (40, 210)            # rubber width in px across observed zooms
# In the centre-field shot the mound sits low in frame and home plate high, so a
# dirt blob above this line is the plate circle, not the mound. Measured across
# the 2025 frames the mound centroid runs 0.70-0.85 of frame height and the
# plate circle 0.40-0.55, so the cut is wide of both.
MOUND_MIN_CY_FRAC = 0.55


def hue_masks(bgr: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)
    return {
        "grass": (H >= 30) & (H <= 90) & (S > 60),
        "dirt": (H >= 5) & (H <= 25) & (S > 50) & (V > 60),
        # Turf and dirt together. A broadcast wipe or a sponsor bumper sits well
        # outside this band, which is what makes it a reliable graphic detector.
        "natural": (H >= 5) & (H <= 95) & (S > 35) & (V > 40),
        "white": (S < 55) & (V > 165),
        "bright_low_sat": (S < 70) & (V > 150),
    }


def classify_frame(bgr: np.ndarray) -> tuple[str, float, dict]:
    m = hue_masks(bgr)
    nat = float(m["natural"].mean())
    stats = {
        "natural_frac": nat,
        "grass_frac": float(m["grass"].mean()),
        "dirt_frac": float(m["dirt"].mean()),
        "white_frac": float(m["white"].mean()),
    }
    if nat < NATURAL_MIN:
        # Confidence that it IS a field view scales with how close it got.
        return "graphic", max(0.0, nat / NATURAL_MIN) * 0.3, stats
    if stats["dirt_frac"] < 0.02:
        return "no_dirt", 0.25, stats
    if stats["white_frac"] > 0.22:
        return "chalk_heavy", 0.3, stats
    return "field", min(1.0, 0.5 + nat / 2), stats


def find_bar(crop: np.ndarray) -> tuple[float, float]:
    """Best horizontal bright bar in the crop: (width_px, confidence 0-1)."""
    m = hue_masks(crop)
    mask = m["bright_low_sat"].astype(np.uint8) * 255
    if mask.sum() == 0:
        return 0.0, 0.0
    # Close along x only: the rubber reads as a wide, thin, near-horizontal run,
    # and closing horizontally joins a bar broken up by the pitcher's foot.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_w, best_c = 0.0, 0.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 2 or w < BAR_W_RANGE[0] or w > BAR_W_RANGE[1]:
            continue
        aspect = w / max(1, h)
        if aspect < 2.0:
            continue
        fill = area / max(1, w * h)
        cy = cents[i][1] / crop.shape[0]
        # Prefer wide, thin, well-filled runs sitting in the lower half of the
        # crop, which is where the rubber falls once the crop is centred on the
        # pitcher's feet.
        conf = (min(1.0, aspect / 8) * 0.4 + fill * 0.3
                + (0.3 if 0.25 < cy < 0.95 else 0.0))
        if conf > best_c:
            best_w, best_c = float(w), float(conf)
    return best_w, best_c


def crop_at(img: np.ndarray, cx: int, cy: int) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    x0 = max(0, min(w - CROP_W, cx - CROP_W // 2))
    y0 = max(0, min(h - CROP_H, cy - CROP_H // 2))
    c = img[y0:y0 + CROP_H, x0:x0 + CROP_W]
    if c.shape[:2] != (CROP_H, CROP_W):
        c = cv2.copyMakeBorder(c, 0, CROP_H - c.shape[0], 0, CROP_W - c.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return c, x0, y0


def score_crop(crop: np.ndarray) -> tuple[float, dict]:
    m = hue_masks(crop)
    dirt = float(m["dirt"].mean())
    grass = float(m["grass"].mean())
    white = float(m["white"].mean())
    bar_w, bar_conf = find_bar(crop)
    d = {"c_dirt": dirt, "c_grass": grass, "c_white": white,
         "bar_w_px": bar_w, "bar_conf": bar_conf}
    if dirt < DIRT_MIN:
        return 0.10 + 0.3 * dirt / DIRT_MIN, d
    if white > WHITE_MAX:
        return 0.25, d
    # A good mound crop is dirt-dominated with a little grass at the top edge
    # (the view past the mound) and a credible bar somewhere in it.
    dirt_term = min(1.0, dirt / 0.45)
    edge_term = 1.0 - abs(min(grass, 0.45) - 0.18) / 0.45
    return float(0.40 * dirt_term + 0.25 * edge_term + 0.35 * bar_conf), d


def dirt_center(img: np.ndarray,
                motion_cx: int | None = None,
                motion_cy: int | None = None) -> tuple[int, int] | None:
    """Crop centre from the mound itself, for when the motion prior is wrong.

    Picking the largest dirt blob is not enough: the infield arc, the first-base
    cutout and above all the HOME PLATE circle are also large dirt regions.
    Home plate is the dangerous one. In the centre-field shot the camera looks
    over the pitcher toward the plate, so the plate circle appears ABOVE the
    mound in the frame -- around 0.40-0.55 of frame height against the mound's
    0.70-0.85 -- while matching the mound on every shape test: compact, wider
    than tall, similar area. An earlier version of this function only excluded
    the top 35% of the frame, which does not touch the plate circle, and it
    duly centred 23 of 66 cells on home plate. Those crops show a batter and a
    catcher and no rubber at all, yet still scored well, because plate dirt and
    the plate itself satisfy the dirt-fraction and white-bar tests.

    The reliable discriminator is vertical position, not shape. The mound is
    always the lower of the two, and when a motion prior exists the mound must
    also sit at the pitcher's feet rather than a third of a frame above them.
    """
    m = hue_masks(img)
    mask = m["dirt"].astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    h, w = img.shape[:2]
    best, best_area = None, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.01 * h * w:
            continue
        if not (0.15 * w <= bw <= 0.65 * w):
            continue
        aspect = bw / max(1, bh)
        if not (1.2 <= aspect <= 5.0):
            continue
        # Lower portion of the frame only: this is what separates the mound from
        # the home-plate circle, which passes every shape test above.
        if cents[i][1] < MOUND_MIN_CY_FRAC * h:
            continue
        if motion_cx is not None and abs(cents[i][0] - motion_cx) > 0.25 * w:
            continue
        # The mound is the dirt the pitcher is standing on, so it must be level
        # with his feet. Without this a park whose mound is cropped out still
        # matches the plate circle and silently returns the wrong centre.
        if motion_cy is not None and abs(cents[i][1] - motion_cy) > 0.18 * h:
            continue
        if area > best_area:
            best_area, best = area, (x, y, bh, bw)
    if best is None:
        return None
    x, y, bh, bw = best
    # Feet sit near the top of the mound crown, so bias upward from centroid.
    return int(x + bw / 2), int(y + bh * 0.35)


def load_priors(season: int) -> dict[tuple[str, str, str], list[float]]:
    priors: dict[tuple[str, str, str], list[float]] = {}
    log_csv = RUBBER_DIR / f"frame_fetch_log_{season}.csv"
    if not log_csv.exists():
        return priors
    with log_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            try:
                box = [float(row[f"pitcher_{k}"]) for k in ("x0", "y0", "x1", "y1")]
            except (KeyError, TypeError, ValueError):
                continue
            if any(math.isnan(v) for v in box):
                continue
            priors[(row["game_pk"], row["pitcher"], row["stand"])] = box
    return priors


FIELDS = ["cell_id", "cand", "frame_file", "frame_class", "frame_score",
          "crop_score", "crop_cx", "crop_cy", "crop_source",
          "natural_frac", "dirt_frac", "grass_frac", "white_frac",
          "bar_w_px", "bar_conf", "score"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--todo", default=None)
    args = ap.parse_args()

    frames_dir = RUBBER_DIR / "frames" / str(args.season)
    pack = RUBBER_DIR / "label_pack"
    todo = Path(args.todo) if args.todo else pack / "labels_todo.csv"
    with todo.open(newline="") as fh:
        cells = list(csv.DictReader(fh))
    priors = load_priors(args.season)

    rows = []
    for c in cells:
        cid = c["cell_id"]
        key = (c["game_pk"], c["pitcher"], c["stand"])
        for k, fp in enumerate(sorted(frames_dir.glob(f"{cid}_c*.jpg"))):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            fclass, fscore, fst = classify_frame(img)

            cands = []
            motion_cx = motion_cy = None
            if key in priors:
                x0f, y0f, x1f, y1f = priors[key]
                motion_cx = int((x0f + x1f) / 2 * w)
                motion_cy = int(y1f * h)
                cands.append(("motion", motion_cx, motion_cy))
            dc = dirt_center(img, motion_cx, motion_cy)
            if dc is not None:
                cands.append(("dirt", dc[0], dc[1]))
                # The pitcher's horizontal position is the one thing the motion
                # blob gets right even when its vertical extent is wrong, which
                # is common from the stretch where the legs barely move before
                # the pitch. Pairing motion-x with mound-y is often better than
                # either estimate alone.
                if motion_cx is not None:
                    cands.append(("motion_x_mound_y", motion_cx, dc[1]))
            if not cands:
                cands.append(("frame_center", w // 2, int(h * 0.72)))

            best = None
            for src, cx, cy in cands:
                crop, _, _ = crop_at(img, cx, cy)
                cs, cd = score_crop(crop)
                if best is None or cs > best[0]:
                    best = (cs, src, cx, cy, cd)
            cs, src, cx, cy, cd = best

            # A bad frame caps the cell no matter how the crop scores, so the
            # two combine multiplicatively rather than additively.
            overall = fscore * (0.35 + 0.65 * cs) if fclass == "field" else fscore * cs

            rows.append({
                "cell_id": cid, "cand": k, "frame_file": fp.name,
                "frame_class": fclass, "frame_score": round(fscore, 4),
                "crop_score": round(cs, 4), "crop_cx": cx, "crop_cy": cy,
                "crop_source": src,
                "natural_frac": round(fst["natural_frac"], 4),
                "dirt_frac": round(cd["c_dirt"], 4),
                "grass_frac": round(cd["c_grass"], 4),
                "white_frac": round(cd["c_white"], 4),
                "bar_w_px": round(cd["bar_w_px"], 1),
                "bar_conf": round(cd["bar_conf"], 4),
                "score": round(overall, 4),
            })

    out = RUBBER_DIR / f"frame_scores_{args.season}.csv"
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)

    print(f"scored {len(rows)} candidate frames across {len(cells)} cells")
    print(f"-> {out}\n")

    cls: dict[str, int] = {}
    for r in rows:
        cls[r["frame_class"]] = cls.get(r["frame_class"], 0) + 1
    print("frame class distribution:")
    for k, v in sorted(cls.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {100*v/len(rows):5.1f}%  {k}")

    per_cell: dict[str, float] = {}
    per_src: dict[str, int] = {}
    for r in rows:
        if r["score"] > per_cell.get(r["cell_id"], -1):
            per_cell[r["cell_id"]] = r["score"]
    for r in rows:
        if per_cell.get(r["cell_id"]) == r["score"]:
            per_src[r["crop_source"]] = per_src.get(r["crop_source"], 0) + 1

    print("\nbest-candidate crop centre came from:")
    for k, v in sorted(per_src.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {k}")

    print("\ncells by best available candidate score:")
    for lo, hi, lab in [(0.6, 1.01, "strong"), (0.45, 0.6, "usable"),
                        (0.3, 0.45, "marginal"), (-0.01, 0.3, "hopeless")]:
        n = sum(1 for v in per_cell.values() if lo <= v < hi)
        print(f"  {n:5d}  {100*n/max(1,len(per_cell)):5.1f}%  {lab} "
              f"(score {max(lo,0):.2f}-{min(hi,1):.2f})")


if __name__ == "__main__":
    main()

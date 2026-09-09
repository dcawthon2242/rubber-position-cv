#!/usr/bin/env python3
"""Measure where on the rubber the pitcher sets up, in inches from its center.

The pitching rubber is a known 24-inch object that appears in the same frame as
the foot we are measuring, so it supplies both the origin and the pixel scale.
That is what makes this measurement portable: park, camera pan and zoom all
cancel out, because everything is expressed relative to the rubber itself.

Everything keys off the median-background image that rubber_03 writes next to
each candidate frame. Colour thresholding cannot separate a pitcher from mound
dirt -- it floods three-quarters of the search box -- but differencing a frame
against its own background isolates him cleanly. The background is equally
valuable for the rubber: the pitcher is absent from it, so the bar appears at
its full 24-inch extent rather than truncated by the shoe standing on it.

Pipeline per frame:

  1. Find the pitcher      threshold |frame - background|, then take the
                           largest blob inside rubber_03's motion bbox.
  2. Find the pivot foot   the bottom of that blob, split into left/right runs.
                           A pitcher faces away from the center-field camera, so
                           his throwing-side foot is on the image side matching
                           his handedness.
  3. Find the rubber       on the *background*, a wide thin bright
                           near-horizontal bar. Isolated with a morphological
                           top-hat using a tall kernel, which deletes thin
                           horizontal structures and leaves the rubber standing
                           out. Mound sponsor logos are bright too, so
                           candidates are scored on proximity to the foot, which
                           the rubber must touch.
  4. Convert               rubber_x_in = -(foot_x - rubber_cx) / px_per_inch.

The negation matters. In the center-field view image-left is the first-base
side: a left-handed batter, who stands on the first-base side of the plate,
appears in the left-hand batter's box. Statcast's x axis is positive toward
first base (LHP mean release_pos_x +2.07, RHP -1.87), so field x runs opposite
to image x.

Scale robustness: the foot partially occludes the rubber, which can truncate
the detected bar and corrupt the per-frame scale. Camera zoom for the pitch
shot is nearly fixed within a park, so a second pass replaces any per-frame
width that disagrees with the park's median by more than a tolerance, and
records which scale was used.

Usage:
    python pipeline/rubber_04_measure.py --season 2025
    python pipeline/rubber_04_measure.py --season 2025 --debug-dir data/rubber/debug
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBBER_DIR = REPO_ROOT / "data" / "rubber"

RUBBER_WIDTH_IN = 24.0
# Rulebook rubber is 24 x 6 inches; a spikes-width allowance keeps the foot
# from being reported as hanging off when it is legally on the edge.
FOOT_WIDTH_IN = 4.0

# Fallback search windows, used only when rubber_03 did not record where the
# pitcher was. Parks vary enough that these are a last resort, not the norm.
MOUND_ROI = (0.18, 0.45, 0.82, 0.98)  # x0, y0, x1, y1
PITCHER_ROI = (0.28, 0.28, 0.72, 0.90)

# How far to grow rubber_03's motion bbox when searching. The motion box spans
# the whole delivery, so the stance is inside it, and the pivot foot sits near
# its bottom edge because the stride carries the other foot up-frame.
PITCHER_PAD_X = 0.06
PITCHER_PAD_Y_TOP = 0.04
PITCHER_PAD_Y_BOTTOM = 0.05
RUBBER_BAND_ABOVE = 0.14
RUBBER_BAND_BELOW = 0.07

# Bar geometry limits in pixels at 1280x720. The reference park measures ~98 px
# for the rubber, and zoom varies by maybe a third either way across parks.
BAR_WIDTH_PX = (45, 210)
BAR_HEIGHT_PX = (2, 24)
BAR_MIN_ASPECT = 4.0
BAR_MAX_ANGLE_DEG = 12.0

# Vertical kernel height for the top-hat. Must exceed the rubber's pixel height
# so the opening erases it.
TOPHAT_KERNEL_H = 21
TOPHAT_THRESH = 14

# Grayscale delta from the median background that counts as the pitcher.
FG_THRESH = 30

# The pivot foot must be in contact with the rubber; these bound "in contact".
MAX_CONTACT_DY_PX = 34
MIN_FOOT_RUBBER_OVERLAP = -18.0  # px of slack outside the bar span

# Per-frame rubber width must be within this fraction of the park median.
PARK_WIDTH_TOL = 0.20
MIN_PARK_FRAMES = 8


@dataclass
class FrameMeasurement:
    game_pk: int
    pitcher: int
    stand: str
    park: str
    p_throws: str
    frame_file: str
    ok: bool = False
    reason: str = ""
    rubber_cx_px: float = float("nan")
    rubber_cy_px: float = float("nan")
    rubber_width_px: float = float("nan")
    rubber_angle_deg: float = float("nan")
    rubber_fill: float = float("nan")
    foot_x_px: float = float("nan")
    foot_y_px: float = float("nan")
    foot_width_px: float = float("nan")
    n_foot_runs: int = 0
    contact_dy_px: float = float("nan")
    px_per_inch: float = float("nan")
    rubber_x_in: float = float("nan")
    scale_source: str = ""
    pitcher_blob_area: int = 0
    has_prior: bool = False


def search_box(shape: tuple[int, int], prior: dict | None) -> tuple[int, int, int, int]:
    """Pixel search window for the pitcher, from rubber_03's motion bbox."""
    h, w = shape
    if prior is None:
        return (int(PITCHER_ROI[0] * w), int(PITCHER_ROI[1] * h),
                int(PITCHER_ROI[2] * w), int(PITCHER_ROI[3] * h))
    x0 = int(max(0.0, prior["x0"] - PITCHER_PAD_X) * w)
    x1 = int(min(1.0, prior["x1"] + PITCHER_PAD_X) * w)
    y0 = int(max(0.0, prior["y0"] - PITCHER_PAD_Y_TOP) * h)
    y1 = int(min(1.0, prior["y1"] + PITCHER_PAD_Y_BOTTOM) * h)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return (int(PITCHER_ROI[0] * w), int(PITCHER_ROI[1] * h),
                int(PITCHER_ROI[2] * w), int(PITCHER_ROI[3] * h))
    return x0, y0, x1, y1


def find_pitcher(bgr: np.ndarray, bg: np.ndarray, prior: dict | None
                 ) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    """Largest foreground blob inside the search box, by background difference."""
    h, w = bgr.shape[:2]
    diff = cv2.absdiff(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    )
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, player = cv2.threshold(diff, FG_THRESH, 255, cv2.THRESH_BINARY)

    roi = np.zeros_like(player)
    x0, y0, x1, y1 = search_box((h, w), prior)
    roi[y0:y1, x0:x1] = 255
    player = cv2.bitwise_and(player, roi)

    player = cv2.morphologyEx(player, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Close vertically to bridge a uniform that matches the background in places.
    player = cv2.morphologyEx(player, cv2.MORPH_CLOSE, np.ones((11, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(player, connectivity=8)
    best_idx, best_score = -1, 0.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 200 or bh < 25:
            continue
        # A standing pitcher is taller than wide, though the glove and arms
        # widen him, so this is a loose check.
        if bh < 0.9 * bw:
            continue
        score = float(area)
        if score > best_score:
            best_idx, best_score = i, score

    if best_idx < 0:
        return None, (0, 0, 0, 0)
    mask = (labels == best_idx).astype(np.uint8) * 255
    x, y, bw, bh, _ = stats[best_idx]
    return mask, (int(x), int(y), int(bw), int(bh))


def find_foot(pitcher_mask: np.ndarray, bbox: tuple[int, int, int, int],
              p_throws: str) -> tuple[float, float, float, int]:
    """Locate the pivot foot along the bottom of the pitcher blob.

    Returns (foot_x, foot_y, foot_width, n_runs). The pitcher faces away from
    the camera, so his throwing-side foot sits on the image side that matches
    his throwing hand: a righty's right foot appears to image-right.
    """
    x, y, bw, bh = bbox
    band_h = max(4, int(0.12 * bh))
    band = pitcher_mask[y + bh - band_h : y + bh, :]
    if band.size == 0:
        return float("nan"), float("nan"), float("nan"), 0

    cols = (band > 0).sum(axis=0)
    active = cols > 0
    if not active.any():
        return float("nan"), float("nan"), float("nan"), 0

    # Contiguous runs of occupied columns are candidate feet.
    runs: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))

    # Merge runs separated by a couple of pixels (JPEG noise splits a shoe).
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= 3:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    runs = [(a, b) for a, b in merged if b - a >= 6]
    if not runs:
        return float("nan"), float("nan"), float("nan"), 0

    if len(runs) == 1:
        pick = runs[0]
    elif p_throws == "R":
        pick = max(runs, key=lambda r: r[1])  # rightmost
    else:
        pick = min(runs, key=lambda r: r[0])  # leftmost

    foot_x = (pick[0] + pick[1]) / 2.0
    foot_width = float(pick[1] - pick[0] + 1)
    foot_y = float(y + bh)
    return foot_x, foot_y, foot_width, len(runs)


def find_rubber(bgr: np.ndarray, foot_x: float, foot_y: float,
                prior: dict | None) -> tuple[dict | None, str]:
    """Detect the rubber as a wide thin bright bar near the pitcher's feet."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lum = lab[..., 0]

    # Opening with a tall, 1-px-wide kernel removes thin horizontal features,
    # so the difference isolates them.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, TOPHAT_KERNEL_H))
    opened = cv2.morphologyEx(lum, cv2.MORPH_OPEN, kernel)
    tophat = cv2.subtract(lum, opened)

    roi = np.zeros((h, w), np.uint8)
    if prior is not None:
        # Band around the detected foot, which is where the rubber must be.
        x0 = int(max(0.0, prior["x0"] - 0.10) * w)
        x1 = int(min(1.0, prior["x1"] + 0.10) * w)
        y0 = int(max(0, foot_y - RUBBER_BAND_ABOVE * h))
        y1 = int(min(h, foot_y + RUBBER_BAND_BELOW * h))
    else:
        x0, y0 = int(MOUND_ROI[0] * w), int(MOUND_ROI[1] * h)
        x1, y1 = int(MOUND_ROI[2] * w), int(MOUND_ROI[3] * h)
    if x1 - x0 < 40 or y1 - y0 < 10:
        return None, "degenerate rubber search band"
    roi[y0:y1, x0:x1] = 255
    tophat = cv2.bitwise_and(tophat, roi)

    _, binary = cv2.threshold(tophat, TOPHAT_THRESH, 255, cv2.THRESH_BINARY)
    # Bridge the gap the shoe punches in the middle of the bar.
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)),
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no bright bars found"

    best, best_score = None, -1e9
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        # Normalise so rw is the long side and angle is off-horizontal.
        if rw < rh:
            rw, rh = rh, rw
            angle += 90.0
        angle = ((angle + 90.0) % 180.0) - 90.0

        if not (BAR_WIDTH_PX[0] <= rw <= BAR_WIDTH_PX[1]):
            continue
        if not (BAR_HEIGHT_PX[0] <= rh <= BAR_HEIGHT_PX[1]):
            continue
        if rh <= 0 or rw / max(rh, 1e-6) < BAR_MIN_ASPECT:
            continue
        if abs(angle) > BAR_MAX_ANGLE_DEG:
            continue

        fill = cv2.contourArea(cnt) / max(rw * rh, 1e-6)

        # The rubber must be at the pitcher's feet. Sponsor logos are lower on
        # the mound's camera-facing slope and further from the foot, so
        # penalising distance separates them reliably.
        dy = abs(cy - foot_y)
        dx = abs(cx - foot_x)
        if dy > 3 * MAX_CONTACT_DY_PX:
            continue
        score = -(dy * 2.0) - dx * 0.5 + rw * 0.35 + fill * 20.0
        if score > best_score:
            best_score = score
            best = {
                "cx": float(cx), "cy": float(cy), "width": float(rw),
                "height": float(rh), "angle": float(angle), "fill": float(fill),
            }

    if best is None:
        return None, "no bar matched rubber geometry"
    return best, ""


def measure_frame(path: Path, bg_path: Path, meta: dict,
                  prior: dict | None) -> FrameMeasurement:
    m = FrameMeasurement(
        game_pk=int(meta["game_pk"]), pitcher=int(meta["pitcher"]),
        stand=meta["stand"], park=meta.get("park", ""),
        p_throws=meta.get("p_throws", ""), frame_file=path.name,
        has_prior=prior is not None,
    )
    bgr = cv2.imread(str(path))
    if bgr is None:
        m.reason = "unreadable frame"
        return m
    if not bg_path.exists():
        m.reason = "missing background"
        return m
    bg = cv2.imread(str(bg_path))
    if bg is None:
        m.reason = "unreadable background"
        return m
    if bg.shape != bgr.shape:
        m.reason = "background size mismatch"
        return m

    pitcher_mask, bbox = find_pitcher(bgr, bg, prior)
    if pitcher_mask is None:
        m.reason = "pitcher not found"
        return m
    m.pitcher_blob_area = int((pitcher_mask > 0).sum())

    foot_x, foot_y, foot_w, n_runs = find_foot(pitcher_mask, bbox,
                                               m.p_throws or "R")
    if math.isnan(foot_x):
        m.reason = "foot not found"
        return m
    m.foot_x_px, m.foot_y_px, m.foot_width_px, m.n_foot_runs = (
        foot_x, foot_y, foot_w, n_runs
    )

    # Detect on the background, where the shoe is not covering the bar.
    bar, why = find_rubber(bg, foot_x, foot_y, prior)
    if bar is None:
        m.reason = why
        return m

    m.rubber_cx_px = bar["cx"]
    m.rubber_cy_px = bar["cy"]
    m.rubber_width_px = bar["width"]
    m.rubber_angle_deg = bar["angle"]
    m.rubber_fill = bar["fill"]

    half = bar["width"] / 2.0
    left, right = bar["cx"] - half, bar["cx"] + half
    m.contact_dy_px = abs(bar["cy"] - foot_y)
    if m.contact_dy_px > MAX_CONTACT_DY_PX:
        m.reason = f"foot not on rubber (dy={m.contact_dy_px:.0f}px)"
        return m
    if foot_x < left + MIN_FOOT_RUBBER_OVERLAP or foot_x > right - MIN_FOOT_RUBBER_OVERLAP:
        m.reason = "foot outside rubber span"
        return m

    m.ok = True
    return m


def annotate(path: Path, m: FrameMeasurement, out_dir: Path) -> None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return
    if not math.isnan(m.rubber_cx_px):
        half = m.rubber_width_px / 2.0
        y = int(m.rubber_cy_px)
        cv2.line(bgr, (int(m.rubber_cx_px - half), y),
                 (int(m.rubber_cx_px + half), y), (0, 255, 255), 2)
        cv2.drawMarker(bgr, (int(m.rubber_cx_px), y), (0, 255, 255),
                       cv2.MARKER_TRIANGLE_UP, 12, 2)
    if not math.isnan(m.foot_x_px):
        cv2.drawMarker(bgr, (int(m.foot_x_px), int(m.foot_y_px)), (0, 0, 255),
                       cv2.MARKER_CROSS, 16, 2)
    label = (
        f"{m.rubber_x_in:+.1f} in" if m.ok and not math.isnan(m.rubber_x_in)
        else (m.reason or "fail")
    )
    cv2.putText(bgr, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Crop tightly around the rubber so the detail is visible.
    cx = int(m.rubber_cx_px) if not math.isnan(m.rubber_cx_px) else bgr.shape[1] // 2
    cy = int(m.rubber_cy_px) if not math.isnan(m.rubber_cy_px) else int(0.75 * bgr.shape[0])
    x0 = max(0, cx - 170); x1 = min(bgr.shape[1], cx + 170)
    y0 = max(0, cy - 110); y1 = min(bgr.shape[0], cy + 60)
    crop = bgr[y0:y1, x0:x1]
    if crop.size:
        crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                          interpolation=cv2.INTER_NEAREST)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), crop)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else (
        RUBBER_DIR / f"clip_manifest_{args.season}.csv"
    )
    frames_dir = RUBBER_DIR / "frames" / str(args.season)
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}")

    with manifest_path.open(newline="") as fh:
        manifest = list(csv.DictReader(fh))
    meta_by_cell = {
        (r["game_pk"], r["pitcher"], r["stand"]): r for r in manifest
    }

    # rubber_03 recorded where the pitcher was in each clip; that prior is what
    # lets this stage work across parks with very different framing.
    priors: dict[tuple[str, str, str], dict] = {}
    log_csv = RUBBER_DIR / f"frame_fetch_log_{args.season}.csv"
    if log_csv.exists():
        with log_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    box = {k: float(row[f"pitcher_{k}"]) for k in
                           ("x0", "y0", "x1", "y1")}
                except (KeyError, TypeError, ValueError):
                    continue
                if any(math.isnan(v) for v in box.values()):
                    continue
                priors[(row["game_pk"], row["pitcher"], row["stand"])] = box
    print(f"loaded {len(priors)} pitcher-position priors")

    frames = sorted(frames_dir.glob("*_c*.jpg"))
    if args.limit:
        frames = frames[: args.limit]
    print(f"measuring {len(frames)} candidate frames")

    measurements: list[FrameMeasurement] = []
    kept_frames: list[Path] = []
    for path in frames:
        # Filename is <game_pk>_<pitcher>_<stand>_c<NN>.jpg
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        key = (parts[0], parts[1], parts[2])
        meta = meta_by_cell.get(key)
        if meta is None:
            continue
        bg_path = frames_dir / f"{parts[0]}_{parts[1]}_{parts[2]}_bg.jpg"
        measurements.append(measure_frame(path, bg_path, meta, priors.get(key)))
        kept_frames.append(path)

    # --- Pass 2: park-level scale ------------------------------------------
    widths_by_park: dict[str, list[float]] = {}
    for m in measurements:
        if m.ok and not math.isnan(m.rubber_width_px):
            widths_by_park.setdefault(m.park, []).append(m.rubber_width_px)
    park_median = {
        park: median(vals)
        for park, vals in widths_by_park.items()
        if len(vals) >= MIN_PARK_FRAMES
    }
    global_median = median([w for vals in widths_by_park.values() for w in vals]) \
        if widths_by_park else float("nan")

    for m in measurements:
        if not m.ok:
            continue
        ref = park_median.get(m.park, global_median)
        if math.isnan(ref) or ref <= 0:
            m.ok = False
            m.reason = "no scale reference"
            continue
        if abs(m.rubber_width_px - ref) / ref <= PARK_WIDTH_TOL:
            width_used = m.rubber_width_px
            m.scale_source = "frame"
        else:
            # Occlusion truncated the bar; fall back to the park's typical
            # width, which is stable because zoom barely moves for this shot.
            width_used = ref
            m.scale_source = "park_median"
        m.px_per_inch = width_used / RUBBER_WIDTH_IN
        # Negated: image-left is the first-base side, field x is positive
        # toward first base.
        m.rubber_x_in = -(m.foot_x_px - m.rubber_cx_px) / m.px_per_inch

    out_csv = RUBBER_DIR / f"rubber_measurements_{args.season}.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FrameMeasurement.__dataclass_fields__))
        writer.writeheader()
        for m in measurements:
            writer.writerow(asdict(m))

    n_ok = sum(1 for m in measurements if m.ok)
    print(f"ok {n_ok}/{len(measurements)} frames ({100*n_ok/max(len(measurements),1):.1f}%)")
    reasons: dict[str, int] = {}
    for m in measurements:
        if not m.ok:
            reasons[m.reason] = reasons.get(m.reason, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {count:5d}  {reason}")
    print(f"park scale references: {len(park_median)} parks, "
          f"global median width {global_median:.1f} px")
    print(f"wrote {out_csv}")

    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        for m, path in zip(measurements, kept_frames):
            annotate(path, m, debug_dir)
        print(f"annotated crops -> {debug_dir}")


if __name__ == "__main__":
    main()

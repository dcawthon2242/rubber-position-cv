#!/usr/bin/env python3
"""Measure rubber position on every fetched frame without new per-pitcher labels.

The classical bar detector in rubber_04_measure.py clears ~5% of frames: the
rubber is often dirt-buried, and a wide search latches onto mound logos.
Foot finding on the same frames already works ~94% of the time.

The rubber's location in the center-field camera is mostly a park property.
Existing hand labels (220 visible rubbers across 20 parks) pin that location
to about 2 inches at a typical park. This script:

  1. Builds a per-park rubber prior (center x, width) from those labels.
  2. Finds the pivot foot on each frame (same background-difference as 04).
  3. Looks for a bar in a tight window around the prior. If one is found,
     that frame gets a real measurement and the game can reuse it.
  4. If no bar is found and the park's labels agree (center SD under ~3.5 in),
     falls back to the park prior. Unstable parks (BAL camera reframe) fail
     closed rather than invent a number.

Output matches rubber_measurements_<season>.csv so rubber_05 can ingest it
as CV truth. Labels still win on any cell that has one.

Usage:
    python pipeline/rubber_04g_cv_measure.py --eval
    python pipeline/rubber_04g_cv_measure.py --season 2025
    python pipeline/rubber_04g_cv_measure.py --season 2026
    python pipeline/rubber_04g_cv_measure.py --season 2025 --debug-dir data/rubber/debug_cv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBBER_DIR = REPO_ROOT / "data" / "rubber"
RUBBER_WIDTH_IN = 24.0

# Park prior is usable as a fallback only when labels agree on where the bar is.
MIN_LABELS_FOR_PRIOR = 3
MAX_PRIOR_CX_SD_IN = 3.5
# Constrained search window around the prior, in full-frame pixels.
PRIOR_CX_WINDOW_PX = 90
PRIOR_WIDTH_TOL = 0.30
MIN_PARK_FRAMES = 8
PARK_WIDTH_TOL = 0.20

# Load the foot / pitcher / annotate helpers from 04 without making that file
# a package. Keep one implementation of the geometry.
_SPEC = importlib.util.spec_from_file_location(
    "rubber_04_measure", Path(__file__).resolve().parent / "rubber_04_measure.py"
)
_M04 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["rubber_04_measure"] = _M04
_SPEC.loader.exec_module(_M04)
find_pitcher = _M04.find_pitcher
find_foot = _M04.find_foot
find_rubber = _M04.find_rubber
annotate = _M04.annotate
FrameMeasurement = _M04.FrameMeasurement


def load_labels() -> list[dict]:
    """Visible rubber labels, mapped to full-frame x (crop is 1:1 with frame)."""
    rows: list[dict] = []
    for done in sorted(RUBBER_DIR.glob("label_pack*/labels_done.csv")):
        with done.open(newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    if int(float(r.get("rubber_visible") or 0)) != 1:
                        continue
                    left = float(r["rubber_left_px"])
                    right = float(r["rubber_right_px"])
                    foot = float(r["foot_center_px"])
                    x0 = float(r["crop_x0"])
                    if right <= left:
                        continue
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append({
                    "pack": done.parent.name,
                    "cell_id": r.get("cell_id", ""),
                    "game_pk": r.get("game_pk", ""),
                    "pitcher": r.get("pitcher", ""),
                    "stand": r.get("stand", ""),
                    "park": r.get("park", ""),
                    "p_throws": r.get("p_throws", ""),
                    "frame_file": r.get("frame_file", ""),
                    "rubber_left_px": x0 + left,
                    "rubber_right_px": x0 + right,
                    "foot_center_px": x0 + foot,
                    "rubber_cx_px": x0 + (left + right) / 2.0,
                    "rubber_width_px": right - left,
                    "rubber_x_in": -(foot - (left + right) / 2.0) / ((right - left) / 24.0),
                })
    return rows


def build_park_priors(labels: list[dict], exclude: set[tuple[str, str]] | None = None
                      ) -> dict[str, dict]:
    """Median rubber center/width per park. exclude is (game_pk, pitcher) pairs."""
    by_park: dict[str, list[dict]] = defaultdict(list)
    for r in labels:
        if exclude and (r["game_pk"], r["pitcher"]) in exclude:
            continue
        if r["park"]:
            by_park[r["park"]].append(r)
    priors: dict[str, dict] = {}
    for park, rs in by_park.items():
        widths = [x["rubber_width_px"] for x in rs]
        cxs = [x["rubber_cx_px"] for x in rs]
        w_med = median(widths)
        cx_med = median(cxs)
        cx_sd = (float(np.std(cxs, ddof=1)) if len(cxs) > 1 else 0.0)
        ppi = w_med / RUBBER_WIDTH_IN
        priors[park] = {
            "n": len(rs),
            "cx": cx_med,
            "width": w_med,
            "cx_sd_px": cx_sd,
            "cx_sd_in": cx_sd / ppi if ppi else float("inf"),
            "stable": (len(rs) >= MIN_LABELS_FOR_PRIOR
                       and (cx_sd / ppi) <= MAX_PRIOR_CX_SD_IN),
        }
    return priors


def find_rubber_constrained(bg: np.ndarray, foot_x: float, foot_y: float,
                            prior_box: dict | None, park_prior: dict | None
                            ) -> tuple[dict | None, str]:
    """Same bar finder as 04, then keep only candidates near the park prior."""
    bar, why = find_rubber(bg, foot_x, foot_y, prior_box)
    if bar is None or park_prior is None:
        return bar, why
    if abs(bar["cx"] - park_prior["cx"]) > PRIOR_CX_WINDOW_PX:
        return None, "bar far from park rubber prior"
    if abs(bar["width"] - park_prior["width"]) / park_prior["width"] > PRIOR_WIDTH_TOL:
        return None, "bar width disagrees with park prior"
    return bar, ""


def load_meta(season: int) -> tuple[dict, dict, dict]:
    """cell key -> clip meta, cell key -> motion bbox, game_pk -> park."""
    meta_by_cell: dict[tuple[str, str, str], dict] = {}
    for path in [
        RUBBER_DIR / f"clip_manifest_{season}.csv",
        RUBBER_DIR / f"fetch_targets_{season}.csv",
    ]:
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if not r.get("game_pk"):
                    continue
                key = (r["game_pk"], r["pitcher"], r.get("stand", ""))
                meta_by_cell[key] = r

    boxes: dict[tuple[str, str, str], dict] = {}
    log_csv = RUBBER_DIR / f"frame_fetch_log_{season}.csv"
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
                boxes[(row["game_pk"], row["pitcher"], row["stand"])] = box

    park_by_game: dict[str, str] = {}
    throws_by_pitcher: dict[str, str] = {}
    pos = RUBBER_DIR / "rubber_position_pitcher_game.csv"
    if pos.exists():
        with pos.open(newline="") as fh:
            for r in csv.DictReader(fh):
                park_by_game[r["game_pk"]] = r["park"]
                throws_by_pitcher[r["pitcher"]] = r["p_throws"]
    return meta_by_cell, boxes, {"park": park_by_game, "throws": throws_by_pitcher}


def finish_inches(m: FrameMeasurement, width_used: float, scale_source: str) -> None:
    m.px_per_inch = width_used / RUBBER_WIDTH_IN
    m.rubber_x_in = -(m.foot_x_px - m.rubber_cx_px) / m.px_per_inch
    m.scale_source = scale_source
    m.ok = True
    m.reason = ""


def measure_one(path: Path, bg_path: Path, meta: dict, motion: dict | None,
                park_prior: dict | None, game_bar: dict | None
                ) -> FrameMeasurement:
    m = FrameMeasurement(
        game_pk=int(meta["game_pk"]), pitcher=int(meta["pitcher"]),
        stand=meta["stand"], park=meta.get("park", ""),
        p_throws=meta.get("p_throws", ""), frame_file=path.name,
        has_prior=motion is not None,
    )
    bgr = cv2.imread(str(path))
    if bgr is None:
        m.reason = "unreadable frame"
        return m
    if not bg_path.exists():
        m.reason = "missing background"
        return m
    bg = cv2.imread(str(bg_path))
    if bg is None or bg.shape != bgr.shape:
        m.reason = "bad background"
        return m

    pitcher_mask, bbox = find_pitcher(bgr, bg, motion)
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

    bar, why = find_rubber_constrained(bg, foot_x, foot_y, motion, park_prior)
    if bar is not None:
        m.rubber_cx_px = bar["cx"]
        m.rubber_cy_px = bar["cy"]
        m.rubber_width_px = bar["width"]
        m.rubber_angle_deg = bar["angle"]
        m.rubber_fill = bar["fill"]
        m.contact_dy_px = abs(bar["cy"] - foot_y)
        finish_inches(m, bar["width"], "constrained")
        return m

    if game_bar is not None:
        m.rubber_cx_px = game_bar["cx"]
        m.rubber_cy_px = game_bar["cy"]
        m.rubber_width_px = game_bar["width"]
        m.contact_dy_px = abs(game_bar["cy"] - foot_y)
        finish_inches(m, game_bar["width"], "game_prior")
        return m

    if park_prior is not None and park_prior["stable"]:
        m.rubber_cx_px = park_prior["cx"]
        m.rubber_cy_px = foot_y
        m.rubber_width_px = park_prior["width"]
        m.contact_dy_px = 0.0
        finish_inches(m, park_prior["width"], "park_prior")
        return m

    m.reason = why or "no rubber prior"
    return m


def iter_frames(season: int):
    frames_dir = RUBBER_DIR / "frames" / str(season)
    for path in sorted(frames_dir.glob("*_c*.jpg")):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        yield path, (parts[0], parts[1], parts[2]), frames_dir / (
            f"{parts[0]}_{parts[1]}_{parts[2]}_bg.jpg"
        )


def resolve_meta(key: tuple[str, str, str], meta_by_cell: dict,
                 lookup: dict) -> dict | None:
    meta = meta_by_cell.get(key)
    if meta:
        if not meta.get("park"):
            meta = dict(meta)
            meta["park"] = lookup["park"].get(key[0], "")
        if not meta.get("p_throws"):
            meta = dict(meta)
            meta["p_throws"] = lookup["throws"].get(key[1], "")
        return meta
    park = lookup["park"].get(key[0])
    if not park:
        return None
    return {
        "game_pk": key[0], "pitcher": key[1], "stand": key[2],
        "park": park, "p_throws": lookup["throws"].get(key[1], ""),
    }


def run_season(season: int, labels: list[dict], debug_dir: Path | None,
               limit: int | None) -> list[FrameMeasurement]:
    priors = build_park_priors(labels)
    print(f"park priors: {len(priors)} parks, "
          f"{sum(1 for p in priors.values() if p['stable'])} stable")
    for park, p in sorted(priors.items()):
        flag = "stable" if p["stable"] else "unstable"
        print(f"  {park:<4} n={p['n']:2d}  w={p['width']:.1f}  "
              f"cx={p['cx']:.0f}  sd={p['cx_sd_in']:.2f} in  {flag}")

    meta_by_cell, boxes, lookup = load_meta(season)
    measurements: list[FrameMeasurement] = []
    kept: list[Path] = []
    game_bars: dict[int, dict] = {}

    frames = list(iter_frames(season))
    if limit:
        frames = frames[:limit]
    print(f"measuring {len(frames)} frames for {season}")

    # Pass 1: detect or park-prior.
    pending_game: list[int] = []
    for path, key, bg_path in frames:
        meta = resolve_meta(key, meta_by_cell, lookup)
        if meta is None:
            continue
        park_prior = priors.get(meta.get("park", ""))
        m = measure_one(path, bg_path, meta, boxes.get(key), park_prior, None)
        measurements.append(m)
        kept.append(path)
        if m.ok and m.scale_source == "constrained":
            game_bars[m.game_pk] = {
                "cx": m.rubber_cx_px, "cy": m.rubber_cy_px,
                "width": m.rubber_width_px,
            }

    # Pass 2: frames that only have a foot, in a game where a bar was found.
    n_upgraded = 0
    for i, m in enumerate(measurements):
        if m.ok or math.isnan(m.foot_x_px):
            continue
        bar = game_bars.get(m.game_pk)
        if bar is None:
            continue
        m.rubber_cx_px = bar["cx"]
        m.rubber_cy_px = bar["cy"]
        m.rubber_width_px = bar["width"]
        m.contact_dy_px = abs(bar["cy"] - m.foot_y_px)
        finish_inches(m, bar["width"], "game_prior")
        n_upgraded += 1
    if n_upgraded:
        print(f"upgraded {n_upgraded} frames via same-game detected rubber")

    out_csv = RUBBER_DIR / f"rubber_measurements_{season}.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FrameMeasurement.__dataclass_fields__))
        writer.writeheader()
        for m in measurements:
            writer.writerow(asdict(m))

    n_ok = sum(1 for m in measurements if m.ok)
    print(f"ok {n_ok}/{len(measurements)} ({100 * n_ok / max(len(measurements), 1):.1f}%)")
    by_src: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    for m in measurements:
        if m.ok:
            by_src[m.scale_source] += 1
        else:
            reasons[m.reason] += 1
    for k, n in sorted(by_src.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {k}")
    for k, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {k}")
    print(f"wrote {out_csv}")

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for m, path in zip(measurements, kept):
            if m.ok:
                annotate(path, m, debug_dir)
        print(f"annotated ok crops -> {debug_dir}")
    return measurements


def eval_against_labels(labels: list[dict]) -> None:
    """Leave-one-pitcher-out park prior + this frame's CV foot, vs the label."""
    print("==== leave-one-pitcher-out vs hand labels ====")
    by_season: dict[int, list[dict]] = defaultdict(list)
    for r in labels:
        try:
            pk = int(r["game_pk"])
        except (TypeError, ValueError):
            continue
        year = 2000 + (pk // 10000) % 100  # 776xxx -> 2025-ish; 82xxxx -> 2026
        if pk >= 820000:
            year = 2026
        elif pk >= 770000:
            year = 2025
        else:
            year = 2024
        by_season[year].append(r)

    preds: list[tuple[float, float, str, str]] = []
    foot_err: list[float] = []
    for season, subset in sorted(by_season.items()):
        frames_dir = RUBBER_DIR / "frames" / str(season)
        if not frames_dir.exists():
            print(f"  skip {season}: no frames dir")
            continue
        meta_by_cell, boxes, lookup = load_meta(season)
        for r in subset:
            priors = build_park_priors(
                labels, exclude={(r["game_pk"], r["pitcher"])}
            )
            park_prior = priors.get(r["park"])
            frame = frames_dir / r["frame_file"]
            if not frame.exists():
                continue
            parts = Path(r["frame_file"]).stem.split("_")
            if len(parts) < 4:
                continue
            key = (parts[0], parts[1], parts[2])
            meta = resolve_meta(key, meta_by_cell, lookup) or {
                "game_pk": r["game_pk"], "pitcher": r["pitcher"],
                "stand": r["stand"], "park": r["park"],
                "p_throws": r["p_throws"],
            }
            bg = frames_dir / f"{parts[0]}_{parts[1]}_{parts[2]}_bg.jpg"
            m = measure_one(frame, bg, meta, boxes.get(key), park_prior, None)
            if not m.ok:
                continue
            preds.append((m.rubber_x_in, r["rubber_x_in"], r["park"],
                          m.scale_source or ""))
            if not math.isnan(m.foot_x_px):
                foot_err.append(abs(m.foot_x_px - r["foot_center_px"]))

    if not preds:
        print("  no overlapping labeled frames on disk")
        return
    err = np.array([a - b for a, b, _, _ in preds])
    print(f"  n={len(preds)}  MAE={np.mean(np.abs(err)):.2f} in  "
          f"RMSE={np.sqrt(np.mean(err ** 2)):.2f} in  "
          f"p90={np.quantile(np.abs(err), 0.9):.2f} in")
    if foot_err:
        ppi = 3.5
        print(f"  foot |dx| median {median(foot_err):.1f} px "
              f"(~{median(foot_err) / ppi:.2f} in at 3.5 px/in)")
    by_src: dict[str, list[float]] = defaultdict(list)
    by_park: dict[str, list[float]] = defaultdict(list)
    for pred, lab, park, src in preds:
        by_src[src].append(abs(pred - lab))
        by_park[park].append(abs(pred - lab))
    print("  by source:")
    for src, vs in sorted(by_src.items()):
        print(f"    {src:<14} n={len(vs):3d}  MAE={sum(vs)/len(vs):.2f}")
    print("  by park:")
    for park, vs in sorted(by_park.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"    {park:<4} n={len(vs):3d}  MAE={sum(vs)/len(vs):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--eval", action="store_true",
                        help="score against held-out labels and exit")
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    labels = load_labels()
    print(f"loaded {len(labels)} visible rubber labels")
    if args.eval or args.season is None:
        eval_against_labels(labels)
        if args.eval:
            return
    if args.season is None:
        print("pass --season YYYY to write measurements", file=sys.stderr)
        return
    debug = Path(args.debug_dir) if args.debug_dir else None
    run_season(args.season, labels, debug, args.limit)


if __name__ == "__main__":
    main()

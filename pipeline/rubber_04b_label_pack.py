#!/usr/bin/env python3
"""Build a hand-labeling pack so CV output can be validated against truth.

Automated rubber detection cleared only 5% of frames across 15 parks, well
short of the plan's 90% gate, because the rubber is dirt-buried at some parks
and camera framing varies enough that pixels-per-inch differs ~2.5x. Hand
labels serve three purposes: they measure the real detection ceiling per park,
they pin down foot-localisation error in inches, and they give the calibration
in rubber_05 a ground-truth anchor that does not depend on the detector.

For each cell this writes a zoomed crop centred on the pitcher's stance, with a
labelled pixel ruler along the top so coordinates can be read off by eye, plus
a row in labels_todo.csv to fill in. Crop origin and zoom are recorded per row,
so crop coordinates map back to full-frame pixels exactly.

Fill in these columns, leaving the rest alone:

    rubber_visible    1 if both ends of the rubber are identifiable, else 0
    rubber_left_px    crop x of the rubber's left end   (ruler coordinate)
    rubber_right_px   crop x of the rubber's right end
    foot_center_px    crop x of the middle of the pivot foot
    foot_on_rubber    1 if the foot is in contact with the rubber, else 0
    notes             anything odd about the frame

Only rows with rubber_visible=1 need coordinates. The rubber is 24 inches wide,
so (rubber_right_px - rubber_left_px) sets the scale for that frame and no
external calibration is needed.

Usage:
    python pipeline/rubber_04b_label_pack.py --season 2025 \
        --manifest data/rubber/label_manifest.csv
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

# Crop geometry. Wide enough to hold the whole rubber at every park's zoom
# (the widest observed bar is ~160 px) with context either side.
CROP_W, CROP_H = 440, 150
ZOOM = 3
RULER_H = 34
TICK_MINOR = 10
TICK_MAJOR = 50

LABEL_FIELDS = [
    "cell_id", "game_pk", "pitcher", "stand", "park", "p_throws",
    "frame_file", "crop_file", "crop_x0", "crop_y0", "zoom",
    "rubber_visible", "rubber_left_px", "rubber_right_px",
    "foot_center_px", "foot_on_rubber", "notes",
]


PARK_ELIGIBILITY = RUBBER_DIR / "park_eligibility.csv"
PRIORITY_CSV = RUBBER_DIR / "label_priority_pitchers.csv"


def park_legibility() -> dict[str, dict]:
    """How often a human could actually read the rubber at each park.

    The automated frame score and the rubber's legibility turn out to be nearly
    unrelated, which is not obvious and is worth stating. rubber_04d scores a
    frame on whether it is a centre-field view of mound dirt, and it is good at
    that. But a frame can be a textbook centre-field view with the rubber
    completely buried in dirt, and it will score well. Measured against the
    labels: Pittsburgh has the single highest median frame score, 0.63, and a
    human found both ends of the rubber in 1 of 11 attempts. Cincinnati scores
    0.60 and went 0 for 10. San Francisco scores 0.50 and went 8 for 10.

    So the share of labeled frames at a park where rubber_visible came back 1 is
    a different and much more useful signal than anything the scorer produces,
    and it is the one to use when choosing WHICH park to ask about a pitcher.
    Most pitchers appear at many parks over a season, so this is close to a free
    improvement in label yield.

    Two quantities per park, because they fail differently:

      vis_rate   how often the rubber is readable at all, i.e. label yield.
                 Shrunk toward the pooled rate, since a park has only 8-11
                 labels and an unshrunk 0.80 on 10 tries is not far from an
                 unshrunk 0.60.
      width_cv   how much the measured rubber WIDTH varies within the park. The
                 camera does not move, so the true width is fixed and all the
                 spread is click error. Since width sets the pixels-per-inch for
                 that label, a park at 0.20 is putting a fifth of a scale error
                 into every inch it reports, even on labels that looked fine.

    Returns {park: {"vis_rate", "width_cv", "n", "score"}}. Empty before any
    labeling has happened, in which case callers should fall back to treating
    all eligible parks alike.
    """
    # Newest round wins when a cell was labeled more than once. Rounds are not
    # interchangeable: the first pack was built before park triage and before
    # rubber_03's frame timing was fixed, so its frames often caught the pitcher
    # mid-stride or showed a broadcast graphic, and the labeler duly marked the
    # rubber invisible. Counting those against a park measures the state of the
    # pipeline in an earlier round rather than how legible the park is now, and
    # doing so understated Milwaukee at 0 for 5 when the current round has it at
    # 5 for 11. Sorted so that a bare "label_pack" precedes "label_pack_v2".
    def round_order(p: Path) -> tuple[int, str]:
        name = p.parent.name
        suffix = name.rsplit("_v", 1)[-1] if "_v" in name else ""
        return (int(suffix) if suffix.isdigit() else 0, name)

    latest: dict[tuple[str, str], tuple[int, float | None]] = {}
    for done in sorted(RUBBER_DIR.glob("label_pack*/labels_done.csv"), key=round_order):
        with done.open(newline="") as fh:
            for r in csv.DictReader(fh):
                park, cid = r.get("park", ""), r.get("cell_id", "")
                if not park:
                    continue
                try:
                    vis = int(float(r.get("rubber_visible") or 0))
                except ValueError:
                    continue
                width = None
                try:
                    width = float(r["rubber_right_px"]) - float(r["rubber_left_px"])
                    if width <= 0:
                        width = None
                except (KeyError, TypeError, ValueError):
                    pass
                latest[(park, cid)] = (vis, width)

    per_park: dict[str, list[tuple[int, float | None]]] = {}
    for (park, _), rec in latest.items():
        per_park.setdefault(park, []).append(rec)
    if not per_park:
        return {}

    tot_vis = sum(v for rows in per_park.values() for v, _ in rows)
    tot_n = sum(len(rows) for rows in per_park.values())
    p0 = tot_vis / tot_n if tot_n else 0.5
    PSEUDO = 3.0

    out: dict[str, dict] = {}
    for park, rows in per_park.items():
        n = len(rows)
        vis = sum(v for v, _ in rows)
        widths = [w for v, w in rows if v == 1 and w]
        cv = None
        if len(widths) > 2:
            mean = sum(widths) / len(widths)
            var = sum((w - mean) ** 2 for w in widths) / (len(widths) - 1)
            cv = (var ** 0.5) / mean if mean else None
        vis_rate = (vis + PSEUDO * p0) / (n + PSEUDO)
        # Scale error enters the reported offset proportionally, so a park is
        # discounted by roughly the fraction of the measurement it corrupts.
        # Unknown CV is treated as the pooled-average park rather than as
        # perfect, so a park with too few visible labels to estimate it is not
        # rewarded for the missing evidence.
        pen = cv if cv is not None else 0.09
        out[park] = {"vis_rate": vis_rate, "width_cv": cv, "n": n,
                     "score": vis_rate * (1.0 - min(pen, 0.5))}
    return out


def load_priority() -> dict[int, int]:
    """Fastball volume per still-unanchored pitcher, from rubber_05's worklist.

    Which cells are worth labeling changed once rubber_05 started anchoring per
    pitcher instead of fitting one pooled regression. A label no longer improves
    the whole model a little; it converts ONE PITCHER'S cells from the ~7.5-inch
    pooled fallback to a ~1.2-inch anchor, and does nothing for anyone else. So
    the second label on an already-anchored pitcher is worth a small fraction of
    the first label on a new one, and a pack built by park order -- which is how
    the earlier rounds were built -- spends most of its frames on cells that
    barely move the deliverable.

    Ordering by fastballs rather than by games matters for the same reason the
    worklist does: a closer appears in far more games than a starter while
    throwing a fraction of the pitches, and it is pitch rows that a Stuff+ model
    consumes.

    Returns {pitcher_id: fastballs}, empty if rubber_05 has not been run.
    """
    pri: dict[int, int] = {}
    if not PRIORITY_CSV.exists():
        return pri
    with PRIORITY_CSV.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                pri[int(r["pitcher"])] = int(r["fastballs"])
            except (KeyError, TypeError, ValueError):
                continue
    return pri


def load_park_pins() -> dict[str, dict]:
    """Per-park mound location and eligibility, from rubber_04f's triage pass.

    This is the most reliable centre available, and it outranks both the frame
    scorer and the motion prior. A broadcast centre-field camera does not move
    for the season, so the rubber occupies the same pixels in every clip from a
    park, whereas the per-cell estimates were being derived afresh from each
    frame and getting it wrong in a systematic way -- the home-plate dirt circle
    outranks the mound on every blob test, so cells were centred on the catcher.

    Returns {park: {fx, fy, status}}; callers should skip parks whose status is
    not "eligible" when building a labeling pack.
    """
    pins: dict[str, dict] = {}
    if not PARK_ELIGIBILITY.exists():
        return pins
    with PARK_ELIGIBILITY.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                fx, fy = float(r["mound_fx"]), float(r["mound_fy"])
            except (TypeError, ValueError):
                continue
            pins[r["park"]] = {"fx": fx, "fy": fy, "status": r.get("status", "")}
    return pins


def draw_ruler(canvas: np.ndarray, crop_w: int, zoom: int) -> None:
    """Ruler in *crop* coordinates, so labels are read directly off the image."""
    canvas[:RULER_H, :] = (32, 32, 32)
    for x in range(0, crop_w + 1, TICK_MINOR):
        sx = x * zoom
        if sx >= canvas.shape[1]:
            break
        major = x % TICK_MAJOR == 0
        cv2.line(canvas, (sx, RULER_H - (14 if major else 7)),
                 (sx, RULER_H - 1), (0, 255, 255) if major else (140, 140, 140),
                 2 if major else 1)
        if major:
            cv2.putText(canvas, str(x), (sx + 3, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)


def build_crop(img: np.ndarray, cx: int, cy: int) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    x0 = max(0, min(w - CROP_W, cx - CROP_W // 2))
    y0 = max(0, min(h - CROP_H, cy - CROP_H // 2))
    crop = img[y0 : y0 + CROP_H, x0 : x0 + CROP_W]
    if crop.shape[:2] != (CROP_H, CROP_W):
        crop = cv2.copyMakeBorder(
            crop, 0, CROP_H - crop.shape[0], 0, CROP_W - crop.shape[1],
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
    big = cv2.resize(crop, (CROP_W * ZOOM, CROP_H * ZOOM),
                     interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((CROP_H * ZOOM + RULER_H, CROP_W * ZOOM, 3), np.uint8)
    canvas[RULER_H:, :] = big
    draw_ruler(canvas, CROP_W, ZOOM)
    return canvas, x0, y0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--priority", action="store_true",
                        help="keep only pitchers rubber_05 lists as unanchored, "
                             "and order by fastball volume so the most valuable "
                             "labels come first")
    parser.add_argument("--one-per-pitcher", action="store_true",
                        help="at most one cell per pitcher; a second label on an "
                             "anchored pitcher buys far less than a first on a "
                             "new one")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many cells (0 = no limit)")
    parser.add_argument("--min-park-legibility", type=float, default=0.0,
                        help="drop parks scoring below this on the legibility "
                             "measure; ~0.25 removes the parks where the rubber "
                             "has never once been readable")
    parser.add_argument("--force-pitchers", default="",
                        help="comma-separated pitcher IDs that must be in the "
                             "pack if they have a frame on disk, even when they "
                             "would lose a volume or park cut")
    args = parser.parse_args()
    force_ids = {int(x) for x in args.force_pitchers.split(",") if x.strip()}

    frames_dir = RUBBER_DIR / "frames" / str(args.season)
    out_dir = Path(args.out_dir) if args.out_dir else RUBBER_DIR / "label_pack"
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.manifest).open(newline="") as fh:
        manifest = {(r["game_pk"], r["pitcher"], r["stand"]): r
                    for r in csv.DictReader(fh)}

    # Stance position comes from rubber_03's motion bbox: its bottom edge is the
    # pivot-foot end of the delivery, since the stride carries the other foot
    # up-frame toward home plate.
    priors: dict[tuple[str, str, str], list[float]] = {}
    log_csv = RUBBER_DIR / f"frame_fetch_log_{args.season}.csv"
    if log_csv.exists():
        with log_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row["status"] != "ok":
                    continue
                try:
                    box = [float(row[f"pitcher_{k}"]) for k in ("x0", "y0", "x1", "y1")]
                except (KeyError, TypeError, ValueError):
                    continue
                if any(math.isnan(v) for v in box):
                    continue
                priors[(row["game_pk"], row["pitcher"], row["stand"])] = box

    # rubber_04d, when it has been run, both ranks the candidate frames and
    # proposes a crop centre that beats the motion prior. Preferring it here
    # keeps these static crops identical to what the interactive labeler shows;
    # without it the two disagree, and a cell can score well on a crop the pack
    # never actually rendered.
    scores: dict[str, list[dict]] = {}
    score_csv = RUBBER_DIR / f"frame_scores_{args.season}.csv"
    if score_csv.exists():
        with score_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                scores.setdefault(row["cell_id"], []).append(row)
        for v in scores.values():
            v.sort(key=lambda r: -float(r["score"]))
        print(f"using frame scores for {len(scores)} cells from {score_csv.name}")

    pins = load_park_pins()
    if pins:
        n_el = sum(1 for v in pins.values() if v["status"] == "eligible")
        print(f"park pins for {len(pins)} parks, {n_el} eligible, from "
              f"{PARK_ELIGIBILITY.name}")

    # Cells already labeled in an earlier round. Re-serving them wastes the
    # labeler's attention on work that is done, and rubber_05 pools every round's
    # labels_done.csv anyway, so a repeat adds nothing.
    already: set[str] = set()
    for done in RUBBER_DIR.glob("label_pack*/labels_done.csv"):
        with done.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("cell_id"):
                    already.add(r["cell_id"])
    anchored: set[str] = {c.split("_")[1] for c in already}

    priority = load_priority() if args.priority else {}
    if args.priority:
        if not priority:
            raise SystemExit(f"--priority needs {PRIORITY_CSV}; run rubber_05 first")
        print(f"priority list: {len(priority)} unanchored pitchers "
              f"from {PRIORITY_CSV.name}")

    leg = park_legibility()
    if leg:
        ranked = sorted(leg.items(), key=lambda kv: -kv[1]["score"])
        print("park legibility (share of labeled frames where the rubber was "
              "readable, shrunk, then discounted for width disagreement):")
        for p, v in ranked:
            cv = f"{v['width_cv']:.3f}" if v["width_cv"] is not None else "  -  "
            print(f"    {p:<4} score {v['score']:.2f}  vis {v['vis_rate']:.2f}  "
                  f"width_cv {cv}  n={v['n']}")

    def order_key(kv):
        key, meta = kv
        forced = 0 if int(key[1]) in force_ids else 1
        if priority:
            # Forced IDs first so --limit cannot drop the pitchers this pack
            # was requested for. Then highest fastball volume, so an
            # interrupted session still bought the labels that upgrade the
            # most pitch rows.
            return (forced, -priority.get(int(key[1]), 0), meta["park"], key)
        return (forced, meta["park"], key)

    # With --one-per-pitcher the loop must not simply take the first cell it
    # meets for a pitcher, because that is an arbitrary park. A pitcher visits
    # many parks in a season and their legibility ranges from 8-in-10 readable
    # to 0-in-10, so choosing among his cells is close to free yield. Candidates
    # are therefore grouped by pitcher first and the most legible park wins,
    # with the frame score breaking ties within a park.
    best_cell: dict[str, tuple] = {}
    if args.one_per_pitcher:
        for key, meta in manifest.items():
            stem = f"{key[0]}_{key[1]}_{key[2]}"
            if stem in already or key[1] in anchored:
                continue
            forced = int(key[1]) in force_ids
            if priority and not forced and int(key[1]) not in priority:
                continue
            pin = pins.get(meta["park"])
            if pins and (pin is None or pin["status"] != "eligible"):
                continue
            if not any(frames_dir.glob(f"{stem}_c*.jpg")):
                continue
            pl = leg.get(meta["park"], {}).get("score", 0.0)
            if (not forced) and leg and pl < args.min_park_legibility:
                continue
            fs = float((scores.get(stem) or [{}])[0].get("score", 0) or 0)
            cand = (pl, fs, stem)
            if key[1] not in best_cell or cand > best_cell[key[1]][0]:
                best_cell[key[1]] = (cand, key)
        keep = {v[1] for v in best_cell.values()}
        manifest = {k: v for k, v in manifest.items() if k in keep}

    rows = []
    skipped = 0
    off_park = 0
    off_priority = 0
    dup_pitcher = 0
    done_already = 0
    seen_pitchers: set[str] = set()
    for key, meta in sorted(manifest.items(), key=order_key):
        stem = f"{key[0]}_{key[1]}_{key[2]}"
        if stem in already:
            done_already += 1
            continue
        if priority and int(key[1]) not in force_ids and int(key[1]) not in priority:
            off_priority += 1
            continue
        if args.one_per_pitcher and (key[1] in seen_pitchers or key[1] in anchored):
            dup_pitcher += 1
            continue
        if args.limit and len(rows) >= args.limit:
            break
        pin = pins.get(meta["park"])
        # A park judged ineligible cannot yield a labelable rubber no matter how
        # a given frame scores, so its cells are dropped before any imaging work.
        if pins and (pin is None or pin["status"] != "eligible"):
            off_park += 1
            continue
        cands = sorted(frames_dir.glob(f"{stem}_c*.jpg"))
        best = (scores.get(stem) or [None])[0]
        if not cands or (key not in priors and best is None and pin is None):
            skipped += 1
            continue
        if best is not None:
            frame_path = frames_dir / best["frame_file"]
        else:
            # Last candidate is the primary: the latest quiet frame before motion.
            frame_path = cands[-1]
        img = cv2.imread(str(frame_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        if pin is not None:
            cx, cy = int(pin["fx"] * w), int(pin["fy"] * h)
        elif best is not None:
            cx, cy = int(best["crop_cx"]), int(best["crop_cy"])
        else:
            x0f, y0f, x1f, y1f = priors[key]
            cx, cy = int((x0f + x1f) / 2 * w), int(y1f * h)
        canvas, cx0, cy0 = build_crop(img, cx, cy)
        crop_name = f"{meta['park']}_{stem}.png"
        cv2.imwrite(str(crops_dir / crop_name), canvas)
        seen_pitchers.add(key[1])

        rows.append({
            "cell_id": stem, "game_pk": key[0], "pitcher": key[1],
            "stand": key[2], "park": meta["park"],
            "p_throws": meta.get("p_throws", ""),
            "frame_file": frame_path.name, "crop_file": crop_name,
            "crop_x0": cx0, "crop_y0": cy0, "zoom": ZOOM,
            "rubber_visible": "", "rubber_left_px": "", "rubber_right_px": "",
            "foot_center_px": "", "foot_on_rubber": "", "notes": "",
        })

    todo = out_dir / "labels_todo.csv"
    with todo.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} crops -> {crops_dir}")
    print(f"label sheet -> {todo}")
    if done_already:
        print(f"skipped {done_already} cells labeled in an earlier round")
    if off_priority:
        print(f"dropped {off_priority} cells whose pitcher is already anchored")
    if dup_pitcher:
        print(f"dropped {dup_pitcher} extra cells for pitchers already in this pack")
    if off_park:
        print(f"dropped {off_park} cells at parks not marked eligible")
    if skipped:
        print(f"skipped {skipped} cells with no frame or no stance prior")
    parks: dict[str, int] = {}
    for r in rows:
        parks[r["park"]] = parks.get(r["park"], 0) + 1
    print("per-park crops:", dict(sorted(parks.items())))
    if priority and rows:
        got = sum(priority.get(int(r["pitcher"]), 0) for r in rows)
        print(f"these {len(rows)} cells anchor pitchers accounting for "
              f"{got:,} fastballs")


if __name__ == "__main__":
    main()

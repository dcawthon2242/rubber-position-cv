#!/usr/bin/env python3
"""Build a tagging pack for training the rubber keypoint model.

The earlier packs picked one cell per unanchored pitcher so a click became an
absolute rubber coordinate for that man. That is the wrong sample for a vision
model: 220 visible labels, many of them the same park and the same zoom, is why
rubber_04h holdout-validated at 6.6 inches.

This pack is stratified for learning. It spends clicks on frames the model has
not seen, spread across parks, games, handedness and camera zoom. Two filters
keep those clicks from being wasted:

  * Early innings. The mound is dressed before first pitch and the rubber's
    front edge buries as dirt kicks up, so a first-inning clip is worth more
    than a seventh-inning one at the same park. Innings 1-3 are taken first,
    then 4-5, then later only if the batch would otherwise come up short.
  * Camera angle. Some parks never show a usable rubber (ineligible in
    park_eligibility.csv, or enough labeled skips to prove it). Those parks
    are skipped and their slots are redistributed across parks that do work,
    so coverage stays high instead of padding the pack with guaranteed skips.

After you tag it, rerun:

    python pipeline/rubber_04h_keypoint.py --train

rubber_05 already pools every label_pack*/labels_done.csv, including this one.

Tagging is the existing browser labeler. Same three clicks (rubber left, rubber
right, pivot-foot centre). Skip with `s` when you cannot see both ends.

    python pipeline/rubber_04i_train_pack.py
    python pipeline/rubber_04c_label_server.py \
        --pack data/rubber/label_pack_train

A useful next batch is about 250 frames. Reliability for replacing the 8-inch
fallback wants holdout MAE under ~3 inches, which in the last run needed a few
hundred new diverse visibles on top of the 220 we have. Parks that are already
well labeled (TEX, MIA, ATH, SF) are down-weighted so new clicks go where the
model is thin.

Usage:
    python pipeline/rubber_04i_train_pack.py
    python pipeline/rubber_04i_train_pack.py --limit 250
    python pipeline/rubber_04i_train_pack.py --seasons 2025,2026
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from rubber_04b_label_pack import (
    CROP_W, CROP_H, LABEL_FIELDS, ZOOM, build_crop,
    load_park_pins, park_legibility,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBBER_DIR = REPO_ROOT / "data" / "rubber"
OUT_DIR = RUBBER_DIR / "label_pack_train"
SEED = 1


def year_of(game_pk: int) -> int:
    if game_pk >= 820000:
        return 2026
    if game_pk >= 770000:
        return 2025
    return 2024


def already_labeled() -> set[str]:
    """cell_ids that already have a verdict, visible or skipped."""
    done: set[str] = set()
    for path in RUBBER_DIR.glob("label_pack*/labels_done.csv"):
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("cell_id"):
                    done.add(r["cell_id"])
    return done


def visible_by_park() -> Counter:
    n: Counter = Counter()
    for path in RUBBER_DIR.glob("label_pack*/labels_done.csv"):
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    if int(float(r.get("rubber_visible") or 0)) == 1 and r.get("park"):
                        n[r["park"]] += 1
                except ValueError:
                    continue
    return n


def load_meta(seasons: list[int]) -> dict[tuple[str, str, str], dict]:
    """Park / handedness for a cell, from whatever tables we have."""
    meta: dict[tuple[str, str, str], dict] = {}
    for season in seasons:
        for name in (f"clip_manifest_{season}.csv", f"fetch_targets_{season}.csv"):
            path = RUBBER_DIR / name
            if not path.exists():
                continue
            with path.open(newline="") as fh:
                for r in csv.DictReader(fh):
                    if not r.get("game_pk"):
                        continue
                    key = (r["game_pk"], r["pitcher"], r.get("stand", ""))
                    inn = r.get("inning")
                    try:
                        inn_i = int(float(inn)) if inn not in (None, "") else None
                    except (TypeError, ValueError):
                        inn_i = None
                    early = str(r.get("sel_early_inning", "")).upper() in ("TRUE", "1")
                    prev = meta.get(key, {})
                    # Prefer the earlier inning when two tables disagree; that
                    # is the clip we want, and fetch_targets is loaded after
                    # the full manifest so it wins on ties.
                    if prev.get("inning") is not None and inn_i is not None:
                        if inn_i > prev["inning"]:
                            inn_i = prev["inning"]
                            early = prev.get("early") or early
                    meta[key] = {
                        "park": r.get("park", "") or prev.get("park", ""),
                        "p_throws": r.get("p_throws", "") or prev.get("p_throws", ""),
                        "season": season,
                        "inning": inn_i,
                        "early": early or bool(prev.get("early")),
                    }
    pos = RUBBER_DIR / "rubber_position_pitcher_game.csv"
    if pos.exists():
        with pos.open(newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["game_pk"], r["pitcher"], r["stand"])
                cur = meta.get(key, {})
                cur.setdefault("park", r.get("park", ""))
                cur.setdefault("p_throws", r.get("p_throws", ""))
                cur.setdefault("inning", None)
                cur.setdefault("early", False)
                try:
                    cur.setdefault("season", int(r.get("season") or year_of(int(r["game_pk"]))))
                except (TypeError, ValueError):
                    pass
                meta[key] = cur
    return meta


def load_scores(seasons: list[int]) -> dict[str, list[dict]]:
    scores: dict[str, list[dict]] = {}
    for season in seasons:
        path = RUBBER_DIR / f"frame_scores_{season}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                scores.setdefault(r["cell_id"], []).append(r)
    for v in scores.values():
        v.sort(key=lambda r: -float(r.get("score") or 0))
    return scores


# Parks the camera never shows a usable rubber. Ineligible is the triage
# verdict. A park with enough labeled skips and a shrunk vis rate under this
# is treated the same — more clicks will not invent a better angle.
UNUSABLE_VIS = 0.30
UNUSABLE_MIN_N = 8


def park_class(park: str, pins: dict, leg: dict) -> str:
    """usable / thin / unusable, from triage plus realized label yield."""
    pin = pins.get(park) or {}
    if pin.get("status") == "ineligible":
        return "unusable"
    info = leg.get(park) or {}
    n = int(info.get("n") or 0)
    vis = info.get("vis_rate")
    if n >= UNUSABLE_MIN_N and vis is not None and vis < UNUSABLE_VIS:
        return "unusable"
    if vis is not None and n >= 5 and vis < 0.35:
        return "thin"
    if pin.get("status") == "eligible":
        return "usable"
    return "thin"


def inning_tier(inning: int | None) -> int:
    """0 = 1st-3rd, 1 = 4th-5th, 2 = later / unknown."""
    if inning is None:
        return 2
    if inning <= 3:
        return 0
    if inning <= 5:
        return 1
    return 2


def inventory(seasons: list[int], meta: dict, scores: dict) -> list[dict]:
    """One row per cell that has a frame on disk."""
    rows: list[dict] = []
    for season in seasons:
        frames_dir = RUBBER_DIR / "frames" / str(season)
        if not frames_dir.exists():
            continue
        cells: dict[str, list[Path]] = defaultdict(list)
        for path in frames_dir.glob("*_c*.jpg"):
            parts = path.stem.split("_")
            if len(parts) < 4:
                continue
            cells[f"{parts[0]}_{parts[1]}_{parts[2]}"].append(path)
        for cid, paths in cells.items():
            parts = cid.split("_")
            key = (parts[0], parts[1], parts[2])
            info = meta.get(key, {})
            try:
                pk = int(parts[0])
            except ValueError:
                continue
            scored = scores.get(cid) or []
            if scored:
                frame = frames_dir / scored[0]["frame_file"]
                if not frame.exists():
                    frame = sorted(paths)[-1]
                score = float(scored[0].get("score") or 0)
            else:
                frame = sorted(paths)[-1]
                score = 0.0
            rows.append({
                "cell_id": cid,
                "game_pk": parts[0],
                "pitcher": parts[1],
                "stand": parts[2],
                "park": info.get("park", ""),
                "p_throws": info.get("p_throws", ""),
                "season": info.get("season") or year_of(pk),
                "inning": info.get("inning"),
                "early": bool(info.get("early")),
                "frame_file": frame.name,
                "frame_path": frame,
                "score": score,
            })
    return rows


def select(rows: list[dict], limit: int, vis: Counter, pins: dict,
           leg: dict, allow_unusable: bool) -> list[dict]:
    """Early innings at parks the camera can actually see.

    Walks (inning tier, park class) so a first-inning clip at a usable park
    always beats a late-inning clip, and unusable camera angles are skipped
    unless the caller explicitly wants them. Leftover slots go back to usable
    parks rather than being filled with guaranteed skips.
    """
    rng = random.Random(SEED)
    usable_parks = {r["park"] for r in rows
                    if r["park"] and park_class(r["park"], pins, leg) == "usable"}
    n_usable = max(len(usable_parks), 1)
    max_park = max(8, math.ceil(limit / n_usable))
    max_pitcher = 2

    def weight(r: dict) -> float:
        novelty = 1.0 / (1.0 + vis[r["park"] or "?"])
        quality = 0.35 + 0.65 * min(max(r["score"], 0.0), 1.0)
        early_bonus = 0.15 if r.get("early") else 0.0
        inn = r.get("inning")
        inn_pen = 0.0 if inn is None else max(0, (inn - 1) * 0.03)
        return novelty * quality + early_bonus - inn_pen + rng.random() * 0.02

    def try_take(cands: list[dict], picked: list[dict],
                 n_park: Counter, n_pitcher: Counter, n_game: Counter,
                 park_cap: int) -> None:
        for r in sorted(cands, key=weight, reverse=True):
            if len(picked) >= limit:
                return
            park = r["park"]
            if n_park[park] >= park_cap:
                continue
            if n_pitcher[r["pitcher"]] >= max_pitcher:
                continue
            if n_game[r["game_pk"]] >= 3:
                continue
            picked.append(r)
            n_park[park] += 1
            n_pitcher[r["pitcher"]] += 1
            n_game[r["game_pk"]] += 1

    # Passes: early usable, mid usable, any usable, then early thin.
    # Unusable parks (bad camera) stay out unless --include-unusable.
    buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
    skipped_unusable = 0
    for r in rows:
        if not r["park"]:
            continue
        klass = park_class(r["park"], pins, leg)
        if klass == "unusable" and not allow_unusable:
            skipped_unusable += 1
            continue
        if klass == "unusable":
            klass = "thin"
        buckets[(inning_tier(r.get("inning")), klass)].append(r)

    picked: list[dict] = []
    n_park: Counter = Counter()
    n_pitcher: Counter = Counter()
    n_game: Counter = Counter()
    order = [
        (0, "usable"), (1, "usable"), (2, "usable"),
        (0, "thin"), (1, "thin"), (2, "thin"),
    ]
    for key in order:
        try_take(buckets.get(key, []), picked, n_park, n_pitcher, n_game, max_park)
        if len(picked) >= limit:
            break

    # Coverage: if dead parks left a hole, raise the per-park cap and refill
    # from usable early innings rather than inventing shots that do not exist.
    cap = max_park
    while len(picked) < limit and cap < limit:
        cap += max(4, math.ceil((limit - len(picked)) / n_usable))
        for key in order:
            if key[1] != "usable":
                continue
            already = {id(x) for x in picked}
            extra = [r for r in buckets.get(key, []) if id(r) not in already]
            try_take(extra, picked, n_park, n_pitcher, n_game, cap)
            if len(picked) >= limit:
                break

    return picked, skipped_unusable


def write_pack(picked: list[dict], pins: dict, out_dir: Path = OUT_DIR) -> None:
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    todo_path = out_dir / "labels_todo.csv"
    extra = ["season", "inning", "frame_score"]
    fields = LABEL_FIELDS + extra

    rows_out: list[dict] = []
    skipped = 0
    for r in picked:
        img = cv2.imread(str(r["frame_path"]))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        pin = pins.get(r["park"])
        if pin is not None:
            cx, cy = int(pin["fx"] * w), int(pin["fy"] * h)
        else:
            cx, cy = w // 2, int(0.82 * h)
        canvas, x0, y0 = build_crop(img, cx, cy)
        crop_name = f"{r['park']}_{r['cell_id']}.png"
        cv2.imwrite(str(crops_dir / crop_name), canvas)
        row = {k: "" for k in fields}
        row.update({
            "cell_id": r["cell_id"],
            "game_pk": r["game_pk"],
            "pitcher": r["pitcher"],
            "stand": r["stand"],
            "park": r["park"],
            "p_throws": r["p_throws"],
            "frame_file": r["frame_file"],
            "crop_file": crop_name,
            "crop_x0": x0,
            "crop_y0": y0,
            "zoom": ZOOM,
            "season": r["season"],
            "inning": "" if r.get("inning") is None else r["inning"],
            "frame_score": f"{r['score']:.3f}",
        })
        rows_out.append(row)

    with todo_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} cells -> {todo_path}  ({skipped} unreadable)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2025,2026")
    ap.add_argument("--limit", type=int, default=250,
                    help="frames to tag in this batch (default 250)")
    ap.add_argument("--include-unusable", action="store_true",
                    help="also sample parks whose camera never shows the rubber "
                         "(default: skip them and refill from usable parks)")
    ap.add_argument("--out", default=None,
                    help="pack directory to write (default data/rubber/label_pack_train)")
    ap.add_argument("--new-pitchers-only", action="store_true",
                    help="exclude pitchers who already have a visible label, so the "
                         "pack is a pitcher-disjoint holdout for the detector")
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    out_dir = Path(args.out) if args.out else OUT_DIR

    pins = load_park_pins()
    vis = visible_by_park()
    leg = park_legibility()
    done = already_labeled()
    meta = load_meta(seasons)
    scores = load_scores(seasons)
    pool = inventory(seasons, meta, scores)
    fresh = [r for r in pool if r["cell_id"] not in done and r["park"]]
    if args.new_pitchers_only:
        seen: set[str] = set()
        for path in RUBBER_DIR.glob("label_pack*/labels_done.csv"):
            with path.open(newline="") as fh:
                for r in csv.DictReader(fh):
                    try:
                        if int(float(r.get("rubber_visible") or 0)) == 1:
                            seen.add(r["pitcher"])
                    except ValueError:
                        continue
        fresh = [r for r in fresh if r["pitcher"] not in seen]
    print(f"frames on disk: {len(pool)} cells  already tagged: {len(done)}  "
          f"unused with a park: {len(fresh)}")

    picked, n_skip_bad = select(fresh, args.limit, vis, pins, leg,
                                args.include_unusable)
    if not picked:
        raise SystemExit("nothing left to tag; fetch more frames or raise --limit")
    print(f"skipped {n_skip_bad} cells at unusable camera-angle parks")

    write_pack(picked, pins, out_dir)

    by_park = Counter(r["park"] for r in picked)
    by_hand = Counter(r["stand"] for r in picked)
    by_year = Counter(r["season"] for r in picked)
    by_inn = Counter(r.get("inning") for r in picked)
    n_early = sum(1 for r in picked if r.get("inning") is not None and r["inning"] <= 3)
    print("\nthis batch")
    print(f"  LHH {by_hand.get('L', 0)}  RHH {by_hand.get('R', 0)}  "
          f"seasons {dict(by_year)}")
    print(f"  inning <= 3: {n_early}/{len(picked)}  "
          f"by inning { {k: by_inn[k] for k in sorted(by_inn, key=lambda x: 99 if x is None else x)} }")
    print("  by park  (existing visible labels -> new frames):")
    for park, n in sorted(by_park.items(), key=lambda kv: -kv[1]):
        klass = park_class(park, pins, leg)
        print(f"    {park:<4} {klass:<9} have {vis[park]:2d} vis   +{n}")

    print("\ntag them:")
    print("  python pipeline/rubber_04c_label_server.py \\")
    print(f"      --pack {out_dir.relative_to(REPO_ROOT) if out_dir.is_relative_to(REPO_ROOT) else out_dir}")
    print("  then open http://127.0.0.1:8765")
    print("  clicks: rubber left, rubber right, pivot-foot centre.  s = skip.")
    print("\nretrain when a batch is done:")
    print("  python pipeline/rubber_04h_keypoint.py --train")
    print("holdout MAE under ~3 in is the bar for using the model as the")
    print("default rubber coordinate. The last run was 6.6 in on 220 visibles.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rubber position from pretrained pose + a small fine-tuned rubber detector.

Why this replaces rubber_04h. The custom keypoint CNN had to learn what a foot
looks like from ~375 crops and plateaued at ~5.8 in. A pretrained YOLO11 pose
model already localises ankles on broadcast frames: on 40 hand-labeled frames
the pivot ankle lands within 1.4 in RMSE of the clicked foot centre once a
per-hand constant (heel-to-centre offset) is removed. So the foot needs no
labels at all. What remains is (a) the rubber bar, which we learn from the
rubber ends already clicked in every label, and (b) rejecting frames that are
the wrong camera or the wrong moment, which pose also answers: a set position
has both feet planted and level, knees under hips, feet close together.

Pipeline per frame:
    pose (yolo11m-pose, full 1280x720)  ->  pitcher = lowest person
    gate: both ankles seen, level, knees down, feet together
    pivot ankle = lower of the two ankles (nearer the camera = on the rubber)
    rubber detector (yolo11n fine-tuned) on a 512x192 crop around the ankle,
        run on the unoccluded _bg.jpg where rubber_03 wrote one
    rubber_x_in = -(foot_x - rubber_cx) / (rubber_w / 24)
        with foot_x = ankle_x - hand_offset[p_throws] * px_per_inch

Sign convention as everywhere else: image-left is first base, Statcast x is
positive toward first base.

Subcommands:
    --build-dataset   derive rubber boxes from labels_done.csv files, write a
                      YOLO dataset to data/rubber/rubber_det/ (pitcher split)
    --train           fine-tune yolo11n on it -> data/rubber/rubber_det.pt
    --eval            end-to-end inches vs the hand labels, holdout pitchers
    --measure SEASON  run on every candidate frame on disk, write
                      data/rubber/rubber_pose_measurements_<season>.csv (per
                      frame, doubles as proposals for the labeler's verify
                      mode) and rubber_pose_cells_<season>.csv (per cell)

Usage:
    python pipeline/rubber_04j_pose_measure.py --build-dataset
    python pipeline/rubber_04j_pose_measure.py --train
    python pipeline/rubber_04j_pose_measure.py --eval
    python pipeline/rubber_04j_pose_measure.py --measure 2025
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBBER_DIR = REPO_ROOT / "data" / "rubber"
DET_DIR = RUBBER_DIR / "rubber_det"
DET_WEIGHTS = RUBBER_DIR / "rubber_det.pt"
CALIB_JSON = RUBBER_DIR / "rubber_pose_calib.json"
POSE_WEIGHTS = "yolo11m-pose.pt"          # downloaded by ultralytics on first use
DET_BASE = "yolo11n.pt"

CROP_W, CROP_H = 440, 150                 # labeler crop (must match rubber_04b)
DET_W, DET_H = 512, 192                   # rubber detector input crop
RUBBER_WIDTH_IN = 24.0
MAX_FOOT_CENTRE_IN = 18.0                 # 12 in half-rubber + 6 in half-shoe
BAR_W_RANGE = (36, 220)                   # plausible rubber width in px
SEED = 1

# Frame acceptance. Two cheap consistency checks remove most gross outliers:
# the detector's own confidence, and the pitcher's bbox height divided by the
# detected rubber width. In the standard CF shot that ratio is ~3.2 (a ~6-ft
# man is about three rubbers tall) with p5-p95 of 2.9-3.8 on hand-labeled
# frames; a close-up replay, or a bar drawn on the wrong object, breaks it.
# Same-game LHH/RHH cell pairs (where the pitcher has not moved) disagreed at
# RMS 5.6 in unfiltered, 3.1 in with these two gates.
MIN_RUBBER_CONF = 0.40
H_OVER_W_RANGE = (2.2, 4.2)
# Feet-level tolerance as a fraction of pitcher bbox height. On the verify
# pass the frames a human flagged "already moving" had ankle_dy median 0.041
# against 0.013 (p90 0.028) for accepted set positions, so 0.05 catches most
# of the leak at a ~3% cost in good frames. (Was 0.10.)
ANKLE_DY_MAX = 0.05

# COCO keypoint indices
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
KP_CONF = 0.30

# Default heel-to-centre offsets (inches, image-x sense: ankle_x - foot_x) from
# the 40-frame probe. --eval refits them on the training split and writes
# rubber_pose_calib.json, which --measure prefers when present.
DEFAULT_HAND_OFFSET_IN = {"L": 2.7, "R": -1.4}


def year_of(game_pk: int) -> int:
    if game_pk >= 820000:
        return 2026
    if game_pk >= 770000:
        return 2025
    return 2024


def frames_dir(season: int) -> Path:
    return RUBBER_DIR / "frames" / str(season)


def bg_path_for(frame_path: Path) -> Path | None:
    stem = frame_path.stem.rsplit("_c", 1)[0]
    p = frame_path.with_name(f"{stem}_bg.jpg")
    return p if p.exists() else None


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

# Packs that must never train the detector or fit the hand offsets. The golden
# set was labeled blind on pitchers with no prior label; it is the one number
# that says what the model does on data it has not seen (1.60 in RMSE on the
# first 84 cells, 2026-09-02). Training on it would quietly make that number
# meaningless.
HOLDOUT_PACKS = {"label_pack_golden", "label_pack_golden2"}


def load_labels(visible_only: bool = True, exclude_packs: set[str] = frozenset()) -> list[dict]:
    """Every hand label, converted to full-frame pixel coordinates."""
    rows: list[dict] = []
    for done in sorted(RUBBER_DIR.glob("label_pack*/labels_done.csv")):
        if done.parent.name in exclude_packs:
            continue
        with done.open(newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    vis = int(float(r.get("rubber_visible") or 0))
                except ValueError:
                    continue
                if visible_only and vis != 1:
                    continue
                try:
                    gp = int(r["game_pk"])
                    x0 = float(r["crop_x0"])
                    y0 = float(r.get("crop_y0") or 0)
                except (TypeError, ValueError, KeyError):
                    continue
                season = year_of(gp)
                frame = frames_dir(season) / r["frame_file"]
                rec = {
                    "pack": done.parent.name,
                    "cell_id": r.get("cell_id", ""),
                    "game_pk": r["game_pk"], "pitcher": r["pitcher"],
                    "stand": r.get("stand", ""), "park": r.get("park", ""),
                    "p_throws": r.get("p_throws", ""), "season": season,
                    "frame_file": r["frame_file"], "frame_path": frame,
                    "visible": vis, "crop_x0": x0, "crop_y0": y0,
                }
                if vis == 1:
                    try:
                        left = x0 + float(r["rubber_left_px"])
                        right = x0 + float(r["rubber_right_px"])
                        foot = x0 + float(r["foot_center_px"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if right - left < 20:
                        continue
                    rec.update(left=left, right=right, foot=foot,
                               ppi=(right - left) / RUBBER_WIDTH_IN,
                               label_in=-(foot - (left + right) / 2) / ((right - left) / RUBBER_WIDTH_IN))
                rows.append(rec)
    return rows


def rubber_y_band(gray: np.ndarray, left: float, right: float,
                  y0: float, y1: float) -> tuple[float, float, float] | None:
    """Find the rubber's row band between y0..y1 given its known x extent.

    The bar is brighter than the dirt around it, so score each row by the mean
    brightness inside the labeled span minus the mean in flanks just outside
    it. The band is the contiguous run around the peak holding >= 50% of peak
    contrast. Returns (cy, height, contrast) or None if nothing stands out.
    """
    h, w = gray.shape[:2]
    li, ri = int(round(left)) + 2, int(round(right)) - 2
    if ri - li < 10:
        return None
    ya, yb = max(0, int(y0)), min(h, int(y1))
    fl0, fl1 = max(0, li - 26), max(0, li - 6)
    fr0, fr1 = min(w, ri + 6), min(w, ri + 26)
    inner = gray[ya:yb, li:ri].astype(np.float32).mean(axis=1)
    flank_parts = []
    if fl1 > fl0:
        flank_parts.append(gray[ya:yb, fl0:fl1].astype(np.float32))
    if fr1 > fr0:
        flank_parts.append(gray[ya:yb, fr0:fr1].astype(np.float32))
    if not flank_parts:
        return None
    flank = np.concatenate(flank_parts, axis=1).mean(axis=1)
    # Also require the bar to be brighter than the rows above/below it.
    contrast = inner - flank
    if len(contrast) < 5:
        return None
    sm = cv2.GaussianBlur(contrast.reshape(-1, 1), (1, 3), 0).ravel()
    k = int(np.argmax(sm))
    peak = float(sm[k])
    if peak < 12.0:
        return None
    thr = 0.5 * peak
    a = k
    while a > 0 and sm[a - 1] >= thr:
        a -= 1
    b = k
    while b < len(sm) - 1 and sm[b + 1] >= thr:
        b += 1
    height = float(b - a + 1)
    if height > 28:  # a wall of brightness, not a bar
        return None
    return ya + (a + b) / 2.0, max(4.0, height), peak


# --------------------------------------------------------------------------
# dataset + training for the rubber detector
# --------------------------------------------------------------------------

def crop_region(img: np.ndarray, cx: float, cy: float) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    x0 = int(max(0, min(w - DET_W, round(cx - DET_W / 2))))
    y0 = int(max(0, min(h - DET_H, round(cy - DET_H / 2))))
    crop = img[y0:y0 + DET_H, x0:x0 + DET_W]
    if crop.shape[:2] != (DET_H, DET_W):
        crop = cv2.copyMakeBorder(crop, 0, DET_H - crop.shape[0], 0,
                                  DET_W - crop.shape[1], cv2.BORDER_CONSTANT, value=0)
    return crop, x0, y0


def build_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(SEED)
    labels = load_labels(visible_only=True, exclude_packs=HOLDOUT_PACKS)
    print(f"visible labels: {len(labels)}  (holdout packs excluded: {sorted(HOLDOUT_PACKS)})")

    boxes: list[dict] = []
    n_nobg = n_noband = 0
    for r in labels:
        if not r["frame_path"].exists():
            continue
        bg = bg_path_for(r["frame_path"])
        src = bg if bg is not None else r["frame_path"]
        if bg is None:
            n_nobg += 1
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        band = rubber_y_band(gray, r["left"], r["right"],
                             r["crop_y0"], r["crop_y0"] + CROP_H)
        if band is None:
            n_noband += 1
            continue
        cy, bh, contrast = band
        boxes.append({**r, "cy": cy, "bh": bh, "contrast": contrast,
                      "sources": [p for p in (bg, r["frame_path"]) if p is not None]})
    print(f"boxes with a y-band: {len(boxes)}   no band: {n_noband}   no bg image: {n_nobg}")
    if boxes:
        c = np.array([b["contrast"] for b in boxes])
        print(f"contrast: median {np.median(c):.0f}  p10 {np.percentile(c, 10):.0f}   "
              f"band height median {np.median([b['bh'] for b in boxes]):.1f} px")

    # pitcher-disjoint split so --eval on val pitchers is honest
    pitchers = sorted({b["pitcher"] for b in boxes})
    rng.shuffle(pitchers)
    val_p = set(pitchers[:max(1, int(0.2 * len(pitchers)))])

    if DET_DIR.exists():
        shutil.rmtree(DET_DIR)
    for split in ("train", "val"):
        (DET_DIR / "images" / split).mkdir(parents=True)
        (DET_DIR / "labels" / split).mkdir(parents=True)

    n_img = Counter()
    for b in boxes:
        split = "val" if b["pitcher"] in val_p else "train"
        cx = (b["left"] + b["right"]) / 2
        bw = b["right"] - b["left"]
        # The detector is run on a crop centred on the pivot ankle, which sits
        # up to ~18 in (70 px) from the rubber centre and ~10-20 px above it.
        # Train on the same distribution of offsets.
        n_jit = 3 if split == "train" else 1
        for j in range(n_jit):
            for src in b["sources"]:
                img = cv2.imread(str(src))
                if img is None:
                    continue
                if j == 0 and split == "val":
                    jx, jy = 0.0, 0.0
                else:
                    jx, jy = rng.uniform(-90, 90), rng.uniform(-45, 35)
                crop, x0, y0 = crop_region(img, cx + jx, b["cy"] + jy)
                bx = (cx - x0) / DET_W
                by = (b["cy"] - y0) / DET_H
                bwn = bw / DET_W
                bhn = max(b["bh"], 6.0) / DET_H
                if not (0 < bx < 1 and 0 < by < 1):
                    continue
                name = f"{b['cell_id'] or b['frame_path'].stem}_{src.stem[-3:]}_{j}"
                cv2.imwrite(str(DET_DIR / "images" / split / f"{name}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 94])
                (DET_DIR / "labels" / split / f"{name}.txt").write_text(
                    f"0 {bx:.6f} {by:.6f} {bwn:.6f} {bhn:.6f}\n")
                n_img[split] += 1

    (DET_DIR / "data.yaml").write_text(
        f"path: {DET_DIR}\ntrain: images/train\nval: images/val\nnames:\n  0: rubber\n")
    json.dump({"val_pitchers": sorted(val_p)}, (DET_DIR / "split.json").open("w"))
    print(f"wrote {n_img['train']} train / {n_img['val']} val crops -> {DET_DIR}")

    # contact sheet of derived boxes for a quick eyeball
    sheet = []
    for b in rng.sample(boxes, min(24, len(boxes))):
        img = cv2.imread(str(b["sources"][0]))
        crop, x0, y0 = crop_region(img, (b["left"] + b["right"]) / 2, b["cy"])
        cv2.rectangle(crop, (int(b["left"] - x0), int(b["cy"] - y0 - b["bh"] / 2)),
                      (int(b["right"] - x0), int(b["cy"] - y0 + b["bh"] / 2)), (0, 255, 255), 1)
        cv2.putText(crop, f"{b['park']} c{b['contrast']:.0f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        sheet.append(crop)
    if sheet:
        rows_ = [np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet) - len(sheet) % 4, 4)]
        out = RUBBER_DIR / "debug" / "rubber_det_boxes.png"
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), np.vstack(rows_))
        print(f"contact sheet -> {out}")


def train_detector(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    # --init lets a retrain start from the previous rubber weights instead of
    # the COCO base, which converges in far fewer epochs on new correction data.
    model = YOLO(args.init or DET_BASE)
    out_w = Path(args.weights) if args.weights else DET_WEIGHTS
    res = model.train(
        data=str(DET_DIR / "data.yaml"), epochs=args.epochs, imgsz=512,
        batch=16, device="mps", project=str(DET_DIR / "runs"), name=out_w.stem,
        exist_ok=True, seed=SEED, verbose=False, plots=False,
        # a rigid symmetric bar: flips are fine, heavy mosaics are not
        fliplr=0.5, mosaic=0.3, mixup=0.0, degrees=2.0, scale=0.25,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, close_mosaic=10, patience=25,
    )
    best = Path(res.save_dir) / "weights" / "best.pt"
    shutil.copy(best, out_w)
    print(f"wrote {out_w}" + ("" if out_w == DET_WEIGHTS else
                              f"   (canonical {DET_WEIGHTS.name} untouched; copy over it to promote)"))


# --------------------------------------------------------------------------
# inference pieces
# --------------------------------------------------------------------------

class Models:
    def __init__(self, need_det: bool = True, weights: Path | None = None) -> None:
        from ultralytics import YOLO
        self.pose = YOLO(POSE_WEIGHTS)
        self.det = None
        if need_det:
            w = weights or DET_WEIGHTS
            if not w.exists():
                raise SystemExit(f"missing {w}; run --build-dataset then --train")
            self.det = YOLO(str(w))
            print(f"rubber detector: {w}")

    def people(self, img: np.ndarray) -> list[dict]:
        res = self.pose.predict(img, imgsz=1280, conf=0.15, verbose=False, device="mps")[0]
        out: list[dict] = []
        if res.keypoints is None or not len(res.boxes):
            return out
        kxy = res.keypoints.xy.cpu().numpy()
        kc = res.keypoints.conf.cpu().numpy()
        bx = res.boxes.xyxy.cpu().numpy()
        for i in range(len(bx)):
            out.append({"box": bx[i], "kp": kxy[i], "kc": kc[i]})
        return out

    def rubber_boxes(self, crop: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        res = self.det.predict(crop, imgsz=512, conf=0.10, verbose=False, device="mps")[0]
        out = []
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            out.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
        return out


def gate_pitcher(people: list[dict], H: int, W: int) -> tuple[dict | None, str, dict]:
    """Pick the pitcher and decide whether this is a usable set-position frame.

    In the centre-field view the pitcher is the lowest person in frame. A set
    position has both ankles visible and level, knees below hips, and the feet
    close together laterally. Anything else is a leg lift, a stride, a replay
    angle, or a graphic, and gets a reason instead of a measurement.
    """
    diag: dict = {}
    if not people:
        return None, "no person", diag
    # Only a full figure with both ankles seen can be the pitcher we want. A
    # torso cut off at the bottom edge of the frame (dugout, on-deck circle,
    # score bug overlap) is often the lowest box and has no legs, so filtering
    # before ranking is what keeps the pitcher.
    cands = []
    for p in people:
        x0, y0, x1, y1 = p["box"]
        h = y1 - y0
        if h < 90:
            continue
        if p["kc"][L_ANK] < KP_CONF or p["kc"][R_ANK] < KP_CONF:
            continue
        cands.append((y1, p))
    if not cands:
        return None, "no person with visible ankles", diag
    cands.sort(key=lambda t: -t[0])
    y1, p = cands[0]
    x0, y0, x1, _ = p["box"]
    h = y1 - y0
    kp, kc = p["kp"], p["kc"]
    diag.update(pitcher_h=float(h), pitcher_cx=float((x0 + x1) / 2 / W),
                pitcher_y1=float(y1 / H),
                n_people_above=sum(1 for q in people
                                   if q["box"][3] < y1 - 0.12 * H and q["box"][3] - q["box"][1] >= 60))
    if y1 < 0.45 * H:
        return p, "lowest person too high in frame (not CF view)", diag
    la, ra = kp[L_ANK], kp[R_ANK]
    ank_dy = abs(la[1] - ra[1])
    ank_dx = abs(la[0] - ra[0])
    diag.update(ankle_dy_frac=float(ank_dy / h), ankle_dx_frac=float(ank_dx / h))
    if ank_dy > ANKLE_DY_MAX * h:
        return p, "feet not level (leg lift / stride)", diag
    if ank_dx > 0.40 * h:
        return p, "feet too far apart (stride)", diag
    for hip, knee in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
        if kc[hip] >= KP_CONF and kc[knee] >= KP_CONF:
            if kp[knee][1] < kp[hip][1] + 0.12 * h:
                return p, "knee lifted", diag
    return p, "", diag


def pivot_ankle(p: dict) -> tuple[float, float, str]:
    la, ra = p["kp"][L_ANK], p["kp"][R_ANK]
    if la[1] >= ra[1]:
        return float(la[0]), float(la[1]), "L"
    return float(ra[0]), float(ra[1]), "R"


def pick_rubber(boxes, ax: float, ay: float, x0: int, y0: int):
    """Best detection near the ankle, in full-frame coords -> (left,right,cy,conf)."""
    best = None
    for bx0, by0, bx1, by1, conf in boxes:
        w = bx1 - bx0
        if not (BAR_W_RANGE[0] <= w <= BAR_W_RANGE[1]):
            continue
        cx, cy = x0 + (bx0 + bx1) / 2, y0 + (by0 + by1) / 2
        ppi = w / RUBBER_WIDTH_IN
        # the ankle sits within ~20 in of the rubber centre and just above the bar
        if abs(cx - ax) > 21 * ppi or not (-45 <= cy - ay <= 60):
            continue
        score = conf - 0.002 * abs(cx - ax) / ppi
        if best is None or score > best[0]:
            best = (score, x0 + bx0, x0 + bx1, cy, conf)
    if best is None:
        return None
    return best[1:]


def measure_frame(models: Models, frame_path: Path, p_throws: str,
                  hand_offset: dict[str, float]) -> dict:
    """One frame -> dict with ok flag, reason, and the measurement."""
    out = {"frame_file": frame_path.name, "ok": False, "reason": ""}
    img = cv2.imread(str(frame_path))
    if img is None:
        out["reason"] = "unreadable frame"
        return out
    H, W = img.shape[:2]
    people = models.people(img)
    p, reason, diag = gate_pitcher(people, H, W)
    out.update({k: diag.get(k, "") for k in
                ("pitcher_h", "pitcher_cx", "pitcher_y1", "n_people_above",
                 "ankle_dy_frac", "ankle_dx_frac")})
    if reason:
        out["reason"] = reason
        return out
    ax, ay, side = pivot_ankle(p)
    out.update(ankle_x=ax, ankle_y=ay, pivot_side=side)

    bg = bg_path_for(frame_path)
    src = cv2.imread(str(bg)) if bg is not None else img
    if src is None:
        src = img
    out["rubber_source"] = "bg" if bg is not None else "frame"
    crop, x0, y0 = crop_region(src, ax, ay)
    rub = pick_rubber(models.rubber_boxes(crop), ax, ay, x0, y0)
    if rub is None and bg is not None:
        # background can be stale/ghosted; try the stance frame itself
        crop, x0, y0 = crop_region(img, ax, ay)
        rub = pick_rubber(models.rubber_boxes(crop), ax, ay, x0, y0)
        out["rubber_source"] = "frame"
    if rub is None:
        out["reason"] = "rubber not detected"
        return out
    left, right, rcy, conf = rub
    ppi = (right - left) / RUBBER_WIDTH_IN
    off_in = hand_offset.get(p_throws, 0.0)
    foot_x = ax - off_in * ppi
    inches = -(foot_x - (left + right) / 2) / ppi
    out.update(rubber_left=left, rubber_right=right, rubber_cy=rcy, rubber_conf=conf,
               px_per_inch=ppi, foot_x=foot_x, rubber_x_in=inches,
               h_over_w=float(diag.get("pitcher_h", 0.0)) / (right - left))
    out["ok"], out["reason"] = accept(out)
    return out


def accept(m: dict) -> tuple[bool, str]:
    """Frame-level acceptance from an already-measured row (dict or CSV row)."""
    try:
        inches = float(m["rubber_x_in"])
        conf = float(m["rubber_conf"])
        ratio = float(m["h_over_w"])
    except (KeyError, TypeError, ValueError):
        return False, "rubber not detected"
    if abs(inches) > MAX_FOOT_CENTRE_IN + 2:
        return False, "implausible offset"
    try:  # re-judge the moment too, so --refilter can tighten the pose gate
        if float(m.get("ankle_dy_frac") or 0) > ANKLE_DY_MAX:
            return False, "feet not level (leg lift / stride)"
    except (TypeError, ValueError):
        pass
    if conf < MIN_RUBBER_CONF:
        return False, f"rubber confidence < {MIN_RUBBER_CONF:.2f}"
    if not (H_OVER_W_RANGE[0] <= ratio <= H_OVER_W_RANGE[1]):
        return False, "pitcher-height / rubber-width ratio off (wrong zoom or wrong bar)"
    return True, ""


# --------------------------------------------------------------------------
# eval against hand labels
# --------------------------------------------------------------------------

def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


def evaluate(args: argparse.Namespace) -> None:
    models = Models(need_det=True, weights=Path(args.weights) if args.weights else None)
    labels = [r for r in load_labels(visible_only=True) if r["frame_path"].exists()]
    split = json.load((DET_DIR / "split.json").open()) if (DET_DIR / "split.json").exists() else {}
    # Detector val pitchers plus every golden-set pitcher count as held out;
    # the hand offsets are fit on the rest.
    val_p = set(split.get("val_pitchers", []))
    val_p |= {r["pitcher"] for r in labels if r["pack"] in HOLDOUT_PACKS}
    if args.limit:
        random.Random(SEED).shuffle(labels)
        labels = labels[:args.limit]
    print(f"evaluating {len(labels)} labeled frames  (detector val pitchers: {len(val_p)})")

    recs = []
    reasons = Counter()
    for i, r in enumerate(labels):
        m = measure_frame(models, r["frame_path"], r["p_throws"], {"L": 0.0, "R": 0.0})
        rec = {**r, **{f"m_{k}": v for k, v in m.items()}}
        rec["is_val"] = r["pitcher"] in val_p
        if not m.get("ankle_x"):
            reasons[m["reason"]] += 1
        recs.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(labels)}", file=sys.stderr)

    gated = [x for x in recs if x.get("m_ankle_x")]
    print(f"\ngate passed {len(gated)}/{len(recs)} labeled frames "
          f"(these are frames a human judged usable, so rejects here are false negatives)")
    for k, n in reasons.most_common():
        print(f"   rejected: {n:3d}  {k}")

    # foot-only (uses labeled rubber for scale): fit hand offsets on train pitchers
    off = {}
    for hand in ("L", "R"):
        e = np.array([(x["m_ankle_x"] - x["foot"]) / x["ppi"] for x in gated
                      if x["p_throws"] == hand and not x["is_val"]])
        off[hand] = float(np.median(e)) if len(e) else DEFAULT_HAND_OFFSET_IN[hand]
    print(f"\nhand offsets (ankle - foot centre, in): L {off['L']:+.2f}  R {off['R']:+.2f}"
          f"   (fit on train pitchers)")
    for tag, sub in (("all", gated), ("val pitchers", [x for x in gated if x["is_val"]])):
        e = np.array([(x["m_ankle_x"] - off.get(x["p_throws"], 0)) / 1.0 for x in sub])
        e = np.array([((x["m_ankle_x"] - off.get(x["p_throws"], 0) * x["ppi"]) - x["foot"]) / x["ppi"] for x in sub])
        print(f"  foot only [{tag:12s}] n={len(e):3d}  RMSE {rmse(e):.2f} in  MAE {np.abs(e).mean() if len(e) else float('nan'):.2f}"
              f"  >3in {100 * np.mean(np.abs(e) > 3) if len(e) else 0:.0f}%")

    # rubber-only and end-to-end
    got = [x for x in gated if x.get("m_rubber_left")]
    print(f"\nrubber detected on {len(got)}/{len(gated)} gated frames")
    for tag, sub in (("all", got), ("val pitchers", [x for x in got if x["is_val"]])):
        if not sub:
            continue
        ec = np.array([((x["m_rubber_left"] + x["m_rubber_right"]) / 2 - (x["left"] + x["right"]) / 2) / x["ppi"] for x in sub])
        ew = np.array([(x["m_rubber_right"] - x["m_rubber_left"]) / (x["right"] - x["left"]) for x in sub])
        print(f"  rubber centre [{tag:12s}] n={len(sub):3d}  RMSE {rmse(ec):.2f} in  >3in {100 * np.mean(np.abs(ec) > 3):.0f}%"
              f"   width ratio median {np.median(ew):.3f}  IQR {np.percentile(ew, 25):.3f}-{np.percentile(ew, 75):.3f}")
        pred = np.array([
            -((x["m_ankle_x"] - off.get(x["p_throws"], 0) * x["m_px_per_inch"])
              - (x["m_rubber_left"] + x["m_rubber_right"]) / 2) / x["m_px_per_inch"] for x in sub])
        lab = np.array([x["label_in"] for x in sub])
        e = pred - lab
        print(f"  END-TO-END    [{tag:12s}] n={len(sub):3d}  RMSE {rmse(e):.2f} in  MAE {np.abs(e).mean():.2f}"
              f"  bias {e.mean():+.2f}  >3in {100 * np.mean(np.abs(e) > 3):.0f}%  >6in {100 * np.mean(np.abs(e) > 6):.0f}%")
        if tag == "val pitchers":
            worst = sorted(zip(np.abs(e), sub), key=lambda t: -t[0])[:8]
            print("  worst val frames:")
            for ae, x in worst:
                print(f"     {ae:5.1f} in  {x['park']:<4} {x['frame_file']}  conf {x['m_rubber_conf']:.2f}  src {x['m_rubber_source']}")

    json.dump({"hand_offset_in": off}, CALIB_JSON.open("w"), indent=1)
    print(f"\nwrote {CALIB_JSON}")

    out = RUBBER_DIR / "debug" / "pose_eval.csv"
    out.parent.mkdir(exist_ok=True)
    keys: list[str] = []
    for x in recs:  # union, in first-seen order; rejected frames lack the m_rubber_* keys
        for k in x.keys():
            if k != "frame_path" and k not in keys:
                keys.append(k)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    print(f"per-frame eval -> {out}")


# --------------------------------------------------------------------------
# measure a whole season
# --------------------------------------------------------------------------

def load_cell_meta(season: int) -> dict[tuple[str, str, str], dict]:
    meta: dict[tuple[str, str, str], dict] = {}
    for name in (f"clip_manifest_{season}.csv", f"fetch_targets_{season}.csv"):
        p = RUBBER_DIR / name
        if not p.exists():
            continue
        with p.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if not r.get("game_pk"):
                    continue
                key = (r["game_pk"], r["pitcher"], r.get("stand", ""))
                cur = meta.setdefault(key, {})
                for k in ("park", "p_throws", "inning"):
                    if r.get(k) and not cur.get(k):
                        cur[k] = r[k]
    pos = RUBBER_DIR / "rubber_position_pitcher_game.csv"
    if pos.exists():
        with pos.open(newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["game_pk"], r["pitcher"], r["stand"])
                cur = meta.setdefault(key, {})
                cur.setdefault("park", r.get("park", ""))
                cur.setdefault("p_throws", r.get("p_throws", ""))
    return meta


MEAS_FIELDS = [
    "cell_id", "game_pk", "pitcher", "stand", "park", "p_throws", "season",
    "frame_file", "cand", "ok", "reason",
    "ankle_x", "ankle_y", "pivot_side", "foot_x",
    "rubber_left", "rubber_right", "rubber_cy", "rubber_conf", "rubber_source",
    "px_per_inch", "rubber_x_in", "h_over_w",
    "pitcher_h", "pitcher_cx", "pitcher_y1", "n_people_above",
    "ankle_dy_frac", "ankle_dx_frac",
    # labeler-space proposal (crop centred on the rubber, same geometry as
    # rubber_04b.build_crop) so the verify mode can pre-place the clicks
    "crop_x0", "crop_y0", "rubber_left_px", "rubber_right_px", "foot_center_px",
]


def finish_row(row: dict) -> None:
    """Attach the labeler-space proposal to an accepted frame row (in place)."""
    rcx = (float(row["rubber_left"]) + float(row["rubber_right"])) / 2
    x0 = int(max(0, min(1280 - CROP_W, rcx - CROP_W // 2)))
    y0 = int(max(0, min(720 - CROP_H, float(row["rubber_cy"]) - CROP_H // 2)))
    row.update(crop_x0=x0, crop_y0=y0,
               rubber_left_px=round(float(row["rubber_left"]) - x0, 2),
               rubber_right_px=round(float(row["rubber_right"]) - x0, 2),
               foot_center_px=round(float(row["foot_x"]) - x0, 2))


def write_season_outputs(season: int, frame_rows: list[dict], n_cells: int) -> None:
    """Per-frame CSV plus the per-cell aggregate, from finished frame rows."""
    out_frames = RUBBER_DIR / f"rubber_pose_measurements_{season}.csv"
    out_cells = RUBBER_DIR / f"rubber_pose_cells_{season}.csv"
    by_cell: dict[str, list[dict]] = defaultdict(list)
    reasons = Counter()
    n_ok = 0
    for row in frame_rows:
        if row.get("ok") in (True, "True"):
            n_ok += 1
            by_cell[row["cell_id"]].append(row)
        else:
            reasons[row.get("reason", "")] += 1
    with out_frames.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MEAS_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(frame_rows)
    n_per_cell = Counter(r["cell_id"] for r in frame_rows)
    cell_rows = []
    for cid, vals in by_cell.items():
        xs = np.array([float(v["rubber_x_in"]) for v in vals])
        confs = np.array([float(v["rubber_conf"]) for v in vals])
        best = vals[int(np.argmax(confs))]
        cell_rows.append({
            "cell_id": cid, "game_pk": best["game_pk"], "pitcher": best["pitcher"],
            "stand": best["stand"], "park": best.get("park", ""),
            "p_throws": best.get("p_throws", ""), "season": season,
            "n_frames_ok": len(vals), "n_frames": n_per_cell[cid],
            "rubber_x_in": float(np.median(xs)),
            "rubber_x_in_spread": float(xs.max() - xs.min()),
            "best_frame": best["frame_file"], "best_conf": float(confs.max()),
            "px_per_inch": float(np.median([float(v["px_per_inch"]) for v in vals])),
        })
    cell_rows.sort(key=lambda r: r["cell_id"])
    with out_cells.open("w", newline="") as fh:
        keys = list(cell_rows[0].keys()) if cell_rows else ["cell_id"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(cell_rows)
    print(f"\nframes ok {n_ok}/{len(frame_rows)}   cells measured {len(cell_rows)}/{n_cells} "
          f"({100 * len(cell_rows) / max(1, n_cells):.0f}%)")
    for k, n in reasons.most_common():
        print(f"   rejected: {n:4d}  {k}")
    if cell_rows:
        sp = np.array([c["rubber_x_in_spread"] for c in cell_rows if c["n_frames_ok"] >= 2])
        if len(sp):
            print(f"within-cell spread across frames (n>=2 ok): median {np.median(sp):.2f} in, "
                  f"p90 {np.percentile(sp, 90):.2f} in")
    print(f"-> {out_frames}\n-> {out_cells}")


def measure_season(args: argparse.Namespace) -> None:
    season = args.measure
    models = Models(need_det=True, weights=Path(args.weights) if args.weights else None)
    hand_offset = dict(DEFAULT_HAND_OFFSET_IN)
    if CALIB_JSON.exists():
        hand_offset.update(json.load(CALIB_JSON.open()).get("hand_offset_in", {}))
    print(f"hand offsets: {hand_offset}")
    meta = load_cell_meta(season)

    fdir = frames_dir(season)
    cells: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(fdir.glob("*_c*.jpg")):
        cells[p.stem.rsplit("_c", 1)[0]].append(p)
    cell_ids = sorted(cells)
    if args.only:
        want = set(args.only.split(","))
        cell_ids = [c for c in cell_ids if c in want or c.split("_")[1] in want]
    if args.limit:
        cell_ids = cell_ids[:args.limit]
    print(f"{len(cell_ids)} cells, {sum(len(cells[c]) for c in cell_ids)} candidate frames")

    frame_rows: list[dict] = []
    for ci, cid in enumerate(cell_ids):
        gp, pit, stand = cid.split("_")[:3]
        m0 = meta.get((gp, pit, stand), {})
        p_throws = m0.get("p_throws", "")
        for fp in cells[cid]:
            m = measure_frame(models, fp, p_throws, hand_offset)
            row = {"cell_id": cid, "game_pk": gp, "pitcher": pit, "stand": stand,
                   "park": m0.get("park", ""), "p_throws": p_throws, "season": season,
                   "cand": fp.stem.rsplit("_c", 1)[1], **m}
            if m.get("ok"):
                finish_row(row)
            frame_rows.append(row)
        if (ci + 1) % 50 == 0:
            n_ok = sum(1 for r in frame_rows if r.get("ok"))
            print(f"  {ci + 1}/{len(cell_ids)} cells, {n_ok} frames ok", file=sys.stderr)
    write_season_outputs(season, frame_rows, len(cell_ids))


def refilter_season(args: argparse.Namespace) -> None:
    """Re-apply accept() to an existing per-frame CSV without re-running models.

    Rows the gate rejected before the detector ran (no ankles, leg lift...) have
    no measurement and stay rejected; rows with a measurement are re-judged
    against the current MIN_RUBBER_CONF / H_OVER_W_RANGE.
    """
    season = args.refilter
    path = RUBBER_DIR / f"rubber_pose_measurements_{season}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run --measure {season} first")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    n_cells = len({r["cell_id"] for r in rows})
    for r in rows:
        if r.get("rubber_left"):
            if not r.get("h_over_w"):
                try:
                    r["h_over_w"] = float(r["pitcher_h"]) / (float(r["rubber_right"]) - float(r["rubber_left"]))
                except (TypeError, ValueError, ZeroDivisionError):
                    r["h_over_w"] = ""
            ok, reason = accept(r)
            r["ok"], r["reason"] = ok, reason
            if ok:
                finish_row(r)
            else:
                for k in ("crop_x0", "crop_y0", "rubber_left_px", "rubber_right_px", "foot_center_px"):
                    r[k] = ""
        else:
            r["ok"] = False
    print(f"refiltered {len(rows)} frames with conf>={MIN_RUBBER_CONF} and "
          f"h/w in {H_OVER_W_RANGE}")
    write_season_outputs(season, rows, n_cells)


# --------------------------------------------------------------------------
# label packs from measured cells (golden set / verify pass)
# --------------------------------------------------------------------------

def make_pack(args: argparse.Namespace) -> None:
    """Write a labeler pack whose frames all passed the pose gate.

    Two uses. A *blind* pack (no proposals.csv) is a golden set: the human
    labels normally, never sees the model, and the result is an honest holdout
    -- pair it with --new-pitchers-only so the detector never trained on those
    pitchers either. A *verify* pack carries proposals.csv, so the labeler
    pre-places the model's answer and the human accepts or corrects it at a few
    seconds per frame. Corrections feed the next --build-dataset; the
    accepted/edited/rejected tally is the model's hit rate.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rubber_04b_label_pack import LABEL_FIELDS, ZOOM, build_crop

    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    out_dir = Path(args.make_pack)
    rng = random.Random(SEED)

    done_cells: set[str] = set()
    seen_pitchers: set[str] = set()
    for path in RUBBER_DIR.glob("label_pack*/labels_done.csv"):
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                done_cells.add(r.get("cell_id", ""))
                try:
                    if int(float(r.get("rubber_visible") or 0)) == 1:
                        seen_pitchers.add(r["pitcher"])
                except ValueError:
                    pass
    # also keep out anything queued in another unfinished pack
    for path in RUBBER_DIR.glob("label_pack*/labels_todo.csv"):
        if path.parent == out_dir:
            continue
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                done_cells.add(r.get("cell_id", ""))

    cells: list[dict] = []
    frame_meas: dict[tuple[str, str], dict] = {}
    for s in seasons:
        cp = RUBBER_DIR / f"rubber_pose_cells_{s}.csv"
        mp = RUBBER_DIR / f"rubber_pose_measurements_{s}.csv"
        if not cp.exists() or not mp.exists():
            print(f"no measurements for {s}; run --measure {s} first")
            continue
        meta = load_cell_meta(s)
        with mp.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("ok") == "True":
                    frame_meas[(r["cell_id"], r["frame_file"])] = r
        with cp.open(newline="") as fh:
            for r in csv.DictReader(fh):
                gp, pit, stand = r["cell_id"].split("_")[:3]
                inn = meta.get((gp, pit, stand), {}).get("inning")
                try:
                    r["inning"] = int(float(inn)) if inn not in (None, "") else None
                except ValueError:
                    r["inning"] = None
                r["best_conf"] = float(r["best_conf"])
                cells.append(r)

    pool = [c for c in cells if c["cell_id"] not in done_cells and c["park"]
            and c["best_conf"] >= args.min_conf]
    if args.new_pitchers_only:
        pool = [c for c in pool if c["pitcher"] not in seen_pitchers]
    print(f"measured cells {len(cells)}  eligible {len(pool)}  "
          f"(excluding {len(done_cells)} labeled/queued cells"
          f"{', pitchers with a label' if args.new_pitchers_only else ''})")

    def tier(c: dict) -> int:
        inn = c.get("inning")
        return 2 if inn is None else (0 if inn <= 3 else 1 if inn <= 5 else 2)

    parks = {c["park"] for c in pool}
    cap = max(6, math.ceil(args.limit / max(1, len(parks))))
    picked: list[dict] = []
    n_park: Counter = Counter()
    n_pitcher: Counter = Counter()
    while len(picked) < args.limit:
        before = len(picked)
        order = sorted(pool, key=lambda c: (tier(c), -c["best_conf"], rng.random()))
        for c in order:
            if len(picked) >= args.limit:
                break
            if c in picked or n_park[c["park"]] >= cap or n_pitcher[c["pitcher"]] >= 1:
                continue
            picked.append(c)
            n_park[c["park"]] += 1
            n_pitcher[c["pitcher"]] += 1
        if len(picked) == before:
            if cap >= args.limit:
                break
            cap += 4  # dead parks left a hole; let good parks fill it
    if not picked:
        raise SystemExit("nothing eligible; run --measure or relax --min-conf")

    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    fields = LABEL_FIELDS + ["season", "inning", "frame_score"]
    todo_rows, prop_rows = [], []
    for c in picked:
        m = frame_meas.get((c["cell_id"], c["best_frame"]))
        if m is None:
            continue
        season = int(c["season"])
        img = cv2.imread(str(frames_dir(season) / c["best_frame"]))
        if img is None:
            continue
        cx = int((float(m["rubber_left"]) + float(m["rubber_right"])) / 2)
        cy = int(float(m["rubber_cy"]))
        canvas, x0, y0 = build_crop(img, cx, cy)
        crop_name = f"{c['park']}_{c['cell_id']}.png"
        cv2.imwrite(str(crops_dir / crop_name), canvas)
        row = {k: "" for k in fields}
        row.update({
            "cell_id": c["cell_id"], "game_pk": c["game_pk"], "pitcher": c["pitcher"],
            "stand": c["stand"], "park": c["park"], "p_throws": c["p_throws"],
            "frame_file": c["best_frame"], "crop_file": crop_name,
            "crop_x0": x0, "crop_y0": y0, "zoom": ZOOM, "season": season,
            "inning": "" if c.get("inning") is None else c["inning"],
            "frame_score": f"{c['best_conf']:.3f}",
        })
        todo_rows.append(row)
        prop_rows.append({
            "cell_id": c["cell_id"], "frame_file": c["best_frame"],
            "rubber_left": m["rubber_left"], "rubber_right": m["rubber_right"],
            "rubber_cy": m["rubber_cy"], "foot_x": m["foot_x"],
            "rubber_conf": m["rubber_conf"], "rubber_x_in": m["rubber_x_in"],
        })

    with (out_dir / "labels_todo.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(todo_rows)
    prop_path = out_dir / "proposals.csv"
    if args.blind:
        if prop_path.exists():
            prop_path.unlink()
    else:
        with prop_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(prop_rows[0].keys()))
            w.writeheader()
            w.writerows(prop_rows)

    by_park = Counter(r["park"] for r in todo_rows)
    n_early = sum(1 for r in todo_rows if r["inning"] != "" and int(r["inning"]) <= 3)
    print(f"wrote {len(todo_rows)} cells -> {out_dir}  "
          f"({'BLIND golden set' if args.blind else 'verify pass with proposals'})")
    print(f"  inning<=3: {n_early}/{len(todo_rows)}   "
          f"LHP {sum(1 for r in todo_rows if r['p_throws'] == 'L')}   "
          f"parks: {dict(sorted(by_park.items(), key=lambda kv: -kv[1]))}")
    print(f"\n  python pipeline/rubber_04c_label_server.py --pack {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dataset", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--init", default=None, metavar="WEIGHTS",
                    help="--train: start from these weights (e.g. data/rubber/rubber_det.pt)")
    ap.add_argument("--weights", default=None, metavar="PATH",
                    help="--train: write the detector here instead of overwriting the "
                         "canonical rubber_det.pt; --eval/--measure: use these weights")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--measure", type=int, default=None, metavar="SEASON")
    ap.add_argument("--refilter", type=int, default=None, metavar="SEASON",
                    help="re-apply the frame acceptance gates to an existing "
                         "measurements CSV (no model inference)")
    ap.add_argument("--make-pack", default=None, metavar="DIR",
                    help="write a labeler pack of gate-passing frames to DIR")
    ap.add_argument("--seasons", default="2025,2026", help="for --make-pack")
    ap.add_argument("--blind", action="store_true",
                    help="--make-pack: no proposals.csv (golden set, labeled blind)")
    ap.add_argument("--new-pitchers-only", action="store_true",
                    help="--make-pack: exclude pitchers with any visible label")
    ap.add_argument("--min-conf", type=float, default=MIN_RUBBER_CONF,
                    help="--make-pack: minimum rubber detector confidence")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma list of cell_ids or pitcher ids")
    args = ap.parse_args()
    if not any((args.build_dataset, args.train, args.eval, args.measure,
                args.refilter, args.make_pack)):
        ap.error("pass one of --build-dataset / --train / --eval / --measure SEASON / "
                 "--refilter SEASON / --make-pack DIR")
    if args.build_dataset:
        build_dataset(args)
    if args.train:
        train_detector(args)
    if args.eval:
        evaluate(args)
    if args.measure:
        measure_season(args)
    if args.refilter:
        refilter_season(args)
    if args.make_pack:
        if not args.limit:
            args.limit = 100
        make_pack(args)


if __name__ == "__main__":
    main()

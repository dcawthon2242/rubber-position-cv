#!/usr/bin/env python3
"""Learn rubber left/right and pivot-foot x from the labels we already have.

Classical bar detection clears ~5% of frames. A park-median rubber plus the
04 foot finder is ~14 inches of MAE — worse than the pooled fallback. The
220 visible-rubber labels are enough to train a tiny CNN on the same 440x150
stance crop the labeler already uses. After that, every fetched frame gets a
position without a new click.

Targets are crop-x of rubber-left, rubber-right, and foot center. Inches are
derived the same way as a hand label:
    rubber_x_in = -(foot - rubber_cx) / ((right-left)/24)

Usage:
    python pipeline/rubber_04h_keypoint.py --train
    python pipeline/rubber_04h_keypoint.py --season 2025
    python pipeline/rubber_04h_keypoint.py --season 2026
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBBER_DIR = REPO_ROOT / "data" / "rubber"
WEIGHTS = RUBBER_DIR / "keypoint_model.pt"
CROP_W, CROP_H = 440, 150
RUBBER_WIDTH_IN = 24.0
SEED = 1


def year_of(game_pk: int) -> int:
    if game_pk >= 820000:
        return 2026
    if game_pk >= 770000:
        return 2025
    return 2024


def load_label_rows() -> list[dict]:
    rows: list[dict] = []
    for done in sorted(RUBBER_DIR.glob("label_pack*/labels_done.csv")):
        with done.open(newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    vis = int(float(r.get("rubber_visible") or 0))
                    x0 = float(r["crop_x0"])
                    y0 = float(r.get("crop_y0") or 0)
                except (TypeError, ValueError, KeyError):
                    continue
                rec = {
                    "park": r.get("park", ""),
                    "game_pk": r.get("game_pk", ""),
                    "pitcher": r.get("pitcher", ""),
                    "stand": r.get("stand", ""),
                    "p_throws": r.get("p_throws", ""),
                    "frame_file": r.get("frame_file", ""),
                    "crop_x0": x0, "crop_y0": y0,
                    "visible": vis,
                    "left": None, "right": None, "foot": None,
                }
                if vis == 1:
                    try:
                        left = float(r["rubber_left_px"])
                        right = float(r["rubber_right_px"])
                        foot = float(r["foot_center_px"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if right <= left:
                        continue
                    rec.update(left=left, right=right, foot=foot)
                rows.append(rec)
    return rows


def read_gray(frame_file: str, game_pk: str) -> np.ndarray | None:
    try:
        season = year_of(int(game_pk))
    except (TypeError, ValueError):
        return None
    path = RUBBER_DIR / "frames" / str(season) / frame_file
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def take_crop(img: np.ndarray, x0: int, y0: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, min(w - CROP_W, x0))
    y0 = max(0, min(h - CROP_H, y0))
    crop = img[y0:y0 + CROP_H, x0:x0 + CROP_W]
    if crop.shape != (CROP_H, CROP_W):
        crop = cv2.copyMakeBorder(
            crop, 0, CROP_H - crop.shape[0], 0, CROP_W - crop.shape[1],
            cv2.BORDER_CONSTANT, value=0,
        )
    return crop


class CropNet(nn.Module):
    """Keep horizontal resolution; 3 keypoints are expected x of a heatmap.

    Adaptive pooling to a handful of bins was throwing away the only axis we
    care about. After a few stride-2 convs we average out y, put a 3-channel
    1-d heatmap on x, and take the softmax expectation. vis is a separate logit.
    Output is (left_frac, right_frac, foot_frac, vis_logit), same as before.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=(2, 2), padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=(2, 1), padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, 3, stride=(2, 1), padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, stride=(2, 1), padding=1), nn.ReLU(inplace=True),
        )
        self.heat = nn.Conv2d(48, 3, 1)
        self.vis = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        heat = self.heat(feat).mean(dim=2)  # B,3,W'
        prob = torch.softmax(heat, dim=-1)
        bins = heat.shape[-1]
        grid = torch.linspace(0.0, 1.0, bins, device=x.device).view(1, 1, bins)
        xs = (prob * grid).sum(dim=-1)
        vis = self.vis(feat)
        return torch.cat([xs, vis], dim=1)


class LabelCrops(Dataset):
    def __init__(self, rows: list[dict], augment: bool) -> None:
        self.rows = rows
        self.augment = augment
        self.crops: list[np.ndarray] = []
        self.keep: list[dict] = []
        for r in rows:
            img = read_gray(r["frame_file"], r["game_pk"])
            if img is None:
                continue
            self.crops.append(take_crop(img, int(r["crop_x0"]), int(r["crop_y0"])))
            self.keep.append(r)

    def __len__(self) -> int:
        return len(self.keep)

    def __getitem__(self, i: int):
        crop = self.crops[i].astype(np.float32)
        r = self.keep[i]
        left, right, foot = r["left"], r["right"], r["foot"]
        vis = float(r["visible"] == 1 and left is not None)
        if self.augment:
            if random.random() < 0.7:
                crop = np.clip(crop * random.uniform(0.7, 1.3) + random.uniform(-15, 15), 0, 255)
            if random.random() < 0.4:
                crop = cv2.GaussianBlur(crop, (3, 3), 0)
            shift = random.randint(-16, 16)
            crop = np.roll(crop, shift, axis=1)
            if vis:
                left += shift
                right += shift
                foot += shift
        crop = (crop / 255.0 - 0.5) * 2.0
        x = torch.from_numpy(crop[None])
        # Fractions of crop width; invisible rows get zeros (masked in loss).
        y = torch.tensor([
            (left / CROP_W) if vis else 0.0,
            (right / CROP_W) if vis else 0.0,
            (foot / CROP_W) if vis else 0.0,
            vis,
        ], dtype=torch.float32)
        return x, y


def inches_from_crop(left: float, right: float, foot: float) -> float | None:
    if right <= left + 8:
        return None
    return -(foot - (left + right) / 2.0) / ((right - left) / RUBBER_WIDTH_IN)


def train(args: argparse.Namespace) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = [r for r in load_label_rows() if r["visible"] == 1]
    pitchers = sorted({r["pitcher"] for r in rows})
    rng = random.Random(SEED)
    rng.shuffle(pitchers)
    cut = max(1, int(0.2 * len(pitchers)))
    val_p = set(pitchers[:cut])
    train_rows = [r for r in rows if r["pitcher"] not in val_p]
    val_rows = [r for r in rows if r["pitcher"] in val_p]
    print(f"train {len(train_rows)} labels / {len(pitchers) - len(val_p)} pitchers, "
          f"val {len(val_rows)} / {len(val_p)}  device={device}")

    tr = LabelCrops(train_rows, augment=True)
    va = LabelCrops(val_rows, augment=False)
    print(f"crops on disk: train {len(tr)} val {len(va)}")
    if len(tr) < 20 or len(va) < 5:
        raise SystemExit("not enough labeled frames on disk to train")

    loader = DataLoader(tr, batch_size=16, shuffle=True, drop_last=False)
    model = CropNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_mae, best_state = 1e9, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            vis = yb[:, 3]
            coord = torch.abs(pred[:, :3] - yb[:, :3]).mean(dim=1)
            vis_loss = nn.functional.binary_cross_entropy_with_logits(pred[:, 3], vis)
            loss = (coord * vis).mean() + 0.1 * vis_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(xb)

        model.eval()
        errs: list[float] = []
        with torch.no_grad():
            for i in range(len(va)):
                x, y = va[i]
                pred = model(x.unsqueeze(0).to(device))[0].cpu()
                left, right, foot = (pred[:3] * CROP_W).tolist()
                lab_left, lab_right, lab_foot = (y[:3] * CROP_W).tolist()
                pred_in = inches_from_crop(left, right, foot)
                lab_in = inches_from_crop(lab_left, lab_right, lab_foot)
                if pred_in is None or lab_in is None:
                    continue
                errs.append(abs(pred_in - lab_in))
        mae = float(np.mean(errs)) if errs else 99.0
        print(f"epoch {epoch:3d}  train_loss {running / max(len(tr), 1):.4f}  "
              f"val MAE {mae:.2f} in  n={len(errs)}")
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise SystemExit("training produced no usable checkpoint")

    # Retrain on all visible labels for the production weights, but keep the
    # holdout MAE as the number we report. A second short pass from the best
    # holdout weights is enough; we do not want to forget the regularisation.
    model.load_state_dict(best_state)
    all_ds = LabelCrops(rows, augment=True)
    all_loader = DataLoader(all_ds, batch_size=16, shuffle=True)
    for _ in range(8):
        model.train()
        for xb, yb in all_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            vis = yb[:, 3]
            coord = torch.abs(pred[:, :3] - yb[:, :3]).mean(dim=1)
            loss = (coord * vis).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "val_mae_in": best_mae,
                "n_train": len(tr), "n_val": len(va)}, WEIGHTS)
    print(f"wrote {WEIGHTS}  holdout MAE {best_mae:.2f} in")


def load_model(device: torch.device) -> CropNet:
    if not WEIGHTS.exists():
        raise SystemExit(f"missing {WEIGHTS}; run --train first")
    blob = torch.load(WEIGHTS, map_location=device, weights_only=False)
    model = CropNet().to(device)
    model.load_state_dict(blob["state"])
    model.eval()
    print(f"loaded {WEIGHTS} (holdout MAE {blob.get('val_mae_in', float('nan')):.2f} in)")
    return model


def load_crop_centers(season: int) -> dict[tuple[str, str, str], tuple[int, int]]:
    """Prefer the same crop centre the label pack used."""
    centers: dict[tuple[str, str, str], tuple[int, int]] = {}
    pins: dict[str, tuple[float, float]] = {}
    elig = RUBBER_DIR / "park_eligibility.csv"
    if elig.exists():
        with elig.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("status") != "eligible":
                    continue
                try:
                    pins[r["park"]] = (float(r["mound_fx"]), float(r["mound_fy"]))
                except (TypeError, ValueError):
                    continue

    scores = RUBBER_DIR / f"frame_scores_{season}.csv"
    if scores.exists():
        with scores.open(newline="") as fh:
            for r in csv.DictReader(fh):
                parts = r.get("frame_file", "").replace(".jpg", "").split("_")
                if len(parts) < 3:
                    continue
                try:
                    centers[(parts[0], parts[1], parts[2])] = (
                        int(float(r["crop_cx"])), int(float(r["crop_cy"]))
                    )
                except (TypeError, ValueError, KeyError):
                    continue

    log_csv = RUBBER_DIR / f"frame_fetch_log_{season}.csv"
    motion: dict[tuple[str, str, str], tuple[float, float, float, float]] = {}
    if log_csv.exists():
        with log_csv.open(newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    motion[(r["game_pk"], r["pitcher"], r["stand"])] = (
                        float(r["pitcher_x0"]), float(r["pitcher_y0"]),
                        float(r["pitcher_x1"]), float(r["pitcher_y1"]),
                    )
                except (TypeError, ValueError, KeyError):
                    continue

    park_by_game: dict[str, str] = {}
    pos = RUBBER_DIR / "rubber_position_pitcher_game.csv"
    if pos.exists():
        with pos.open(newline="") as fh:
            for r in csv.DictReader(fh):
                park_by_game[r["game_pk"]] = r["park"]

    return {"centers": centers, "pins": pins, "motion": motion,
            "park": park_by_game}


def crop_origin_for(img: np.ndarray, key: tuple[str, str, str], park: str,
                    aux: dict) -> tuple[int, int]:
    h, w = img.shape[:2]
    if key in aux["centers"]:
        cx, cy = aux["centers"][key]
    elif park in aux["pins"]:
        fx, fy = aux["pins"][park]
        cx, cy = int(fx * w), int(fy * h)
    elif key in aux["motion"]:
        x0, y0, x1, y1 = aux["motion"][key]
        cx, cy = int((x0 + x1) / 2 * w), int(y1 * h)
    else:
        cx, cy = w // 2, int(0.82 * h)
    x0 = max(0, min(w - CROP_W, cx - CROP_W // 2))
    y0 = max(0, min(h - CROP_H, cy - CROP_H // 2))
    return x0, y0


@torch.no_grad()
def infer_season(season: int) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(device)
    aux = load_crop_centers(season)
    frames_dir = RUBBER_DIR / "frames" / str(season)
    out_csv = RUBBER_DIR / f"rubber_measurements_{season}.csv"
    fields = [
        "game_pk", "pitcher", "stand", "park", "p_throws", "frame_file",
        "ok", "reason", "rubber_cx_px", "rubber_cy_px", "rubber_width_px",
        "foot_x_px", "px_per_inch", "rubber_x_in", "scale_source",
        "vis_logit",
    ]
    n_ok = n_tot = 0
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for path in sorted(frames_dir.glob("*_c*.jpg")):
            parts = path.stem.split("_")
            if len(parts) < 4:
                continue
            key = (parts[0], parts[1], parts[2])
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            park = aux["park"].get(key[0], "")
            x0, y0 = crop_origin_for(img, key, park, aux)
            crop = take_crop(img, x0, y0)
            tens = torch.from_numpy(((crop.astype(np.float32) / 255.0 - 0.5) * 2.0)[None, None])
            pred = model(tens.to(device))[0].cpu()
            vis_logit = float(pred[3])
            left, right, foot = (pred[:3] * CROP_W).tolist()
            left_f, right_f, foot_f = x0 + left, x0 + right, x0 + foot
            inches = inches_from_crop(left, right, foot)
            ok = vis_logit > 0 and inches is not None and abs(inches) <= 22
            n_tot += 1
            n_ok += int(ok)
            w.writerow({
                "game_pk": key[0], "pitcher": key[1], "stand": key[2],
                "park": park, "p_throws": "", "frame_file": path.name,
                "ok": ok,
                "reason": "" if ok else ("not visible" if vis_logit <= 0 else "implausible"),
                "rubber_cx_px": (left_f + right_f) / 2.0,
                "rubber_cy_px": y0 + CROP_H * 0.55,
                "rubber_width_px": right_f - left_f,
                "foot_x_px": foot_f,
                "px_per_inch": (right - left) / RUBBER_WIDTH_IN if right > left else "",
                "rubber_x_in": "" if inches is None else f"{inches:.4f}",
                "scale_source": "keypoint",
                "vis_logit": f"{vis_logit:.3f}",
            })
    print(f"ok {n_ok}/{n_tot} ({100 * n_ok / max(n_tot, 1):.1f}%) -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    if args.train:
        train(args)
    if args.season is not None:
        infer_season(args.season)
    if not args.train and args.season is None:
        parser.error("pass --train and/or --season")


if __name__ == "__main__":
    main()

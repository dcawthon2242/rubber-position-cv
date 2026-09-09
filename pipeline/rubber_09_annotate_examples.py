#!/usr/bin/env python3
"""Render the human rubber-position annotations back onto their crop frames.

Every measurement in the pipeline traces to three clicks on one frame: the two
ends of the rubber and the centre of the pivot foot. Those clicks are stored in
`labels_done.csv` as `rubber_left_px`, `rubber_right_px` and `foot_center_px`,
and the offset is

    px_per_inch = (rubber_right_px - rubber_left_px) / 24
    rubber_x_in = (foot_center_px - rubber_centre_px) / px_per_inch

with the sign flipped for the third-base-side camera so that positive is always
toward first base. This script draws those three clicks back on the image so a
measurement can be checked by eye rather than trusted from a number.

The pairing is the point. A pitcher labelled against both batter sides in the
same game produces two frames whose difference is his by-handedness shift, and
seeing them side by side makes clear how small that shift usually is relative to
the width of a shoe.

Note that px_per_inch is derived from the clicked rubber width, so the scale is
self-calibrating per frame -- no assumption about camera distance survives into
the measurement. A miscliked rubber end therefore biases the scale and the
offset together, which is why repeat labels of one frame are the honest estimate
of precision.

Usage: python3 baseball/rubber_09_annotate_examples.py [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from PIL import Image, ImageDraw

RUBBER_IN = 24.0
PACK_GLOB = os.path.join("data", "rubber", "label_pack*")
OUT_DEFAULT = os.path.join("data", "rubber", "annotated_examples")

# Colors chosen to read on dirt: rubber ends in cyan, centre dashed, foot in red.
C_RUBBER = (0, 220, 255)
C_CENTRE = (255, 255, 255)
C_FOOT = (255, 60, 60)
C_TEXT = (255, 255, 255)
C_SHADOW = (0, 0, 0)


def load_labels() -> list[dict]:
    rows: list[dict] = []
    for pack in sorted(glob.glob(PACK_GLOB)):
        f = os.path.join(pack, "labels_done.csv")
        if not os.path.exists(f):
            continue
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                r["pack"] = os.path.basename(pack)
                r["crop_path"] = os.path.join(pack, "crops", r.get("crop_file", ""))
                rows.append(r)
    return rows


def usable(r: dict) -> bool:
    if r.get("rubber_visible") != "1":
        return False
    for k in ("rubber_left_px", "rubber_right_px", "foot_center_px", "rubber_x_in"):
        if not r.get(k):
            return False
    return os.path.exists(r["crop_path"])


def draw_text(d: ImageDraw.ImageDraw, xy, text, fill=C_TEXT):
    x, y = xy
    # Cheap outline so labels stay legible over both dirt and grass.
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        d.text((x + dx, y + dy), text, fill=C_SHADOW)
    d.text((x, y), text, fill=fill)


def annotate(r: dict) -> Image.Image:
    # The crop PNG is already rendered at `zoom` times its source resolution and
    # carries a pixel ruler in that source scale, which is the space the clicks
    # were recorded in. So click coordinates map to PNG pixels by multiplying by
    # zoom exactly once. Resizing the image and scaling the clicks both would
    # apply the factor twice and drop every marker a third of the way left of
    # the rubber, which is what the first version of this did.
    src = Image.open(r["crop_path"]).convert("RGB")
    zoom = int(float(r.get("zoom") or 1))

    xl = float(r["rubber_left_px"]) * zoom
    xr = float(r["rubber_right_px"]) * zoom
    xf = float(r["foot_center_px"]) * zoom
    xc = (xl + xr) / 2.0
    x_in = float(r["rubber_x_in"])
    ppi = (xr - xl) / RUBBER_IN / zoom

    # The full crop is a very wide, short strip, so at chat size the markers are
    # unreadable. Crop HORIZONTALLY around the rubber, with a margin wide enough
    # to keep the foot and both rubber ends in view whatever the offset. Full
    # height is kept deliberately: the rubber's vertical position in the frame
    # varies by park and camera, and a fixed vertical band cut it out of some
    # frames entirely while looking fine on others.
    span = xr - xl
    pad = max(0.7 * span, abs(xf - xc) + 0.35 * span)
    x0 = max(0, int(min(xl, xf) - pad))
    x1 = min(src.width, int(max(xr, xf) + pad))
    im = src.crop((x0, 0, x1, src.height))
    xl, xr, xf, xc = (v - x0 for v in (xl, xr, xf, xc))

    CAPTION_H = 40
    canvas = Image.new("RGB", (im.width, im.height + CAPTION_H), (16, 16, 16))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)

    h = im.height
    # Only the x of each click is recorded, so everything is drawn as a full-height
    # vertical rather than pinned to a guessed rubber height.
    top = h * 0.22
    for x in (xl, xr):
        d.line([(x, top), (x, h)], fill=C_RUBBER, width=2)
    # Rubber centre, dashed so it is not mistaken for a clicked point.
    y = top
    while y < h:
        d.line([(xc, y), (xc, min(y + 6, h))], fill=C_CENTRE, width=1)
        y += 12
    d.line([(xf, top), (xf, h)], fill=C_FOOT, width=3)
    # The measured quantity itself: centre-to-foot, drawn as a bar with end caps.
    yo = h * 0.30
    d.line([(xc, yo), (xf, yo)], fill=C_FOOT, width=3)
    for x in (xc, xf):
        d.line([(x, yo - 4), (x, yo + 4)], fill=C_FOOT, width=3)

    side = "1B" if x_in > 0 else "3B"
    draw_text(d, (8, h + 6),
              f'{r["park"]}  {r["p_throws"]}HP vs {r["stand"]}HH     '
              f'foot centre {abs(x_in):.1f} in toward {side}')
    draw_text(d, (8, h + 22),
              f"clicked rubber width = 24 in  ->  {ppi:.2f} px/in", C_RUBBER)
    draw_text(d, (max(4.0, xf - 14), h * 0.10), "foot", C_FOOT)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [r for r in load_labels() if usable(r)]
    # Prefer the most recent pack when one frame was labelled more than once, so
    # a pitcher does not get two near-identical panels.
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        best[(r["pitcher"], r["stand"])] = r
    # Pair within a GAME, not merely within a pitcher. Two labels of the same
    # pitcher at different parks differ by whatever he does park to park plus any
    # park calibration offset, and that is not a by-handedness shift: pairing
    # McGough's ATH frame against his PIT frame produced an apparent 16.5-inch
    # "shift" that is really a different outing. Only same-game pairs isolate the
    # batter's hand, so cross-game pairs are counted and set aside.
    by_game: dict[tuple[str, str], dict[str, dict]] = {}
    pitchers = set()
    for (p, s), r in best.items():
        pitchers.add(p)
        by_game.setdefault((p, r["game_pk"]), {})[s] = r

    pairs = {k: d for k, d in by_game.items() if {"L", "R"} <= set(d)}
    cross = sum(1 for p in pitchers
                if {"L", "R"} <= {s for (pp, s) in best if pp == p}) - len(pairs)
    print(f"usable annotations: {len(rows)} | pitchers: {len(pitchers)} | "
          f"same-game L/R pairs: {len(pairs)} | cross-game only: {max(0, cross)}")

    n = 0
    for (p, _game), d in sorted(pairs.items()):
        panels = [annotate(d["L"]), annotate(d["R"])]
        gap = 10
        w = sum(im.width for im in panels) + gap
        hgt = max(im.height for im in panels)
        sheet = Image.new("RGB", (w, hgt), (16, 16, 16))
        x = 0
        for im in panels:
            sheet.paste(im, (x, 0))
            x += im.width + gap
        dl, dr = float(d["L"]["rubber_x_in"]), float(d["R"]["rubber_x_in"])
        out = os.path.join(args.out, f"pair_{p}_shift{dl - dr:+.1f}in.png")
        sheet.save(out)
        n += 1
        print(f"  {p}  vs LHH {dl:+6.1f}  vs RHH {dr:+6.1f}  "
              f"shift {dl - dr:+5.1f} in  -> {os.path.basename(out)}")
    print(f"wrote {n} paired panels to {args.out}")


if __name__ == "__main__":
    main()

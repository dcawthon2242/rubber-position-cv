#!/usr/bin/env python3
"""Publication figures for the rubber-position work.

Everything is drawn from the pipeline's own outputs so the figures regenerate
when the data does. Written to data/rubber/figures/ as PNG (200 dpi) and PDF.

    python pipeline/rubber_10_paper_figures.py

Figures
  fig1_method        annotated frame: pose ankles, detected rubber, the measurement
  fig2_validation    blind golden set: CV vs hand label, error distribution
  fig3_parks         per-park measurement yield (camera readability)
  fig4_distribution  league distribution of absolute rubber position by hand
  fig5_movers        pitchers who shift by batter side; camera confirmation
  fig6_ramirez       Ramírez 2024-26 game-by-game position, both sides
  fig7_sweep         Stuff+ response to a lateral shift, by pitch type
  fig8_validation2   natural-experiment slope vs model-implied slope
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[1]
RUB = REPO / "data" / "rubber"
MOD = REPO / "data" / "statcast_model"
OUT = RUB / "figures"
OUT.mkdir(exist_ok=True)

# ---- palette (validated default; see dataviz references/palette.md) ----------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

_fam = [f for f in ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
        if any(f == x.name for x in font_manager.fontManager.ttflist)]
plt.rcParams.update({
    "font.family": _fam[0] if _fam else "sans-serif", "font.size": 9,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 8,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def year_of(pk: int) -> int:
    return 2026 if pk >= 820000 else 2025 if pk >= 770000 else 2024


def rd(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


NAMES = {r["pitcher"]: r["player_name"] for r in rd(RUB / "rubber_position_pitcher_game.csv")}
CELLS = [r for s in (2025, 2026) for r in rd(RUB / f"rubber_pose_cells_{s}.csv")]
MEAS = {s: rd(RUB / f"rubber_pose_measurements_{s}.csv") for s in (2025, 2026)}


# ============================================================ fig 1: method
def fig1_method():
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rubber_04j_pose_measure as M
    # a clean, high-confidence measured frame at a bright park
    cand = [r for r in MEAS[2025] if r["ok"] == "True" and r["park"] in ("BOS", "MIA", "TEX")
            and float(r["rubber_conf"]) > 0.8 and 280 < float(r["pitcher_h"]) < 330
            and abs(float(r["rubber_x_in"])) < 8]
    cand.sort(key=lambda r: -float(r["rubber_conf"]))
    r = cand[0]
    fp = RUB / "frames" / "2025" / r["frame_file"]
    img = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
    models = M.Models(need_det=False)
    people = models.people(cv2.imread(str(fp)))
    H, W = img.shape[:2]
    p, reason, diag = M.gate_pitcher(people, H, W)
    kp, kc = p["kp"], p["kc"]
    x0, y0, x1, y1 = p["box"]
    L, R, cy = float(r["rubber_left"]), float(r["rubber_right"]), float(r["rubber_cy"])
    ax_, ay_ = float(r["ankle_x"]), float(r["ankle_y"])
    foot = float(r["foot_x"])
    ppi = (R - L) / 24
    inches = float(r["rubber_x_in"])

    fig = plt.figure(figsize=(10, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.04)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
    for axx in (a, b):
        axx.grid(False); axx.set_xticks([]); axx.set_yticks([])
        for s in axx.spines.values(): s.set_visible(False)
    a.imshow(img)
    a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=BLUE, lw=1.4))
    # skeleton legs
    for i, j in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12)):
        if kc[i] > 0.3 and kc[j] > 0.3:
            a.plot([kp[i][0], kp[j][0]], [kp[i][1], kp[j][1]], color=BLUE, lw=1.4)
    a.scatter([kp[15][0], kp[16][0]], [kp[15][1], kp[16][1]], s=28, color=BLUE, ec="white", lw=1, zorder=5)
    a.plot([L, R], [cy, cy], color=ORANGE, lw=2.2, solid_capstyle="round")
    # zoom window with the same aspect as the full frame so the panels match in height
    zw = 300; zh = int(zw * H / W)
    zcx = (L + R) / 2; zx0, zx1 = int(zcx - zw / 2), int(zcx + zw / 2)
    zy0, zy1 = int(cy - 0.68 * zh), int(cy + 0.32 * zh)
    a.add_patch(plt.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec="white", lw=1.0, ls=(0, (3, 3))))
    a.text(x0 - 6, y0 + 4, "pitcher\n(lowest full\nfigure with\nboth ankles)", color="white", fontsize=7, va="top", ha="right")
    a.set_title(f"(a) centre-field frame ({r['park']}): pose + detected rubber", loc="left")

    b.imshow(img[zy0:zy1, zx0:zx1], extent=(zx0, zx1, zy1, zy0))
    b.plot([L, R], [cy, cy], color=ORANGE, lw=3, solid_capstyle="round")
    for xx in (L, R):
        b.plot([xx, xx], [cy - 8, cy + 8], color=ORANGE, lw=1.6)
    b.scatter([ax_], [ay_], s=60, color=BLUE, ec="white", lw=1.2, zorder=5)
    b.scatter([foot], [cy], s=60, color=INK, ec="white", lw=1.2, zorder=6, marker="D")
    b.plot([zcx, zcx], [cy - 40, cy + 12], color="white", lw=0.9, ls=(0, (2, 2)))
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
    # scale bar and offset arrow
    b.annotate("", xy=(R, cy + 20), xytext=(L, cy + 20), arrowprops=dict(arrowstyle="<->", color="white", lw=1.0))
    b.text(zcx, cy + 26, f"24 in = {R - L:.0f} px  ({ppi:.2f} px/in)", ha="center", va="top", fontsize=8, color=INK, bbox=box)
    b.annotate("", xy=(foot, cy - 30), xytext=(zcx, cy - 30), arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    b.annotate(f"foot centre: {inches:+.1f} in from rubber centre", xy=(foot, cy - 30), xytext=(zx0 + 8, zy0 + 14),
               fontsize=8.5, color=INK, ha="left", va="top", fontweight="semibold", bbox=box,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    b.annotate("pivot ankle (pose)", xy=(ax_, ay_), xytext=(zx1 - 8, zy0 + 14), fontsize=7.5, color=INK, ha="right", va="top",
               bbox=box, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    b.annotate("detected rubber", xy=(R, cy), xytext=(zx1 - 8, cy + 40), fontsize=7.5, color=INK, ha="right",
               bbox=box, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    b.set_title("(b) the measurement", loc="left")
    fig.text(0.01, -0.02, "Positive = toward first base. Foot centre = pivot-ankle x minus a per-hand constant "
             "(RHP −1.9 in, LHP +3.0 in) fitted on hand labels. Rubber from a YOLO11n detector fine-tuned on 716 labeled bars.",
             fontsize=7.5, color=MUTED)
    save(fig, "fig1_method")


# ============================================================ fig 2: blind validation
def fig2_validation():
    lab = {}
    for r in rd(RUB / "label_pack_golden" / "labels_done.csv"):
        if r.get("rubber_visible", "").strip() in ("1", "1.0") and r.get("rubber_x_in"):
            lab[r["cell_id"]] = (float(r["rubber_x_in"]), r["park"])
    cell = {c["cell_id"]: float(c["rubber_x_in"]) for c in CELLS}
    pts = [(lab[k][0], cell[k], lab[k][1]) for k in lab if k in cell]
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts]); e = y - x
    rmse = np.sqrt(np.mean(e ** 2)); mae = np.abs(e).mean()

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.3))
    lim = (-20, 20)
    a.fill_between(lim, [lim[0] - 3, lim[1] - 3], [lim[0] + 3, lim[1] + 3], color=BLUE, alpha=0.08, lw=0)
    a.plot(lim, lim, color=BASE, lw=1)
    a.scatter(x, y, s=26, color=BLUE, ec="white", lw=0.8, zorder=4)
    a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
    a.set_xlabel("hand label (inches from rubber centre, + = first base)")
    a.set_ylabel("camera measurement (inches)")
    a.set_title(f"(a) blind holdout, {len(x)} cells, {len({p[2] for p in pts})} parks", loc="left")
    a.text(0.03, 0.97, f"RMSE {rmse:.2f} in\nMAE {mae:.2f} in\n{100 * np.mean(np.abs(e) > 3):.0f}% beyond 3 in",
           transform=a.transAxes, va="top", fontsize=9, color=INK)
    a.text(0.97, 0.03, "band = ±3 in", transform=a.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    for xx in (-12, 12):
        a.axvline(xx, color=GRID, lw=0.8); a.axhline(xx, color=GRID, lw=0.8)
    a.text(12.3, 19.5, "rubber edge", fontsize=7, color=MUTED, va="top", rotation=90)

    bins = np.arange(-8, 8.5, 1.0)
    b.hist(e, bins=bins, color=BLUE, ec="white", lw=1.2, rwidth=1.0)
    b.axvline(0, color=BASE, lw=1)
    b.set_xlabel("camera − label (inches)")
    b.set_ylabel("cells")
    b.set_title("(b) error distribution", loc="left")
    b.text(0.97, 0.95, f"target: 3 in RMSE\nachieved: {rmse:.2f} in", transform=b.transAxes, ha="right", va="top", fontsize=9, color=INK)
    fig.text(0.01, -0.03, "Pitchers in the holdout had no prior label and were never seen by the detector; labels were made without the model's output visible. "
             "Label repeatability is ~0.6 in.", fontsize=7.5, color=MUTED)
    save(fig, "fig2_validation")


# ============================================================ fig 3: park yield
def fig3_parks():
    att = Counter(); got = Counter()
    for s in (2025, 2026):
        seen = set()
        for r in MEAS[s]:
            if r["cell_id"] not in seen:
                seen.add(r["cell_id"]); att[r["park"]] += 1
    for c in CELLS:
        got[c["park"]] += 1
    parks = [p for p in att if att[p] >= 10 and p]
    parks.sort(key=lambda p: got[p] / att[p])
    y = np.arange(len(parks)); yl = np.array([got[p] / att[p] for p in parks])
    fig, a = plt.subplots(figsize=(6.2, 7.2))
    cols = [SEQ[1] if v < 0.3 else SEQ[3] if v < 0.6 else SEQ[5] for v in yl]
    a.barh(y, 100 * yl, height=0.62, color=cols)
    for i, p in enumerate(parks):
        a.text(100 * yl[i] + 1, i, f"{100 * yl[i]:.0f}%  (n={att[p]})", va="center", fontsize=7.5, color=INK2)
    a.set_yticks(y); a.set_yticklabels(parks, fontsize=8, color=INK)
    a.set_xlim(0, 108); a.set_xlabel("cells measured / cells with a fetched clip")
    a.grid(axis="y", visible=False)
    a.set_title("Camera readability of the rubber by home park (2025–26)", loc="left")
    a.axvline(30, color=BASE, lw=0.8, ls=(0, (3, 3)))
    a.set_ylim(-1.6, len(parks) - 0.4)
    a.text(30.8, -1.2, "parks below 30% are excluded from clip targeting", fontsize=7.5, color=MUTED, va="center")
    fig.text(0.01, -0.01, "A cell is one pitcher × game × batter side. Failures are dominated by dirt-covered rubbers and camera "
             "angles that never show the mound flat-on, not by the model.", fontsize=7.5, color=MUTED)
    save(fig, "fig3_parks")


# ============================================================ fig 4: league distribution
def fig4_distribution():
    pos = defaultdict(list); hand = {}
    for r in rd(RUB / "rubber_position_pitcher_game.csv"):
        if r["season"] in ("2025", "2026") and r["source"] in ("label", "cv", "anchor") and r["rubber_x_in"] not in ("", "NA"):
            pos[r["pitcher"]].append(float(r["rubber_x_in"])); hand[r["pitcher"]] = r["p_throws"]
    med = {p: np.median(v) for p, v in pos.items() if len(v) >= 3}
    R = np.array([m for p, m in med.items() if hand[p] == "R"]); L = np.array([m for p, m in med.items() if hand[p] == "L"])
    fig, a = plt.subplots(figsize=(8, 4))
    bins = np.arange(-20, 21, 2)
    hR, _ = np.histogram(R, bins=bins); hL, _ = np.histogram(L, bins=bins)
    ctr = (bins[:-1] + bins[1:]) / 2
    a.bar(ctr - 0.48, hR, width=0.9, color=BLUE, label=f"RHP (n={len(R)}, median {np.median(R):+.1f} in)")
    a.bar(ctr + 0.48, hL, width=0.9, color=ORANGE, label=f"LHP (n={len(L)}, median {np.median(L):+.1f} in)")
    for xx, lab_ in ((-12, "rubber edge"), (12, "rubber edge"), (-18, "toe on edge"), (18, "toe on edge")):
        a.axvline(xx, color=BASE if abs(xx) == 12 else GRID, lw=1 if abs(xx) == 12 else 0.8)
    ymax = a.get_ylim()[1]
    a.text(-12, ymax * 0.98, "  3B edge", fontsize=7.5, color=MUTED, va="top")
    a.text(12, ymax * 0.98, "1B edge  ", fontsize=7.5, color=MUTED, va="top", ha="right")
    a.text(-18, ymax * 0.98, "legal limit\n(toe touching) ", fontsize=7, color=MUTED, va="top", ha="right")
    a.set_xlabel("pivot-foot centre, inches from rubber centre (+ = toward first base)")
    a.set_ylabel("pitchers")
    a.set_xlim(-22, 22)
    a.set_title("Where pitchers actually stand: median measured position, 2025–26", loc="left")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.text(0.01, -0.03, f"One value per pitcher (median over ≥3 camera-anchored cells; {len(med)} pitchers). "
             "Both hands favour the glove side: right-handers toward first base, left-handers toward third.", fontsize=7.5, color=MUTED)
    save(fig, "fig4_distribution")


# ============================================================ fig 5: platoon movers
def fig5_movers():
    cells = defaultdict(dict); hand = {}
    for r in rd(RUB / "rubber_position_pitcher_game.csv"):
        if r["season"] in ("2025", "2026") and r["release_pos_x_med"] not in ("", "NA") and int(r["n_fb"]) >= 3:
            cells[(r["pitcher"], r["game_pk"])][r["stand"]] = float(r["release_pos_x_med"]); hand[r["pitcher"]] = r["p_throws"]
    gaps = defaultdict(list)
    for (p, g), d in cells.items():
        if "L" in d and "R" in d:
            gaps[p].append(12 * (d["L"] - d["R"]))
    rows = []
    for p, v in gaps.items():
        v = np.array(v); med = np.median(v)
        if len(v) >= 8 and abs(med) >= 3 and np.mean(np.sign(v) == np.sign(med)) >= 0.75:
            rows.append((p, med, len(v), np.percentile(v, 25), np.percentile(v, 75)))
    # Ramírez: modes alternate, add explicitly from platoon-mode games
    rows.sort(key=lambda t: t[1])
    # camera same-game pairs
    cam = defaultdict(dict)
    for c in CELLS:
        cam[(c["pitcher"], c["game_pk"])][c["stand"]] = float(c["rubber_x_in"])
    camgap = defaultdict(list)
    for (p, g), d in cam.items():
        if "L" in d and "R" in d:
            camgap[p].append(d["L"] - d["R"])
    fig, a = plt.subplots(figsize=(8, 0.34 * len(rows) + 1.6))
    y = np.arange(len(rows))
    for i, (p, med, n, q1, q3) in enumerate(rows):
        a.plot([q1, q3], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        a.scatter([med], [i], s=46, color=BLUE if hand[p] == "R" else ORANGE, ec="white", lw=1, zorder=4)
        if p in camgap:
            for cg in camgap[p]:
                a.scatter([cg], [i], s=70, marker="D", facecolor="none", ec=INK, lw=1.2, zorder=5)
    a.axvline(0, color=BASE, lw=1)
    a.set_yticks(y); a.set_yticklabels([f"{NAMES.get(p, p)}  ({hand[p]}HP, {n} g)" for p, _, n, _, _ in rows], fontsize=8, color=INK)
    a.set_xlabel("in-game shift, position vs LHH minus vs RHH (inches; + = toward the hitter's box)")
    a.grid(axis="y", visible=False)
    a.set_title("Pitchers who set up differently for left- and right-handed hitters (2025–26)", loc="left")
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", ms=7, label="RHP, release-based median"),
                      Line2D([], [], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="white", ms=7, label="LHP, release-based median"),
                      Line2D([], [], marker="D", color="none", markerfacecolor="none", markeredgecolor=INK, ms=7, label="camera, same game both sides"),
                      Line2D([], [], color=GRID, lw=3, label="interquartile range over games")],
             loc="lower right", fontsize=7.5)
    a.set_ylim(-1.2, len(rows) - 0.4)
    a.text(0.6, -0.9, "toward the hitter", fontsize=7.5, color=MUTED, ha="left", va="center")
    a.text(-0.6, -0.9, "away from the hitter", fontsize=7.5, color=MUTED, ha="right", va="center")
    fig.text(0.01, -0.02, "Same pitcher, same game, both sides, so park and mid-season changes cancel. Cut: ≥8 mixed games, |shift| ≥3 in, same direction in ≥75% of games. "
             "Ramírez (alternating platoon/fixed modes) is shown separately in the next figure.", fontsize=7.5, color=MUTED)
    save(fig, "fig5_movers")


# ============================================================ fig 6: Ramírez
def fig6_ramirez():
    pid = "670990"
    # per-pitch files from the analysis scratchpads, else grep the season files
    rows = []
    for y in (2024, 2025, 2026):
        found = None
        for cand in glob.glob(f"/private/tmp/claude-502/*/**/scratchpad/ramirez_{y}.csv", recursive=True):
            found = cand; break
        if found:
            rows += [r | {"season": y} for r in rd(found)]
        else:
            with open(REPO / "data" / f"statcast_{y}" / f"statcast_{y}_all.csv", newline="") as fh:
                for r in csv.DictReader(fh):
                    if r["pitcher"] == pid:
                        rows.append(r | {"season": y})
    rows = [r for r in rows if r.get("release_pos_x") not in ("", "NA", None) and r.get("game_type", "R") == "R"
            and r["game_date"] >= "2024-03-20"]
    games = defaultdict(lambda: defaultdict(list))
    for r in rows:
        games[(r["game_date"], r["game_pk"])][r["stand"]].append(12 * float(r["release_pos_x"]))
    keys = sorted(games)
    xs = np.arange(len(keys))
    fig, a = plt.subplots(figsize=(11, 4.2))
    for side, col, lab_ in (("R", BLUE, "vs RHH"), ("L", ORANGE, "vs LHH")):
        xv = [i for i, k in enumerate(keys) if len(games[k][side]) >= 3]
        yv = [np.median(games[k][side]) for k in keys if len(games[k][side]) >= 3]
        a.scatter(xv, yv, s=26, color=col, ec="white", lw=0.8, label=lab_, zorder=4)
    # season boundaries and annotations
    dates = [k[0] for k in keys]
    for y in (2025, 2026):
        i = next((i for i, d in enumerate(dates) if d >= f"{y}-01-01"), None)
        if i is not None:
            a.axvline(i - 0.5, color=BASE, lw=1)
            a.text(i + 0.5, -46.5, str(y), fontsize=9, color=INK2, fontweight="semibold", va="bottom")
    a.text(0, -46.5, "2024", fontsize=9, color=INK2, fontweight="semibold", va="bottom")
    def mark(date, text, y_from, y_text):
        i = next((i for i, d in enumerate(dates) if d >= date), None)
        if i is None: return
        a.axvline(i - 0.5, color=INK2, lw=0.8, ls=(0, (2, 2)), ymin=0.02, ymax=0.74)
        a.text(i + 1, y_text, text, fontsize=8, color=INK, ha="left", va="top")
    mark("2024-06-15", "first game with LAD:\nwhole-body move, −37 to −23 in", -47, -2.5)
    mark("2025-08-19", "platoon split begins:\nLHH from the 1B end, RHH from the 3B end", -47, -2.5)
    a.set_ylim(-48, 0); a.set_xlim(-1, len(keys))
    a.set_xticks([]); a.set_xlabel("games, in order")
    a.set_ylabel("release point, inches (− = toward third base)")
    for y in (2025, 2026):
        pass
    a.set_title("Yohan Ramírez, 2024–26: one spot for everyone, then two spots by batter side", loc="left")
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.98))
    fig.text(0.01, -0.03, "Per-game medians of Statcast horizontal release point (≥3 pitches). Camera measurements put the LHH spot at about +16 in "
             "(toe past the first-base end) and the RHH spot at −11 to −17 in.", fontsize=7.5, color=MUTED)
    save(fig, "fig6_ramirez")


# ============================================================ fig 7: sweep by pitch type
def fig7_sweep():
    rows = rd(MOD / "rubber_sweep_by_pitchtype.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    order = ["ST", "SL", "CU", "CH", "FC", "FF", "SI"]
    cols = {"ST": BLUE, "SL": "#256abf", "CU": AQUA, "CH": ORANGE, "FC": "#4a3aa7", "FF": MUTED, "SI": BASE}
    for a, hand in zip(axes, ("R", "L")):
        sub = [r for r in rows if r["p_throws"] == hand]
        ends = []
        for pt in order:
            pts = sorted([(float(r["delta_3b_in"]), float(r["d_stuff"])) for r in sub if r["pt"] == pt and abs(float(r["delta_3b_in"])) <= 10])
            if not pts: continue
            x = [p[0] for p in pts]; y = [p[1] for p in pts]
            a.plot(x, y, color=cols[pt], lw=2 if pt in ("ST", "SL", "CU", "CH") else 1.4, solid_capstyle="round")
            ends.append([pt, x[-1], y[-1], y[-1]])
        # end labels: push apart when they collide, connect with a leader
        ends.sort(key=lambda e: e[2])
        for i in range(1, len(ends)):
            if ends[i][3] - ends[i - 1][3] < 0.11:
                ends[i][3] = ends[i - 1][3] + 0.11
        for pt, xe, ye, yl in ends:
            a.plot([xe, xe + 0.5], [ye, yl], color=GRID, lw=0.7)
            a.text(xe + 0.6, yl, pt, fontsize=8, color=INK2, va="center")
        a.axhline(0, color=BASE, lw=1); a.axvline(0, color=GRID, lw=0.8)
        a.set_xlim(-10.5, 12); a.set_xlabel("shift toward third base (inches)")
        a.set_title(f"({'a' if hand == 'R' else 'b'}) {'right' if hand == 'R' else 'left'}-handed pitchers", loc="left")
    axes[0].set_ylabel("Δ Stuff+ (model-implied, 1 pt ≈ 1.5 pp whiff)")
    fig.suptitle("How a lateral shift changes modelled stuff, by pitch type (within ±10 in, where the model has data)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.text(0.01, -0.03, "Sweeping pitches respond; fastballs and sinkers are flat. The natural experiment measures about half the model's slope, so read magnitudes as upper bounds.",
             fontsize=7.5, color=MUTED)
    save(fig, "fig7_sweep")


# ============================================================ fig 8: model vs measured
def fig8_validation2():
    rows = rd(MOD / "rubber_natural_experiment.csv")
    outs = sorted({r["outcome"] for r in rows})
    print("natural-experiment outcomes:", outs)
    pick = [o for o in outs if "miss" in o] or outs[:1]
    sub = [r for r in rows if r["outcome"] == pick[0] and r["regressor"] == "toward_batter_in"]
    fig, a = plt.subplots(figsize=(7.5, 0.5 * len(sub) + 1.8))
    y = np.arange(len(sub))
    for i, r in enumerate(sub):
        a.plot([float(r["ci_lo"]), float(r["ci_hi"])], [i, i], color=BLUE, lw=2, solid_capstyle="round")
        a.scatter([float(r["slope_per_in"])], [i], s=50, color=BLUE, ec="white", lw=1, zorder=4)
    a.axvline(0, color=BASE, lw=1)
    a.axvline(0.00414, color=ORANGE, lw=2)
    a.set_ylim(-0.7, len(sub) - 0.3)
    a.text(0.00414 + 0.00008, -0.55, "model-implied slope +0.0041", color=INK2, fontsize=8, va="center", ha="left")
    pretty = {"none": "no controls", "velo_med": "controls for velocity"}
    a.set_yticks(y); a.set_yticklabels([f"{r['design']}\n({pretty.get(r['controls'], r['controls'])})" for r in sub], fontsize=8, color=INK)
    a.set_xlabel("change in mean miss distance (inches) per inch moved toward the batter, 95% CI")
    a.grid(axis="y", visible=False)
    a.set_title("Does moving actually change outcomes? Measured slope vs the model's prediction", loc="left")
    fig.text(0.01, -0.04, "Within-pitcher designs on pitchers who changed position between games. Every estimate is positive and consistent with the model, "
             "but none is separable from zero: consistent, underpowered.", fontsize=7.5, color=MUTED)
    save(fig, "fig8_validation2")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8"]
    for k in want:
        {"1": fig1_method, "2": fig2_validation, "3": fig3_parks, "4": fig4_distribution,
         "5": fig5_movers, "6": fig6_ramirez, "7": fig7_sweep, "8": fig8_validation2}[k]()

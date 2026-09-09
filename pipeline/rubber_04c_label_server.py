#!/usr/bin/env python3
"""Click-based labeler for rubber position, served to the browser.

Why a browser and not an OpenCV window: the CV venv pins
opencv-python-headless, which ships no GUI backend, so cv2.imshow is
unavailable. Swapping in full opencv-python would risk the libavdevice
collision documented in requirements-cv.txt. A canvas in the browser also
clicks more precisely than a scaled cv2 window and needs no new dependency --
this uses only the standard library plus the cv2/numpy already installed.

What it fixes about the CSV workflow. rubber_04b writes one crop per cell using
the LAST candidate frame, but rubber_03 saves up to three, and roughly a third
of the last-frame crops turn out to be the wrong moment or the wrong camera
entirely. Here every candidate is reachable with the arrow keys, so a cell whose
default crop is junk can usually be rescued by stepping to another frame instead
of being lost. Junk cells get skipped in one keystroke.

Each label is self-calibrating: the rubber is 24 inches wide in the same image,
so clicking its two ends sets pixels-per-inch for that frame and no per-park
camera calibration is needed. The live readout shows the resulting offset in
inches as you click, which is the fastest way to catch a misclick.

Sign convention matches rubber_05_calibrate.R: in the centre-field view,
image-left is the first-base side, while Statcast x is positive toward first
base, so    rubber_x_in = -(foot_center - rubber_center) / px_per_inch.

Contact means any part of the foot touching the rubber, so the foot's centre may
legitimately fall outside the rubber's own 24-inch span -- up to about half a
shoe past either end, since the pivot foot lies lengthwise along it. The live
readout therefore treats offsets past 12 inches as valid but past the end, and
only warns past 18 inches, where no part of the foot could still reach.

Controls:
    click 1     left end of the rubber
    click 2     right end of the rubber
    click 3     centre of the pivot foot
    Enter       save and advance
    s           skip this cell (rubber not identifiable)
    u           undo last click
    r           reset all clicks
    left/right  previous / next candidate frame for this cell
    j / k       previous / next cell
    o           toggle the foot-on-rubber flag (default on)

Progress is written to labels_done.csv after every action, so closing the tab
loses nothing and rerunning resumes where you stopped.

Usage:
    python pipeline/rubber_04c_label_server.py --season 2025
    then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from rubber_04b_label_pack import (LABEL_FIELDS, ZOOM, build_crop,
                                   load_park_pins)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBBER_DIR = REPO_ROOT / "data" / "rubber"
OUT_FIELDS = LABEL_FIELDS + ["cand", "px_per_inch", "rubber_x_in",
                             # verify mode: what the model proposed and whether
                             # the human accepted it untouched, edited it, or
                             # labeled without a proposal
                             "verify", "prop_left_px", "prop_right_px",
                             "prop_foot_px", "prop_conf"]
WIDE_W = 1180                      # full-frame width in the re-centring view

# Typed skips. rubber_04j reads the notes to exclude bad frames from detector
# training and to score its frame gate, so the wording is load-bearing.
SKIP_NOTES = {
    "s": "skipped: rubber not identifiable",
    "w": "skipped: wrong camera",
    "m": "skipped: already moving / mid-delivery",
    "b": "skipped: rubber buried",
}


def year_of(game_pk: int) -> int:
    if game_pk >= 820000:
        return 2026
    if game_pk >= 770000:
        return 2025
    return 2024


def load_priors(seasons: list[int]) -> dict[tuple[str, str, str], list[float]]:
    """Stance bounding boxes from rubber_03's motion detection."""
    priors: dict[tuple[str, str, str], list[float]] = {}
    for season in seasons:
        log_csv = RUBBER_DIR / f"frame_fetch_log_{season}.csv"
        if not log_csv.exists():
            continue
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


class Store:
    """Cell list, candidate frames, and the growing label file."""

    def __init__(self, season: int, todo: Path, out: Path,
                 min_score: float = 0.0, lhp_first: bool = False) -> None:
        self.season = season
        self.out = out
        self.min_score = min_score
        self.lhp_first = lhp_first
        self.pins = load_park_pins()

        with todo.open(newline="") as fh:
            self.cells = list(csv.DictReader(fh))

        seasons = {season}
        for c in self.cells:
            if c.get("season"):
                try:
                    seasons.add(int(c["season"]))
                except ValueError:
                    pass
            else:
                try:
                    seasons.add(year_of(int(c["game_pk"])))
                except (TypeError, ValueError):
                    pass
        self.seasons = sorted(seasons)
        self.priors = load_priors(self.seasons)

        # rubber_04d scores every candidate frame and also proposes a better crop
        # centre than the raw motion prior. Where it exists, candidates are
        # presented best-first, so the default view of each cell is the most
        # promising frame rather than an arbitrary one.
        self.scores: dict[str, list[dict]] = {}
        for yr in self.seasons:
            sf = RUBBER_DIR / f"frame_scores_{yr}.csv"
            if not sf.exists():
                continue
            with sf.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    self.scores.setdefault(row["cell_id"], []).append(row)
        for v in self.scores.values():
            v.sort(key=lambda r: -float(r["score"]))

        # Verify mode. rubber_04j --make-pack writes proposals.csv next to
        # labels_todo.csv: the pose+detector measurement for one frame per cell
        # in full-frame pixels. The labeler pre-places the three clicks from it,
        # so a correct proposal costs one keystroke instead of three clicks, and
        # a wrong one is corrected by clicking the point that is off.
        self.proposals: dict[str, dict] = {}
        prop_path = todo.parent / "proposals.csv"
        if prop_path.exists():
            with prop_path.open(newline="") as fh:
                for r in csv.DictReader(fh):
                    self.proposals[r["cell_id"]] = r

        for c in self.cells:
            key = (c["game_pk"], c["pitcher"], c["stand"])
            scored = self.scores.get(c["cell_id"])
            frames_dir = self.frames_dir_for(c)
            if scored:
                c["_cands"] = [r["frame_file"] for r in scored]
                c["_scored"] = scored
                c["_best"] = float(scored[0]["score"])
            else:
                c["_cands"] = [p.name for p in
                               sorted(frames_dir.glob(f"{c['cell_id']}_c*.jpg"))]
                c["_scored"] = []
                c["_best"] = None
            pr = self.proposals.get(c["cell_id"])
            if pr and pr.get("frame_file") in c["_cands"]:
                # the proposed frame becomes the default view
                if c["_scored"]:
                    c["_scored"].sort(key=lambda r: r["frame_file"] != pr["frame_file"])
                    c["_cands"] = [r["frame_file"] for r in c["_scored"]]
                else:
                    c["_cands"].remove(pr["frame_file"])
                    c["_cands"].insert(0, pr["frame_file"])
            c["_has_prior"] = key in self.priors

        # Pack order is meaningful and is therefore the default. A pack built
        # with rubber_04b --priority is already sorted by how many fastballs each
        # label would upgrade from the pooled fallback to a per-pitcher anchor,
        # so working straight down the file means an interrupted session still
        # bought the most valuable labels available.
        if self.lhp_first:
            # Overrides that order to put left-handers first. This mattered more
            # under the old pooled fit, where a single release_pos_x slope was
            # estimated mostly from right-handers and extrapolated left-handers
            # off the end of the rubber. Anchoring per pitcher removed that
            # failure, so the flag is now a targeting convenience rather than a
            # fix, and it costs pack ordering to use.
            self.cells.sort(key=lambda c: (c.get("p_throws") != "L",
                                           c.get("park", ""), c["cell_id"]))

        # Shared with rubber_04f's triage pass: same file, same meaning, so a
        # frame judged there does not have to be judged again here.
        self.moments_path = RUBBER_DIR / f"frame_moments_{season}.csv"
        self.moments: dict[str, int] = {}
        self.moment_park: dict[str, str] = {}
        if self.moments_path.exists():
            with self.moments_path.open(newline="") as fh:
                for r in csv.DictReader(fh):
                    self.moments[r["frame_file"]] = int(r["toeing"])
                    self.moment_park[r["frame_file"]] = r.get("park", "?")

        self.done: dict[str, dict] = {}
        if out.exists():
            with out.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    self.done[row["cell_id"]] = row

    def frames_dir_for(self, cell: dict) -> Path:
        if cell.get("season"):
            try:
                return RUBBER_DIR / "frames" / str(int(cell["season"]))
            except ValueError:
                pass
        try:
            return RUBBER_DIR / "frames" / str(year_of(int(cell["game_pk"])))
        except (TypeError, ValueError):
            return RUBBER_DIR / "frames" / str(self.season)

    def crop_png(self, cell_id: str, cand: int, fx: float | None = None,
                 fy: float | None = None, wide: bool = False
                 ) -> tuple[bytes, tuple[int, int] | None] | None:
        """PNG bytes plus the crop origin in full-frame pixels (None for wide)."""
        cell = next((c for c in self.cells if c["cell_id"] == cell_id), None)
        if cell is None or not cell["_cands"]:
            return None
        cand = max(0, min(len(cell["_cands"]) - 1, cand))
        frame_name = cell["_cands"][cand]
        img = cv2.imread(str(self.frames_dir_for(cell) / frame_name))
        if img is None:
            return None
        h, w = img.shape[:2]

        if wide:
            # Whole frame, for finding the mound when the park pin misses. The
            # pin is one point from one clip, and a broadcast can zoom between
            # clips at the same park, which leaves the crop on bare grass or
            # below the rubber. Marked not labelable so a click here cannot be
            # mistaken for a measurement.
            scale = WIDE_W / w
            big = cv2.resize(img, (WIDE_W, int(h * scale)),
                             interpolation=cv2.INTER_AREA)
            cell["_last_origin"] = None
            ok, buf = cv2.imencode(".png", big)
            return (buf.tobytes(), None) if ok else None

        scored = cell.get("_scored") or []
        pin = self.pins.get(cell.get("park", ""))
        pr = self.proposals.get(cell_id)
        if fx is not None and fy is not None:
            # Hand re-centre for this cell, which outranks everything else.
            cx, cy = int(fx * w), int(fy * h)
        elif pr is not None and pr.get("frame_file") == frame_name:
            # Centre on the detected rubber so the proposal sits mid-crop.
            cx = int((float(pr["rubber_left"]) + float(pr["rubber_right"])) / 2)
            cy = int(float(pr["rubber_cy"]))
        elif pin is not None:
            # Hand-pinned mound for this park. It outranks the scorer because the
            # camera is fixed for the season, so this centre is right for every
            # clip from the park, while the per-frame estimates were reliably
            # landing on the home-plate circle instead of the mound.
            cx, cy = int(pin["fx"] * w), int(pin["fy"] * h)
        elif cand < len(scored):
            # Scorer-chosen centre: it picks between the motion prior and a
            # dirt-blob estimate, which is what rescues cells whose motion blob
            # was the batter rather than the pitcher.
            cx, cy = int(scored[cand]["crop_cx"]), int(scored[cand]["crop_cy"])
        else:
            key = (cell["game_pk"], cell["pitcher"], cell["stand"])
            if key in self.priors:
                x0f, y0f, x1f, y1f = self.priors[key]
                cx, cy = int((x0f + x1f) / 2 * w), int(y1f * h)
            else:
                cx, cy = w // 2, int(h * 0.72)
        canvas, cx0, cy0 = build_crop(img, cx, cy)
        cell["_last_origin"] = (cx0, cy0)
        ok, buf = cv2.imencode(".png", canvas)
        return (buf.tobytes(), (cx0, cy0)) if ok else None

    def save(self, payload: dict) -> None:
        cid = payload["cell_id"]
        cell = next((c for c in self.cells if c["cell_id"] == cid), None)
        if cell is None:
            return
        cand = int(payload.get("cand", 0))
        frame_file = cell["_cands"][cand] if cell["_cands"] else ""
        origin = cell.get("_last_origin", (cell.get("crop_x0", 0), cell.get("crop_y0", 0)))

        row = {k: cell.get(k, "") for k in LABEL_FIELDS}
        row.update({
            "cell_id": cid, "frame_file": frame_file,
            "crop_x0": origin[0], "crop_y0": origin[1], "zoom": ZOOM,
            "rubber_visible": payload.get("rubber_visible", 0),
            "rubber_left_px": payload.get("rubber_left_px", ""),
            "rubber_right_px": payload.get("rubber_right_px", ""),
            "foot_center_px": payload.get("foot_center_px", ""),
            "foot_on_rubber": payload.get("foot_on_rubber", ""),
            "notes": payload.get("notes", ""),
            "cand": cand,
            "px_per_inch": payload.get("px_per_inch", ""),
            "rubber_x_in": payload.get("rubber_x_in", ""),
            "verify": payload.get("verify", ""),
            "prop_left_px": payload.get("prop_left_px", ""),
            "prop_right_px": payload.get("prop_right_px", ""),
            "prop_foot_px": payload.get("prop_foot_px", ""),
            "prop_conf": payload.get("prop_conf", ""),
        })
        self.done[cid] = row
        self.flush()

    def flush(self) -> None:
        tmp = self.out.with_suffix(".tmp")
        with tmp.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for cell in self.cells:
                if cell["cell_id"] in self.done:
                    writer.writerow(self.done[cell["cell_id"]])
        tmp.replace(self.out)

    def autoskip(self) -> int:
        """Pre-skip cells whose best candidate is below the usability threshold."""
        if self.min_score <= 0:
            return 0
        n = 0
        for c in self.cells:
            if c["cell_id"] in self.done or c["_best"] is None:
                continue
            if c["_best"] < self.min_score:
                self.done[c["cell_id"]] = {
                    **{k: c.get(k, "") for k in LABEL_FIELDS},
                    "cell_id": c["cell_id"], "rubber_visible": 0,
                    "notes": f"auto-skipped: best frame score "
                             f"{c['_best']:.3f} < {self.min_score:.2f}",
                    "cand": "", "px_per_inch": "", "rubber_x_in": "",
                }
                n += 1
        if n:
            self.flush()
        return n

    def save_moment(self, frame_file: str, park: str, toeing) -> None:
        """Record whether the pivot foot is against the rubber in this frame.

        Separate from the position label on purpose. A frame can show the rubber
        perfectly and still be the wrong instant, and conflating the two is what
        produced labels of a stride foot half a stride off the rubber.
        """
        if toeing is None:
            self.moments.pop(frame_file, None)
        else:
            self.moments[frame_file] = int(toeing)
            self.moment_park[frame_file] = park
        tmp = self.moments_path.with_suffix(".tmp")
        with tmp.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["park", "frame_file", "toeing"])
            wr.writeheader()
            for f, t in sorted(self.moments.items()):
                wr.writerow({"park": self.moment_park.get(f, "?"),
                             "frame_file": f, "toeing": t})
        tmp.replace(self.moments_path)

    def manifest(self) -> list[dict]:
        return [{
            "cell_id": c["cell_id"], "park": c["park"], "stand": c["stand"],
            "cand_files": c["_cands"],
            "cand_toeing": [self.moments.get(f) for f in c["_cands"]],
            "p_throws": c.get("p_throws", ""), "pitcher": c["pitcher"],
            "n_cands": len(c["_cands"]), "has_prior": c["_has_prior"],
            "best_score": c["_best"],
            "cand_scores": [round(float(r["score"]), 3) for r in (c.get("_scored") or [])],
            "cand_class": [r["frame_class"] for r in (c.get("_scored") or [])],
            "done": c["cell_id"] in self.done,
            "labeled_visible": self.done.get(c["cell_id"], {}).get("rubber_visible", ""),
            "proposal": self._proposal_json(c),
        } for c in self.cells]

    def _proposal_json(self, c: dict) -> dict | None:
        pr = self.proposals.get(c["cell_id"])
        if not pr or pr.get("frame_file") not in c["_cands"]:
            return None
        try:
            return {
                "frame_file": pr["frame_file"],
                "left": float(pr["rubber_left"]), "right": float(pr["rubber_right"]),
                "foot": float(pr["foot_x"]), "conf": float(pr.get("rubber_conf") or 0),
                "inches": float(pr.get("rubber_x_in") or "nan"),
            }
        except (KeyError, ValueError):
            return None


PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>Rubber position labeler</title>
<style>
 :root{--bg:#12141a;--fg:#e8eaf0;--dim:#8a90a0;--acc:#4ea1ff;--ok:#3ddc84;--warn:#ffb454}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
 header{display:flex;gap:18px;align-items:center;padding:10px 16px;
        background:#191c24;border-bottom:1px solid #262a35;position:sticky;top:0;z-index:5}
 h1{font-size:15px;margin:0;font-weight:600}
 .pill{background:#232734;padding:3px 9px;border-radius:99px;color:var(--dim);font-size:12px}
 .pill b{color:var(--fg)}
 #wrap{padding:16px;display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
 #stage{position:relative;line-height:0;border:1px solid #2b3040;border-radius:6px;
        overflow:hidden;background:#000}
 canvas{display:block;cursor:crosshair}
 aside{min-width:290px;max-width:340px;display:flex;flex-direction:column;gap:12px}
 .card{background:#191c24;border:1px solid #262a35;border-radius:8px;padding:12px 14px}
 .card h2{margin:0 0 8px;font-size:12px;text-transform:uppercase;
          letter-spacing:.08em;color:var(--dim);font-weight:600}
 .row{display:flex;justify-content:space-between;gap:10px;padding:2px 0}
 .row span:first-child{color:var(--dim)}
 .big{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}
 kbd{background:#2a2f3d;border:1px solid #39405219;border-radius:4px;
     padding:1px 6px;font:12px ui-monospace,monospace}
 .step{color:var(--acc);font-weight:600}
 .muted{color:var(--dim)}
 .bar{height:6px;background:#232734;border-radius:99px;overflow:hidden}
 .bar>i{display:block;height:100%;background:var(--ok);width:0}
 button{background:#232734;color:var(--fg);border:1px solid #313747;
        border-radius:6px;padding:6px 10px;cursor:pointer;font:inherit}
 button:hover{border-color:var(--acc)}
 .warn{color:var(--warn)}
</style>
<header>
  <h1>Rubber position labeler</h1>
  <span class="pill">cell <b id="idx">-</b> / <b id="tot">-</b></span>
  <span class="pill" id="parkpill">park <b id="park">-</b></span>
  <span class="pill">frame <b id="cand">-</b>/<b id="ncand">-</b></span>
  <span class="pill" id="scorepill">score <b id="score">-</b> <i id="fclass" style="font-style:normal;color:#8a90a0"></i></span>
  <span class="pill" id="toepill">moment <b id="toe">-</b></span>
  <span class="pill" id="proppill" hidden>proposal <b id="prop">-</b></span>
  <span class="pill">done <b id="ndone">0</b></span>
  <span class="pill">skipped <b id="nskip">0</b></span>
</header>
<div id="wrap">
  <div id="stage"><canvas id="cv"></canvas></div>
  <aside>
    <div class="card">
      <h2>Next click</h2>
      <div class="step" id="step">1 &mdash; left end of the rubber</div>
      <div class="bar" style="margin-top:8px"><i id="prog"></i></div>
    </div>
    <div class="card">
      <h2>Live measurement</h2>
      <div class="row"><span>rubber width</span><span id="w">&mdash;</span></div>
      <div class="row"><span>px per inch</span><span id="ppi">&mdash;</span></div>
      <div class="row"><span>foot on rubber</span><span id="onrub">yes</span></div>
      <div class="row" style="margin-top:6px">
        <span>offset</span><span class="big" id="off">&mdash;</span></div>
      <div class="muted" id="offhint">+ is the first-base side</div>
    </div>
    <div class="card">
      <h2>Keys</h2>
      <div class="row"><span><kbd>Enter</kbd> save + next</span>
                       <span><kbd>s</kbd> skip: can't see rubber</span></div>
      <div class="row"><span><kbd>w</kbd> skip: wrong camera</span>
                       <span><kbd>m</kbd> skip: already moving</span></div>
      <div class="row"><span><kbd>b</kbd> skip: rubber buried</span>
                       <span><kbd>u</kbd> undo &nbsp;<kbd>r</kbd> reset</span></div>
      <div class="row muted" id="prophelp" hidden>
        <span>proposal shown: <kbd>Enter</kbd> accepts; click near a marker to move it</span></div>
      <div class="row"><span><kbd>&larr;</kbd><kbd>&rarr;</kbd> frame</span>
                       <span><kbd>j</kbd><kbd>k</kbd> cell</span></div>
      <div class="row"><span><kbd>o</kbd> toggle contact</span>
                       <span><kbd>t</kbd> moment: toeing?</span></div>
      <div class="row"><span><kbd>z</kbd> wide view / re-centre</span><span></span></div>
      <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
        <button id="jumpnext">First unlabeled</button>
        <button id="jumpgood">Next scoring &ge;0.45</button>
      </div>
    </div>
    <div class="card"><h2>Status</h2><div id="status" class="muted">loading&hellip;</div></div>
  </aside>
</div>
<script>
const ZOOM = __ZOOM__;
let cells = [], i = 0, cand = 0, pts = [], onRubber = true, img = new Image();
let wide = false;
let origin = null;        // crop origin (full-frame px) of the frame on screen
let propPts = null;       // proposal converted to canvas px, or null
let edited = false;       // did the user move any proposed point
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const $ = id => document.getElementById(id);

async function boot(){
  cells = await (await fetch('/api/cells')).json();
  $('tot').textContent = cells.length;
  i = Math.max(0, cells.findIndex(c => !c.done));
  if (i < 0) i = 0;
  load();
}

function load(){
  pts = []; onRubber = true; wide = false;
  const c = cells[i];
  cand = 0;
  $('idx').textContent = i + 1;
  $('park').textContent = c.park + ' ' + c.pitcher + ' vs ' + c.stand + 'HH';
  $('ncand').textContent = c.n_cands;
  if (!c.n_cands){ $('status').innerHTML = '<span class="warn">no candidate frames on disk \u2014 press s to skip</span>'; }
  else { $('status').textContent = c.done ? 'already labeled, re-labeling overwrites' : 'ready'; }
  showFrame();
  tally();
}

function scoreBadge(){
  const c = cells[i];
  const s = (c.cand_scores && c.cand_scores.length > cand) ? c.cand_scores[cand] : null;
  const k = (c.cand_class && c.cand_class.length > cand) ? c.cand_class[cand] : '';
  $('score').textContent = s === null ? '\u2014' : s.toFixed(2);
  $('fclass').textContent = k ? ' ' + k.replace('_',' ') : '';
  // Colour is the whole point of the threshold: red means do not bother
  // labeling this frame, try another candidate or skip.
  const col = s === null ? '#8a90a0'
            : s >= 0.60 ? '#3ddc84' : s >= 0.45 ? '#ffb454' : '#ff6b6b';
  $('score').style.color = col;
  $('scorepill').style.borderLeft = '3px solid ' + col;
  toeBadge();
}

function toeBadge(){
  const c = cells[i];
  const v = (c.cand_toeing && c.cand_toeing.length > cand) ? c.cand_toeing[cand] : null;
  const txt = v === 1 ? 'toeing' : v === 0 ? 'NOT toeing' : 'unmarked';
  const col = v === 1 ? '#3ddc84' : v === 0 ? '#ff6b6b' : '#8a90a0';
  $('toe').textContent = txt; $('toe').style.color = col;
  $('toepill').style.borderLeft = '3px solid ' + col;
}

// Cycles unmarked -> toeing -> not toeing. Recorded per FRAME, not per cell,
// because the whole point is to tell rubber_03 which candidate it should have
// picked.
async function cycleToe(){
  const c = cells[i];
  if (!c.n_cands) return;
  if (!c.cand_toeing) c.cand_toeing = [];
  const cur = c.cand_toeing[cand];
  c.cand_toeing[cand] = (cur === null || cur === undefined) ? 1 : (cur === 1 ? 0 : null);
  toeBadge();
  await fetch('/api/moment', {method:'POST', body: JSON.stringify({
    frame_file: c.cand_files[cand], park: c.park, toeing: c.cand_toeing[cand]})});
}

async function showFrame(){
  const c = cells[i];
  if (!c.n_cands){ cv.width = 900; cv.height = 200; draw(); scoreBadge(); return; }
  $('cand').textContent = cand + 1;
  scoreBadge();
  let u = `/crop?cell_id=${encodeURIComponent(c.cell_id)}&cand=${cand}&t=${Date.now()}`;
  if (wide) u += '&wide=1';
  else if (c._cx !== undefined) u += `&cx=${c._cx}&cy=${c._cy}`;
  // fetch (not <img src>) so the crop origin header is readable; the proposal
  // is stored in full-frame pixels and has to be mapped into this crop.
  const resp = await fetch(u);
  const o = resp.headers.get('X-Crop-Origin');
  origin = o ? o.split(',').map(Number) : null;
  const blob = await resp.blob();
  img = new Image();
  img.onload = () => { cv.width = img.width; cv.height = img.height; applyProposal(); draw(); };
  img.src = URL.createObjectURL(blob);
  $('status').innerHTML = wide
    ? '<span class="warn">wide view \u2014 click the rubber to re-centre, z to go back</span>'
    : (c._cx !== undefined ? 're-centred by hand' : (c.done ? 'already labeled, re-labeling overwrites' : 'ready'));
}

// Pre-place the three clicks from the model's proposal when we are looking at
// the frame it was measured on. Enter then records an accepted proposal; any
// moved marker records an edit. Both go into labels_done.csv so the model's
// hit rate can be read straight off the verify pass.
function applyProposal(){
  const c = cells[i];
  propPts = null; edited = false;
  const p = c.proposal;
  const onPropFrame = p && !wide && origin && c.cand_files[cand] === p.frame_file;
  $('proppill').hidden = !p;
  $('prophelp').hidden = !onPropFrame;
  if (p){
    $('prop').textContent = (isFinite(p.inches) ? (p.inches>=0?'+':'') + p.inches.toFixed(1) + '"' : '?')
                          + '  conf ' + p.conf.toFixed(2);
    $('prop').style.color = p.conf >= 0.6 ? '#3ddc84' : p.conf >= 0.35 ? '#ffb454' : '#ff6b6b';
  }
  if (!onPropFrame) return;
  propPts = [p.left, p.right, p.foot].map(x => (x - origin[0]) * ZOOM);
  if (pts.length === 0) pts = propPts.slice();
  $('status').innerHTML = '<span style="color:#3ddc84">proposal placed</span> \u2014 Enter accepts, click near a marker to move it, w/m/b/s to skip';
}

function draw(){
  ctx.fillStyle = '#000'; ctx.fillRect(0,0,cv.width,cv.height);
  if (img.width) ctx.drawImage(img,0,0);
  const cols = ['#4ea1ff','#4ea1ff','#3ddc84'];
  if (propPts){
    // faint ghost of the original proposal so an edit is visible as a delta
    ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1; ctx.setLineDash([2,4]);
    propPts.forEach(p => { ctx.beginPath(); ctx.moveTo(p,0); ctx.lineTo(p,cv.height); ctx.stroke(); });
    ctx.setLineDash([]);
  }
  pts.forEach((p,k) => {
    ctx.strokeStyle = cols[k]; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(p,0); ctx.lineTo(p,cv.height); ctx.stroke();
    ctx.fillStyle = cols[k]; ctx.fillRect(p-3, cv.height-14, 6, 12);
  });
  if (pts.length >= 2){
    ctx.strokeStyle = '#4ea1ff'; ctx.setLineDash([5,4]); ctx.lineWidth = 1;
    const mid = (pts[0]+pts[1])/2;
    ctx.beginPath(); ctx.moveTo(mid,0); ctx.lineTo(mid,cv.height); ctx.stroke();
    ctx.setLineDash([]);
  }
  readout();
}

function readout(){
  const steps = ['1 \u2014 left end of the rubber',
                 '2 \u2014 right end of the rubber',
                 '3 \u2014 centre of the pivot foot',
                 'ready \u2014 press Enter to save'];
  $('step').textContent = steps[Math.min(3, pts.length)];
  $('prog').style.width = (100*Math.min(3,pts.length)/3) + '%';
  $('onrub').textContent = onRubber ? 'yes' : 'no';
  if (pts.length >= 2){
    const wpx = Math.abs(pts[1]-pts[0]) / ZOOM;
    const ppi = wpx / 24;
    $('w').textContent = wpx.toFixed(1) + ' px';
    $('ppi').textContent = ppi.toFixed(2);
    if (pts.length >= 3){
      const centre = (pts[0]+pts[1])/2;
      const off = -((pts[2]-centre)/ZOOM)/ppi;
      $('off').textContent = (off>=0?'+':'') + off.toFixed(2) + '"';
      // Contact only needs a toe or heel on the rubber, so the foot's centre can
      // legally sit up to half a shoe (~6in) past either 12in end. Only past
      // 18in is the click impossible.
      const a = Math.abs(off);
      $('offhint').textContent = a > 18
        ? 'impossible \u2014 foot cannot reach, check the clicks'
        : a > 12 ? 'past the end but still touching \u2014 valid'
                 : '+ is the first-base side';
      $('offhint').className = a > 18 ? 'warn' : 'muted';
      return;
    }
  } else { $('w').textContent = $('ppi').textContent = '\u2014'; }
  $('off').textContent = '\u2014';
}

cv.addEventListener('click', e => {
  const r = cv.getBoundingClientRect();
  if (wide){
    // Re-centre this cell only. Sent as fractions so the server can map back to
    // the source frame whatever its dimensions.
    const c = cells[i];
    c._cx = ((e.clientX - r.left) / r.width).toFixed(5);
    c._cy = ((e.clientY - r.top) / r.height).toFixed(5);
    wide = false; pts = []; showFrame();
    return;
  }
  const x = (e.clientX - r.left) * (cv.width / r.width);
  if (pts.length >= 3){
    // all three placed (typically a proposal): move whichever marker is nearest
    let k = 0, best = 1e9;
    pts.forEach((p, j) => { const d = Math.abs(p - x); if (d < best){ best = d; k = j; } });
    pts[k] = x; edited = true;
    // keep rubber ends ordered left/right
    if (pts[0] > pts[1]) { const t = pts[0]; pts[0] = pts[1]; pts[1] = t; }
    draw();
    return;
  }
  pts.push(x);
  draw();
});

function tally(){
  $('ndone').textContent = cells.filter(c => c.done && c.labeled_visible != '0').length;
  $('nskip').textContent = cells.filter(c => c.done && c.labeled_visible == '0').length;
}

const SKIP_NOTES = __SKIP_NOTES__;

async function save(skip){
  const c = cells[i];
  let body = { cell_id: c.cell_id, cand: cand, foot_on_rubber: onRubber ? 1 : 0 };
  if (propPts){
    body.prop_left_px = (propPts[0]/ZOOM).toFixed(2);
    body.prop_right_px = (propPts[1]/ZOOM).toFixed(2);
    body.prop_foot_px = (propPts[2]/ZOOM).toFixed(2);
    body.prop_conf = c.proposal ? c.proposal.conf.toFixed(3) : '';
  }
  if (skip || pts.length < 3){
    body.rubber_visible = 0;
    body.notes = skip ? (SKIP_NOTES[skip] || SKIP_NOTES['s']) : 'incomplete';
    body.verify = propPts ? 'rejected' : '';
  } else {
    body.verify = propPts ? (edited ? 'edited' : 'accepted') : 'none';
    const wpx = Math.abs(pts[1]-pts[0]) / ZOOM, ppi = wpx / 24;
    const centre = (pts[0]+pts[1])/2;
    Object.assign(body, {
      rubber_visible: 1,
      rubber_left_px: (Math.min(pts[0],pts[1])/ZOOM).toFixed(2),
      rubber_right_px: (Math.max(pts[0],pts[1])/ZOOM).toFixed(2),
      foot_center_px: (pts[2]/ZOOM).toFixed(2),
      px_per_inch: ppi.toFixed(4),
      rubber_x_in: (-((pts[2]-centre)/ZOOM)/ppi).toFixed(3),
    });
  }
  await fetch('/api/label', {method:'POST', body: JSON.stringify(body)});
  c.done = true; c.labeled_visible = String(body.rubber_visible);
  $('status').textContent = skip ? 'skipped' : 'saved';
  tally();
  if (i < cells.length - 1){ i++; load(); }
  else $('status').textContent = 'all cells visited \u2014 labels_done.csv is complete';
}

document.addEventListener('keydown', e => {
  const c = cells[i];
  if (e.key === 'Enter'){ save(false); e.preventDefault(); }
  else if (e.key in SKIP_NOTES){ save(e.key); }
  else if (e.key === 'u'){ pts.pop(); edited = true; draw(); }
  else if (e.key === 'r'){ pts = []; edited = true; draw(); }
  else if (e.key === 'o'){ onRubber = !onRubber; readout(); }
  else if (e.key === 'ArrowRight'){ if (cand < c.n_cands-1){ cand++; pts=[]; showFrame(); } }
  else if (e.key === 'ArrowLeft'){ if (cand > 0){ cand--; pts=[]; showFrame(); } }
  else if (e.key === 'k'){ if (i < cells.length-1){ i++; load(); } }
  else if (e.key === 'j'){ if (i > 0){ i--; load(); } }
  else if (e.key === 't'){ cycleToe(); }
  else if (e.key === 'z'){ wide = !wide; pts = []; showFrame(); }
});
$('jumpnext').onclick = () => {
  const n = cells.findIndex(c => !c.done);
  if (n >= 0){ i = n; load(); } else $('status').textContent = 'nothing left unlabeled';
};
$('jumpgood').onclick = () => {
  const n = cells.findIndex(c => !c.done && c.best_score !== null && c.best_score >= 0.45);
  if (n >= 0){ i = n; load(); }
  else $('status').textContent = 'no unlabeled cells score 0.45 or better';
};
boot();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    store: Store

    def log_message(self, *a) -> None:  # keep the console quiet
        pass

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/":
            page = (PAGE.replace("__ZOOM__", str(ZOOM))
                        .replace("__SKIP_NOTES__", json.dumps(SKIP_NOTES)))
            self._send(200, "text/html; charset=utf-8", page.encode())
        elif u.path == "/api/cells":
            self._send(200, "application/json",
                       json.dumps(self.store.manifest()).encode())
        elif u.path == "/crop":
            q = parse_qs(u.query)
            cid = q.get("cell_id", [""])[0]
            if not re.fullmatch(r"[0-9]+_[0-9]+_[LR]", cid):
                self._send(400, "text/plain", b"bad cell_id")
                return
            def _frac(name):
                v = q.get(name, [""])[0]
                try:
                    f = float(v)
                except ValueError:
                    return None
                return f if 0.0 <= f <= 1.0 else None

            got = self.store.crop_png(cid, int(q.get("cand", ["0"])[0]),
                                      fx=_frac("cx"), fy=_frac("cy"),
                                      wide=q.get("wide", ["0"])[0] == "1")
            if got is None:
                self._send(404, "text/plain", b"no frame")
            else:
                png, origin = got
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.send_header("Cache-Control", "no-store")
                if origin is not None:
                    self.send_header("X-Crop-Origin", f"{origin[0]},{origin[1]}")
                self.end_headers()
                self.wfile.write(png)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/label", "/api/moment"):
            self._send(404, "text/plain", b"not found")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, "text/plain", b"bad json")
            return
        if path == "/api/moment":
            self.store.save_moment(payload["frame_file"], payload.get("park", "?"),
                                   payload.get("toeing"))
        else:
            self.store.save(payload)
        self._send(200, "application/json", b'{"ok":true}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--todo", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="pre-skip cells whose best candidate scores below this "
                         "(0.30 drops the broadcast-graphic cells; 0 disables)")
    ap.add_argument("--pack", default=None,
                    help="label pack directory (default data/rubber/label_pack)")
    ap.add_argument("--lhp-first", action="store_true",
                    help="order left-handed pitchers first; they are the thin "
                         "part of the calibration anchor")
    args = ap.parse_args()

    pack = Path(args.pack) if args.pack else RUBBER_DIR / "label_pack"
    todo = Path(args.todo) if args.todo else pack / "labels_todo.csv"
    out = Path(args.out) if args.out else pack / "labels_done.csv"
    if not todo.exists():
        raise SystemExit(
            f"missing {todo}; run baseball/rubber_04i_train_pack.py "
            f"(training tags) or baseball/rubber_04b_label_pack.py"
        )

    Handler.store = Store(args.season, todo, out, min_score=args.min_score,
                          lhp_first=args.lhp_first)
    store = Handler.store
    n_cand = sum(1 for c in store.cells if c["_cands"])
    scored = [c["_best"] for c in store.cells if c["_best"] is not None]
    print(f"cells: {len(store.cells)}  with frames on disk: {n_cand}  "
          f"already labeled: {len(store.done)}")
    if store.proposals:
        n_prop = sum(1 for c in store.cells if store._proposal_json(c))
        print(f"verify mode: proposals for {n_prop} cells (Enter accepts, click moves a marker)")
    if scored:
        strong = sum(1 for v in scored if v >= 0.60)
        usable = sum(1 for v in scored if 0.45 <= v < 0.60)
        print(f"frame scores loaded: {strong} strong, {usable} usable, "
              f"{len(scored) - strong - usable} below 0.45")
        n_skip = store.autoskip()
        if n_skip:
            print(f"auto-skipped {n_skip} cells below --min-score {args.min_score}")
    else:
        print("no frame_scores CSV found; run rubber_04d_frame_score.py to enable "
              "scoring and crop re-centring")
    print(f"writing -> {out}")
    url = f"http://127.0.0.1:{args.port}"
    print(f"serving {url}   (ctrl-c to stop)")
    if not args.no_open:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

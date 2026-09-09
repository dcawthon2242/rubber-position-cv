#!/usr/bin/env python3
"""Pull the set-position frame out of each Savant clip in the manifest.

The frame we need is the one where the pitcher is stationary in his stance with
the pivot foot backed lengthwise against the rubber. That is *not* the first
frame of the clip and it is *not* the poster frame: Savant clips open with a
broadcast graphic or sometimes a wide establishing shot, and a mid-clip poster
lands in the follow-through, when the pivot foot has already dragged off the
rubber.

Finding that frame has to survive large differences in framing between parks.
Milwaukee's center-field camera puts the pitcher at x=0.49 of frame width in a
fairly tight shot; Angel Stadium's is offset and zoomed out, putting him near
x=0.33. So nothing here keys off a fixed screen position. Instead we locate the
pitcher by *motion*: summing inter-frame differences across a shot makes the
delivery light up as a single tall blob, which gives both his position and his
apparent size. Apparent size is also what rejects wide establishing shots,
where the pitcher is only a handful of pixels tall.

One decode pass per clip:

  * ffmpeg writes the opening seconds as full-resolution JPEGs to a temp dir.
  * Those are read back, downscaled in memory, and used to pick frames.
  * The chosen frames are moved into place and the rest are discarded.

Full resolution matters because the rubber is only ~98 px wide at 1280x720,
about 4 px per inch, and that is what sets the measurement's precision. Because
clips carry `moov` at the end of the file, ffmpeg range-fetches the index first
and then streams only the leading packets.

We emit up to three candidate frames per cell rather than one. Deciding which
frame truly has the pivot foot against the rubber needs the rubber located at
full resolution, which is rubber_04_measure.py's job, so this stage keeps a
shortlist instead of committing.

Alongside them we write a `_bg.jpg`: the per-pixel median across the probe
window. Over four seconds the pitcher occupies any given spot for well under
half the frames, so the median resolves to an empty field. That single image
solves the two hardest problems downstream. Colour thresholding cannot separate
a pitcher from the mound (it floods three-quarters of the search box), but
differencing against this background gives a clean silhouette. And because the
pitcher is absent from it, the rubber appears unoccluded at its full 24-inch
extent, instead of truncated by the shoe standing on it.

Usage:
    python pipeline/rubber_03_fetch_frames.py --season 2025
    python pipeline/rubber_03_fetch_frames.py --season 2025 --limit 200
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

import cv2
import imageio_ffmpeg
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBBER_DIR = REPO_ROOT / "data" / "rubber"

SPORTY_PAGE = "https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
MP4_RE = re.compile(r"https://sporty-clips\.mlb\.com/[A-Za-z0-9_\-=]+\.mp4")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
# sporty-clips.mlb.com returns 403 without a Savant referer.
REFERER = "https://baseballsavant.mlb.com/"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# How much of the clip to inspect. Clips run ~6.7 s. Four seconds was enough at
# most parks, but the first labeling pass found whole parks -- Detroit, Tampa
# Bay, Seattle, Kansas City -- where every extracted frame was a broadcast
# graphic, because those feeds open with a longer replay wipe or sponsor bumper
# that pushes the delivery past the four-second mark. Probing the full clip costs
# a little more of an already-downloaded file and recovers those parks entirely.
PROBE_SECONDS = 6.6

# A broadcast graphic is not merely un-green, it is off-palette: turf and dirt
# occupy hue 5-95 with real saturation, while wipes and bumpers are saturated
# magenta, cyan, or pure white. Requiring most of the frame to sit in natural
# hues rejects graphics that happen to carry a green-ish brand colour, which the
# grass-fraction test alone let through.
MIN_NATURAL_FRAC = 0.45
# Resolution used for the selection maths only; frames are kept at full size.
WORK_W, WORK_H = 320, 180

# Mean absolute inter-frame difference above this is a broadcast cut.
SCENE_CUT_DIFF = 22.0
# A live field view is mostly grass; graphics and dugout shots are not.
MIN_GREEN_FRAC = 0.20
MIN_SEGMENT_FRAMES = 12

# The motion blob left by a delivery, as fractions of frame size. The height
# floor is what discriminates the real center-field shot from a wide
# establishing shot, where the pitcher is only a few pixels tall.
MIN_PITCHER_H_FRAC = 0.10
MAX_PITCHER_H_FRAC = 0.85
PITCHER_CX_FRAC = (0.12, 0.88)
# The batter's swing also lights up the motion map, and can outweigh the
# delivery, so blob area alone picks the wrong person. In the center-field view
# the pitcher is nearest the camera and therefore the lowest figure in frame,
# which is the discriminator we use. The upper bound keeps the broadcast score
# bug at the very bottom of frame out of contention.
PITCHER_BOTTOM_FRAC = (0.55, 0.93)
# A delivery's motion blob is roughly as tall as it is wide; the score bug and
# other overlay strips are far wider than tall.
MIN_PITCHER_ASPECT = 0.45
MOTION_MAP_PCTL = 96.0

# Motion is "real" once energy exceeds this multiple of the quiet baseline.
MOTION_ONSET_MULT = 3.0
MOTION_ONSET_FLOOR = 1.2
MOTION_SUSTAIN = 3
# Among near-tied quiet frames prefer the latest, i.e. the settled stance.
QUIET_TIE_MULT = 1.5

N_CANDIDATES = 3

# Frames sampled for the median background. More is cleaner but each one is a
# 2.7 MB array, so this caps peak memory per worker.
BG_SAMPLE_FRAMES = 25

MAX_RETRIES = 4
SLEEP_BETWEEN = 0.2


@dataclass
class Result:
    game_pk: int
    pitcher: int
    stand: str
    play_id: str
    status: str
    frame_path: str = ""
    mp4_url: str = ""
    n_frames_written: int = 0
    has_background: bool = False
    candidate_frames: str = ""
    chosen_frame: int = -1
    chosen_time_s: float = float("nan")
    n_probe_frames: int = 0
    n_scene_cuts: int = 0
    segment_start: int = -1
    segment_end: int = -1
    motion_onset: int = -1
    energy_chosen: float = float("nan")
    energy_baseline: float = float("nan")
    green_frac: float = float("nan")
    # Fractional bbox of the pitcher, handed to the measurement stage so it can
    # search the right neighbourhood instead of assuming a fixed layout.
    pitcher_x0: float = float("nan")
    pitcher_y0: float = float("nan")
    pitcher_x1: float = float("nan")
    pitcher_y1: float = float("nan")
    note: str = ""


def resolve_mp4(session: requests.Session, play_id: str) -> str:
    """Scrape the clip page for its mp4 URL."""
    url = SPORTY_PAGE.format(play_id=play_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            match = MP4_RE.search(resp.text)
            return match.group(0) if match else ""
        except Exception:  # noqa: BLE001
            if attempt == MAX_RETRIES:
                return ""
            time.sleep(min(20.0, 2.0**attempt) + random.uniform(0, 0.5))
    return ""


def _headers_arg() -> str:
    return f"Referer: {REFERER}\r\nUser-Agent: {USER_AGENT}\r\n"


def decode_opening(mp4_url: str, work_dir: Path) -> tuple[list[Path], str]:
    """Write the opening PROBE_SECONDS as full-resolution JPEGs."""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-headers", _headers_arg(),
        "-i", mp4_url,
        "-t", str(PROBE_SECONDS),
        "-fps_mode", "passthrough",
        "-q:v", "3",
        "-y", str(work_dir / "f%05d.jpg"),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=240)
    except subprocess.TimeoutExpired:
        return [], "ffmpeg timeout"
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        return [], (err[:300] if err else f"ffmpeg rc={proc.returncode}")
    return sorted(work_dir.glob("f*.jpg")), ""


def median_background(jpegs: list[Path], lo: int, hi: int) -> np.ndarray | None:
    """Per-pixel median over frames in [lo, hi): an empty-field image.

    Callers pass the window starting at motion onset rather than the whole clip.
    A pitcher holds his stance long enough to survive a median taken over
    everything, leaving a ghost exactly where we need to look. Once he has
    started his delivery he is off the rubber and out of the stance position, so
    a median over that window is clean in both places that matter.
    """
    window = jpegs[lo:hi]
    if len(window) < 8:
        window = jpegs
    if len(window) < 5:
        return None

    step = max(1, len(window) // BG_SAMPLE_FRAMES)
    picks = window[::step][:BG_SAMPLE_FRAMES]

    stack = []
    for path in picks:
        img = cv2.imread(str(path))
        if img is not None:
            stack.append(img)
    if len(stack) < 5:
        return None
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def green_fraction(frame_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= 30) & (h <= 95) & (s >= 40) & (v >= 30)
    return float(mask.mean())


def natural_fraction(frame_bgr: np.ndarray) -> float:
    """Share of the frame in turf/dirt hues, i.e. plausibly a real field view."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= 5) & (h <= 95) & (s > 35) & (v > 40)
    return float(mask.mean())


def pitcher_from_motion(gray_seg: np.ndarray) -> tuple[tuple[int, int, int, int] | None,
                                                       np.ndarray | None]:
    """Find the delivery's motion blob within one shot.

    Summing consecutive absolute differences over the shot accumulates a bright
    region wherever something moved a lot; in the live pitch view that is
    overwhelmingly the pitcher. Returns his bounding box and the motion map.
    """
    if gray_seg.shape[0] < 3:
        return None, None
    h, w = gray_seg.shape[1:3]

    motion = np.abs(np.diff(gray_seg, axis=0)).sum(axis=0)
    if motion.max() <= 0:
        return None, None
    motion = (motion / motion.max() * 255.0).astype(np.uint8)

    thresh = max(20.0, float(np.percentile(motion, MOTION_MAP_PCTL)))
    _, binary = cv2.threshold(motion, thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best, best_key = None, (-1.0, -1.0)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        cx_frac = (x + bw / 2.0) / w
        if not (PITCHER_CX_FRAC[0] <= cx_frac <= PITCHER_CX_FRAC[1]):
            continue
        bottom_frac = (y + bh) / h
        if not (PITCHER_BOTTOM_FRAC[0] <= bottom_frac <= PITCHER_BOTTOM_FRAC[1]):
            continue
        h_frac = bh / h
        if not (MIN_PITCHER_H_FRAC <= h_frac <= MAX_PITCHER_H_FRAC):
            continue
        if bh < MIN_PITCHER_ASPECT * bw:
            continue
        # Lowest in frame wins; blob area only breaks ties.
        key = (bottom_frac, float(area))
        if key > best_key:
            best, best_key = (int(x), int(y), int(bw), int(bh)), key

    return best, motion


def choose_frames(work: np.ndarray) -> tuple[list[int], dict]:
    """Return candidate set-position frame indices plus diagnostics.

    Isolate the first shot that is grass-dominated *and* contains a
    pitcher-sized motion blob, then inside it find the delivery's motion onset
    and take frames from the stationary window before it. The primary candidate
    is the latest quiet frame, because that is the settled stance immediately
    before the delivery, when the pivot foot is definitely still on the rubber.
    """
    n, h, w = work.shape[:3]
    fail = {
        "segment_start": -1, "segment_end": -1, "motion_onset": -1,
        "energy_chosen": float("nan"), "energy_baseline": float("nan"),
        "n_scene_cuts": 0, "green_frac": float("nan"),
        "pitcher_x0": float("nan"), "pitcher_y0": float("nan"),
        "pitcher_x1": float("nan"), "pitcher_y1": float("nan"),
        "note": "",
    }
    if n < MIN_SEGMENT_FRAMES:
        fail["note"] = "too few frames decoded"
        return [], fail

    gray = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in work]
    ).astype(np.float32)

    global_diff = np.zeros(n, dtype=np.float32)
    global_diff[1:] = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
    cuts = [i for i in range(1, n) if global_diff[i] > SCENE_CUT_DIFF]
    fail["n_scene_cuts"] = len(cuts)

    greens = np.array([green_fraction(f) for f in work], dtype=np.float32)
    naturals = np.array([natural_fraction(f) for f in work], dtype=np.float32)
    bounds = [0, *cuts, n]

    chosen_seg = None
    bbox = None
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < MIN_SEGMENT_FRAMES:
            continue
        if float(np.median(greens[a:b])) < MIN_GREEN_FRAC:
            continue
        # Off-palette segments are wipes and bumpers, not the field.
        if float(np.median(naturals[a:b])) < MIN_NATURAL_FRAC:
            continue
        cand_bbox, _ = pitcher_from_motion(gray[a:b])
        if cand_bbox is None:
            continue
        chosen_seg, bbox = (a, b), cand_bbox
        break

    if chosen_seg is None:
        fail["green_frac"] = float(np.median(greens))
        fail["note"] = (
            "no center-field pitch segment found"
            if float(np.max(naturals)) >= MIN_NATURAL_FRAC
            else "clip is entirely broadcast graphics"
        )
        return [], fail

    a, b = chosen_seg
    px, py, pw, ph = bbox

    diag = {
        "segment_start": a,
        "segment_end": b,
        "n_scene_cuts": len(cuts),
        "green_frac": float(np.median(greens[a:b])),
        "pitcher_x0": px / w,
        "pitcher_y0": py / h,
        "pitcher_x1": (px + pw) / w,
        "pitcher_y1": (py + ph) / h,
    }

    # Motion energy measured inside the pitcher's own box, so it adapts to each
    # park's framing rather than assuming he is centred.
    roi = gray[:, py : py + ph, px : px + pw]
    energy = np.zeros(n, dtype=np.float32)
    energy[1:] = np.abs(roi[1:] - roi[:-1]).mean(axis=(1, 2))

    seg_idx = np.arange(a + 1, b)  # frame after a cut has a meaningless diff
    if len(seg_idx) < 3:
        diag.update({"motion_onset": -1, "energy_chosen": float("nan"),
                     "energy_baseline": float("nan"), "note": "segment too short"})
        return [a], diag

    seg_energy = energy[seg_idx]
    baseline = float(np.percentile(seg_energy, 33))
    threshold = max(MOTION_ONSET_FLOOR, MOTION_ONSET_MULT * baseline)

    onset = -1
    for k in range(len(seg_idx) - MOTION_SUSTAIN):
        if np.all(seg_energy[k : k + MOTION_SUSTAIN] > threshold):
            onset = int(seg_idx[k])
            break

    if onset > seg_idx[0] + 1:
        cand = np.arange(seg_idx[0], onset)
    else:
        # Cut came in late and the pitcher is already moving; fall back to the
        # calmest part of the front of the shot.
        cutoff = seg_idx[0] + max(3, int(0.6 * len(seg_idx)))
        cand = np.arange(seg_idx[0], min(cutoff, b))

    cand_energy = energy[cand]
    emin = float(cand_energy.min())
    quiet = cand[cand_energy <= max(emin * QUIET_TIE_MULT, emin + 0.25)]
    if len(quiet) == 0:
        quiet = cand[[int(cand_energy.argmin())]]

    primary = int(quiet.max())
    picks = [primary]
    if len(quiet) > 1 and N_CANDIDATES > 1:
        for frac in np.linspace(0.0, 0.6, N_CANDIDATES - 1):
            alt = int(quiet[int(frac * (len(quiet) - 1))])
            if alt not in picks:
                picks.append(alt)
    picks = sorted(set(picks))

    diag.update({
        "motion_onset": onset,
        "energy_chosen": float(energy[primary]),
        "energy_baseline": baseline,
        "note": "" if onset > 0 else "no motion onset detected",
    })
    return picks, diag


def process(session: requests.Session, row: dict, frames_dir: Path,
            overwrite: bool) -> Result:
    game_pk = int(row["game_pk"])
    pitcher = int(row["pitcher"])
    stand = row["stand"]
    play_id = row["play_id"]

    stem = f"{game_pk}_{pitcher}_{stand}"
    res = Result(game_pk=game_pk, pitcher=pitcher, stand=stand, play_id=play_id,
                 status="pending")

    existing = sorted(frames_dir.glob(f"{stem}_c*.jpg"))
    if existing and not overwrite:
        res.status = "cached"
        res.n_frames_written = len(existing)
        res.frame_path = ";".join(p.name for p in existing)
        return res

    mp4_url = resolve_mp4(session, play_id)
    if not mp4_url:
        res.status = "no_mp4"
        return res
    res.mp4_url = mp4_url

    with tempfile.TemporaryDirectory(prefix="rubber_") as tmp:
        work_dir = Path(tmp)
        jpegs, decode_err = decode_opening(mp4_url, work_dir)
        if not jpegs:
            res.status = "decode_failed"
            res.note = decode_err
            return res
        res.n_probe_frames = len(jpegs)

        work = np.zeros((len(jpegs), WORK_H, WORK_W, 3), dtype=np.uint8)
        for i, path in enumerate(jpegs):
            img = cv2.imread(str(path))
            if img is None:
                continue
            work[i] = cv2.resize(img, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

        picks, diag = choose_frames(work)
        for key, val in diag.items():
            setattr(res, key, val)
        if not picks:
            res.status = "no_set_frame"
            return res

        res.chosen_frame = picks[-1]
        res.candidate_frames = ";".join(str(p) for p in picks)
        fps = res.n_probe_frames / PROBE_SECONDS
        res.chosen_time_s = res.chosen_frame / fps if fps > 0 else 0.0

        frames_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for k, idx in enumerate(picks):
            if idx >= len(jpegs):
                continue
            dest = frames_dir / f"{stem}_c{k:02d}.jpg"
            shutil.copyfile(jpegs[idx], dest)
            written.append(dest)

        bg_lo = (res.motion_onset + 3) if res.motion_onset > 0 else res.segment_start
        bg_hi = res.segment_end if res.segment_end > 0 else len(jpegs)
        bg = median_background(jpegs, max(0, bg_lo), bg_hi)
        if bg is not None:
            cv2.imwrite(str(frames_dir / f"{stem}_bg.jpg"), bg,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            res.has_background = True

    if not written:
        res.status = "extract_failed"
        return res

    res.n_frames_written = len(written)
    res.frame_path = ";".join(p.name for p in written)
    res.status = "ok"
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else (
        RUBBER_DIR / f"clip_manifest_{args.season}.csv"
    )
    if not manifest_path.exists():
        raise SystemExit(
            f"missing {manifest_path} - run baseball/rubber_02_select_clips.R first"
        )

    with manifest_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    frames_dir = RUBBER_DIR / "frames" / str(args.season)
    frames_dir.mkdir(parents=True, exist_ok=True)
    log_csv = RUBBER_DIR / f"frame_fetch_log_{args.season}.csv"

    print(f"{len(rows)} manifest rows -> {frames_dir}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Referer": REFERER})

    lock = Lock()
    counts: dict[str, int] = {}
    done = 0

    with log_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(Result.__dataclass_fields__))
        writer.writeheader()

        def work_one(row: dict) -> None:
            nonlocal done
            try:
                res = process(session, row, frames_dir, args.overwrite)
            except Exception as exc:  # noqa: BLE001
                res = Result(
                    game_pk=int(row["game_pk"]), pitcher=int(row["pitcher"]),
                    stand=row["stand"], play_id=row["play_id"],
                    status="error", note=str(exc)[:200],
                )
            time.sleep(SLEEP_BETWEEN)
            with lock:
                writer.writerow(asdict(res))
                counts[res.status] = counts.get(res.status, 0) + 1
                done += 1
                if done % 25 == 0:
                    fh.flush()
                    print(f"  {done}/{len(rows)} {counts}", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work_one, rows))

    print(f"done: {counts}")
    print(f"log -> {log_csv}")


if __name__ == "__main__":
    main()

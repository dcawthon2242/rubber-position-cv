#!/usr/bin/env python3
"""Per-park triage: judge which stadiums show a usable rubber, and pin the mound.

Why this exists. Automatic crop centring keeps failing in a specific way: in the
centre-field shot the camera looks over the pitcher toward the plate, so the
home-plate dirt circle appears ABOVE the mound while matching it on size and
shape. Blob heuristics pick the plate, and the resulting crop shows a catcher
and no rubber. rubber_04d now excludes the plate by vertical position, but the
motion priors it falls back on cover only 66 of 555 fetched cells, so most cells
still have no reliable centre.

The fix is to stop solving it per cell. A broadcast centre-field camera is
bolted in place for the season, so the mound lands in essentially the same spot
in every frame from a given park. One click per park therefore pins the mound
for every cell at that park, and 29 clicks replace a detector that does not
work. The same pass answers the other open question -- which parks show the
rubber at all -- because parks vary from a clean unobstructed bar to one buried
in dirt or hidden behind a sponsor logo.

Full frames are shown rather than tight crops, deliberately. A tight crop can
only be built from a centre estimate, which is the thing being established here,
so showing crops would beg the question.

Two independent judgements are collected, and it matters not to conflate them:

  PARK eligibility -- can this stadium's camera ever show a usable rubber? This
  is a property of the fixed camera and the groundskeeping, so it generalises to
  every clip from the park.

  FRAME moment -- is the pitcher actually toeing the rubber in this particular
  frame? This is a property of when rubber_03 chose to sample, and it varies clip
  to clip at a park that is otherwise perfectly good.

A park can be eligible while most of its frames are mistimed, which is exactly
the failure the earlier labeling rounds hit. Marking the moment per frame here
gives rubber_03's motion-onset detector the ground truth it never had.

Note for the mound click: prefer a frame where the pitcher is NOT on the rubber.
The camera is fixed, so the bar sits in the same pixels either way, and with
nobody standing on it both ends are visible.

Writes data/rubber/park_eligibility.csv:

    park          three-letter park code
    status        eligible | ineligible | unsure   (blank until decided)
    mound_fx      x of the rubber, as a fraction of frame width
    mound_fy      y of the rubber, as a fraction of frame height
    ref_frame     frame the click was made on, for auditing
    n_frames      how many frames exist for the park
    notes         free text

and data/rubber/frame_moments_<season>.csv:

    park          three-letter park code
    frame_file    the candidate frame judged
    toeing        1 if the pivot foot is against the rubber, 0 if not

Downstream: rubber_04b and rubber_04c prefer (mound_fx, mound_fy) over their own
estimates whenever the park has a row here, rubber_05 drops parks whose status is
not `eligible`, and the moment labels both filter candidates and score whether
rubber_03 is picking the right instant.

Usage:
    python pipeline/rubber_04f_park_triage.py --season 2025
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBBER_DIR = REPO_ROOT / "data" / "rubber"

GRID_MAX = 8          # frames shown per park
ZOOM_HALF_W = 110     # half-width, in source px, of the confirmation inset
ZOOM_HALF_H = 42
ZOOM_FACTOR = 4

FIELDS = ["park", "status", "mound_fx", "mound_fy", "ref_frame", "n_frames", "notes"]
MOMENT_FIELDS = ["park", "frame_file", "toeing"]


class Store:
    """Park -> frame list, plus the decisions accumulated so far."""

    def __init__(self, season: int, out: Path, moments: Path | None = None) -> None:
        self.season = season
        self.out = out
        self.moments_path = moments or RUBBER_DIR / f"frame_moments_{season}.csv"
        self.frames_dir = RUBBER_DIR / "frames" / str(season)

        cell_park: dict[str, str] = {}
        man = RUBBER_DIR / f"clip_manifest_{season}.csv"
        if man.exists():
            with man.open(newline="") as fh:
                for r in csv.DictReader(fh):
                    cell_park[f"{r['game_pk']}_{r['pitcher']}_{r['stand']}"] = \
                        r.get("park", "?")

        # Frames whose cell is absent from the current manifest are kept under
        # "?" rather than dropped: they are real frames from earlier selection
        # rounds, and a park with only such frames would otherwise look empty.
        self.by_park: dict[str, list[str]] = {}
        for p in sorted(glob.glob(str(self.frames_dir / "*_c*.jpg"))):
            stem = os.path.basename(p).rsplit("_c", 1)[0]
            self.by_park.setdefault(cell_park.get(stem, "?"), []).append(
                os.path.basename(p))

        self.parks = sorted(k for k in self.by_park if k != "?")
        if "?" in self.by_park:
            self.parks.append("?")

        self.rows: dict[str, dict] = {}
        if out.exists():
            with out.open(newline="") as fh:
                for r in csv.DictReader(fh):
                    self.rows[r["park"]] = r

        self.moments: dict[str, int] = {}
        if self.moments_path.exists():
            with self.moments_path.open(newline="") as fh:
                for r in csv.DictReader(fh):
                    self.moments[r["frame_file"]] = int(r["toeing"])

    def grid(self, park: str) -> list[str]:
        """Evenly spaced sample, so the grid spans games rather than one clip."""
        files = self.by_park.get(park, [])
        if len(files) <= GRID_MAX:
            return files
        step = len(files) / GRID_MAX
        return [files[int(i * step)] for i in range(GRID_MAX)]

    def frame_png(self, name: str, max_w: int = 640) -> bytes | None:
        img = cv2.imread(str(self.frames_dir / name))
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > max_w:
            img = cv2.resize(img, (max_w, int(h * max_w / w)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes() if ok else None

    def zoom_png(self, name: str, fx: float, fy: float) -> bytes | None:
        """Inset at the clicked point, to confirm a rubber is actually there."""
        img = cv2.imread(str(self.frames_dir / name))
        if img is None:
            return None
        h, w = img.shape[:2]
        cx, cy = int(fx * w), int(fy * h)
        x0, x1 = max(0, cx - ZOOM_HALF_W), min(w, cx + ZOOM_HALF_W)
        y0, y1 = max(0, cy - ZOOM_HALF_H), min(h, cy + ZOOM_HALF_H)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        big = cv2.resize(crop, ((x1 - x0) * ZOOM_FACTOR, (y1 - y0) * ZOOM_FACTOR),
                         interpolation=cv2.INTER_NEAREST)
        # Crosshair marks the exact click so a near-miss is visible.
        ccx, ccy = (cx - x0) * ZOOM_FACTOR, (cy - y0) * ZOOM_FACTOR
        cv2.line(big, (ccx - 26, ccy), (ccx + 26, ccy), (0, 255, 255), 1)
        cv2.line(big, (ccx, ccy - 26), (ccx, ccy + 26), (0, 255, 255), 1)
        ok, buf = cv2.imencode(".png", big)
        return buf.tobytes() if ok else None

    def save_moment(self, park: str, frame_file: str, toeing) -> None:
        if toeing is None:
            self.moments.pop(frame_file, None)
        else:
            self.moments[frame_file] = int(toeing)
        self.frame_park = getattr(self, "frame_park", None) or {
            f: p for p, fs in self.by_park.items() for f in fs}
        tmp = self.moments_path.with_suffix(".tmp")
        with tmp.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=MOMENT_FIELDS)
            wr.writeheader()
            for f, t in sorted(self.moments.items()):
                wr.writerow({"park": self.frame_park.get(f, "?"),
                             "frame_file": f, "toeing": t})
        tmp.replace(self.moments_path)

    def save(self, payload: dict) -> None:
        park = payload["park"]
        row = self.rows.get(park, {k: "" for k in FIELDS})
        row["park"] = park
        row["n_frames"] = str(len(self.by_park.get(park, [])))
        for k in ("status", "ref_frame", "notes"):
            if payload.get(k) is not None:
                row[k] = str(payload[k])
        for k in ("mound_fx", "mound_fy"):
            if payload.get(k) is not None:
                row[k] = f"{float(payload[k]):.5f}"
        self.rows[park] = row
        self.flush()

    def flush(self) -> None:
        tmp = self.out.with_suffix(".tmp")
        with tmp.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            for p in self.parks:
                if p in self.rows:
                    wr.writerow({k: self.rows[p].get(k, "") for k in FIELDS})
        tmp.replace(self.out)

    def manifest(self) -> list[dict]:
        out = []
        for p in self.parks:
            r = self.rows.get(p, {})
            frames = self.grid(p)
            out.append({
                "park": p,
                "n_frames": len(self.by_park.get(p, [])),
                "frames": frames,
                "toeing": [self.moments.get(f) for f in frames],
                "status": r.get("status", ""),
                "mound_fx": r.get("mound_fx", ""),
                "mound_fy": r.get("mound_fy", ""),
                "ref_frame": r.get("ref_frame", ""),
                "notes": r.get("notes", ""),
            })
        return out


PAGE = """<!doctype html><meta charset=utf-8><title>Park triage</title>
<style>
 body{background:#14161c;color:#e8e8ea;font:14px/1.45 -apple-system,Segoe UI,sans-serif;margin:0;padding:14px}
 header{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
 h1{font-size:16px;margin:0 12px 0 0}
 button{background:#252a35;color:#e8e8ea;border:1px solid #39404f;border-radius:6px;
        padding:6px 11px;cursor:pointer;font-size:13px}
 button:hover{background:#2f3542}
 button.on{background:#2b6cb0;border-color:#3b82d6}
 .pill{background:#1c212b;border:1px solid #333a48;border-radius:6px;padding:5px 10px}
 .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
 .cell{position:relative;cursor:crosshair;border:2px solid transparent;border-radius:4px;overflow:hidden}
 .cell.sel{border-color:#3ddc84}
 .cell img{width:100%;display:block}
 .cell span{position:absolute;top:3px;left:5px;font-size:11px;color:#ffd866;
            text-shadow:0 0 4px #000}
 .toe{position:absolute;bottom:5px;left:5px;font-size:11px;padding:3px 7px;border-radius:4px;
      border:1px solid #4a5262;background:rgba(12,14,19,.82);color:#c6ccd8;cursor:pointer}
 .toe.yes{background:#1e4d33;border-color:#3ddc84;color:#8ff0b8}
 .toe.no{background:#4d1e1e;border-color:#ff6b6b;color:#ffb3b3}
 .dot{position:absolute;width:11px;height:11px;margin:-6px 0 0 -6px;border-radius:50%;
      border:2px solid #3ddc84;background:rgba(61,220,132,.35);pointer-events:none}
 aside{position:fixed;right:14px;top:14px;width:330px;background:#1a1f28;
       border:1px solid #333a48;border-radius:8px;padding:10px}
 aside img{width:100%;border-radius:4px;background:#000}
 main{margin-right:360px}
 .status{margin-left:auto}
 #list{display:flex;flex-wrap:wrap;gap:4px;margin-top:10px}
 #list b{font-size:11px;padding:3px 6px;border-radius:4px;background:#252a35;cursor:pointer;font-weight:400}
 #list b.el{background:#1e4d33;color:#8ff0b8} #list b.no{background:#4d1e1e;color:#ffb3b3}
 #list b.un{background:#4d431e;color:#f0dc8f} #list b.cur{outline:2px solid #3b82d6}
</style>
<header>
 <h1>Park triage</h1>
 <button id=prev>&larr; prev</button><button id=next>next &rarr;</button>
 <span class=pill>park <b id=park>-</b></span>
 <span class=pill><b id=idx>0</b>/<b id=tot>0</b></span>
 <span class=pill>frames <b id=nfr>0</b></span>
 <span class=pill>mound <b id=mound>not set</b></span>
 <span class=status id=msg></span>
</header>
<main>
 <div class=grid id=grid></div>
 <div id=list></div>
</main>
<aside>
 <div style="margin-bottom:6px;color:#8a90a0">click the rubber on any frame<br>
  <i style="font-style:normal;color:#6d7484">prefer one with nobody standing on it</i></div>
 <img id=zoom src="">
 <div style="margin:9px 0 6px;color:#8a90a0">is the rubber usable at this park?</div>
 <div style="display:flex;gap:6px;flex-wrap:wrap">
  <button id=bel>Eligible (e)</button>
  <button id=bno>Ineligible (i)</button>
  <button id=bun>Unsure (u)</button>
 </div>
 <input id=notes placeholder="notes" style="width:100%;margin-top:8px;background:#12151b;
   color:#e8e8ea;border:1px solid #333a48;border-radius:5px;padding:6px">
</aside>
<script>
let P=[], i=0, sel=null;
const $=id=>document.getElementById(id);
async function boot(){ P=await (await fetch('/api/parks')).json(); $('tot').textContent=P.length; render(); }
function render(){
  const p=P[i];
  $('park').textContent=p.park; $('idx').textContent=i+1; $('nfr').textContent=p.n_frames;
  $('mound').textContent = p.mound_fx ? (+p.mound_fx).toFixed(3)+', '+(+p.mound_fy).toFixed(3) : 'not set';
  $('notes').value=p.notes||'';
  $('bel').className=p.status==='eligible'?'on':''; $('bno').className=p.status==='ineligible'?'on':'';
  $('bun').className=p.status==='unsure'?'on':'';
  const g=$('grid'); g.innerHTML='';
  p.frames.forEach((f,k)=>{
    const d=document.createElement('div'); d.className='cell';
    d.innerHTML='<img src="/img?f='+encodeURIComponent(f)+'"><span>'+f+'</span>';
    d.onclick=ev=>{
      const im=d.querySelector('img'), r=im.getBoundingClientRect();
      const fx=(ev.clientX-r.left)/r.width, fy=(ev.clientY-r.top)/r.height;
      setMound(f,fx,fy,d);
    };
    // Moment toggle. stopPropagation matters: without it the same click would
    // also drop a mound point wherever the badge happens to sit.
    const t=document.createElement('div');
    t.onclick=ev=>{ ev.stopPropagation();
      const cur=p.toeing[k];
      p.toeing[k] = cur===null||cur===undefined ? 1 : (cur===1 ? 0 : null);
      paintToe(t,p.toeing[k]);
      post({park:p.park, frame_file:f, toeing:p.toeing[k]}, '/moment');
    };
    paintToe(t,p.toeing[k]); d.appendChild(t);
    if(p.ref_frame===f && p.mound_fx){ d.classList.add('sel'); sel=d; addDot(d,+p.mound_fx,+p.mound_fy); }
    g.appendChild(d);
  });
  if(p.mound_fx&&p.ref_frame) $('zoom').src='/zoom?f='+encodeURIComponent(p.ref_frame)+'&x='+p.mound_fx+'&y='+p.mound_fy;
  else $('zoom').removeAttribute('src');
  drawList();
}
function paintToe(el,v){
  el.className = 'toe' + (v===1?' yes':v===0?' no':'');
  el.textContent = v===1?'toeing':v===0?'not toeing':'moment?';
}
function addDot(cell,fx,fy){
  const o=cell.querySelector('.dot'); if(o)o.remove();
  const dot=document.createElement('div'); dot.className='dot';
  dot.style.left=(fx*100)+'%'; dot.style.top=(fy*100)+'%'; cell.appendChild(dot);
}
function setMound(f,fx,fy,cell){
  const p=P[i]; p.ref_frame=f; p.mound_fx=fx; p.mound_fy=fy;
  document.querySelectorAll('.cell').forEach(c=>c.classList.remove('sel'));
  cell.classList.add('sel'); addDot(cell,fx,fy);
  $('mound').textContent=fx.toFixed(3)+', '+fy.toFixed(3);
  $('zoom').src='/zoom?f='+encodeURIComponent(f)+'&x='+fx+'&y='+fy;
  post({park:p.park, ref_frame:f, mound_fx:fx, mound_fy:fy});
}
function mark(s){ const p=P[i]; p.status=s; post({park:p.park,status:s,notes:$('notes').value}); render();
  setTimeout(()=>{ if(i<P.length-1){i++;render();} },140); }
async function post(b,path){ await fetch(path||'/save',{method:'POST',body:JSON.stringify(b)});
  $('msg').textContent='saved '+b.park; setTimeout(()=>$('msg').textContent='',900); }
function drawList(){
  const l=$('list'); l.innerHTML='';
  P.forEach((p,k)=>{ const b=document.createElement('b'); b.textContent=p.park;
    if(p.status==='eligible')b.className='el'; else if(p.status==='ineligible')b.className='no';
    else if(p.status==='unsure')b.className='un';
    if(k===i)b.className+=' cur'; b.onclick=()=>{i=k;render();}; l.appendChild(b); });
}
$('prev').onclick=()=>{ if(i>0){i--;render();} };
$('next').onclick=()=>{ if(i<P.length-1){i++;render();} };
$('bel').onclick=()=>mark('eligible'); $('bno').onclick=()=>mark('ineligible');
$('bun').onclick=()=>mark('unsure');
$('notes').onchange=()=>post({park:P[i].park,notes:$('notes').value});
document.onkeydown=ev=>{ if(ev.target.tagName==='INPUT')return;
  if(ev.key==='e')mark('eligible'); else if(ev.key==='i')mark('ineligible');
  else if(ev.key==='u')mark('unsure');
  else if(ev.key==='ArrowLeft'&&i>0){i--;render();}
  else if(ev.key==='ArrowRight'&&i<P.length-1){i++;render();} };
boot();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    store: Store

    def log_message(self, *a) -> None:
        pass

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif u.path == "/api/parks":
            self._send(200, "application/json",
                       json.dumps(self.store.manifest()).encode())
        elif u.path == "/img":
            png = self.store.frame_png(q.get("f", [""])[0])
            self._send(200, "image/png", png) if png else self._send(404, "text/plain", b"")
        elif u.path == "/zoom":
            png = self.store.zoom_png(q.get("f", [""])[0],
                                      float(q.get("x", ["0.5"])[0]),
                                      float(q.get("y", ["0.7"])[0]))
            self._send(200, "image/png", png) if png else self._send(404, "text/plain", b"")
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        if urlparse(self.path).path == "/moment":
            self.store.save_moment(payload["park"], payload["frame_file"],
                                   payload.get("toeing"))
        else:
            self.store.save(payload)
        self._send(200, "application/json", b'{"ok":true}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    out = Path(args.out) if args.out else RUBBER_DIR / "park_eligibility.csv"
    store = Store(args.season, out)
    if not store.parks:
        raise SystemExit(f"no frames under {store.frames_dir}")
    Handler.store = store

    done = sum(1 for p in store.parks if store.rows.get(p, {}).get("status"))
    print(f"{len(store.parks)} parks, {sum(len(v) for v in store.by_park.values())} frames")
    print(f"{done} already decided -> {out}")
    url = f"http://127.0.0.1:{args.port}/"
    print(f"open {url}   (e eligible, i ineligible, u unsure, arrows navigate)")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

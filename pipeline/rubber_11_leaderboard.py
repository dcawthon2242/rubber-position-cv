#!/usr/bin/env python3
"""Write docs/LEADERBOARD.md: who gains most from a rubber adjustment, in depth.

Two lists, because they are different recommendations:
  A. single-spot movers   - pitchers who stand in one place today and whose
                            whole arsenal wants a different single spot
  B. platoon switchers    - pitchers who stand in one place today but whose
                            LHH and RHH optima are different spots (what Strahm
                            and Adams already do)
Pitchers who already shift by batter side are excluded from both and reported
separately, with the measured value of the shift they make.

Inputs (pipeline outputs): rubber_move_gains.csv, rubber_arsenal_detail.csv,
rubber_move_sweep.csv, rubber_position_pitcher_game.csv, rubber_pose_cells_*.csv,
and (optional) per-pitcher season volumes for the tangible projections.

    python pipeline/rubber_11_leaderboard.py [--out docs/LEADERBOARD.md]
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUB = REPO / "data" / "rubber"
MOD = REPO / "data" / "statcast_model"
RES = REPO / "results"

# conversions from the Stuff+ model on 2026 held-out swings (see README)
WHIFF_PER_PT = 0.0148        # whiff probability per swing per +1 Stuff+
KPCT_PER_PT = 0.0128         # K% per +1 Stuff+ (pitcher-season regression)
RUNS_PER_SWING_PT = 0.00207  # runs allowed per swing per +1 Stuff+
RUNS_PER_WAR = 9.7
PA_PER_IP = 4.3
MPH_PER_PT = (1.0, 1.5)      # literature-anchored range; the model's own 1.56 is an upper bound

PT_NAME = {"FF": "four-seam", "SI": "sinker", "FC": "cutter", "SL": "slider", "ST": "sweeper",
           "CU": "curveball", "CH": "changeup", "FS": "splitter", "KC": "knuckle-curve", "SV": "slurve"}


def rd(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def first(*paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    raise FileNotFoundError(paths)


def side_word(delta_in: float) -> str:
    return "toward third base" if delta_in < 0 else "toward first base"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "docs" / "LEADERBOARD.md"))
    ap.add_argument("--volumes", default=None, help="csv: pitcher,year,pa,swings,swings_L,swings_R,k")
    args = ap.parse_args()

    gains = rd(first(MOD / "rubber_move_gains.csv", RES / "rubber_move_gains.csv"))
    ars = rd(first(MOD / "rubber_arsenal_detail.csv", RES / "rubber_arsenal_detail.csv"))
    sweep = rd(first(MOD / "rubber_move_sweep.csv", RES / "rubber_move_sweep.csv"))
    pos = rd(first(RUB / "rubber_position_pitcher_game.csv"))
    cells = [c for s in (2025, 2026) for c in rd(RUB / f"rubber_pose_cells_{s}.csv")] if (RUB / "rubber_pose_cells_2025.csv").exists() else []
    names = {r["pitcher"]: r["player_name"] for r in pos}

    # ---- volumes: from a cached csv, else built from the Statcast season files --
    vol = defaultdict(dict)
    if args.volumes and Path(args.volumes).exists():
        for r in rd(args.volumes):
            vol[r["pitcher"]][int(r["year"])] = {k: float(v) for k, v in r.items() if k not in ("pitcher", "year")}
    else:
        WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
        SWING = WHIFF | {"foul", "hit_into_play", "foul_bunt", "bunt_foul_tip"}
        for y in (2025, 2026):
            f = REPO / "data" / f"statcast_{y}" / f"statcast_{y}_all.csv"
            if not f.exists():
                continue
            acc = defaultdict(Counter)
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("game_type", "R") != "R":
                        continue
                    v = acc[r["pitcher"]]; side = "L" if r["stand"] == "L" else "R"
                    if r["events"] not in ("", "NA"):
                        v["pa"] += 1; v["k"] += r["events"] in ("strikeout", "strikeout_double_play")
                    if r["description"] in SWING:
                        v["swings"] += 1; v["swings_" + side] += 1
            for p, v in acc.items():
                vol[p][y] = {k: float(v[k]) for k in ("pa", "k", "swings", "swings_L", "swings_R")}
            print(f"volumes built from statcast_{y}: {len(acc)} pitchers")

    def season(p):
        c = []
        v25 = vol[p].get(2025); v26 = vol[p].get(2026)
        if v25 and v25["pa"] >= 150: c.append((v25["pa"], v25))
        if v26: c.append((v26["pa"] * 1.33, {k: x * 1.33 for k, x in v26.items()}))
        return max(c, key=lambda t: t[0])[1] if c else None

    # ---- current behaviour: in-game platoon gap, absolute position ----------
    game = defaultdict(dict); hand = {}
    absx = defaultdict(list); src = defaultdict(Counter)
    for r in pos:
        if r["season"] not in ("2025", "2026"):
            continue
        hand[r["pitcher"]] = r["p_throws"]
        if r["release_pos_x_med"] not in ("", "NA") and int(r["n_fb"]) >= 3:
            game[(r["pitcher"], r["game_pk"])][r["stand"]] = float(r["release_pos_x_med"])
        if r["rubber_x_in"] not in ("", "NA"):
            absx[r["pitcher"]].append(float(r["rubber_x_in"])); src[r["pitcher"]][r["source"]] += 1
    gaps = defaultdict(list)
    for (p, g), d in game.items():
        if "L" in d and "R" in d:
            gaps[p].append(12 * (d["L"] - d["R"]))
    def behaviour(p):
        v = np.array(gaps.get(p, []))
        if len(v) < 6:
            return "fixed", float("nan"), len(v)
        med = float(np.median(v)); cons = float(np.mean(np.sign(v) == np.sign(med)))
        if abs(med) >= 3 and cons >= 0.75: return "moves", med, len(v)
        if abs(med) >= 1.5 and cons >= 0.7: return "partial", med, len(v)
        return "fixed", med, len(v)
    manual_movers = {"670990"}  # Ramírez alternates platoon/fixed modes; season median hides it
    ncam = Counter(c["pitcher"] for c in cells)

    def cred(p, rng):
        n = ncam.get(p, 0)
        a = f"{n} camera cell{'s' if n != 1 else ''}"
        if rng <= 6: return f"{a}; move inside a {rng:.0f}-in observed range (dense data) — **credible**"
        if rng <= 10: return f"{a}; {rng:.0f}-in observed range — moderate"
        return f"{a}; {rng:.0f}-in observed range comes from release scatter, largely unconfirmed — **treat with caution**"

    def project(p, g_total, gL=None, gR=None):
        v = season(p)
        if not v: return None
        pa = v["pa"]; ip = pa / PA_PER_IP
        if gL is None:
            runs = RUNS_PER_SWING_PT * g_total * v["swings"]
        else:
            runs = RUNS_PER_SWING_PT * (gL * v.get("swings_L", v["swings"] * 0.4) + gR * v.get("swings_R", v["swings"] * 0.6))
        dk = KPCT_PER_PT * g_total * pa
        return dict(pa=pa, ip=ip, k=v["k"], dk=dk, runs=runs, era=runs / ip * 9, war=runs / RUNS_PER_WAR,
                    kpct=v["k"] / pa, whiff=WHIFF_PER_PT * g_total)

    ars_by = defaultdict(list)
    for r in ars:
        ars_by[r["pitcher"]].append(r)

    # per-side gain at a given shift, from the sweep (the move_gains vs-side
    # columns are for the unconstrained optimum and can exceed the in-support total)
    sweep_at = {(r["pitcher"], int(float(r["delta_in"]))): (float(r["d_stuff_L"]), float(r["d_stuff_R"])) for r in sweep}

    # ---- list A: single-spot movers -------------------------------------------
    A = []
    for r in gains:
        if r["gain_stuff_in_support"] in ("", "NA"):
            continue
        p = r["pitcher"]; beh, gap, ng = behaviour(p)
        if beh == "moves" or p in manual_movers:
            continue
        delta = float(r["best_delta_in_in_support"])
        gL, gR = sweep_at.get((p, int(round(delta))), (float("nan"), float("nan")))
        A.append(dict(p=p, name=r["player_name"], T=r["p_throws"], gain=float(r["gain_stuff_in_support"]),
                      delta=delta, gL=gL, gR=gR, w=float(r["w_lhh"]), rng=float(r["rel_own_p95"]), beh=beh, gap=gap))
    A.sort(key=lambda d: -d["gain"])
    A = [d for d in A if d["gain"] >= 1.0] or A[:12]

    # ---- list B: platoon switchers (per-side optima differ) -------------------
    sw = defaultdict(list)
    for r in sweep:
        if r["in_support"] == "TRUE" and r["feasible"] == "TRUE":
            sw[r["pitcher"]].append(r)
    B = []
    for p, rows in sw.items():
        beh, gap, ng = behaviour(p)
        if beh == "moves" or p in manual_movers:
            continue
        nL, nR = int(rows[0]["n_L"]), int(rows[0]["n_R"])
        if nL < 50 or nR < 50: continue
        w = float(rows[0]["w_lhh"])
        bL = max(rows, key=lambda r: float(r["d_stuff_L"])); bR = max(rows, key=lambda r: float(r["d_stuff_R"]))
        b1 = max(rows, key=lambda r: float(r["d_stuff_wtd"]))
        gL, gR = max(0.0, float(bL["d_stuff_L"])), max(0.0, float(bR["d_stuff_R"]))
        dL, dR = (int(bL["delta_in"]) if gL > 0 else 0), (int(bR["delta_in"]) if gR > 0 else 0)
        if abs(dL - dR) < 6 or min(gL, gR) <= 0.05: continue
        split = w * gL + (1 - w) * gR; single = max(0.0, float(b1["d_stuff_wtd"]))
        B.append(dict(p=p, name=rows[0]["player_name"], T=rows[0]["p_throws"], split=split, single=single, incr=split - single,
                      dL=dL, gL=gL, dR=dR, gR=gR, w=w, rng=float(rows[0]["rel_own_p95"]), beh=beh, gap=gap,
                      pattern="toward the hitter" if dL > dR else "away from the hitter"))
    B.sort(key=lambda d: -d["split"])
    B = B[:12]

    # ---- write ----------------------------------------------------------------
    L = []
    w = L.append
    w("# Leaderboard: who gains most from a rubber adjustment\n")
    w("Generated by `pipeline/rubber_11_leaderboard.py` from the counterfactual sweep (`rubber_07`/`rubber_08`), the camera measurements, "
      "and 2025–26 Statcast. Every gain below is **in-support**: the optimizer may only move a pitcher within the range of positions "
      "the model has actually seen him use, so nothing here is an extrapolation.\n")
    w("## How to read the numbers\n")
    w("| Quantity | Meaning |\n|---|---|")
    w("| **Gain (Stuff+)** | Change in the platoon Stuff+ model's score for the pitcher's whole arsenal at the recommended spot vs where he stands now. Scale: 1 point ≈ 1.5 pp whiff per swing ≈ 1.3 pp K% ≈ −0.0021 runs per swing (all fitted on 2026 held-out swings). |")
    w("| **Move** | Inches from the current spot; negative = toward third base, positive = toward first base. Foot centre may legally sit anywhere within ±18 in of rubber centre (24-in rubber, toe or heel in contact). |")
    w("| **Current position** | Median camera-anchored position, 2025–26, inches from rubber centre (+ = first base). |")
    w(f"| **mph-equivalent** | Gain × {MPH_PER_PT[0]}–{MPH_PER_PT[1]} mph per Stuff+ point (literature-anchored whiff-per-mph; the model's own arsenal-wide figure, 1.56 mph/pt, is an upper bound because it under-weights raw fastball velocity). |")
    w("| **Projection** | Full-season workload (2025 actuals, or 2026-to-date × 1.33 if larger). Extra K = 1.3 pp × gain × PA; runs = 0.0021 × gain × swings; ERA = runs / IP × 9; WAR = runs / 9.7. |")
    w("| **Credibility** | How much of the recommended range the model has real data for, and whether the camera has measured him. |")
    w("\n**Discount everything by about half** if you take the natural experiment at face value: pitchers who actually changed position "
      "realized ~55% of the model-implied slope (+0.0023 ± 0.0014 vs +0.0041 in of miss per inch), with an interval that includes zero.\n")

    # ---------------- A
    w("## A. Single-spot movers — pitchers who should stand somewhere else, for everyone\n")
    w("Excluded: anyone who already shifts by batter side (Strahm, Adams, Ramírez, Kolek, Civale, Greene, Ryan Johnson) — fixing them to one spot would be wrong; see section C.\n")
    w("| # | Pitcher | T | Now | Move | Gain | vs LHH / vs RHH | mph-eq | Season PA | +K | Runs | ERA | WAR | Credibility |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    totA = Counter()
    for i, d in enumerate(A, 1):
        p = d["p"]; cur = np.median(absx[p]) if absx[p] else float("nan")
        pr = project(p, d["gain"])
        mph = f"+{d['gain'] * MPH_PER_PT[0]:.1f}–{d['gain'] * MPH_PER_PT[1]:.1f}"
        proj = (f"{pr['pa']:.0f} | +{pr['dk']:.0f} | {pr['runs']:.1f} | −{pr['era']:.2f} | +{pr['war']:.2f}" if pr else "— | — | — | — | —")
        if pr: totA.update(dk=pr["dk"], runs=pr["runs"], war=pr["war"])
        w(f"| {i} | {d['name']} | {d['T']} | {cur:+.1f} in | {abs(d['delta']):.0f} in toward {'3B' if d['delta'] < 0 else '1B'} | **+{d['gain']:.2f}** | "
          f"{d['gL']:+.2f} / {d['gR']:+.2f} | {mph} | {proj} | {cred(p, d['rng'])} |")
    w("")
    if totA:
        w(f"**Totals across the list: ≈ +{totA['dk']:.0f} strikeouts, ≈ {totA['runs']:.0f} runs saved, ≈ +{totA['war']:.1f} WAR per season.**\n")

    # deep dives A
    w("### In depth\n")
    for d in A[:8]:
        p = d["p"]; cur = np.median(absx[p]) if absx[p] else float("nan")
        rows = sorted(ars_by.get(p, []), key=lambda r: -float(r["wt"]))
        pr = project(p, d["gain"])
        w(f"#### {d['name']} ({d['T']}HP) — +{d['gain']:.2f} Stuff+ from {abs(d['delta']):.0f} in {side_word(d['delta'])}\n")
        bits = []
        if absx[p]:
            s = src[p].most_common(1)[0][0]
            bits.append(f"Stands at **{cur:+.1f} in** today ({'camera-anchored' if s in ('anchor', 'label', 'cv') else 'model-estimated'}), "
                        f"recommended spot ≈ **{cur + d['delta']:+.1f} in**.")
        both = d["gL"] > 0 and d["gR"] > 0
        bits.append(f"At that spot the gain is {d['gL']:+.2f} vs LHH and {d['gR']:+.2f} vs RHH (he faces {100 * d['w']:.0f}% LHH)"
                    + (", so one spot serves both sides." if both else " — the gain is one-sided, so a platoon split (section B) may suit him better."))
        if pr:
            bits.append(f"Over a {pr['pa']:.0f}-PA season that projects to **≈ +{pr['dk']:.0f} K** (K% {100 * pr['kpct']:.1f} → {100 * (pr['kpct'] + KPCT_PER_PT * d['gain']):.1f}), "
                        f"**≈ {pr['runs']:.1f} runs**, ERA −{pr['era']:.2f}, **+{pr['war']:.2f} WAR**; velocity-equivalent ≈ +{d['gain'] * MPH_PER_PT[0]:.1f} to +{d['gain'] * MPH_PER_PT[1]:.1f} mph across the arsenal.")
        w(" ".join(bits) + "\n")
        if rows:
            w("| Pitch | Usage | Gain at the recommended spot | Pitch's own optimum |")
            w("|---|---|---|---|")
            for r in rows[:6]:
                w(f"| {PT_NAME.get(r['pt'], r['pt'])} | {100 * float(r['wt']):.0f}% | {float(r['at_comp']):+.2f} | {float(r['own_best']):+.2f} at {float(r['own_delta']):+.0f} in |")
            top = max(rows, key=lambda r: float(r["at_comp"]) * float(r["wt"]))
            neg = [r for r in rows if float(r["at_comp"]) < -0.3]
            note = f"The gain is the **{PT_NAME.get(top['pt'], top['pt'])}** ({100 * float(top['wt']):.0f}% usage, {float(top['at_comp']):+.2f})"
            if neg:
                note += f"; it costs the {', '.join(PT_NAME.get(r['pt'], r['pt']) for r in neg)} ({', '.join(f'{float(r['at_comp']):+.2f}' for r in neg)}), which caps how far he should go"
            w(f"\n{note}. {cred(p, d['rng']).replace('**', '')}.\n")

    # ---------------- B
    w("## B. Platoon switchers — pitchers who should adopt two spots, like Strahm\n")
    w("Same exclusions as above. Listed only where the LHH and RHH optima are ≥ 6 in apart and each side gains something; "
      "\"vs one spot\" is the part of the gain that comes specifically from splitting rather than from a single better spot.\n")
    w("| # | Pitcher | T | LHH spot | RHH spot | Pattern | Gain | vs one spot | Season PA | +K | Runs | ERA | WAR | Credibility |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    totB = Counter()
    for i, d in enumerate(B, 1):
        p = d["p"]; pr = project(p, d["split"], d["gL"], d["gR"])
        proj = (f"{pr['pa']:.0f} | +{pr['dk']:.0f} | {pr['runs']:.1f} | −{pr['era']:.2f} | +{pr['war']:.2f}" if pr else "— | — | — | — | —")
        if pr: totB.update(dk=pr["dk"], runs=pr["runs"], war=pr["war"])
        w(f"| {i} | {d['name']} | {d['T']} | {d['dL']:+d} in (+{d['gL']:.1f}) | {d['dR']:+d} in (+{d['gR']:.1f}) | {d['pattern']} | **+{d['split']:.2f}** | +{d['incr']:.2f} | {proj} | {cred(p, d['rng'])} |")
    w("")
    if totB:
        w(f"**Totals: ≈ +{totB['dk']:.0f} K, ≈ {totB['runs']:.0f} runs, ≈ +{totB['war']:.1f} WAR per season.** Smaller than list A because each move applies to only part of the batters faced.\n")
    w("### In depth\n")
    for d in B[:5]:
        p = d["p"]; rows = sorted(ars_by.get(p, []), key=lambda r: -float(r["wt"]))
        w(f"#### {d['name']} ({d['T']}HP) — +{d['split']:.2f} Stuff+ from a {abs(d['dL'] - d['dR'])}-in platoon split\n")
        w(f"Model wants **{d['dL']:+d} in vs LHH** (+{d['gL']:.2f}) and **{d['dR']:+d} in vs RHH** (+{d['gR']:.2f}) — a *{d['pattern']}* pattern. "
          f"Splitting is worth +{d['incr']:.2f} over the best single spot (+{d['single']:.2f}). "
          f"Today his in-game shift is {d['gap']:+.1f} in" + (" (essentially none)." if abs(d["gap"]) < 1.5 else " (a partial version already).") + "\n")
        if rows:
            w("| Pitch | Usage | Pitch's own optimum |")
            w("|---|---|---|")
            for r in rows[:5]:
                w(f"| {PT_NAME.get(r['pt'], r['pt'])} | {100 * float(r['wt']):.0f}% | {float(r['own_best']):+.2f} at {float(r['own_delta']):+.0f} in |")
            w("")

    # ---------------- C
    w("## C. Pitchers who already move — what their shift is worth\n")
    w("Undoing the actual in-game shift on their own 2025–26 swings (moving each side's pitches to the other side's spot, recomputing release point and approach angle, rescoring):\n")
    w("| Pitcher | What he does | Platoon spots vs everyone at his RHH spot | vs the midpoint | Mechanism |")
    w("|---|---|---|---|---|")
    w("| Matt Strahm (L) | LHH from the 1B end, RHH from the 3B side, every game (21-in gap, 54/54 games) | **+0.40** Stuff+ | +0.28 | Slider vs LHH gains +2.45 from the 1B end; every pitch vs RHH would lose 1.2–1.8 there. The split is what lets him have both. ≈ +1.3 K, 0.4 runs a season. |")
    w("| Yohan Ramírez (R) | Alternates platoon mode (1B end vs LHH, 3B end vs RHH) with fixed mode (3B end for all) | **−0.22** Stuff+ | +0.45 | Sweeper vs RHH gains +2.26 from the 3B end but sweeper vs LHH loses 1.53 at the 1B end. His best pitch works to both sides from the 3B end — his fixed mode. Results vs LHH were nonetheless better when shifted (wOBA .284 vs .389, ~70 PA each). |")
    w("| Travis Adams (R) | +18 in toward the hitter, 80% of games; camera-confirmed +24 in same game | not separately scored | | |")
    w("\nThe general rule these two illustrate: shifting toward the hitter helps a sweeping pitch against same-side hitters *and* hurts it against opposite-side hitters; fastballs are indifferent. "
      "Whether a pitcher should split depends on whether his best breaking ball is a weapon to both sides (stand where it is best) or only one (split).\n")

    # ---------------- caveats
    w("## Caveats\n")
    w("- **Model-implied, not observed.** The natural experiment recovers ~55% of the model slope with an interval that includes zero. These are rankings of who has the most to gain *if* the mechanism holds.")
    w("- **Wide observed ranges are not all real.** Several 12–20 in ranges come from `release_pos_x` scatter with few camera cells; the names flagged *credible* rank on 3–8 in moves inside dense data.")
    w("- **Stuff is not results.** Location, sequencing and comfort at a new spot are outside the model. Ramírez changed spots and his walk rate rose.")
    w("- **Position and attack plan are confounded** for current movers: pitchers throw different mixes from different spots.")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}  (A: {len(A)} pitchers, B: {len(B)} pitchers)")


if __name__ == "__main__":
    main()

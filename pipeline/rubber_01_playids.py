#!/usr/bin/env python3
"""Harvest per-pitch playId GUIDs so Statcast rows can be linked to video.

Statcast's CSV export has no video handle. The GUID that Baseball Savant's
video player keys on lives in MLB's StatsAPI play-by-play feed, which also
carries the at-bat / pitch ordinals we need to join back to Statcast.

Two endpoints expose the same GUID:

    statsapi.mlb.com/api/v1/game/<pk>/playByPlay     ~566 KB/game
    baseballsavant.mlb.com/gf?game_pk=<pk>          ~2.6 MB/game

They were verified to return byte-identical playId values, so this uses the
StatsAPI one and moves ~4.6x less data.

Join key: StatsAPI numbers at-bats from zero, Statcast from one, so
``at_bat_number = atBatIndex + 1``. Verified against game 776572 /
atBatIndex 3, which lands on the Statcast row for pitcher 669194 vs a
left-handed batter with release_pos_x = -1.28.

Output: data/rubber/play_ids_<season>.csv with one row per pitch. Resumable
at game granularity, so an interrupted run can be restarted safely.

Usage:
    python pipeline/rubber_01_playids.py --season 2025
    python pipeline/rubber_01_playids.py --season 2025 --limit-games 50
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "rubber"

PBP_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

FIELDS = ["game_pk", "at_bat_number", "pitch_number", "play_id"]

# StatsAPI is not documented as rate-limited but is a courtesy-access public
# API, so keep concurrency low and pause between requests.
DEFAULT_WORKERS = 4
SLEEP_BETWEEN = 0.25
MAX_RETRIES = 5


def statcast_game_pks(season: int, game_types: set[str]) -> list[int]:
    """Read the distinct game_pk values for a season out of the Statcast CSV.

    Streams the file rather than loading it; these are ~750k-row CSVs and we
    only need two columns' worth of information.

    Spring training parks use temporary camera positions that do not match the
    regular-season center-field mount, so by default only game_type "R" is
    kept. Including them poisons the rubber measurement.
    """
    path = REPO_ROOT / f"data/statcast_{season}/statcast_{season}_all.csv"
    if not path.exists():
        raise SystemExit(f"missing Statcast season file: {path}")

    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        try:
            pk_idx = header.index("game_pk")
            type_idx = header.index("game_type")
        except ValueError as exc:
            raise SystemExit(f"missing expected column in {path}: {exc}")
        pks = {
            int(row[pk_idx])
            for row in reader
            if row and row[pk_idx] and row[type_idx] in game_types
        }

    return sorted(pks)


def read_wanted_games(path: Path) -> set[int]:
    """game_pks named in a target list, as written by rubber_01b."""
    if not path.exists():
        raise SystemExit(f"--games-from file not found: {path}")
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "game_pk" not in reader.fieldnames:
            raise SystemExit(f"{path} has no game_pk column")
        return {int(row["game_pk"]) for row in reader if row.get("game_pk")}


def already_done(out_csv: Path) -> set[int]:
    """game_pks already present in the output, so reruns skip them."""
    if not out_csv.exists():
        return set()
    done: set[int] = set()
    with out_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                done.add(int(row["game_pk"]))
            except (KeyError, TypeError, ValueError):
                continue
    return done


def fetch_game(session: requests.Session, game_pk: int) -> list[dict]:
    """Return one dict per pitch in the game, or [] if unavailable.

    Retries with exponential backoff and jitter on transport errors and 5xx.
    A 404 means the game has no play-by-play (postponed, spring split squad)
    and is not retried.
    """
    url = PBP_URL.format(game_pk=game_pk)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return []
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as exc:  # noqa: BLE001 - transport/JSON errors alike
            if attempt == MAX_RETRIES:
                print(f"  game {game_pk}: giving up ({exc})", file=sys.stderr)
                return []
            backoff = min(30.0, 2.0**attempt) + random.uniform(0, 1.0)
            time.sleep(backoff)
    else:  # pragma: no cover - loop always breaks or returns
        return []

    rows: list[dict] = []
    for play in payload.get("allPlays", []):
        at_bat_index = play.get("atBatIndex")
        if at_bat_index is None:
            continue
        for event in play.get("playEvents", []):
            if not event.get("isPitch"):
                continue
            play_id = event.get("playId")
            pitch_number = event.get("pitchNumber")
            if not play_id or pitch_number is None:
                continue
            rows.append(
                {
                    "game_pk": game_pk,
                    # StatsAPI is 0-based, Statcast is 1-based.
                    "at_bat_number": at_bat_index + 1,
                    "pitch_number": pitch_number,
                    "play_id": play_id,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--limit-games",
        type=int,
        default=None,
        help=(
            "process only the first N outstanding games. This is a SMOKE TEST "
            "flag: 'first' means lowest game_pk, i.e. chronologically earliest, "
            "so it yields a biased slice of the season rather than a sample. "
            "For a partial harvest use --games-from instead."
        ),
    )
    parser.add_argument(
        "--games-from",
        default=None,
        help=(
            "CSV with a game_pk column listing exactly which games to harvest, "
            "as emitted by rubber_01b_target_games.R. Use this to spend the "
            "fetch on games that unlock unanchored pitchers."
        ),
    )
    parser.add_argument(
        "--game-types",
        default="R",
        help="comma-separated Statcast game_type codes to include (default R)",
    )
    args = parser.parse_args()

    if args.games_from and args.limit_games:
        parser.error("--games-from and --limit-games are mutually exclusive")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"play_ids_{args.season}.csv"

    game_types = {t.strip() for t in args.game_types.split(",") if t.strip()}
    all_pks = statcast_game_pks(args.season, game_types)
    done = already_done(out_csv)
    todo = [pk for pk in all_pks if pk not in done]

    if args.games_from:
        wanted = read_wanted_games(Path(args.games_from))
        unknown = wanted - set(all_pks)
        if unknown:
            print(
                f"  note: {len(unknown)} requested game_pks are not "
                f"{sorted(game_types)} games in season {args.season}; skipping"
            )
        todo = [pk for pk in todo if pk in wanted]
        print(f"targeting {len(wanted)} games from {args.games_from}")
    elif args.limit_games:
        todo = todo[: args.limit_games]

    print(
        f"season {args.season}: {len(all_pks)} games of type "
        f"{sorted(game_types)} in Statcast, {len(done)} already harvested, "
        f"{len(todo)} to fetch"
    )
    if not todo:
        return

    write_header = not out_csv.exists()
    lock = Lock()
    counters = {"games": 0, "pitches": 0, "empty": 0}

    with out_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

        def work(game_pk: int) -> None:
            rows = fetch_game(session, game_pk)
            time.sleep(SLEEP_BETWEEN)
            with lock:
                if rows:
                    writer.writerows(rows)
                    counters["pitches"] += len(rows)
                else:
                    counters["empty"] += 1
                counters["games"] += 1
                if counters["games"] % 50 == 0:
                    fh.flush()
                    print(
                        f"  {counters['games']}/{len(todo)} games, "
                        f"{counters['pitches']} pitches, {counters['empty']} empty"
                    )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, todo))

    print(
        f"done: {counters['games']} games, {counters['pitches']} pitches, "
        f"{counters['empty']} with no play-by-play -> {out_csv}"
    )


if __name__ == "__main__":
    main()

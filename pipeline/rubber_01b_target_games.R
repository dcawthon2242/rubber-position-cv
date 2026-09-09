#!/usr/bin/env Rscript

# Choose which GAMES to harvest play_ids for, so the clip supply covers the
# pitchers who actually need anchoring.
#
# Why this exists. rubber_01 harvests play_ids and has a --limit-games flag that
# was written for smoke tests: it takes the first N outstanding games in game_pk
# order, which is chronological. That flag ended up being used for the real run,
# so 2025 has play_ids for 300 of 2430 games (12%) -- all from the opening weeks
# -- and 2023, 2024 and 2026 have none at all. Every clip the pipeline has ever
# fetched comes from that 12% slice of one season.
#
# The consequence is not a small loss of yield, it is a hard hole in coverage.
# Caleb Kilian threw no MLB pitches in 2025 at all; his innings are in 2026,
# where there are no play_ids, so no clip of him can ever be selected and his
# rubber position falls back to the pooled model at ~4 inches of error. Travis
# Adams pitched 18 games in 2025 but only 8 fall inside the harvested slice.
# Both then show up in the counterfactual with a 13-19 inch "observed range"
# that is fallback noise rather than measured movement, and that range is what
# admits them to the top of the leaderboard.
#
# So rank games by how much anchor error they would remove, and harvest those
# instead of the chronologically first ones. A game is worth the summed value of
# the unanchored pitchers who appear in it with a usable cell, scaled by whether
# the park is one where the rubber is legible and how early the pitchers in
# question take the mound.
#
# Usage:
#   Rscript pipeline/rubber_01b_target_games.R --season 2026 --top 400
#   python pipeline/rubber_01_playids.py --season 2026 \
#       --games-from data/rubber/target_games_2026.csv

suppressPackageStartupMessages({ library(data.table) })

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[[i + 1L]]
}
season <- as.integer(get_arg("--season", "2026"))
top_games <- as.integer(get_arg("--top", "400"))
min_fb_in_cell <- as.integer(get_arg("--min-fb", "3"))

RUBBER_DIR <- file.path("data", "rubber")
OUT <- file.path(RUBBER_DIR, sprintf("target_games_%d.csv", season))

# Same list rubber_02b ranks by, for the same reason: these are the venues whose
# centre-field camera shows the mound clearly enough to place both ends of the
# rubber. Kept in sync by hand; if one moves, change both.
PREFERRED_PARKS <- c(
  "ATL", "BAL", "BOS", "CIN", "COL", "HOU", "MIA",
  "PHI", "SD", "SEA", "SF", "TB", "TEX", "TOR", "WSH"
)
EARLY_INNING_MAX <- 3L
FASTBALLS <- c("FF", "SI")

# Same error model rubber_02b uses to value a pitcher: a label moves his cells
# off the pooled fallback and onto an anchor, and the fallback is far worse for
# left-handers, so a LHP label removes about twice the error per pitch.
FALLBACK_MAE_IN <- c(L = 5.93, R = 3.54)
ANCHOR_MAE_IN <- 1.20

PRI <- file.path(RUBBER_DIR, "label_priority_pitchers.csv")
if (!file.exists(PRI))
  stop("missing ", PRI, "; run rubber_05_calibrate.R first")
pri <- fread(PRI, showProgress = FALSE)
cat(sprintf("unanchored pitchers needing a label: %d\n", nrow(pri)))

sc_f <- file.path("data", sprintf("statcast_%d", season),
                  sprintf("statcast_%d_all.csv", season))
if (!file.exists(sc_f)) stop("missing ", sc_f)
d <- fread(sc_f, select = c("game_pk", "game_type", "pitcher", "p_throws",
                            "stand", "pitch_type", "inning", "home_team",
                            "release_pos_x"),
           showProgress = FALSE)
d <- d[game_type == "R" & pitcher %in% pri$pitcher & !is.na(release_pos_x)]
d[, park := home_team]
cat(sprintf("season %d: %d pitches by unanchored pitchers in %d games\n",
            season, nrow(d), uniqueN(d$game_pk)))

# A cell is only usable if it has enough fastballs to fix a reference release
# point, which is the same gate rubber_02 applies.
cells <- d[, .(n_fb = sum(pitch_type %in% FASTBALLS),
               min_inning = min(inning, na.rm = TRUE),
               park = park[1L], p_throws = p_throws[1L]),
           by = .(game_pk, pitcher, stand)]
cells <- cells[n_fb >= min_fb_in_cell]
cat(sprintf("usable cells (>= %d fastballs): %d across %d games\n",
            min_fb_in_cell, nrow(cells), uniqueN(cells$game_pk)))

cells <- merge(cells, pri[, .(pitcher, fastballs)], by = "pitcher")
cells[, value := fastballs * (FALLBACK_MAE_IN[p_throws] - ANCHOR_MAE_IN)]

# Park and inning multipliers, deliberately mild. They break ties toward legible
# venues and early innings without letting a preferred park outrank a game that
# is the only chance at an otherwise unreachable pitcher.
cells[, park_mult := fifelse(park %in% PREFERRED_PARKS, 1.5, 1.0)]
cells[, inning_mult := fifelse(min_inning <= EARLY_INNING_MAX, 1.3, 1.0)]
cells[, w := value * park_mult * inning_mult]

# This is a set-cover problem, not a ranking one. A pitcher needs anchoring
# ONCE, so the second game containing him is worth nothing extra, and scoring
# games independently gets that badly wrong in both directions: it double-counts
# durable starters across all their games, and if you correct for that by
# dividing each pitcher's value over his appearances it then buries them, since
# a 30-game starter contributes a thirtieth of his value to each game and never
# clears the cut. That is what the first version of this script did, and the
# result was that the highest-volume pitchers in the league -- Berrios, Morton,
# Lopez -- were all reported unreachable.
#
# Greedy set cover instead: repeatedly take the game adding the most uncovered
# value. It is the standard (1 - 1/e) approximation and it is fast enough here
# because the candidate set is only a few thousand games.
pool <- cells[, .(game_pk, pitcher, w, park, min_inning)]
covered <- integer(0)
picked <- integer(top_games)
gain <- numeric(top_games)
n_new <- integer(top_games)
for (i in seq_len(top_games)) {
  rem <- pool[!pitcher %in% covered]
  if (!nrow(rem)) { picked <- picked[seq_len(i - 1L)]
                    gain <- gain[seq_len(i - 1L)]
                    n_new <- n_new[seq_len(i - 1L)]; break }
  # A game can contain the same pitcher twice (once per batter side); collapse
  # to his best cell so a platoon split does not count as two pitchers.
  best <- rem[, .(w = max(w)), by = .(game_pk, pitcher)][
    , .(gain = sum(w), n_new = .N), by = game_pk][order(-gain)][1L]
  picked[i] <- best$game_pk
  gain[i] <- best$gain
  n_new[i] <- best$n_new
  covered <- c(covered, rem[game_pk == best$game_pk]$pitcher)
  if (i %% 100L == 0L)
    cat(sprintf("  greedy: %d games picked, %d pitchers covered\n",
                i, length(unique(covered))))
}

meta <- cells[, .(park = park[1L], earliest_inning = min(min_inning),
                  cells = .N), by = game_pk]
sel <- merge(data.table(game_pk = picked, score = gain, pitchers = n_new),
             meta, by = "game_pk", sort = FALSE)
setorder(sel, -score)

fwrite(sel[, .(game_pk, park, pitchers, cells, earliest_inning,
               score = round(score, 1))], OUT)

covered <- unique(cells[game_pk %in% sel$game_pk]$pitcher)

cat(sprintf("\nselected %d games -> %s\n", nrow(sel), OUT))
cat(sprintf("they cover %d of %d unanchored pitchers (%.0f%%)\n",
            length(covered), nrow(pri), 100 * length(covered) / nrow(pri)))
cat(sprintf("at a preferred park: %d of %d games (%.0f%%)\n",
            sel[park %in% PREFERRED_PARKS, .N], nrow(sel),
            100 * mean(sel$park %in% PREFERRED_PARKS)))

cat("\nper-park:\n")
print(sel[, .N, by = park][order(-N)])

# Which pitchers this still fails to reach, named rather than counted, because
# the ones that fall through are the ones worth a manual look.
missed <- pri[!pitcher %in% covered]
cat(sprintf("\nstill unreachable this season: %d pitchers\n", nrow(missed)))
if (nrow(missed)) {
  setorder(missed, -fastballs)
  print(head(missed[, .(player_name, p_throws, fastballs, cells)], 15))
}

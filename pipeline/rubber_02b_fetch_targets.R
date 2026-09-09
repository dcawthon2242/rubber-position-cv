#!/usr/bin/env Rscript

# Choose which clips to fetch next, so video time is spent where it buys the
# most accuracy.
#
# Why this exists. rubber_03 downloads a clip and extracts frames, which is the
# slow, network-bound step in the pipeline. Until now it was pointed at the clip
# manifest in whatever order that file happened to be in, which spread the fetch
# evenly over parks and pitchers. Two findings make that the wrong order:
#
#   1. rubber_05 anchors per pitcher. A label converts ONE pitcher's cells from
#      the ~4.4-inch pooled fallback to a ~1.2-inch anchor and does nothing for
#      anybody else. So the unit of value is a distinct unanchored pitcher, and
#      a pitcher is worth what he throws -- fastball count, not game count,
#      since a closer appears in far more games than a starter while throwing a
#      fraction of the pitches.
#
#   2. Parks differ enormously in whether the rubber is legible at all. Across
#      the labeled rounds the share of frames where a human could find both ends
#      of the rubber runs from about 8 in 10 at the best parks to 0 in 10 at the
#      worst, and the automated frame score does not predict it -- Pittsburgh
#      has the highest median frame score and went 1 for 11. Most pitchers work
#      at many parks in a season, so choosing the park is close to free yield.
#
#   3. The mound degrades over a game. It is dressed before first pitch and then
#      digs out: the landing area erodes and the displaced dirt gets kicked up
#      over the front edge of the rubber, burying the edge a labeler has to
#      click. Innings are therefore ranked, earliest first. This is free in the
#      same way the park is -- a pitcher with 30 games offers many innings to
#      choose from, and picking a first-inning clip over a seventh-inning one
#      costs nothing.
#
# So: for each unanchored pitcher, take his best available clips, ranked by park
# preference, then stretch, then how early in the game the pitch was thrown, then
# park legibility. Output is a manifest in exactly rubber_03's input format.
#
# Usage:
#   Rscript pipeline/rubber_02b_fetch_targets.R --season 2025 --top 400
#   python pipeline/rubber_03_fetch_frames.py --season 2025 \
#       --manifest data/rubber/fetch_targets_2025.csv

suppressPackageStartupMessages({ library(data.table) })

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[[i + 1L]]
}
season <- as.integer(get_arg("--season", "2025"))
top_n <- as.integer(get_arg("--top", "400"))
# More than one clip per pitcher is deliberate but small. A clip can fail to
# yield a usable frame for reasons nothing to do with the park -- a replay wipe
# over the delivery, a mid-stride motion trigger -- and at roughly a 50% label
# yield a single clip per pitcher leaves half the target list unanchored after
# all the downloading is done. Two clips at different games make that much less
# likely while barely more than doubling the fetch.
per_pitcher <- as.integer(get_arg("--per-pitcher", "2"))

# Force a pitcher's clips into DIFFERENT parks. Ranking by park legibility alone
# put both of his picks at his single best park, which wastes the second clip on
# a near-duplicate of the first. Spreading them buys a second thing for free: the
# gap between two labels of the same pitcher at two parks is an estimate of the
# park's lateral calibration offset. That offset is currently the largest
# unquantified error in the whole pipeline, because an anchor is measured at one
# park and then applied to every park the pitcher visits. The existing labels
# hint the offset is large -- 13 pitchers with repeats inside one park agree to a
# median 0.62 in, while the 2 with repeats across parks differ by 8.91 in -- but
# n=2 cannot separate a park offset from two mislabels or two real moves.
distinct_parks <- as.integer(get_arg("--distinct-parks", "1")) == 1L

# Rank pitchers by expected error REDUCTION rather than raw fastball count. A
# label moves a cell from the pooled fallback to the anchor, and the fallback is
# far worse for left-handers (held-out MAE 5.93 in vs 3.54 for right-handers,
# against 1.20 for the anchor). So a left-hander's label removes about twice the
# error per pitch, and weighting by volume alone systematically under-serves
# them. This replaces an arbitrary LHP quota with the reason one would be wanted.
weight_by_error <- as.integer(get_arg("--weight-by-error", "1")) == 1L
FALLBACK_MAE_IN <- c(L = 5.93, R = 3.54)
ANCHOR_MAE_IN <- 1.20

RUBBER_DIR <- file.path("data", "rubber")
CLIPS <- file.path(RUBBER_DIR, sprintf("clip_manifest_%d.csv", season))
OUT <- file.path(RUBBER_DIR, sprintf("fetch_targets_%d.csv", season))
MIN_PARK_LEGIBILITY <- 0.25

# Hand-picked parks where the centre-field camera shows the mound clearly enough
# to place both ends of the rubber. This is a human judgement over the venues
# rather than a statistic, so it takes precedence over the fitted park_score
# below: park_score is estimated from a handful of labels per park and is noisy
# where a park has few, whereas the camera geometry either shows the rubber or
# it does not. Parks outside the list are still eligible, just ranked last.
PREFERRED_PARKS <- c(
  "ATL", "BAL", "BOS", "CIN", "COL", "HOU", "MIA",
  "PHI", "SD", "SEA", "SF", "TB", "TEX", "TOR", "WSH"
)
prefer_parks <- as.integer(get_arg("--prefer-parks", "1")) == 1L
# Rank earliest innings first, for the mound-condition reason in the header.
prefer_early <- as.integer(get_arg("--prefer-early-innings", "1")) == 1L

clips <- fread(CLIPS, showProgress = FALSE)
cat(sprintf("clip manifest: %d cells, %d pitchers, %d parks\n",
            nrow(clips), uniqueN(clips$pitcher), uniqueN(clips$park)))

# ---- park legibility, from the labels themselves --------------------------
#
# Same measure rubber_04b uses when it builds a pack: the share of labeled
# frames at a park where the rubber turned out to be readable, shrunk toward the
# pooled rate because a park has only a handful of labels, then discounted by
# how much the measured rubber WIDTH disagrees within the park. Width sets the
# pixels-per-inch for a label, so a park whose widths scatter by 20% is putting
# a fifth of a scale error into every inch it reports even when the label looked
# fine at the time.

paths <- Sys.glob(file.path(RUBBER_DIR, "label_pack*", "labels_done.csv"))
# Newest round wins for a cell labeled twice: the earliest pack predates the
# park triage and the frame-timing fix, so its misses measure the pipeline of
# the day rather than the park.
ord <- order(as.integer(sub(".*_v", "", sub("/labels_done.csv", "", paths))),
             na.last = FALSE)
L <- rbindlist(lapply(paths[ord], function(p)
  fread(p, showProgress = FALSE, colClasses = "character")),
  use.names = TRUE, fill = TRUE)
for (cc in c("rubber_visible", "rubber_left_px", "rubber_right_px"))
  set(L, j = cc, value = suppressWarnings(as.numeric(L[[cc]])))
L <- unique(L, by = "cell_id", fromLast = TRUE)
set(L, j = "w", value = L$rubber_right_px - L$rubber_left_px)

p0 <- L[, mean(rubber_visible == 1, na.rm = TRUE)]
PSEUDO <- 3.0
leg <- L[, {
  wv <- w[rubber_visible == 1 & !is.na(w) & w > 0]
  cv <- if (length(wv) > 2) sd(wv) / mean(wv) else NA_real_
  .(n = .N,
    vis_rate = (sum(rubber_visible == 1, na.rm = TRUE) + PSEUDO * p0) / (.N + PSEUDO),
    width_cv = cv)
}, by = park]
leg[, pen := fifelse(is.na(width_cv), 0.09, pmin(width_cv, 0.5))]
leg[, park_score := vis_rate * (1 - pen)]
setorder(leg, -park_score)
cat("\npark legibility (share of labeled frames with a readable rubber):\n")
print(leg[, .(park, n, vis_rate = round(vis_rate, 2),
              width_cv = round(width_cv, 3), park_score = round(park_score, 2))])

elig <- fread(file.path(RUBBER_DIR, "park_eligibility.csv"), showProgress = FALSE)
ok_parks <- elig[status == "eligible", unique(park)]

# ---- who still needs a label ---------------------------------------------

PRI <- file.path(RUBBER_DIR, "label_priority_pitchers.csv")
if (!file.exists(PRI))
  stop("missing ", PRI, "; run rubber_05_calibrate.R first")
pri <- fread(PRI, showProgress = FALSE)
cat(sprintf("\nunanchored pitchers: %d\n", nrow(pri)))

# Cells already fetched, so the same clip is not downloaded twice.
have <- list.files(file.path(RUBBER_DIR, "frames", season), pattern = "_c\\d+\\.jpg$")
have_cells <- unique(sub("_c\\d+\\.jpg$", "", have))

# ---- rank and pick --------------------------------------------------------

d <- merge(clips, leg[, .(park, park_score)], by = "park", all.x = TRUE)
d[is.na(park_score), park_score := p0 * (1 - 0.09)]   # unlabeled park: pooled rate
d <- d[park %in% ok_parks & park_score >= MIN_PARK_LEGIBILITY]
d <- merge(d, pri[, .(pitcher, fastballs)], by = "pitcher")   # p_throws is already on clips
d[, cell_id := paste(game_pk, pitcher, stand, sep = "_")]
d <- d[!cell_id %in% have_cells]
cat(sprintf("candidate clips at legible parks for unanchored pitchers: %d\n", nrow(d)))

# Inches of error removed if this pitcher gets anchored, summed over his pitches.
if (weight_by_error) {
  d[, value := fastballs * (FALLBACK_MAE_IN[p_throws] - ANCHOR_MAE_IN)]
} else {
  d[, value := as.numeric(fastballs)]
}

# Within a pitcher the order is: a preferred park, then a stretch pitch (the
# mandated stop puts the pivot foot flat against the rubber, without which the
# measurement means nothing), then an early inning while the mound is still
# groomed, then the fitted park legibility, then a representative delivery.
d[, park_preferred := if (prefer_parks) park %in% PREFERRED_PARKS else FALSE]
if (!"sel_early_inning" %in% names(d)) {
  # Manifest predates the inning column; fall back to no early-inning signal
  # rather than silently ranking every clip as late.
  d[, `:=`(sel_early_inning = FALSE, inning = NA_integer_)]
  warning("clip manifest has no inning column -- rerun rubber_02_select_clips.R ",
          "to enable the early-inning preference")
}
d[, inning_key := if (prefer_early) fifelse(is.na(inning), 99L, inning) else 0L]
d[, early_key := if (prefer_early) !sel_early_inning else FALSE]

rank_keys <- function(dt) setorderv(
  dt,
  c("value", "pitcher", "park_preferred", "sel_stretch", "early_key",
    "park_score", "inning_key", "sel_representative", "sel_fresh_count",
    "rpx_dev"),
  c(-1L, 1L, -1L, -1L, 1L, -1L, 1L, -1L, -1L, 1L))

rank_keys(d)
if (distinct_parks) {
  # Best clip per park first, then take the top few, so the picks span parks.
  d <- d[, .SD[1L], by = .(pitcher, park)]
  rank_keys(d)
}
d[, rank_in_pitcher := seq_len(.N), by = pitcher]
sel <- d[rank_in_pitcher <= per_pitcher]

keep_pitchers <- head(unique(sel$pitcher), top_n)
sel <- sel[pitcher %in% keep_pitchers]
setorder(sel, -value, pitcher, rank_in_pitcher)

out_cols <- intersect(names(clips), names(sel))
fwrite(sel[, ..out_cols], OUT)

cat(sprintf("\nselected %d clips for %d pitchers -> %s\n",
            nrow(sel), uniqueN(sel$pitcher), OUT))
cat(sprintf("those pitchers account for %s fastballs (%.1f%% of the unanchored total)\n",
            format(sum(unique(sel[, .(pitcher, fastballs)])$fastballs), big.mark = ","),
            100 * sum(unique(sel[, .(pitcher, fastballs)])$fastballs) / sum(pri$fastballs)))
cat("\nper-park:\n")
print(sel[, .N, by = .(park)][order(-N)])
cat(sprintf("\nhandedness of selected pitchers: %s\n", paste(
  sprintf("%s %d", names(table(unique(sel[, .(pitcher, p_throws)])$p_throws)),
          table(unique(sel[, .(pitcher, p_throws)])$p_throws)), collapse = " / ")))
np <- sel[, .(parks = uniqueN(park)), by = pitcher]
cat(sprintf("pitchers whose clips span >1 park: %d of %d (these are the ones that\n",
            np[parks > 1, .N], nrow(np)))
cat("  can measure a park calibration offset)\n")
cat("\nselection quality:\n")
cat(sprintf("  from stretch:   %.0f%%\n", 100 * mean(sel$sel_stretch)))
cat(sprintf("  representative: %.0f%%\n", 100 * mean(sel$sel_representative)))
cat(sprintf("  median park legibility: %.2f\n", median(sel$park_score)))
cat(sprintf("  at a preferred park: %.0f%% (%d of %d clips)\n",
            100 * mean(sel$park_preferred), sum(sel$park_preferred), nrow(sel)))
cat(sprintf("  inning <= 3: %.0f%% | median inning %s\n",
            100 * mean(sel$sel_early_inning),
            ifelse(all(is.na(sel$inning)), "n/a",
                   as.character(as.integer(median(sel$inning, na.rm = TRUE))))))
cat("\ninning distribution of the selected clips:\n")
print(sel[, .N, by = inning][order(inning)])
# Pitchers we could not serve well, so the shortfall is visible rather than
# buried in the averages. A reliever who only ever works the eighth cannot be
# given an early inning, and a pitcher who never visits a preferred park cannot
# be given one; both are facts about his season, not failures of the ranking.
short <- sel[, .(any_pref = any(park_preferred), any_early = any(sel_early_inning)),
             by = .(pitcher, player_name)]
cat(sprintf("\npitchers with no preferred-park clip: %d of %d\n",
            short[!(any_pref), .N], nrow(short)))
cat(sprintf("pitchers with no early-inning clip:  %d of %d\n",
            short[!(any_early), .N], nrow(short)))

#!/usr/bin/env Rscript

# Pick one video clip to measure per pitcher-game-batter-handedness cell.
#
# The rubber measurement only needs a single frame per cell, so the choice of
# pitch matters more than the count. We want a pitch whose setup is typical of
# what the pitcher did in that cell, because one weird pickoff-hold or
# quick-pitch would otherwise become that cell's "rubber position".
#
# Selection priority within each (game_pk, pitcher, stand) cell:
#   1. four-seam or sinker      - the pitcher's default delivery
#   2. runner on first or second - the STRETCH, not the windup (see below)
#   3. release_pos_x within 0.15 ft of the cell median  - guards against
#      sampling an outlier delivery; 0.15 ft is ~1.5 pitch-to-pitch SDs
#   4. 0-0 count                - start of a PA, cleanest broadcast framing
#   5. earliest in the game     - deterministic tiebreak
#
# Why the stretch, having originally preferred bases empty. The measurement
# needs a frame where the pivot foot is planted lengthwise against the rubber,
# and from the stretch the rules require the pitcher to come to a complete stop
# with that foot in contact. That produces a genuinely still, sustained set
# position, which is both the moment we want and an easy target for phase 3's
# motion-onset detection. From the windup there is no mandated stop: by the time
# the delivery is legible the pitcher has often already strided off the rubber,
# which is exactly how the first labeling pass produced frames where the foot
# was nowhere near it. A runner on third alone is excluded because many pitchers
# still use the windup in that situation.
#
# The obvious worry is that a pitcher sets up differently from the stretch, so
# the measurement would not describe his normal delivery. Measured on 2025
# fastballs across 664 pitchers with at least 30 pitches in each state, the
# stretch-minus-windup shift in release_pos_x has a median of 0.00 inches and a
# mean absolute value of 0.75 inches, with only 2.4% of pitchers exceeding 3
# inches. That is far below the measurement error we are trying to reduce, so
# the two states are pooled rather than modelled separately.
#
# Criteria 1-4 are soft: they order candidates rather than filter them, so a
# cell still yields a clip when (say) the pitcher never pitched from the
# stretch. The `sel_*` flag columns record which criteria the chosen pitch
# actually met so downstream QC can drop compromised cells.
#
# --stratified samples the calibration subset instead of taking every cell,
# spreading draws across park x pitcher-hand x release_pos_x decile so the
# release_pos_x -> rubber_x regression is fit over the full range rather than
# concentrated in the middle.
#
# Usage:
#   Rscript pipeline/rubber_02_select_clips.R --season 2025
#   Rscript pipeline/rubber_02_select_clips.R --season 2025 --stratified 4000

suppressPackageStartupMessages({
  library(data.table)
})

# ---- Args -----------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[[i + 1L]]
}

season <- as.integer(get_arg("--season", "2025"))
stratified_n <- get_arg("--stratified", NA)
stratified_n <- if (is.na(stratified_n)) NA_integer_ else as.integer(stratified_n)
min_fb_in_cell <- as.integer(get_arg("--min-fb", "3"))
seed <- as.integer(get_arg("--seed", "20260827"))

set.seed(seed)

REPRESENTATIVE_TOL_FT <- 0.15
FASTBALLS <- c("FF", "SI")
# Innings counted as "early" for mound condition. The mound is dressed before
# first pitch and degrades monotonically: the landing area digs out and the
# displaced dirt gets kicked up over the front edge of the rubber, burying the
# very edge a labeler has to click. Bucketed rather than ranked outright so a
# first-inning pitch does not outrank a third-inning one that is a better
# representative of the delivery.
EARLY_INNING_MAX <- as.integer(get_arg("--early-inning-max", "3"))

statcast_csv <- file.path("data", sprintf("statcast_%d", season),
                          sprintf("statcast_%d_all.csv", season))
playid_csv <- file.path("data", "rubber", sprintf("play_ids_%d.csv", season))
out_csv <- file.path("data", "rubber", sprintf("clip_manifest_%d.csv", season))

stopifnot(file.exists(statcast_csv))
if (!file.exists(playid_csv)) {
  stop("missing ", playid_csv, " - run baseball/rubber_01_playids.py --season ", season)
}

# ---- Load -----------------------------------------------------------------

keep_cols <- c(
  "game_pk", "game_date", "game_type", "at_bat_number", "pitch_number",
  "pitcher", "player_name", "p_throws", "stand", "pitch_type",
  "release_pos_x", "release_pos_z", "release_extension", "arm_angle",
  "on_1b", "on_2b", "on_3b", "balls", "strikes",
  "home_team", "away_team", "inning_topbot",
  # Carried through so rubber_02b can prefer early innings. The mound is
  # groomed before first pitch and degrades from there: by the middle innings
  # the landing area is dug out and the dirt in front of the rubber is kicked
  # over its front edge, which is exactly the edge a labeler needs to see.
  "inning"
)

d <- fread(statcast_csv, select = keep_cols, showProgress = FALSE)
pid <- fread(playid_csv, showProgress = FALSE)

cat(sprintf("season %d: %d Statcast pitches, %d harvested play_ids\n",
            season, nrow(d), nrow(pid)))

# Spring training and exhibition venues mount the center-field camera
# differently (and sometimes not at all), which breaks the rubber geometry.
d <- d[game_type == "R"]
cat(sprintf("  %d pitches in regular-season games\n", nrow(d)))

# Only games we actually have GUIDs for can produce clips.
d <- d[game_pk %in% unique(pid$game_pk)]
cat(sprintf("  %d pitches in %d games with play_ids\n", nrow(d), uniqueN(d$game_pk)))

d <- merge(d, pid, by = c("game_pk", "at_bat_number", "pitch_number"), all.x = FALSE)
cat(sprintf("  %d pitches joined to a play_id\n", nrow(d)))

# The park is what drives camera geometry, and the home team identifies it.
# inning_topbot tells us nothing extra here since home_team is already the venue.
d[, park := home_team]
d[, bases_empty := is.na(on_1b) & is.na(on_2b) & is.na(on_3b)]
# A runner on first or second forces the stretch and a mandated complete stop.
# Third base alone does not, so it is not counted.
d[, from_stretch := !is.na(on_1b) | !is.na(on_2b)]
d[, is_fastball := pitch_type %in% FASTBALLS]

# ---- Cell medians ---------------------------------------------------------

# The reference median comes from fastballs only, so a cell where the pitcher
# threw mostly breaking balls is still judged against a consistent baseline.
# Fall back to all pitch types when the cell has too few fastballs.
d[is_fastball & !is.na(release_pos_x),
  fb_med := median(release_pos_x), by = .(game_pk, pitcher, stand)]
d[, fb_med := fb_med[!is.na(fb_med)][1L], by = .(game_pk, pitcher, stand)]
d[, n_fb := sum(is_fastball & !is.na(release_pos_x)), by = .(game_pk, pitcher, stand)]

d[, cell_ref := fifelse(is.na(fb_med), median(release_pos_x, na.rm = TRUE), fb_med),
  by = .(game_pk, pitcher, stand)]
d[, rpx_dev := abs(release_pos_x - cell_ref)]

# ---- Rank candidates within each cell -------------------------------------

d[, `:=`(
  sel_fastball      = is_fastball,
  sel_stretch       = from_stretch,
  sel_bases_empty   = bases_empty,
  sel_representative = !is.na(rpx_dev) & rpx_dev <= REPRESENTATIVE_TOL_FT,
  sel_fresh_count   = balls == 0L & strikes == 0L,
  sel_early_inning  = !is.na(inning) & inning <= EARLY_INNING_MAX
)]

# Ordering intent: stretch first, because a mandated stop is what puts the pivot
# foot flat along the rubber and makes the measurement mean anything at all.
# Mound condition comes next, since an unreadable rubber costs the whole label.
# Only then delivery representativeness and count.
setorder(d,
         game_pk, pitcher, stand,
         -sel_fastball, -sel_stretch, -sel_early_inning,
         -sel_representative, -sel_fresh_count,
         rpx_dev, inning, at_bat_number, pitch_number)

manifest <- d[, .SD[1L], by = .(game_pk, pitcher, stand)]

cat(sprintf("  %d pitcher-game-hand cells\n", nrow(manifest)))
manifest <- manifest[n_fb >= min_fb_in_cell]
cat(sprintf("  %d cells with >= %d fastballs\n", nrow(manifest), min_fb_in_cell))

cat("\ncriteria met by the selected pitch:\n")
cat(sprintf("  fastball:        %.1f%%\n", 100 * mean(manifest$sel_fastball)))
cat(sprintf("  from stretch:    %.1f%%\n", 100 * mean(manifest$sel_stretch)))
cat(sprintf("  representative:  %.1f%%\n", 100 * mean(manifest$sel_representative)))
cat(sprintf("  0-0 count:       %.1f%%\n", 100 * mean(manifest$sel_fresh_count)))
cat(sprintf("  inning <= %d:     %.1f%% (median inning %d)\n", EARLY_INNING_MAX,
            100 * mean(manifest$sel_early_inning),
            as.integer(median(manifest$inning, na.rm = TRUE))))
cat(sprintf("  all five:        %.1f%%\n", 100 * mean(
  manifest$sel_fastball & manifest$sel_stretch & manifest$sel_early_inning &
    manifest$sel_representative & manifest$sel_fresh_count)))

# ---- Optional stratified subsample ----------------------------------------

if (!is.na(stratified_n) && stratified_n < nrow(manifest)) {
  # Deciles are computed on the cell reference (not the single pitch) because
  # that is the quantity the calibration regression actually uses.
  manifest[, rpx_decile := cut(cell_ref,
                               breaks = quantile(cell_ref, probs = seq(0, 1, 0.1),
                                                 na.rm = TRUE),
                               include.lowest = TRUE, labels = FALSE)]
  manifest[, stratum := paste(park, p_throws, rpx_decile, sep = "|")]

  n_strata <- uniqueN(manifest$stratum)
  per_stratum <- max(1L, ceiling(stratified_n / n_strata))
  cat(sprintf("\nstratifying: %d strata (park x hand x rpx decile), up to %d per stratum\n",
              n_strata, per_stratum))

  # Prefer cells that met every selection criterion, then fill with the rest.
  manifest[, clean := sel_fastball & sel_stretch & sel_representative]
  manifest <- manifest[sample(.N)]
  setorder(manifest, stratum, -clean)
  manifest <- manifest[, head(.SD, per_stratum), by = stratum]

  if (nrow(manifest) > stratified_n) {
    manifest <- manifest[sample(.N, stratified_n)]
  }
  cat(sprintf("  sampled %d cells\n", nrow(manifest)))
}

# ---- Write ----------------------------------------------------------------

out_cols <- c(
  "game_pk", "game_date", "park", "inning", "at_bat_number", "pitch_number",
  "play_id",
  "pitcher", "player_name", "p_throws", "stand", "pitch_type",
  "release_pos_x", "release_pos_z", "release_extension", "arm_angle",
  "cell_ref", "n_fb", "rpx_dev",
  "sel_fastball", "sel_stretch", "sel_bases_empty", "sel_representative",
  "sel_fresh_count", "sel_early_inning"
)
manifest <- manifest[, ..out_cols]
setorder(manifest, game_pk, pitcher, stand)

dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
fwrite(manifest, out_csv)
cat(sprintf("\nwrote %d rows -> %s\n", nrow(manifest), out_csv))

cat("\nby pitcher hand:\n")
print(manifest[, .N, by = .(p_throws, stand)][order(p_throws, stand)])
cat("\nparks represented:", uniqueN(manifest$park), "\n")

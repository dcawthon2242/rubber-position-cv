#!/usr/bin/env Rscript

# Turn measured rubber positions into a rubber position for every
# pitcher-game-handedness cell, and say how much room each pitcher has left.
#
# Why a calibration step exists at all. Video is the only thing that can locate
# a pitcher's foot *relative to the rubber's 24-inch edges*, which is the
# binding constraint on "how far could he move". But video is expensive and, as
# the Phase 0 spike showed, only reliable at parks where the rubber is not
# buried in mound dirt. Meanwhile release_pos_x already tracks lateral movement
# with a pitch-to-pitch SD of ~0.10 ft, so a 15-pitch game median pins relative
# position to well under half an inch. So: measure absolute position on a
# sample, learn the release_pos_x -> rubber_x_in mapping, apply it everywhere.
#
# Truth sources, in order of preference per cell:
#   1. hand labels      data/rubber/label_pack/labels_done.csv
#   2. CV measurement   data/rubber/rubber_measurements_<season>.csv, restricted
#                       to parks that clear a reliability bar
#
# How a measurement is extended to unmeasured cells. This is the part that
# matters most and the part an earlier version of this script got wrong, so it
# is worth stating plainly.
#
# release_pos_x is an excellent RELATIVE ruler and a poor ABSOLUTE one. For one
# pitcher with a fixed delivery, moving a foot along the rubber moves the
# release point with it, near 1:1, so a change in release_pos_x is a change in
# setup. Across pitchers the same number is dominated by arm slot, height and
# extension: a tall over-the-top righty and a short sidearmer standing on the
# very same spot release the ball a foot apart.
#
# Measured on the labels themselves, the difference is not subtle. Predicting a
# held-out label from a pooled cross-pitcher regression gives 8.1 inches of MAE.
# Predicting the same held-out labels from the SAME PITCHER'S other label, by
# carrying it across on the release_pos_x difference, gives 1.2 inches. The
# rubber is 24 inches wide, so the first number cannot answer any question we
# care about and the second can.
#
# So the mapping is per pitcher. One label fixes that pitcher's offset
#   anchor_in = rubber_x_in - 12 * release_pos_x_med
# and every other cell of his is then anchor_in + 12 * release_pos_x_med. The
# pooled regression is kept, but demoted to a fallback for pitchers who have no
# label at all, and its rows are marked as such so downstream work can weight or
# drop them rather than treat 8-inch guesses as measurements.
#
# The practical consequence is that labeling should chase DISTINCT PITCHERS, not
# more games. A second label on a pitcher already anchored buys a small variance
# reduction; a first label on a new pitcher converts all of his cells from the
# 8-inch fallback to the 1-inch anchor.
#
# Hand labels are in crop pixel coordinates, which are 1:1 with full-frame
# pixels (the label crops are zoomed for display only). Because the rubber is a
# known 24 inches wide in the same image, each label carries its own scale:
#   px_per_inch = (rubber_right_px - rubber_left_px) / 24
#   rubber_x_in = -(foot_center_px - rubber_center_px) / px_per_inch
# The negation converts image x to field x. In the center-field view image-left
# is the first-base side (a left-handed batter, who stands on the first-base
# side of the plate, appears in the left-hand box), while Statcast's x is
# positive toward first base.
#
# Note on what rubber_x_in can legally be. It is the offset of the pivot foot's
# CENTRE from the rubber's centre. Because the rules only require part of the
# foot to touch, that centre can sit up to half a shoe (~6 in) past either end of
# the 24-inch rubber, so the physically valid range is about +/-18 inches, not
# +/-12. Offsets past 12 inches are real setups, not mislabels.
#
# Output: data/rubber/rubber_position_pitcher_game.csv, one row per
# (game_pk, pitcher, stand), ready to join onto pitch rows as columns.
#
# Usage:
#   Rscript pipeline/rubber_05_calibrate.R
#   Rscript pipeline/rubber_05_calibrate.R --seasons 2024,2025,2026

suppressPackageStartupMessages({ library(data.table) })

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[[i + 1L]]
}

seasons <- as.integer(strsplit(get_arg("--seasons", "2021,2022,2023,2024,2025,2026"),
                               ",")[[1]])
RUBBER_DIR <- file.path("data", "rubber")
OUT_CSV <- file.path(RUBBER_DIR, "rubber_position_pitcher_game.csv")

RUBBER_HALF_IN <- 12.0   # rubber is 24 inches wide
# The pivot foot sits lengthwise along the rubber, so the dimension that matters
# along the rubber's axis is shoe LENGTH, not spike width. A typical MLB cleat is
# about 12 inches, giving a 6-inch half-length.
FOOT_HALF_LEN_IN <- 6.0
# Contact only requires part of the foot on the rubber -- a toe or a heel is
# enough. So the foot's CENTRE can legally sit outside the rubber's own 24-inch
# span, up to half a shoe beyond either end. This is why a measured offset of,
# say, 17 inches is a perfectly real setup rather than a mislabel, and an earlier
# version of this script wrongly discarded such labels at the 12-inch line.
MAX_FOOT_CENTRE_IN <- RUBBER_HALF_IN + FOOT_HALF_LEN_IN   # 18
# Fastballs needed before a cell's median release_pos_x is trusted. Five was the
# original bar, chosen for a stable median, but it silently discarded 13 of 56
# hand labels -- all of them one-inning relievers who threw three or four
# fastballs. Those labels are perfectly good measurements and cost as much to
# make as any other. At three pitches the median carries about 0.7 inches of
# sampling error, which is small against even the 1.2-inch anchor path and
# invisible against the 8-inch fallback, so the trade is clearly worth it. n_fb
# rides along in the output for anyone who wants to re-impose a stricter bar.
MIN_FB_PER_CELL <- 3L
# A foot moves with the release point about 1:1, so a foot of release_pos_x is
# twelve inches of rubber travel. Used only WITHIN a pitcher; see the header.
IN_PER_FT <- 12.0
# Two labels of one pitcher imply two estimates of his offset. Real setup change
# between games is small -- the labels put the median within-pitcher spread
# under an inch -- so a large disagreement means one of the two clicks is wrong,
# and there is no way to tell which. Such a pitcher is dropped to the fallback
# rather than anchored on a coin flip. The bar is set well above ordinary label
# noise so it catches genuine contradictions only.
MAX_ANCHOR_SPREAD_IN <- 6.0
FASTBALLS <- c("FF", "SI")

# A park qualifies as CV-reliable only with enough successful frames and a
# stable detected rubber width; unstable width means the detector is latching
# onto sponsor logos or chalk lines rather than the rubber.
PARK_MIN_OK_FRAMES <- 12L
PARK_MAX_WIDTH_CV <- 0.12
# rubber_04j (pose + fine-tuned bar detector) measurements: frame acceptance.
# Detector confidence, plus the pitcher's bbox height over the detected rubber
# width, which is ~3.2 in the standard CF shot and breaks on close-up replays
# or a bar drawn on the wrong object. Same-game LHH/RHH cell pairs -- where the
# pitcher has not moved -- disagreed at RMS 5.6 in unfiltered and 3.1 in with
# these gates (an implied ~2.2 in per cell). Mirrors rubber_04j.accept().
POSE_MIN_CONF <- 0.40
POSE_H_OVER_W <- c(2.2, 4.2)
# Per-pitcher anchor outliers: a cell whose implied anchor sits this far from
# the pitcher's median anchor is dropped as a bad measurement (needs >= 3 cells).
ANCHOR_CELL_OUTLIER_IN <- 6.0

# Hand triage from rubber_04f, which is authoritative where it exists. The two
# park filters answer different questions and both are applied: the CV gate asks
# whether the DETECTOR works at a park, this asks whether the rubber is legible
# to a human at all. A park can pass the first and fail the second when the
# detector is confidently measuring the wrong white object.
ELIG_CSV <- file.path(RUBBER_DIR, "park_eligibility.csv")
park_elig <- NULL
if (file.exists(ELIG_CSV)) {
  park_elig <- fread(ELIG_CSV, showProgress = FALSE)
  ok_parks <- park_elig[status == "eligible", unique(park)]
  cat(sprintf("park triage: %d of %d parks eligible (%s)\n",
              length(ok_parks), nrow(park_elig),
              paste(sort(park_elig[status != "eligible", park]), collapse = " ")))
}

# ---- 1. Cell-level Statcast aggregates ------------------------------------

cell_cols <- c("game_pk", "game_date", "game_type", "pitcher", "player_name",
               "p_throws", "stand", "pitch_type", "release_pos_x",
               "release_pos_z", "release_extension", "arm_angle", "home_team")

load_cells <- function(yr) {
  f <- file.path("data", sprintf("statcast_%d", yr),
                 sprintf("statcast_%d_all.csv", yr))
  if (!file.exists(f)) { message("missing season file: ", f); return(NULL) }
  d <- fread(f, select = cell_cols, showProgress = FALSE)
  d <- d[game_type == "R" & pitch_type %in% FASTBALLS & !is.na(release_pos_x)]
  d[, season := yr]
  d[, .(
    season      = season[1L],
    game_date   = game_date[1L],
    park        = home_team[1L],
    player_name = player_name[1L],
    p_throws    = p_throws[1L],
    n_fb        = .N,
    release_pos_x_med   = median(release_pos_x),
    release_pos_x_sd    = if (.N > 1L) sd(release_pos_x) else NA_real_,
    release_pos_z_med   = median(release_pos_z, na.rm = TRUE),
    release_extension_med = median(release_extension, na.rm = TRUE),
    arm_angle_med       = median(arm_angle, na.rm = TRUE)
  ), by = .(game_pk, pitcher, stand)]
}

cells <- rbindlist(lapply(seasons, load_cells), use.names = TRUE, fill = TRUE)
cells <- cells[n_fb >= MIN_FB_PER_CELL]
cat(sprintf("cells with >= %d fastballs: %d across seasons %s\n",
            MIN_FB_PER_CELL, nrow(cells), paste(range(cells$season), collapse = "-")))

# ---- 2. Truth from hand labels -------------------------------------------

# Labeling happens in rounds: a pack is built, labeled, then the selection or
# frame extraction improves and a fresh pack covers the parks the last round
# missed. Each round keeps its own labels_done.csv so a rebuild cannot clobber
# earlier work, and all of them are pooled here.
labels_paths <- sort(Sys.glob(file.path(RUBBER_DIR, "label_pack*", "labels_done.csv")))
truth_lab <- NULL
if (length(labels_paths)) {
  parts <- lapply(labels_paths, function(p) {
    x <- fread(p, showProgress = FALSE, colClasses = "character")
    x[, label_file := basename(dirname(p))]
    x
  })
  L <- rbindlist(parts, use.names = TRUE, fill = TRUE)
  num_cols <- c("rubber_visible", "rubber_left_px", "rubber_right_px",
                "foot_center_px", "foot_on_rubber")
  for (cc in num_cols) if (cc %in% names(L)) L[[cc]] <- suppressWarnings(as.numeric(L[[cc]]))
  for (cc in c("game_pk", "pitcher")) if (cc %in% names(L)) L[[cc]] <- as.integer(L[[cc]])
  cat(sprintf("label files pooled: %s\n",
              paste(basename(dirname(labels_paths)), collapse = ", ")))

  L <- L[!is.na(rubber_visible) & rubber_visible == 1 &
           !is.na(rubber_left_px) & !is.na(rubber_right_px) &
           !is.na(foot_center_px) & rubber_right_px > rubber_left_px]
  if (nrow(L)) {
    L[, rubber_width_px := rubber_right_px - rubber_left_px]
    L[, px_per_inch := rubber_width_px / 24]
    L[, rubber_center_px := (rubber_left_px + rubber_right_px) / 2]
    # Negated: image-left is the first-base side, field x is +1B.
    L[, rubber_x_in := -(foot_center_px - rubber_center_px) / px_per_inch]
    L[, on_rubber := is.na(foot_on_rubber) | foot_on_rubber == 1]
    # The gate is contact, not containment: the foot centre may sit up to half a
    # shoe past either end of the rubber and still be legally touching it. Only
    # beyond that is the label impossible, which in practice means the stride
    # foot was clicked instead of the pivot foot -- an easy confusion in a frame
    # caught after the stride has begun.
    L[, plausible := abs(rubber_x_in) <= MAX_FOOT_CENTRE_IN]
    n_bad <- L[on_rubber == TRUE & plausible == FALSE, .N]
    # Labels made before the park triage include parks later judged illegible.
    # Those clicks were guesses at a rubber the labeler could not actually see,
    # so they are dropped rather than trusted.
    n_offpark <- 0L
    if (!is.null(park_elig) && "park" %in% names(L)) {
      n_offpark <- L[on_rubber == TRUE & plausible == TRUE &
                       !(park %in% ok_parks), .N]
      L <- L[park %in% ok_parks]
    }
    truth_lab <- L[on_rubber == TRUE & plausible == TRUE,
                   .(game_pk, pitcher, stand, rubber_x_in,
                     px_per_inch, source = "label")]
    truth_lab <- unique(truth_lab, by = c("game_pk", "pitcher", "stand"))
    if (n_offpark) cat(sprintf("dropped %d labels at ineligible parks\n", n_offpark))
    cat(sprintf("hand labels usable: %d (of %d rows; dropped %d off-rubber)\n",
                nrow(truth_lab), nrow(L), n_bad))
  }
} else {
  cat("no hand labels found under ", RUBBER_DIR, "/label_pack*/\n", sep = "")
  cat("  (generate crops with baseball/rubber_04b_label_pack.py, label with\n")
  cat("   rubber_04c_label_server.py, which writes labels_done.csv)\n")
}

# ---- 3. Truth from CV, restricted to reliable parks -----------------------

truth_cv <- NULL
# Prefer rubber_04j's pose-based measurements when they exist; the older
# rubber_04 / rubber_04h files are the fallback so an old run still works.
pose_files <- file.path(RUBBER_DIR, sprintf("rubber_pose_measurements_%d.csv", seasons))
pose_files <- pose_files[file.exists(pose_files)]
use_pose <- length(pose_files) > 0L
meas_files <- if (use_pose) pose_files else
  file.path(RUBBER_DIR, sprintf("rubber_measurements_%d.csv", seasons))
meas_files <- meas_files[file.exists(meas_files)]
if (length(meas_files)) {
  M <- rbindlist(lapply(meas_files, fread, showProgress = FALSE),
                 use.names = TRUE, fill = TRUE)
  M[, ok := as.logical(ok)]
  M <- M[ok == TRUE & is.finite(rubber_x_in)]
  if (use_pose && nrow(M)) {
    cat(sprintf("CV source: rubber_04j pose measurements (%s)\n",
                paste(basename(meas_files), collapse = ", ")))
    M[, rubber_width_px := rubber_right - rubber_left]
    if (!"h_over_w" %in% names(M)) M[, h_over_w := pitcher_h / rubber_width_px]
    n_lowconf <- M[rubber_conf < POSE_MIN_CONF, .N]
    n_ratio <- M[rubber_conf >= POSE_MIN_CONF &
                   !(h_over_w >= POSE_H_OVER_W[1] & h_over_w <= POSE_H_OVER_W[2]), .N]
    M <- M[rubber_conf >= POSE_MIN_CONF &
             h_over_w >= POSE_H_OVER_W[1] & h_over_w <= POSE_H_OVER_W[2]]
    cat(sprintf("dropped %d frames with detector confidence < %.2f, %d more with height/width outside %.1f-%.1f\n",
                n_lowconf, POSE_MIN_CONF, n_ratio, POSE_H_OVER_W[1], POSE_H_OVER_W[2]))
  }
  if (nrow(M)) {
    park_qc <- M[, .(n_ok = .N,
                     width_cv = sd(rubber_width_px, na.rm = TRUE) /
                                mean(rubber_width_px, na.rm = TRUE)),
                 by = park]
    # The pose path is self-calibrating per frame (the detected bar sets the
    # scale), so width consistency across clips is not a reliability signal
    # there -- broadcasts zoom differently between clips at one park. Only the
    # old fixed-scale path is gated on it.
    park_qc[, reliable := n_ok >= PARK_MIN_OK_FRAMES &
              (use_pose | (is.finite(width_cv) & width_cv <= PARK_MAX_WIDTH_CV))]
    cat("\nCV reliability by park:\n")
    print(park_qc[order(-n_ok)])
    good <- park_qc[reliable == TRUE]$park
    if (!is.null(park_elig)) good <- intersect(good, ok_parks)
    cat(sprintf("reliable parks: %d of %d\n", length(good), nrow(park_qc)))

    M <- M[park %in% good]
    if (nrow(M)) {
      # One value per cell: median across that cell's candidate frames, and
      # the spread across frames is a per-cell precision estimate.
      truth_cv <- M[, .(rubber_x_in = median(rubber_x_in),
                        frame_spread_in = if (.N > 1L) sd(rubber_x_in) else NA_real_,
                        n_frames = .N,
                        px_per_inch = median(px_per_inch),
                        source = "cv"),
                    by = .(game_pk, pitcher, stand)]
      cat(sprintf("CV cell measurements from reliable parks: %d\n", nrow(truth_cv)))
    }
  }
}

# ---- 4. Assemble the calibration sample ----------------------------------

truth <- rbindlist(list(truth_lab, truth_cv), use.names = TRUE, fill = TRUE)
if (!is.null(truth) && nrow(truth)) {
  # Prefer hand labels when a cell has both.
  setorder(truth, game_pk, pitcher, stand, source)  # "cv" < "label" alphabetically
  truth[, keep := source == "label" | !any(source == "label"),
        by = .(game_pk, pitcher, stand)]
  truth <- truth[keep == TRUE][, keep := NULL]
  truth <- unique(truth, by = c("game_pk", "pitcher", "stand"))
}

if (is.null(truth) || nrow(truth) < 25L) {
  cat("\n----------------------------------------------------------------\n")
  cat("Calibration sample too small to fit the mapping.\n")
  cat(sprintf("Have %d measured cells; need >= 25.\n",
              if (is.null(truth)) 0L else nrow(truth)))
  cat("Writing the deliverable with rubber_x_in_rel_own only (within-pitcher\n")
  cat("inches, no absolute anchor), so phases 6-8 can run. rubber_x_in and\n")
  cat("room_* stay NA; phase 8 states the room fallback it uses.\n")
  cat("----------------------------------------------------------------\n")
  fit <- NULL
} else {
  cal <- merge(truth, cells, by = c("game_pk", "pitcher", "stand"))
  cat(sprintf("\ncalibration cells joined to Statcast: %d\n", nrow(cal)))

  # Pooled fit. A pitcher random effect is not identified here: most pitchers
  # appear once in a sample this size, so a per-pitcher term would just absorb
  # the outcome.
  #
  # Handedness is handled by MIRRORING rather than by its own slope. A left-hander
  # setting up 5 inches toward first base is the reflection of a right-hander 5
  # inches toward third, and both release_pos_x and rubber_x_in flip sign with the
  # throwing hand (RHP medians -2.2 ft / +4.1 in, LHP +1.9 ft / -5.8 in). So
  # reflecting the horizontal quantities for left-handers lets one slope be fitted
  # on all the labels instead of splitting them. Release height, arm angle and
  # extension are not reflected, being vertical or already hand-agnostic.
  #
  # This matters because left-handers are a fifth of the labels, and an
  # interaction term spent their slope on that fifth alone. Measured by held-out
  # pitcher CV, mirroring beats the interaction everywhere: LHP MAE 7.51 -> 5.93
  # in, RHP 3.77 -> 3.54, overall 4.70 -> 4.13, physically impossible LHP
  # predictions 2.1% -> 0.0%, and with 5 parameters rather than 7.
  cal[, mirror := fifelse(p_throws == "R", 1, -1)]
  cal[, rubber_x_in_m := mirror * rubber_x_in]
  cal[, release_pos_x_med_m := mirror * release_pos_x_med]
  form <- rubber_x_in_m ~ release_pos_x_med_m + arm_angle_med +
    release_pos_z_med + release_extension_med
  cal_fit <- cal[stats::complete.cases(
    cal[, .(rubber_x_in_m, release_pos_x_med_m, arm_angle_med,
            release_pos_z_med, release_extension_med)])]
  fit <- stats::lm(form, data = cal_fit)

  cat("\n=== calibration fit: mirrored rubber_x_in ~ release_pos_x + covariates ===\n")
  print(summary(fit))
  resid_sd <- stats::sd(stats::residuals(fit))
  cat(sprintf("\nresidual SD: %.2f inches  (this is the accuracy of an\n", resid_sd))
  cat("  extrapolated rubber_x_in for an unmeasured cell)\n")
  cat(sprintf("R-squared: %.3f  |  n = %d\n",
              summary(fit)$r.squared, nrow(cal_fit)))

  # Leave-one-out style check on pitchers held out entirely, so the reported
  # error is not flattered by having seen that pitcher. This is the number that
  # justifies demoting the pooled fit: it is what an unlabeled pitcher gets.
  set.seed(42)
  pitchers <- unique(cal_fit$pitcher)
  if (length(pitchers) >= 10L) {
    folds <- split(pitchers, sample(rep_len(1:5, length(pitchers))))
    errs <- unlist(lapply(folds, function(pp) {
      tr <- cal_fit[!pitcher %in% pp]; te <- cal_fit[pitcher %in% pp]
      if (nrow(tr) < 15L || nrow(te) < 1L) return(numeric(0))
      m <- try(stats::lm(form, data = tr), silent = TRUE)
      if (inherits(m, "try-error")) return(numeric(0))
      # Predict on the mirrored scale, then reflect back before scoring.
      te$rubber_x_in - te$mirror * stats::predict(m, newdata = te)
    }))
    if (length(errs) > 5L) {
      pooled_mae <- mean(abs(errs))
      cat(sprintf("\npooled fallback, by-pitcher CV: MAE %.2f in, RMSE %.2f in, n=%d\n",
                  mean(abs(errs)), sqrt(mean(errs^2)), length(errs)))
    }
  }
}

# ---- 4b. Per-pitcher anchors ---------------------------------------------
#
# The offset that turns this pitcher's release_pos_x into rubber position. See
# the header for why this is per pitcher rather than pooled.

anchors <- NULL
if (!is.null(truth) && nrow(truth) && exists("cal") && nrow(cal)) {
  a <- cal[!is.na(release_pos_x_med),
           .(game_pk, pitcher, season, stand,
             anchor_in = rubber_x_in - IN_PER_FT * release_pos_x_med)]
  # With several CV cells per pitcher, one bad frame should cost that cell, not
  # the pitcher. Drop cells far from the pitcher's own median anchor first.
  a[, `:=`(med_anchor = median(anchor_in), n_cells = .N), by = pitcher]
  n_outlier_cells <- a[n_cells >= 3L & abs(anchor_in - med_anchor) > ANCHOR_CELL_OUTLIER_IN, .N]
  a <- a[!(n_cells >= 3L & abs(anchor_in - med_anchor) > ANCHOR_CELL_OUTLIER_IN)]
  a[, c("med_anchor", "n_cells") := NULL]
  if (n_outlier_cells)
    cat(sprintf("dropped %d measured cells > %.0f in from their pitcher's median anchor\n",
                n_outlier_cells, ANCHOR_CELL_OUTLIER_IN))
  # Spread: full range for the 2-3 hand labels a pitcher used to have; with the
  # CV path giving many cells per pitcher, one bad frame should not discard
  # him, so from four observations up it is the 10th-90th percentile span.
  anchors <- a[, .(anchor_in = median(anchor_in),
                   n_labels = .N,
                   anchor_spread_in = if (.N <= 1L) 0 else if (.N <= 3L)
                     diff(range(anchor_in)) else
                     diff(stats::quantile(anchor_in, c(0.1, 0.9), names = FALSE)),
                   anchor_season = as.numeric(median(season))),
               by = pitcher]
  bad <- anchors[anchor_spread_in > MAX_ANCHOR_SPREAD_IN]
  if (nrow(bad)) {
    cat(sprintf("\n%d pitcher(s) dropped: their labels disagree by more than %.0f in\n",
                nrow(bad), MAX_ANCHOR_SPREAD_IN))
    print(bad[order(-anchor_spread_in), .(pitcher, n_labels,
                                          spread_in = round(anchor_spread_in, 1))])
    cat("  re-label these; one of the two clicks caught the stride foot\n")
  }
  anchors <- anchors[anchor_spread_in <= MAX_ANCHOR_SPREAD_IN]

  # Honest accuracy for the anchor path: hold out one label of a repeat-labeled
  # pitcher and rebuild his anchor from the rest. Only repeats can be scored
  # this way, which is exactly why labeling a handful of pitchers twice is worth
  # the frames it costs.
  reps <- cal[, .N, by = pitcher][N > 1L]$pitcher
  aerr <- unlist(lapply(reps, function(p) {
    s <- cal[pitcher == p & !is.na(release_pos_x_med)]
    if (nrow(s) < 2L) return(numeric(0))
    vapply(seq_len(nrow(s)), function(i) {
      o <- s[-i]
      s$rubber_x_in[i] - mean(o$rubber_x_in +
        (s$release_pos_x_med[i] - o$release_pos_x_med) * IN_PER_FT)
    }, numeric(1))
  }))
  if (length(aerr) > 5L) {
    cat(sprintf("\nanchor path, leave-one-label-out: MAE %.2f in, RMSE %.2f in, n=%d\n",
                mean(abs(aerr)), sqrt(mean(aerr^2)), length(aerr)))
  }
  cat(sprintf("usable anchors: %d pitchers\n", nrow(anchors)))
}

# ---- 5. Predict for every cell, coalesce, compute room -------------------

cells[, `:=`(rubber_x_in_cv = NA_real_, rubber_x_in_anchor = NA_real_,
             rubber_x_in_est = NA_real_, se = NA_real_, source = "none",
             meas_source = NA_character_)]

if (!is.null(truth) && nrow(truth)) {
  cells[, meas_source := NULL]
  cells <- merge(cells, truth[, .(game_pk, pitcher, stand,
                                  measured = rubber_x_in, meas_source = source)],
                 by = c("game_pk", "pitcher", "stand"), all.x = TRUE)
  cells[!is.na(measured), rubber_x_in_cv := measured]
}

# Anchor path. Carries a pitcher's own measurement to his other cells on the
# release_pos_x difference, which the leave-one-label-out check above scores at
# ~1 inch against the pooled fit's ~8.
if (!is.null(anchors) && nrow(anchors)) {
  cells <- merge(cells, anchors[, .(pitcher, anchor_in, n_labels,
                                    anchor_season)],
                 by = "pitcher", all.x = TRUE)
  cells[!is.na(anchor_in) & !is.na(release_pos_x_med),
        rubber_x_in_anchor := anchor_in + IN_PER_FT * release_pos_x_med]
  # A label anchors its own season cleanly. Applying it to a different season
  # assumes the pitcher did not change his setup over the winter, which is a
  # real assumption and sometimes wrong, so those cells are marked separately
  # rather than pooled in with the ones the label actually covers.
  cells[, anchor_xseason := !is.na(rubber_x_in_anchor) & season != anchor_season]
}

if (!is.null(fit)) {
  # Same mirroring as the fit: reflect left-handers in, predict, reflect back out.
  cells[, mirror := fifelse(p_throws == "R", 1, -1)]
  cells[, release_pos_x_med_m := mirror * release_pos_x_med]
  need <- cells[, .(release_pos_x_med_m, arm_angle_med,
                    release_pos_z_med, release_extension_med)]
  ok <- stats::complete.cases(need)
  cells[ok, rubber_x_in_est := mirror * stats::predict(fit, newdata = cells[ok])]
}

if (!("anchor_xseason" %in% names(cells))) cells[, anchor_xseason := FALSE]
cells[, rubber_x_in := fcoalesce(rubber_x_in_cv, rubber_x_in_anchor,
                                 rubber_x_in_est)]
cells[, source := fcase(
  !is.na(rubber_x_in_cv), fifelse(!is.na(meas_source), meas_source, "measured"),
  !is.na(rubber_x_in_anchor) & anchor_xseason == FALSE, "anchor",
  !is.na(rubber_x_in_anchor), "anchor_xseason",
  !is.na(rubber_x_in_est), "model",
  default = "none")]

# Per-row accuracy, so downstream work can weight or filter instead of treating
# an 8-inch guess as a measurement. The label, anchor and model figures come
# from the holdout checks printed above rather than from nominal standard
# errors, which would understate the fallback badly.
#
# The cross-season figure is the one number here that is ASSUMED rather than
# measured, and it should be treated with suspicion until it can be checked.
# Every label so far is from 2025, so there is no pitcher measured in two
# seasons and therefore no way to observe how much an anchor decays over a
# winter. Doubling the within-season error is a placeholder, not a finding. The
# check becomes possible as soon as a handful of pitchers are labeled in a
# second season, which is a cheap and worthwhile addition to the next round.
anchor_mae <- if (exists("aerr") && length(aerr) > 5L) mean(abs(aerr)) else NA_real_
if (!exists("pooled_mae")) pooled_mae <- NA_real_
cells[, se := fcase(source %in% c("label", "cv", "measured"), 0.6,
                    source == "anchor", anchor_mae,
                    source == "anchor_xseason", anchor_mae * 2,
                    source == "model", pooled_mae,
                    default = NA_real_)]

# Clipping to the physical range. This is almost entirely a fallback-path
# concern: the pooled regression is unconstrained and happily predicts positions
# no foot can occupy, whereas an anchored cell starts from a real measurement
# and only moves by the pitcher's own release_pos_x drift. A large clip share is
# therefore a signal that the fallback is carrying too much of the output, not a
# nuisance to be silenced.
cells[, clipped := !is.na(rubber_x_in) & abs(rubber_x_in) > MAX_FOOT_CENTRE_IN]
n_clip <- cells[clipped == TRUE, .N]
if (n_clip) {
  cat(sprintf("\nclipped %d of %d cells (%.1f%%) to the +/-%.0fin physical range\n",
              n_clip, cells[!is.na(rubber_x_in), .N],
              100 * n_clip / cells[!is.na(rubber_x_in), .N], MAX_FOOT_CENTRE_IN))
  print(cells[!is.na(rubber_x_in),
              .(cells = .N, clipped = sum(clipped),
                pct = round(100 * mean(clipped), 1)), by = p_throws][order(p_throws)])
  cells[, rubber_x_in := pmax(-MAX_FOOT_CENTRE_IN,
                              pmin(MAX_FOOT_CENTRE_IN, rubber_x_in))]
}

# Within-pitcher relative position, always available because it needs no
# absolute anchor. For a fixed delivery the release point tracks the rubber
# roughly 1:1, so a pitcher-season-demeaned release_pos_x converts to inches of
# lateral movement. This is only meaningful *within* pitcher: across pitchers,
# most release_pos_x spread is arm slot and height, not setup position. Phase 6
# depends on exactly this quantity.
cells[, rubber_x_in_rel_own := (release_pos_x_med -
        median(release_pos_x_med, na.rm = TRUE)) * 12, by = .(pitcher, season)]

# Remaining travel to each edge, on two definitions because they answer
# different questions and differ by a full foot-length.
#
# room_*_in is the legal limit: slide until only the toe (or heel) still touches,
# i.e. until the foot centre reaches 18 inches from rubber centre. This is the
# constraint the plan means by "no pitcher is moved off the rubber", so it is
# what phase 8 clips against.
#
# room_*_full_foot_in is the practical limit: keep the whole foot inside the
# rubber's 24-inch span. A pitcher with only a toe on the edge has little to
# push against, so real-world movement is closer to this, and the gap between
# the two columns is the honest uncertainty in "how far could he actually go".
#
# Both are NA without an absolute anchor rather than guessed; phase 8 states the
# fallback it uses when they are missing.
cells[, room_first_base_in := pmax(0, MAX_FOOT_CENTRE_IN - rubber_x_in)]
cells[, room_third_base_in := pmax(0, MAX_FOOT_CENTRE_IN + rubber_x_in)]
cells[, room_first_base_full_foot_in :=
        pmax(0, RUBBER_HALF_IN - FOOT_HALF_LEN_IN - rubber_x_in)]
cells[, room_third_base_full_foot_in :=
        pmax(0, RUBBER_HALF_IN - FOOT_HALF_LEN_IN + rubber_x_in)]

out_cols <- c("game_pk", "game_date", "season", "park", "pitcher", "player_name",
              "p_throws", "stand", "n_fb", "release_pos_x_med", "release_pos_x_sd",
              "release_pos_z_med", "release_extension_med", "arm_angle_med",
              "rubber_x_in_cv", "rubber_x_in_anchor", "rubber_x_in_est",
              "rubber_x_in", "source", "se",
              "rubber_x_in_rel_own", "room_first_base_in", "room_third_base_in",
              "room_first_base_full_foot_in", "room_third_base_full_foot_in")
out_cols <- intersect(out_cols, names(cells))
res <- cells[, ..out_cols]
setorder(res, season, game_pk, pitcher, stand)

dir.create(RUBBER_DIR, recursive = TRUE, showWarnings = FALSE)
fwrite(res, OUT_CSV)
cat(sprintf("\nwrote %d rows -> %s\n", nrow(res), OUT_CSV))
cat("\nsource breakdown (se is the holdout MAE for that path, in inches):\n")
print(res[, .(cells = .N, pct = round(100 * .N / nrow(res), 1),
              se_in = round(se[1], 2)), by = source][order(-cells)])
if (res[source != "none", .N] > 0L) {
  cat("\nrubber_x_in distribution (inches from rubber center, + = first-base side):\n")
  print(round(quantile(res$rubber_x_in, c(.05,.25,.5,.75,.95), na.rm = TRUE), 2))
  cat("\nby pitcher hand:\n")
  print(res[!is.na(rubber_x_in), .(n = .N, med = round(median(rubber_x_in), 2)),
            by = p_throws])
}

# What another round of labeling would buy. Every cell on the "model" path is
# there because its pitcher has no label at all, so it is sitting at ~8 inches
# of error on a 24-inch rubber. The count of DISTINCT such pitchers, ordered by
# how many cells each would convert, is the labeling worklist -- and it is why
# the next round should chase new pitchers rather than new games of the ones
# already anchored.
#
# Ranked by FASTBALLS, not by cells. Cells count pitcher-games, which flatters
# relievers: a closer appears in 70 games a season and throws 15 pitches in
# each, so he tops a cell ranking while contributing a fraction of the pitches a
# starter does. Since the point of this table is to feed a pitch-level Stuff+
# model, the quantity that matters is how many pitch rows an anchor would
# upgrade from the 7.5-inch fallback to the 1.2-inch anchor.
todo <- res[source == "model",
            .(fastballs = sum(n_fb), cells = .N, seasons = uniqueN(season)),
            by = .(pitcher, player_name, p_throws)][order(-fastballs)]
if (nrow(todo)) {
  cat(sprintf("\nunanchored pitchers: %d, covering %d cells (%.1f%% of output)\n",
              nrow(todo), sum(todo$cells), 100 * sum(todo$cells) / nrow(res)))
  for (k in c(100L, 300L, 600L)) {
    if (nrow(todo) < k) next
    cat(sprintf("  labeling the top %d would upgrade %.1f%% of all fastballs\n",
                k, 100 * sum(head(todo, k)$fastballs) / sum(res$n_fb)))
  }
  # Left-handers are worth calling out separately: they are a quarter of the
  # data and the pooled fallback fits them worst, since its release_pos_x slope
  # is estimated mostly from right-handers.
  cat(sprintf("  of the top 300, %d are left-handed\n",
              head(todo, 300)[p_throws == "L", .N]))
  print(head(todo, 15))
  fwrite(todo, file.path(RUBBER_DIR, "label_priority_pitchers.csv"))
  cat(sprintf("wrote worklist -> %s\n",
              file.path(RUBBER_DIR, "label_priority_pitchers.csv")))
}

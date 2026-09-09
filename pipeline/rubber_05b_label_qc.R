#!/usr/bin/env Rscript

# Diagnose how much of rubber_05's residual error is label noise rather than a
# genuine limit on what release_pos_x can tell us.
#
# The calibration reports a ~5.4 inch by-pitcher CV error. That number is only
# actionable once it is split into its two possible causes, which call for
# opposite responses:
#
#   label noise    the same pitcher measured twice lands in two different
#                  places, so the anchor itself is wrong and more labels of the
#                  same quality will not help. Fix the labeling.
#   model limit    the labels agree with each other but release_pos_x cannot
#                  separate setup position from arm slot and height. Fix the
#                  features, or accept the error.
#
# Three checks, in order of how much they can be trusted:
#
#   1. Repeat measurements of one pitcher. A pitcher does not meaningfully
#      change where he stands between games, so the spread across a pitcher's
#      own labeled cells is close to a direct read on labeling precision. This
#      is the only check here that needs no modelling assumption at all.
#   2. Rubber width in pixels within a park. The camera does not move, so every
#      clip from a park should give the same rubber width. A wide spread means
#      the two clicks were not landing on the same two edges, which corrupts
#      the per-label scale and hence every inch value derived from it.
#   3. Residuals against the fit, by park, by hand, and by label round, to see
#      whether the error is concentrated somewhere specific.
#
# Usage:  Rscript pipeline/rubber_05b_label_qc.R

suppressPackageStartupMessages({ library(data.table) })

RUBBER_DIR <- file.path("data", "rubber")
MAX_FOOT_CENTRE_IN <- 18.0
FASTBALLS <- c("FF", "SI")
MIN_FB_PER_CELL <- 5L

# ---- labels ---------------------------------------------------------------

paths <- sort(Sys.glob(file.path(RUBBER_DIR, "label_pack*", "labels_done.csv")))
L <- rbindlist(lapply(paths, function(p) {
  x <- fread(p, showProgress = FALSE, colClasses = "character")
  x[, round := basename(dirname(p))]
  x
}), use.names = TRUE, fill = TRUE)

for (cc in c("rubber_visible", "rubber_left_px", "rubber_right_px",
             "foot_center_px", "foot_on_rubber"))
  L[[cc]] <- suppressWarnings(as.numeric(L[[cc]]))
for (cc in c("game_pk", "pitcher")) L[[cc]] <- as.integer(L[[cc]])

L <- L[!is.na(rubber_visible) & rubber_visible == 1 &
         !is.na(rubber_left_px) & !is.na(rubber_right_px) &
         !is.na(foot_center_px) & rubber_right_px > rubber_left_px]
L[, width_px := rubber_right_px - rubber_left_px]
L[, px_per_inch := width_px / 24]
L[, rubber_x_in := -(foot_center_px - (rubber_left_px + rubber_right_px) / 2) /
     px_per_inch]
L[, on_rubber := is.na(foot_on_rubber) | foot_on_rubber == 1]
L <- L[on_rubber == TRUE & abs(rubber_x_in) <= MAX_FOOT_CENTRE_IN]

elig <- file.path(RUBBER_DIR, "park_eligibility.csv")
if (file.exists(elig)) {
  ok_parks <- fread(elig, showProgress = FALSE)[status == "eligible", unique(park)]
  L <- L[park %in% ok_parks]
}
L <- unique(L, by = c("game_pk", "pitcher", "stand"))
cat(sprintf("usable labels: %d over %d pitchers, %d parks, rounds: %s\n",
            nrow(L), uniqueN(L$pitcher), uniqueN(L$park),
            paste(unique(L$round), collapse = ", ")))

# ---- 1. repeat measurements of the same pitcher ---------------------------
#
# The cleanest available read on labeling precision. Two labels of one pitcher
# should agree to within how precisely a click can be placed; anything larger is
# the labeler measuring different things. Reported as the SD across a pitcher's
# cells, pooled over pitchers with at least two.

cat("\n=== 1. repeat labels of the same pitcher ===\n")
rep <- L[, .(n = .N, sd_in = if (.N > 1L) sd(rubber_x_in) else NA_real_,
             range_in = if (.N > 1L) diff(range(rubber_x_in)) else NA_real_,
             parks = uniqueN(park)),
         by = pitcher][n > 1L]
if (nrow(rep)) {
  cat(sprintf("pitchers labeled more than once: %d\n", nrow(rep)))
  cat(sprintf("within-pitcher SD:   median %.2f in, mean %.2f in\n",
              median(rep$sd_in), mean(rep$sd_in)))
  cat(sprintf("within-pitcher range: median %.2f in, max %.2f in\n",
              median(rep$range_in), max(rep$range_in)))
  cat("\nworst offenders (a real pitcher does not move this much):\n")
  print(head(rep[order(-range_in)], 8))
  # A pitcher's two cells often differ only in batter handedness, which is a
  # real (small) effect, so this is an upper bound on pure click noise rather
  # than a clean estimate of it.
  cat("\nsplit by whether the repeats share a park:\n")
  print(rep[, .(pitchers = .N, med_range = round(median(range_in), 2)),
            by = .(same_park = parks == 1L)])
} else {
  cat("no pitcher has two usable labels; cannot estimate labeling precision\n")
  cat("this is itself the finding: deliberately re-label a few pitchers twice\n")
}

# ---- 2. rubber width within a park ----------------------------------------
#
# The centre-field camera is bolted down for the season, so the rubber subtends
# the same pixel width in every clip from a park. Spread here is pure click
# error on the two edge points, and it propagates straight into the scale.

cat("\n=== 2. rubber width in pixels, within park ===\n")
w <- L[, .(n = .N, med_w = round(median(width_px), 1),
           cv = round(sd(width_px) / mean(width_px), 3)),
       by = park][n > 1L][order(-cv)]
print(w)
if (nrow(w)) {
  # A 10% error in width is a 10% error in every inch value from that label.
  cat(sprintf("\nmedian within-park width CV: %.1f%%\n", 100 * median(w$cv, na.rm = TRUE)))
  cat("each 1% of width error is ~1% of the measured offset, so a 10-inch\n")
  cat("offset measured with a 10% width error carries a 1-inch scale error\n")
}

# ---- 3. residuals against the fit -----------------------------------------

cell_cols <- c("game_pk", "game_type", "pitcher", "p_throws", "stand",
               "pitch_type", "release_pos_x", "release_pos_z",
               "release_extension", "arm_angle", "home_team")
load_cells <- function(yr) {
  f <- file.path("data", sprintf("statcast_%d", yr), sprintf("statcast_%d_all.csv", yr))
  if (!file.exists(f)) return(NULL)
  d <- fread(f, select = cell_cols, showProgress = FALSE)
  d <- d[game_type == "R" & pitch_type %in% FASTBALLS & !is.na(release_pos_x)]
  d[, .(park = home_team[1L], p_throws = p_throws[1L], n_fb = .N,
        release_pos_x_med = median(release_pos_x),
        release_pos_z_med = median(release_pos_z, na.rm = TRUE),
        release_extension_med = median(release_extension, na.rm = TRUE),
        arm_angle_med = median(arm_angle, na.rm = TRUE)),
    by = .(game_pk, pitcher, stand)]
}
cells <- rbindlist(lapply(2021:2026, load_cells), use.names = TRUE, fill = TRUE)
cells <- cells[n_fb >= MIN_FB_PER_CELL]

cal <- merge(L[, .(game_pk, pitcher, stand, rubber_x_in, width_px, round,
                   lab_park = park)],
             cells, by = c("game_pk", "pitcher", "stand"))
cat(sprintf("\n=== 3. residuals ===\njoined to Statcast: %d of %d labels\n",
            nrow(cal), nrow(L)))
lost <- L[!paste(game_pk, pitcher, stand) %in%
            cal[, paste(game_pk, pitcher, stand)]]
if (nrow(lost)) {
  # Usually a cell with fewer than MIN_FB_PER_CELL fastballs, i.e. a short
  # outing. Worth knowing because those labels cost the same to make.
  cat(sprintf("labels with no Statcast cell: %d\n", nrow(lost)))
  print(lost[, .N, by = .(round, park)][order(-N)])
}

form <- rubber_x_in ~ release_pos_x_med * p_throws + arm_angle_med +
  release_pos_z_med + release_extension_med
cal <- cal[complete.cases(cal[, .(rubber_x_in, release_pos_x_med, p_throws,
                                  arm_angle_med, release_pos_z_med,
                                  release_extension_med)])]
fit <- lm(form, data = cal)
cal[, resid := residuals(fit)]

cat(sprintf("\nn = %d, R2 = %.3f, residual SD = %.2f in\n",
            nrow(cal), summary(fit)$r.squared, sd(cal$resid)))
cat("\nby pitcher hand:\n")
print(cal[, .(n = .N, mae = round(mean(abs(resid)), 2),
              sd_truth = round(sd(rubber_x_in), 2)), by = p_throws])
cat("\nby label round:\n")
print(cal[, .(n = .N, mae = round(mean(abs(resid)), 2)), by = round])
cat("\nby park (n >= 3):\n")
print(cal[, .(n = .N, mae = round(mean(abs(resid)), 2)),
          by = lab_park][n >= 3L][order(-mae)])
cat("\nlargest residuals -- inspect these crops first:\n")
print(head(cal[order(-abs(resid)),
               .(game_pk, pitcher, stand, lab_park, round,
                 measured = round(rubber_x_in, 1),
                 pred = round(rubber_x_in - resid, 1),
                 resid = round(resid, 1))], 10))

# A simple ceiling check: how well could ANY model do if the labels are noisy?
# If within-pitcher SD is s, no model can beat s even with perfect features.
if (exists("rep") && nrow(rep) && sum(!is.na(rep$sd_in))) {
  s <- median(rep$sd_in, na.rm = TRUE)
  cat(sprintf("\nceiling: with within-pitcher label SD of %.2f in, a perfect\n", s))
  cat(sprintf("model still shows ~%.2f in of apparent error. Observed is %.2f in,\n",
              s, sd(cal$resid)))
  cat(sprintf("so roughly %.0f%% of the residual is label noise.\n",
              100 * min(1, (s^2) / var(cal$resid))))
}

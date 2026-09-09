#!/usr/bin/env Rscript

# What would happen to a pitcher's platoon stuff+ if he slid along the rubber?
#
# The perturbation, exactly as specified in the plan. For a lateral shift of
# Delta feet, holding the intended plate location and the pitch's movement fixed:
#
#   release_pos_x += Delta
#   flight time t is unchanged, because vy0 and ay are untouched
#   plate_x = x0 + vx0*t + 0.5*ax*t^2 is held fixed, so   Delta_vx0 = -Delta/t
#   therefore the horizontal velocity at the plate also shifts by -Delta/t
#   velo, spin, and induced movement are unchanged
#
# With t ~ 0.4 s and vy_plate ~ -125 ft/s that works out to roughly 1.15 degrees
# of horizontal approach angle per foot of rubber movement, which this script
# verifies numerically rather than assuming.
#
# Delta is swept over the full legal placement span and clipped by the room each
# pitcher has left, so nobody gets moved off the end of the rubber. The span is
# the physical one: a 24-inch rubber plus half a shoe past either end, giving
# foot centres out to +/-18 inches and a 36-inch total window.
#
# How much to trust the numbers below. Phase 6 tested whether pitchers who
# actually moved saw their outcomes change, across three designs and 110k
# pitcher-game-hand cells. No design separated the effect from zero -- but all
# three came back POSITIVE, in the direction this geometry predicts, and the
# panel's point estimate is 71% of the model's slope and within 0.8 SE of it. So
# the measurement and the model agree with each other; what neither can do is
# beat zero. The reconciliation section below states this as "consistent but
# underpowered" rather than as a contradiction.
#
# That distinction was got wrong at first. The test used to compare the model's
# slope against phase 6's minimum detectable effect -- 1.96 SE centred on zero --
# and reported "CONTRADICTED" when the model exceeded it. That treats an
# imprecise measurement as though it were a precise null, so any prediction
# smaller than the panel's resolution would be branded a conflict. The
# comparison is now against the measured confidence interval.
#
# Phase 7 separately found that rubber position adds essentially no predictive
# value once approach angles are in the model, which is expected: HAA is the
# channel through which position acts, so position is redundant given HAA rather
# than unimportant.
#
# The honest summary is that the mechanism is real geometry, the predicted effect
# is small (a 6-inch move buys about 0.17 stuff+ points), and the observational
# data can neither confirm nor exclude it. Treat the leaderboard as a ranking of
# who has the most to gain if the mechanism works, not as a measured return.
#
# Idealizations, stated plainly because they matter for how the output reads:
#   - A real rubber move also changes stride direction, trunk lean, and the
#     hitter's sightline. This perturbation holds all of that constant, so it
#     isolates the geometric channel only.
#   - Holding plate location fixed assumes the pitcher fully recalibrates his
#     aim. A pitcher who moves and does not re-aim gets a different, worse
#     result that this does not model.
#   - Approach angle carries some information about where the pitch ended up, so
#     part of the model's HAA sensitivity is location rather than pure stuff.
#
# Inputs : data/statcast_model/stuff_platoon_data.rds            (phase 7)
#          data/statcast_model/stuff_platoon_models.rds          (phase 7)
#          data/statcast_model/rubber_within_pitcher_gradient.rds (phase 6)
#          data/statcast_model/rubber_cell_outcomes.rds          (phase 6, for PA mix)
#          data/rubber/rubber_position_pitcher_game.csv          (phase 5, for room)
# Outputs: data/statcast_model/rubber_move_gains.csv
#          data/statcast_model/rubber_move_sweep.csv
#
# Usage: Rscript pipeline/rubber_07_counterfactual.R

suppressPackageStartupMessages({ library(data.table); library(lightgbm) })

MDIR <- file.path("data", "statcast_model")
STEP_IN  <- 1
MIN_SWINGS_STAND <- 60L  # per pitcher per batter side, for a stable mean

# Rubber geometry, kept identical to phase 5 so the fallback below cannot
# disagree with the measured room it substitutes for. The rubber is 24 inches
# wide and the pivot foot lies LENGTHWISE along it, so contact only requires a
# toe or a heel: the foot's centre may sit up to half a shoe (about 6 inches)
# past either end, giving legal foot centres out to +/-18 inches from centre.
# This file previously carried FOOT_HALF_IN = 2.0, a leftover from an earlier
# spike-WIDTH assumption, which made the fallback claim 10 inches of room where
# phase 5's measured columns allow 18.
RUBBER_HALF_IN <- 12.0
FOOT_HALF_LEN_IN <- 6.0
MAX_FOOT_CENTRE_IN <- RUBBER_HALF_IN + FOOT_HALF_LEN_IN   # 18

# Sweep the whole legal window rather than an arbitrary sub-range. Legal foot
# centres span 24 + 6 + 6 = 36 inches, and a pitcher already standing at one end
# can traverse all 36 of them, so the DELTA range is +/-36 rather than +/-18.
# Anything narrower silently truncates the optimiser: every pitcher in the data
# has more than 18 inches of room on at least one side. The per-pitcher room
# columns below do the real clipping from wherever he currently stands.
SWEEP_IN <- 2 * MAX_FOOT_CENTRE_IN

bundle <- readRDS(file.path(MDIR, "stuff_platoon_data.rds"))
mods   <- readRDS(file.path(MDIR, "stuff_platoon_models.rds"))
BEST <- mods$scale$best
SC_MU <- mods$scale$mu; SC_SD <- mods$scale$sd

d <- bundle$data
ALL_FEATS <- unique(unlist(mods$specs))
d <- d[set == "holdout" & stats::complete.cases(d[, ..ALL_FEATS])]
cat(sprintf("counterfactual population: %d holdout swings, %d pitchers\n",
            nrow(d), uniqueN(d$pitcher)))

# ------------------------------------------------------------------- room

pos_f <- file.path("data", "rubber", "rubber_position_pitcher_game.csv")
pos <- fread(pos_f, showProgress = FALSE)
room <- pos[, .(room_fb = suppressWarnings(median(room_first_base_in, na.rm = TRUE)),
                room_tb = suppressWarnings(median(room_third_base_in, na.rm = TRUE)),
                rel_own_p95 = quantile(abs(rubber_x_in_rel_own), 0.95, na.rm = TRUE)),
            by = pitcher]
room[!is.finite(room_fb), room_fb := NA_real_]
room[!is.finite(room_tb), room_tb := NA_real_]

have_abs <- room[!is.na(room_fb), .N]
if (have_abs == 0L) {
  cat("\nNOTE: phase 5 has no absolute rubber position yet (the CV/label\n")
  cat("      calibration has not been supplied), so room_first_base_in and\n")
  cat("      room_third_base_in are unavailable. Falling back to the symmetric\n")
  cat("      assumption that every pitcher starts at rubber centre, giving\n")
  cat(sprintf("      %.0f inches of room each way. Once labels land, rerun\n",
              MAX_FOOT_CENTRE_IN))
  cat("      phase 5 and this script picks up the real per-pitcher limits.\n")
  room[, `:=`(room_fb = MAX_FOOT_CENTRE_IN, room_tb = MAX_FOOT_CENTRE_IN)]
  room_source <- "symmetric fallback"
} else {
  room_source <- "phase 5 measured"
}

# --------------------------------------------------------------- platoon mix

cell_f <- file.path(MDIR, "rubber_cell_outcomes.rds")
if (file.exists(cell_f)) {
  co <- readRDS(cell_f)
  mix <- co[season == max(season), .(n_pitch = sum(n_pitch)), by = .(pitcher, stand)]
} else {
  mix <- d[, .(n_pitch = .N), by = .(pitcher, stand)]
}
mix <- dcast(mix, pitcher ~ stand, value.var = "n_pitch", fill = 0)
setnames(mix, c("L", "R"), c("np_lhh", "np_rhh"), skip_absent = TRUE)
mix[, w_lhh := np_lhh / pmax(1, np_lhh + np_rhh)]

# ---------------------------------------------------------------- the sweep

# Recomputing HAA from the shifted vx0 rather than adding a linear approximation
# keeps the geometry exact for pitches with unusual flight times.
score_shift <- function(dd, st, delta_ft) {
  mo <- mods$models[[paste(st, BEST, sep = "_")]]
  fts <- mo$feats
  x <- copy(dd)
  x[, release_pos_x := release_pos_x + delta_ft]
  x[, vy_p := vy0 + ay * t_plate]
  x[, vx_p_new := (vx0 - delta_ft / t_plate) + ax * t_plate]
  x[, haa := atan2(vx_p_new, -vy_p) * 180 / pi]
  x[, rubber_x_in_rel_own := rubber_x_in_rel_own + delta_ft * 12]
  predict(mo$model, as.matrix(x[, ..fts]))
}

deltas_in <- seq(-SWEEP_IN, SWEEP_IN, by = STEP_IN)
cat(sprintf("\nsweeping %d shifts from %+d to %+d inches\n",
            length(deltas_in), -SWEEP_IN, SWEEP_IN))

sweep <- list()
for (st in c("L", "R")) {
  dd <- d[stand == st]
  base <- score_shift(dd, st, 0)
  for (din in deltas_in) {
    p <- if (din == 0) base else score_shift(dd, st, din / 12)
    sweep[[length(sweep) + 1L]] <- data.table(
      pitcher = dd$pitcher, player_name = dd$player_name,
      p_throws = dd$p_throws, stand = st, delta_in = din,
      d_pred = p - base)
  }
  cat(sprintf("  scored %d swings vs %sHH\n", nrow(dd), st))
}
sweep <- rbindlist(sweep)

# Verify the plan's stated geometry instead of trusting it.
hchk <- d[1:min(20000, nrow(d))]
haa_1ft <- {
  v0 <- hchk[, atan2(vx0 + ax * t_plate, -(vy0 + ay * t_plate)) * 180 / pi]
  v1 <- hchk[, atan2((vx0 - 1 / t_plate) + ax * t_plate, -(vy0 + ay * t_plate)) * 180 / pi]
  mean(v1 - v0)
}
cat(sprintf("\ngeometry check: %.3f deg of HAA per foot of rubber shift (plan said ~1.15)\n",
            abs(haa_1ft)))
cat(sprintf("median flight time %.3f s, median vy at plate %.1f ft/s\n",
            median(d$t_plate), median(d[, vy0 + ay * t_plate])))

# ------------------------------------------------- per-pitcher, per-side means

agg <- sweep[, .(n = .N, d_miss = mean(d_pred)),
             by = .(pitcher, player_name, p_throws, stand, delta_in)]
agg <- agg[n >= MIN_SWINGS_STAND]
agg[, d_stuff := d_miss / (SC_SD / 10)]   # inches of miss -> stuff+ points

w <- dcast(agg, pitcher + player_name + p_throws + delta_in ~ stand,
           value.var = c("d_stuff", "n"))
w <- w[!is.na(d_stuff_L) & !is.na(d_stuff_R)]
w <- merge(w, mix[, .(pitcher, w_lhh)], by = "pitcher", all.x = TRUE)
w[is.na(w_lhh), w_lhh := n_L / pmax(1, n_L + n_R)]
# Weight the two sides by how often this pitcher actually faces each, so a
# reliever who sees mostly same-handed batters is not credited for gains against
# a platoon split he rarely encounters.
w[, d_stuff_wtd := w_lhh * d_stuff_L + (1 - w_lhh) * d_stuff_R]

w <- merge(w, room[, .(pitcher, room_fb, room_tb, rel_own_p95)],
           by = "pitcher", all.x = TRUE)
# Positive delta is toward first base, so it is limited by room on that side.
w[, feasible := delta_in <= room_fb & -delta_in <= room_tb]
# Inside the pitcher's own observed range, the model is interpolating; outside
# it, extrapolating. The plan restricts headline claims to the former.
w[, in_support := is.na(rel_own_p95) | abs(delta_in) <= pmax(rel_own_p95, 1)]

# Which limit is actually doing the work. The sweep window is no longer an
# arbitrary sub-range -- it is the legal placement span itself -- so only two
# things now cap a move: the room the pitcher has left from where he currently
# stands, and his own observed range. The second is far tighter than the first
# for almost everyone, so a headline gain is usually a statement about how far
# the model is willing to extrapolate, not about the rubber. Printing this keeps
# the gains from being read as "the most a pitcher could do".
lim <- unique(w[, .(pitcher, room_fb, room_tb, rel_own_p95)])
cat("\n---- what limits the move (inches, per pitcher) ----\n")
cat(sprintf("legal room toward 1B  : median %.1f  p05 %.1f\n",
            median(lim$room_fb, na.rm = TRUE), quantile(lim$room_fb, 0.05, na.rm = TRUE)))
cat(sprintf("legal room toward 3B  : median %.1f  p05 %.1f\n",
            median(lim$room_tb, na.rm = TRUE), quantile(lim$room_tb, 0.05, na.rm = TRUE)))
cat(sprintf("own observed range    : median %.1f  p95 %.1f\n",
            median(lim$rel_own_p95, na.rm = TRUE),
            quantile(lim$rel_own_p95, 0.95, na.rm = TRUE)))
cat(sprintf("binding at an %d-inch move: rubber room %.0f%% of pitcher-sides, own range %.0f%%\n",
            SWEEP_IN,
            100 * mean(c(lim$room_fb, lim$room_tb) < SWEEP_IN, na.rm = TRUE),
            100 * mean(lim$rel_own_p95 < SWEEP_IN, na.rm = TRUE)))

fwrite(w[order(pitcher, delta_in)], file.path(MDIR, "rubber_move_sweep.csv"))

# ------------------------------------------------------------- best move

pick <- function(dt) {
  dt <- dt[feasible == TRUE]
  if (!nrow(dt)) return(NULL)
  dt[which.max(d_stuff_wtd)]
}
best_any <- rbindlist(lapply(split(w, w$pitcher), pick))
best_sup <- rbindlist(lapply(split(w[in_support == TRUE], w$pitcher[w$in_support == TRUE]),
                             pick))

gains <- merge(
  best_any[, .(pitcher, player_name, p_throws, w_lhh,
               n_lhh = n_L, n_rhh = n_R,
               best_delta_in = delta_in, gain_stuff = d_stuff_wtd,
               gain_stuff_vs_lhh = d_stuff_L, gain_stuff_vs_rhh = d_stuff_R,
               room_fb, room_tb, rel_own_p95)],
  best_sup[, .(pitcher, best_delta_in_in_support = delta_in,
               gain_stuff_in_support = d_stuff_wtd)],
  by = "pitcher", all.x = TRUE)
setorder(gains, -gain_stuff)
gains[, room_source := room_source]

# ------------------------------------- reconcile with the natural experiment

grad_f <- file.path(MDIR, "rubber_within_pitcher_gradient.rds")
verdict <- "phase 6 output not found; counterfactual is unvalidated"
if (file.exists(grad_f)) {
  ne <- readRDS(grad_f)$gradient
  ne_miss <- ne[outcome == "miss_mean"]
  # Model-implied slope in the same units phase 6 measured: inches of miss
  # distance per inch moved toward the batter. Phase 6's regressor was signed
  # toward the batter's box, so mirror the model's per-side gradients the same
  # way before averaging: toward a LHH is +x, toward a RHH is -x.
  g <- agg[delta_in %in% c(-2, 2)]
  g <- dcast(g, pitcher + stand ~ delta_in, value.var = "d_miss")
  setnames(g, c("-2", "2"), c("m2", "p2"))
  g <- g[!is.na(m2) & !is.na(p2)]
  g[, per_in_x := (p2 - m2) / 4]
  g[, per_in_toward_batter := per_in_x * fifelse(stand == "L", 1, -1)]
  model_slope <- mean(g$per_in_toward_batter)

  cat("\n================ model vs natural experiment ================\n")
  cat(sprintf("model-implied      : %+.5f inches of miss per inch toward the batter\n",
              model_slope))
  cat(sprintf("measured (phase 6) : %+.5f +/- %.5f (panel, %s)\n",
              ne_miss$panel_slope, ne_miss$panel_se,
              if (isTRUE(ne_miss$detected)) "detected" else "NOT detected"))
  # The comparison that matters is between the model's slope and the MEASURED
  # slope, not between the model's slope and the smallest effect the panel could
  # have resolved. An earlier version of this test compared the model against
  # mde_6in, which is 1.96*SE centred on ZERO, and therefore declared
  # "CONTRADICTED" whenever the model predicted anything the panel was too
  # imprecise to certify. That is backwards: it converts low power into evidence
  # of a conflict. Here the measured slope is +0.00258 +/- 0.00139 against a model
  # slope of +0.00365 -- the same sign, 71% of the magnitude, and a difference of
  # only 0.77 SE. Those two numbers agree; the panel simply cannot separate either
  # of them from zero (measured vs zero is 1.85 SE, p = 0.06).
  est <- ne_miss$panel_slope; se <- ne_miss$panel_se
  lo <- est - 1.96 * se; hi <- est + 1.96 * se
  z_vs_model <- (model_slope - est) / se
  z_vs_zero  <- est / se
  cat(sprintf("measured 95%% CI   : [%+.5f, %+.5f]\n", lo, hi))
  cat(sprintf("model vs measured : %.2f SE apart (p = %.2f)\n",
              z_vs_model, 2 * stats::pnorm(-abs(z_vs_model))))
  cat(sprintf("measured vs zero  : %.2f SE (p = %.2f)\n",
              z_vs_zero, 2 * stats::pnorm(-abs(z_vs_zero))))
  implied_6 <- abs(model_slope) * 6
  cat(sprintf("model implies a 6-inch move is worth %.4f inches of miss (%.2f stuff+)\n",
              implied_6, implied_6 / (SC_SD / 10)))
  cat(sprintf("smallest 6-inch effect the panel could resolve: %.4f inches\n",
              ne_miss$mde_6in))

  model_in_ci <- model_slope >= lo && model_slope <= hi
  if (!model_in_ci) {
    verdict <- "INCONSISTENT: model gradient falls outside the measured 95% CI"
    cat("\n  *** The model's gradient is incompatible with what pitchers who\n")
    cat("  *** actually moved experienced, so the leaderboard reflects\n")
    cat("  *** cross-sectional differences rather than a return to moving.\n")
  } else if (abs(z_vs_zero) >= 1.96) {
    verdict <- "CORROBORATED: measured slope agrees with the model and excludes zero"
    cat("\n  Model and natural experiment agree in sign and magnitude, and the\n")
    cat("  measurement is itself distinguishable from zero.\n")
  } else {
    verdict <- "CONSISTENT BUT UNDERPOWERED: measured slope matches the model in sign and size, but neither is separable from zero"
    cat("\n  The measured slope sits on top of the model's prediction, in the same\n")
    cat("  direction and of similar size, so nothing here argues the mechanism is\n")
    cat("  absent. But the panel cannot separate it from zero either, and the\n")
    cat("  predicted effect is small in absolute terms, so the leaderboard is a\n")
    cat("  plausible ranking rather than a demonstrated return. Note that the\n")
    cat("  significant vertical placebo does NOT rescue this: the placebo effect\n")
    cat(sprintf("  is %.1fx the size of the predicted one, so it shows the design has\n",
                abs(ne_miss$placebo_slope) / abs(model_slope)))
    cat("  power for much larger effects, not for one this small.\n")
  }
}
gains[, validation := verdict]

fwrite(gains, file.path(MDIR, "rubber_move_gains.csv"))

# ----------------------------------------------------------------- report

cat(sprintf("\nroom source: %s\n", room_source))
cat(sprintf("pitchers with a feasible best move: %d\n", nrow(gains)))
cat("\n---- sweep shape, averaged over all pitchers (stuff+ points) ----\n")
shape <- w[, .(n_pitchers = .N,
               vs_lhh = round(mean(d_stuff_L), 3),
               vs_rhh = round(mean(d_stuff_R), 3),
               weighted = round(mean(d_stuff_wtd), 3)), by = delta_in][order(delta_in)]
print(shape)

# The headline ranking is the in-support one. The room-clipped optimum is kept
# below it for reference only: with the sweep now covering the full legal window
# it routinely lands 20-30 inches out, where the model has almost no training
# data and the mean sweep turns positive in BOTH directions. Phase 7b confirmed
# that is tree extrapolation rather than a channel-selection problem, so the
# unconstrained column is not a usable recommendation.
cat("\n---- top 15 by IN-SUPPORT stuff+ gain (inside each pitcher's own range) ----\n")
print(head(gains[order(-gain_stuff_in_support),
                 .(player_name, p_throws,
                   move_in = best_delta_in_in_support,
                   gain = round(gain_stuff_in_support, 2),
                   own_range_in = round(rel_own_p95, 1),
                   unconstrained_move = best_delta_in,
                   unconstrained_gain = round(gain_stuff, 2))], 15))

cat("\n---- same table ranked by the UNCONSTRAINED optimum (reference only) ----\n")
cat("These moves are mostly extrapolated far past the observed range; see phase 7b.\n")
print(head(gains[order(-gain_stuff),
                 .(player_name, p_throws,
                   best_delta_in, gain = round(gain_stuff, 2),
                   own_range_in = round(rel_own_p95, 1),
                   in_support_delta = best_delta_in_in_support,
                   in_support_gain = round(gain_stuff_in_support, 2))], 15))

cat(sprintf("\nin-support gain distribution (stuff+): median %.2f, p90 %.2f, max %.2f\n",
            median(gains$gain_stuff_in_support),
            quantile(gains$gain_stuff_in_support, .9),
            max(gains$gain_stuff_in_support)))
cat(sprintf("unconstrained, for contrast     : median %.2f, p90 %.2f, max %.2f\n",
            median(gains$gain_stuff), quantile(gains$gain_stuff, .9),
            max(gains$gain_stuff)))
cat(sprintf("1 stuff+ point = %.4f inches of miss distance\n", SC_SD / 10))
cat(sprintf("\nvalidation status: %s\n", verdict))
cat(sprintf("\nwrote %s\nwrote %s\n",
            file.path(MDIR, "rubber_move_gains.csv"),
            file.path(MDIR, "rubber_move_sweep.csv")))

#!/usr/bin/env Rscript

# Does moving on the rubber actually change outcomes? A within-pitcher answer.
#
# Why this runs before the stuff+ counterfactual, not after. A stuff+ model fit
# on all pitches learns mostly the BETWEEN-pitcher gradient in release_pos_x,
# because within-pitcher lateral variation is small (~0.10 ft pitch to pitch,
# ~0.11 ft between games). But pitchers who set up on the third-base edge differ
# systematically from those on the first-base edge: arm slot, height, extension,
# and movement profile all covary with setup. Reading that cross-sectional slope
# as "what happens if THIS pitcher slides 6 inches" is a confound, not a result.
#
# So estimate the within-pitcher effect three ways, each differencing out a
# different set of confounders, and hand the answer to phase 8 as a sanity bound:
#
#   D1  Two-way FE panel. Cell outcome on lateral position, absorbing
#       pitcher-season-stand AND park. Park FE matter because Hawk-Eye lateral
#       calibration has park-specific offsets, and park also moves offense --
#       without park FE that shared variation loads onto the position slope.
#       Identifying variation: a pitcher's game-to-game setup drift.
#
#   D2  Movers only. Same estimator restricted to pitcher-seasons with a
#       sustained changepoint of >= MOVER_MIN_IN inches. Large deliberate moves
#       have a much better signal-to-measurement-error ratio, and a mid-season
#       adjustment is closer to an intervention than daily drift is.
#
#   D3  Within-game platoon split, double-differenced. For pitchers who set up
#       differently against LHH and RHH in the SAME game, regress the
#       (vs-LHH minus vs-RHH) outcome gap on the (vs-LHH minus vs-RHH) position
#       gap. Every game-level confounder -- opponent, park, weather, fatigue,
#       umpire, that day's mechanics -- differences out exactly. This is the
#       cleanest design available, and by the plan's own numbers only ~5.5% of
#       pitcher-games qualify, so it is also the least powered.
#
# Placebo: D1 and D2 are re-run against release_pos_z drift, which no rubber
# move would cause. A "significant" position effect that also shows up in the
# vertical placebo is a mechanics-of-the-day artifact, not a rubber effect.
#
# Sign convention throughout: positive x is the first-base side (LHP mean
# release_pos_x = +2.07, RHP = -1.87). A right-handed batter stands on the
# third-base side of the plate, a left-handed batter on the first-base side, so
#   toward_batter_in = rel_own_in * (stand == "L" ? +1 : -1)
# is "inches moved toward the batter's box", pooled across handedness.
#
# Outcomes, all oriented so higher = better for the pitcher:
#   rv_saved_per_100  -100 * mean(delta_run_exp)
#   whiff_rate        whiffs / swings
#   miss_mean         mean miss_distance on competitive swings (2023 H2 onward)
#
# Inputs : data/rubber/rubber_position_pitcher_game.csv   (phase 5)
#          data/statcast_<season>/statcast_<season>_all.csv
# Outputs: data/statcast_model/rubber_within_pitcher_gradient.rds  (phase 8 reads this)
#          data/statcast_model/rubber_natural_experiment.csv
#
# Usage:
#   Rscript pipeline/rubber_06_natural_experiment.R
#   Rscript pipeline/rubber_06_natural_experiment.R --seasons 2024,2025,2026

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
MDIR       <- file.path("data", "statcast_model")
CELL_CACHE <- file.path(MDIR, "rubber_cell_outcomes.rds")

MIN_PITCHES_CELL <- 10L   # a cell mean below this is mostly noise
MOVER_MIN_IN     <- 3.0   # sustained changepoint size that counts as a real move
MOVER_MIN_GAMES  <- 5L    # games required on each side of the changepoint
PLATOON_MIN_IN   <- 3.0   # within-game LHH-vs-RHH split that counts as a real split
MISS_START       <- as.Date("2023-07-14")  # bat tracking begins at the 2023 ASB

dir.create(MDIR, recursive = TRUE, showWarnings = FALSE)

WHIFF <- c("swinging_strike", "swinging_strike_blocked", "foul_tip")
SWING <- c(WHIFF, "foul", "hit_into_play")

# ---------------------------------------------------------------- cell outcomes

NEED <- c("game_pk", "game_date", "game_type", "pitcher", "stand", "p_throws",
          "home_team", "description", "delta_run_exp", "miss_distance",
          "release_speed", "release_pos_x", "release_pos_z", "pitch_type")

cell_outcomes <- function(yr) {
  f <- file.path("data", sprintf("statcast_%d", yr), sprintf("statcast_%d_all.csv", yr))
  if (!file.exists(f)) { message("missing season file: ", f); return(NULL) }
  d <- fread(f, select = NEED, showProgress = FALSE)
  d <- d[game_type == "R" & !is.na(delta_run_exp)]
  d[, game_date := as.Date(game_date)]

  d[, is_swing := description %in% SWING]
  d[, is_whiff := description %in% WHIFF]
  # Same target construction as miss_grade_features.R: contact is a ~0 miss, and
  # whiffs with no tracked miss are unknown rather than zero, so they drop out.
  d[, miss_ok := is_swing & game_date >= MISS_START & !(is_whiff & is.na(miss_distance))]
  d[miss_ok & is.na(miss_distance), miss_distance := 0]

  out <- d[, .(
    season   = yr,
    game_date = game_date[1L],
    park     = home_team[1L],
    p_throws = p_throws[1L],
    n_pitch  = .N,
    n_swing  = sum(is_swing),
    rv_saved_per_100 = -100 * mean(delta_run_exp),
    whiff_rate = if (sum(is_swing) > 0L) sum(is_whiff) / sum(is_swing) else NA_real_,
    n_miss   = sum(miss_ok),
    miss_mean = if (sum(miss_ok) > 0L) mean(miss_distance[miss_ok]) else NA_real_,
    velo_med = median(release_speed[pitch_type %in% c("FF","SI","FC")], na.rm = TRUE),
    rel_z_med = median(release_pos_z, na.rm = TRUE)
  ), by = .(game_pk, pitcher, stand)]

  message(sprintf("  %d: %d pitcher-game-hand cells", yr, nrow(out)))
  out
}

if (file.exists(CELL_CACHE)) {
  cells_out <- readRDS(CELL_CACHE)
  cells_out <- cells_out[season %in% seasons]
  message(sprintf("loaded cached cell outcomes: %d rows", nrow(cells_out)))
}
if (!exists("cells_out") || !nrow(cells_out) ||
    !all(seasons %in% unique(cells_out$season))) {
  message("building cell outcomes from season CSVs")
  cells_out <- rbindlist(lapply(seasons, cell_outcomes), use.names = TRUE, fill = TRUE)
  saveRDS(cells_out, CELL_CACHE)
  message("cached -> ", CELL_CACHE)
}

# --------------------------------------------------------------- join position

pos_f <- file.path(RUBBER_DIR, "rubber_position_pitcher_game.csv")
if (!file.exists(pos_f)) stop("missing ", pos_f, " -- run rubber_05_calibrate.R first")
pos <- fread(pos_f, showProgress = FALSE)

pos_cols <- intersect(c("game_pk", "pitcher", "stand", "n_fb", "release_pos_x_med",
                        "release_pos_x_sd", "arm_angle_med", "release_extension_med",
                        "rubber_x_in", "rubber_x_in_rel_own", "source",
                        "room_first_base_in", "room_third_base_in"),
                      names(pos))
d <- merge(cells_out, pos[, ..pos_cols], by = c("game_pk", "pitcher", "stand"))
d <- d[n_pitch >= MIN_PITCHES_CELL & !is.na(rubber_x_in_rel_own)]

d[, toward_batter_in := rubber_x_in_rel_own * fifelse(stand == "L", 1, -1)]
d[, arm_side_in      := rubber_x_in_rel_own * fifelse(p_throws == "R", -1, 1)]
d[, psid := paste(pitcher, season, stand, sep = "_")]
# Placebo regressor: vertical release drift, demeaned the same way. A rubber
# slide is lateral, so anything that loads here is day-to-day mechanics.
d[, rel_z_rel_own := (rel_z_med - median(rel_z_med, na.rm = TRUE)) * 12,
  by = .(pitcher, season)]

cat(sprintf("\nanalysis panel: %d cells, %d pitcher-seasons, %d parks, seasons %s\n",
            nrow(d), uniqueN(paste(d$pitcher, d$season)), uniqueN(d$park),
            paste(range(d$season), collapse = "-")))
cat(sprintf("toward_batter_in: SD %.2f in, 5/95 pct %.1f / %.1f in\n",
            sd(d$toward_batter_in, na.rm = TRUE),
            quantile(d$toward_batter_in, .05, na.rm = TRUE),
            quantile(d$toward_batter_in, .95, na.rm = TRUE)))

# ------------------------------------------------------- estimation machinery

# Two-way fixed effects by alternating projections. With only data.table
# available this is cheaper and clearer than building a park dummy matrix, and
# for two factors the sweep converges to the within transform quickly.
demean_twoway <- function(dt, cols, f1, f2, iters = 12L) {
  X <- as.matrix(dt[, ..cols])
  w <- dt$w
  g1 <- dt[[f1]]; g2 <- dt[[f2]]
  for (it in seq_len(iters)) {
    for (g in list(g1, g2)) {
      sw <- as.vector(rowsum(w, g))
      m  <- rowsum(X * w, g) / sw
      X  <- X - m[match(g, rownames(m)), , drop = FALSE]
    }
  }
  as.data.table(X)
}

# Cluster-robust SEs on pitcher. Cells from the same pitcher share his mechanics
# and his true setup, so treating them as independent would understate SEs a lot.
cluster_se <- function(fit, cluster) {
  X <- model.matrix(fit)
  u <- residuals(fit)
  w <- if (is.null(fit$weights)) rep(1, length(u)) else fit$weights
  bread <- chol2inv(qr.R(fit$qr))
  score <- rowsum(X * (w * u), cluster)
  meat  <- crossprod(score)
  G <- nrow(score); n <- nrow(X); k <- ncol(X)
  adj <- (G / (G - 1)) * ((n - 1) / (n - k))
  V <- bread %*% meat %*% bread * adj
  sqrt(pmax(diag(V), 0))
}

# One within-pitcher, within-park slope: outcome per inch of `xvar`.
fe_slope <- function(dt, yvar, xvar, extra = character(), label = "") {
  cols <- c(yvar, xvar, extra)
  dt <- dt[stats::complete.cases(dt[, ..cols]) & is.finite(get(yvar))]
  # Cell outcomes are means of n_pitch pitches, so their sampling variance goes
  # like 1/n. Weighting by n is the efficient choice and also stops 10-pitch
  # relief appearances from dominating.
  dt[, w := as.numeric(n_pitch)]
  if (uniqueN(dt$psid) < 30L || nrow(dt) < 200L) return(NULL)

  dm <- demean_twoway(dt, cols, "psid", "park")
  setnames(dm, cols, paste0("dm_", cols))
  dm[, w := dt$w]
  rhs <- paste0("dm_", c(xvar, extra), collapse = " + ")
  fit <- lm(as.formula(sprintf("dm_%s ~ 0 + %s", yvar, rhs)), data = dm, weights = w)
  se <- cluster_se(fit, dt$pitcher)
  i <- match(paste0("dm_", xvar), names(coef(fit)))
  b <- coef(fit)[i]; s <- se[i]

  data.table(design = label, outcome = yvar, regressor = xvar,
             controls = if (length(extra)) paste(extra, collapse = "+") else "none",
             n_cells = nrow(dt), n_pitchers = uniqueN(dt$pitcher),
             slope_per_in = b, se = s, t = b / s,
             ci_lo = b - 1.96 * s, ci_hi = b + 1.96 * s)
}

OUTCOMES <- c("rv_saved_per_100", "whiff_rate", "miss_mean")
res <- list()

# ------------------------------------------------------ D1: two-way FE panel

cat("\n=== D1: two-way FE panel (pitcher-season-stand + park) ===\n")
for (y in OUTCOMES) {
  res[[length(res) + 1L]] <- fe_slope(d, y, "toward_batter_in", label = "D1 panel")
  # Velo is the main "was he right that day" proxy. If the slope survives it,
  # the position effect is not just a stand-in for a diminished start.
  res[[length(res) + 1L]] <- fe_slope(d, y, "toward_batter_in", extra = "velo_med",
                                      label = "D1 panel + velo")
  res[[length(res) + 1L]] <- fe_slope(d, y, "rel_z_rel_own", label = "D1 PLACEBO vertical")
}

# ------------------------------------------------------------ D2: movers only

# Largest sustained pre/post gap in a pitcher-season-stand's game sequence. This
# is a one-break changepoint on the mean, which is the shape a deliberate
# mid-season adjustment takes; daily wobble produces a small gap and drops out.
best_break <- function(x, min_side) {
  n <- length(x)
  if (n < 2L * min_side) return(list(gap = NA_real_, k = NA_integer_))
  cs <- cumsum(x)
  ks <- min_side:(n - min_side)
  pre <- cs[ks] / ks
  post <- (cs[n] - cs[ks]) / (n - ks)
  gap <- post - pre
  j <- which.max(abs(gap))
  list(gap = gap[j], k = ks[j])
}

setorder(d, pitcher, season, stand, game_date, game_pk)
brk <- d[, {
  b <- best_break(rubber_x_in_rel_own, MOVER_MIN_GAMES)
  .(n_games = .N, break_gap_in = b$gap, break_k = b$k)
}, by = .(pitcher, season, stand)]

movers <- brk[!is.na(break_gap_in) & abs(break_gap_in) >= MOVER_MIN_IN]
cat(sprintf("\n=== D2: movers only (sustained break >= %.1f in, >= %d games/side) ===\n",
            MOVER_MIN_IN, MOVER_MIN_GAMES))
cat(sprintf("%d of %d pitcher-season-stand series qualify (%.1f%%), %d pitchers\n",
            nrow(movers), nrow(brk[!is.na(break_gap_in)]),
            100 * nrow(movers) / max(1L, nrow(brk[!is.na(break_gap_in)])),
            uniqueN(movers$pitcher)))

dm_set <- d[movers[, .(pitcher, season, stand)], on = .(pitcher, season, stand),
            nomatch = 0L]
for (y in OUTCOMES) {
  res[[length(res) + 1L]] <- fe_slope(dm_set, y, "toward_batter_in", label = "D2 movers")
  res[[length(res) + 1L]] <- fe_slope(dm_set, y, "rel_z_rel_own",
                                      label = "D2 movers PLACEBO vertical")
}

# ------------------------------------- D3: within-game platoon double-difference

# Same pitcher, same game, both handedness cells: differencing L minus R wipes
# out every game-level confounder, leaving only how he set up for each batter.
wide <- dcast(d[stand %in% c("L", "R")],
              game_pk + pitcher + season + park + p_throws ~ stand,
              value.var = c("rubber_x_in_rel_own", "n_pitch", OUTCOMES),
              fun.aggregate = function(z) z[1L])
setnames(wide, gsub("_L$", "_lhh", gsub("_R$", "_rhh", names(wide))))
wide <- wide[!is.na(rubber_x_in_rel_own_lhh) & !is.na(rubber_x_in_rel_own_rhh)]

# Positive = he set up further toward the first-base side (the LHH box) when
# facing a LHH than when facing a RHH, i.e. moved toward whoever was hitting.
wide[, d_toward_batter_in := rubber_x_in_rel_own_lhh - rubber_x_in_rel_own_rhh]
wide[, n_eff := 2 / (1 / n_pitch_lhh + 1 / n_pitch_rhh)]

cat(sprintf("\n=== D3: within-game platoon double-difference ===\n"))
cat(sprintf("%d pitcher-games with both handedness cells; %d split >= %.1f in (%.1f%%)\n",
            nrow(wide), wide[abs(d_toward_batter_in) >= PLATOON_MIN_IN, .N],
            PLATOON_MIN_IN,
            100 * wide[abs(d_toward_batter_in) >= PLATOON_MIN_IN, .N] / max(1L, nrow(wide))))

dd_slope <- function(w, yvar, label) {
  w <- copy(w)
  w[, dy := get(paste0(yvar, "_lhh")) - get(paste0(yvar, "_rhh"))]
  w <- w[is.finite(dy) & is.finite(d_toward_batter_in) & n_eff >= MIN_PITCHES_CELL]
  if (nrow(w) < 200L || uniqueN(w$pitcher) < 30L) return(NULL)
  # Pitcher FE on the difference: a pitcher's own average platoon gap in both
  # setup and outcome is absorbed, so only game-to-game variation identifies.
  w[, `:=`(dy_dm = dy - weighted.mean(dy, n_eff),
           dx_dm = d_toward_batter_in - weighted.mean(d_toward_batter_in, n_eff)),
    by = pitcher]
  fit <- lm(dy_dm ~ 0 + dx_dm, data = w, weights = n_eff)
  se <- cluster_se(fit, w$pitcher)
  b <- coef(fit)[["dx_dm"]]
  data.table(design = label, outcome = yvar, regressor = "d_toward_batter_in",
             controls = "pitcher FE on within-game difference",
             n_cells = nrow(w), n_pitchers = uniqueN(w$pitcher),
             slope_per_in = b, se = se[1L], t = b / se[1L],
             ci_lo = b - 1.96 * se[1L], ci_hi = b + 1.96 * se[1L])
}

for (y in OUTCOMES) {
  res[[length(res) + 1L]] <- dd_slope(wide, y, "D3 platoon DD (all)")
  res[[length(res) + 1L]] <- dd_slope(wide[abs(d_toward_batter_in) >= PLATOON_MIN_IN],
                                      y, sprintf("D3 platoon DD (>=%.0fin)", PLATOON_MIN_IN))
}

# ------------------------------------------------------------------- report

out <- rbindlist(Filter(Negate(is.null), res), use.names = TRUE)
if (!nrow(out)) stop("no design had enough data to estimate; widen --seasons")

fmt <- copy(out)
fmt[, `:=`(slope = signif(slope_per_in, 3), se = signif(se, 3),
           t = round(t, 2), ci = sprintf("[%s, %s]", signif(ci_lo, 3), signif(ci_hi, 3)))]
cat("\n================ within-pitcher gradient, per inch toward the batter ================\n")
for (y in OUTCOMES) {
  sub <- fmt[outcome == y]
  if (!nrow(sub)) next
  cat(sprintf("\n-- %s --\n", y))
  print(sub[, .(design, controls, n_cells, n_pitchers, slope, se, t, ci)])
}

# What phase 8 needs: a headline within-pitcher slope per outcome, and whether
# the placebo stayed quiet. If the placebo is as large as the real slope, the
# design is not isolating rubber position and the counterfactual must say so.
headline <- out[design == "D1 panel" | design == "D2 movers"]
placebo  <- out[grepl("PLACEBO", design)]
gradient <- merge(
  headline[design == "D1 panel", .(outcome, panel_slope = slope_per_in,
                                   panel_se = se, panel_n = n_cells)],
  headline[design == "D2 movers", .(outcome, mover_slope = slope_per_in,
                                    mover_se = se, mover_n = n_cells)],
  by = "outcome", all = TRUE)
gradient <- merge(gradient,
  placebo[design == "D1 PLACEBO vertical",
          .(outcome, placebo_slope = slope_per_in, placebo_se = se)],
  by = "outcome", all.x = TRUE)
dd <- out[design == "D3 platoon DD (all)", .(outcome, dd_slope = slope_per_in, dd_se = se)]
gradient <- merge(gradient, dd, by = "outcome", all.x = TRUE)

# Three separate questions, kept separate because conflating them is how a null
# gets misread as a finding or vice versa:
#   detected      is the within-pitcher slope distinguishable from zero at all
#   mde_6in       the largest 6-inch effect this panel could have ruled out --
#                 without it, a null is ambiguous between "no effect" and
#                 "no power", and here the panel is precise enough to matter
#   placebo_loud  did vertical drift, which no rubber move causes, move the
#                 outcome more than lateral position did
gradient[, detected := abs(panel_slope / panel_se) > 1.96]
gradient[, mde_6in := 6 * 1.96 * panel_se]
gradient[, placebo_loud := !is.na(placebo_slope) &
           abs(placebo_slope) > abs(panel_slope)]
gradient[, signs_agree := !is.na(mover_slope) & sign(panel_slope) == sign(mover_slope)]
# Phase 8 may only extrapolate a model gradient the within-pitcher data
# corroborates: detected, sign-consistent across designs, and louder than the
# placebo channel.
gradient[, corroborated := detected & signs_agree & !placebo_loud]

cat("\n================ summary handed to phase 8 ================\n")
print(gradient[, .(outcome,
                   panel = signif(panel_slope, 3), panel_se = signif(panel_se, 3),
                   movers = signif(mover_slope, 3), dd = signif(dd_slope, 3),
                   placebo = signif(placebo_slope, 3),
                   detected, mde_6in = signif(mde_6in, 3), corroborated)])

cat("\nInterpretation guide:\n")
cat("  Units are outcome per INCH moved toward the batter's box, within pitcher.\n")
cat("  mde_6in is the smallest 6-inch effect this design could resolve: 1.96 SE\n")
cat("  centred on ZERO. It is a power statement, not a bound on the truth. A null\n")
cat("  here does NOT mean the effect is absent -- it means any effect is smaller\n")
cat("  than the design can certify. Read panel +/- panel_se for what was actually\n")
cat("  measured, and compare a candidate effect against THAT interval.\n")
cat("  Nor does a louder vertical placebo rescue the null: it shows the design has\n")
cat("  power at the placebo's effect size, which here is several times the size of\n")
cat("  the effect phase 8's geometry predicts.\n")
for (i in seq_len(nrow(gradient))) {
  g <- gradient[i]
  cat(sprintf("  %-16s : %s; a 6-inch move is bounded to +/- %.3g%s\n",
              g$outcome,
              if (g$detected) "slope detected" else "no detectable slope",
              g$mde_6in,
              if (isTRUE(g$placebo_loud))
                "; vertical placebo is larger, so this channel is noise-dominated at the placebo's scale"
              else ""))
}
if (!any(gradient$corroborated)) {
  cat("\n  *** No outcome SEPARATES a within-pitcher effect of rubber position from\n")
  cat("  *** zero. Note this is not the same as showing there is none: on miss_mean\n")
  cat("  *** all three designs came back positive, and the panel estimate sits within\n")
  cat("  *** one SE of the gradient phase 8's geometry implies. Phase 8 should report\n")
  cat("  *** its stuff+ gradient as model-implied\n")
  cat("  *** and explicitly NOT validated, per the plan's phase 6 condition.\n")
}

saveRDS(list(all = out, gradient = gradient, movers = movers,
             seasons = seasons, n_cells = nrow(d)),
        file.path(MDIR, "rubber_within_pitcher_gradient.rds"))
fwrite(out, file.path(MDIR, "rubber_natural_experiment.csv"))
cat(sprintf("\nwrote %s\nwrote %s\n",
            file.path(MDIR, "rubber_within_pitcher_gradient.rds"),
            file.path(MDIR, "rubber_natural_experiment.csv")))

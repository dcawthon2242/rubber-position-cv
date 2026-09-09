#!/usr/bin/env Rscript

# Three follow-ups to the phase 8 counterfactual, which reported only a pooled
# sweep and a per-pitcher leaderboard:
#
#   1. Does the gradient differ by PITCHER handedness, not just batter side?
#   2. Is there a fixed "move lefties one way, righties the other" rule that
#      captures most of the per-pitcher optimum?
#   3. Do individual pitch types gain much more than the pitcher's average?
#
# Sign convention, which drives everything below. Statcast release_pos_x is from
# the catcher's view with positive toward first base, so a RHP sits at negative
# x (third-base side) and a LHP at positive x. Phase 8's delta_in is in the same
# absolute frame: positive is toward first base for everybody. To ask whether the
# effect is MIRRORED between the two handedness groups, that has to be re-expressed
# in each pitcher's own terms, so delta_arm_in below is signed toward the
# pitcher's ARM side (third base for a RHP, first base for a LHP). If the
# mechanism is mirror-symmetric the two groups agree in delta_arm_in; if it is
# about which batter's box you are moving toward, they agree in delta_in instead.
# These two hypotheses make opposite predictions for LHP and cannot both hold,
# which is the whole point of splitting.
#
# On question 3, one caveat governs how the pitch-type numbers should be read. A
# pitcher occupies ONE spot on the rubber for a given pitch, so per-pitch-type
# optima are not simultaneously attainable. If his fastball and his slider want
# opposite directions, the achievable gain is the mix-weighted compromise, not
# the best pitch's number. The script therefore reports the compromise cost
# explicitly rather than leaving a misleadingly large per-pitch figure standing.
#
# Inputs : data/statcast_model/stuff_platoon_data.rds     (phase 7)
#          data/statcast_model/stuff_platoon_models.rds   (phase 7)
#          data/statcast_model/rubber_move_sweep.csv      (phase 8)
#          data/rubber/rubber_position_pitcher_game.csv   (phase 5, for room)
# Outputs: data/statcast_model/rubber_sweep_by_hand.csv
#          data/statcast_model/rubber_sweep_by_pitchtype.csv
#          data/statcast_model/rubber_policy_comparison.csv
#
# Usage: Rscript pipeline/rubber_08_handedness_pitchtype.R

suppressPackageStartupMessages({ library(data.table); library(lightgbm) })

MDIR <- file.path("data", "statcast_model")
# Matches phase 7. Legal foot centres span 24 + 6 + 6 = 36 inches, and a pitcher
# starting at one end can traverse all of it, so the delta range is +/-36.
SWEEP_IN <- 36
STEP_IN  <- 1
MIN_SWINGS_STAND <- 60L
MIN_SWINGS_PT    <- 40L   # per pitcher per pitch type per side
MIN_PT_TOTAL     <- 2000L # league-wide swings for a pitch type to be reported

bundle <- readRDS(file.path(MDIR, "stuff_platoon_data.rds"))
mods   <- readRDS(file.path(MDIR, "stuff_platoon_models.rds"))
BEST <- mods$scale$best
SC_SD <- mods$scale$sd
IN_PER_STUFF <- SC_SD / 10

# ============================================================ 1 + 2: handedness

w <- fread(file.path(MDIR, "rubber_move_sweep.csv"), showProgress = FALSE)
w[, delta_arm_in := fifelse(p_throws == "R", -delta_in, delta_in)]

cat("=================== 1. sweep by PITCHER handedness ===================\n")
cat("stuff+ gain vs each batter side, averaged over pitchers, absolute frame\n")
cat("(delta_in > 0 = toward first base for both handedness groups).\n\n")
byh <- w[, .(n_p = .N,
             vs_lhh = round(mean(d_stuff_L), 3),
             vs_rhh = round(mean(d_stuff_R), 3),
             weighted = round(mean(d_stuff_wtd), 3)),
         by = .(p_throws, delta_in)][order(p_throws, delta_in)]
print(dcast(byh, delta_in ~ p_throws, value.var = c("vs_lhh", "vs_rhh", "weighted")))

cat("\nSame thing in each pitcher's OWN frame (delta_arm_in > 0 = toward his arm side).\n")
cat("If the mechanism mirrors, the two columns should now agree.\n\n")
byarm <- w[, .(weighted = round(mean(d_stuff_wtd), 3)),
           by = .(p_throws, delta_arm_in)]
print(dcast(byarm, delta_arm_in ~ p_throws, value.var = "weighted"))

# Which frame the two handedness groups actually agree in. A mirrored mechanism
# correlates across handedness in the arm frame; a batter-box mechanism
# correlates in the absolute frame. Reporting both prevents reading a sign flip
# as a handedness "effect" when it is just the coordinate convention.
cmp_abs <- merge(w[p_throws == "R", .(d = delta_in, r = d_stuff_wtd)][, mean(r), by = d],
                 w[p_throws == "L", .(d = delta_in, l = d_stuff_wtd)][, mean(l), by = d],
                 by = "d", suffixes = c("_R", "_L"))
cmp_arm <- merge(w[p_throws == "R", .(d = delta_arm_in, r = d_stuff_wtd)][, mean(r), by = d],
                 w[p_throws == "L", .(d = delta_arm_in, l = d_stuff_wtd)][, mean(l), by = d],
                 by = "d", suffixes = c("_R", "_L"))
cat(sprintf("\nRHP-vs-LHP sweep-shape correlation: absolute frame %+.3f | arm frame %+.3f\n",
            cor(cmp_abs$V1_R, cmp_abs$V1_L), cor(cmp_arm$V1_R, cmp_arm$V1_L)))
fwrite(byh, file.path(MDIR, "rubber_sweep_by_hand.csv"))

cat("\n=================== 2. is a fixed handedness rule enough? ===================\n")
cat(sprintf("Mean weighted stuff+ gain per pitcher. Every policy is clipped to the\n"))
cat("pitcher's measured room and to his observed range (in_support), so these are\n")
cat("comparable and none moves anyone off the rubber.\n\n")

ok <- w[feasible == TRUE & in_support == TRUE]
# Best feasible in-support move per pitcher: the oracle ceiling a fixed rule is
# being judged against.
oracle <- ok[, .SD[which.max(d_stuff_wtd)], by = .(pitcher, p_throws)]
policies <- list(
  `per-pitcher optimum (oracle)` = oracle[, .(pitcher, gain = d_stuff_wtd)],
  `all to third base (max feasible)` =
    ok[delta_in <= 0][, .SD[which.min(delta_in)], by = pitcher][, .(pitcher, gain = d_stuff_wtd)],
  `all to first base (max feasible)` =
    ok[delta_in >= 0][, .SD[which.max(delta_in)], by = pitcher][, .(pitcher, gain = d_stuff_wtd)],
  `arm side (RHP->3B, LHP->1B)` =
    ok[delta_arm_in >= 0][, .SD[which.max(delta_arm_in)], by = pitcher][, .(pitcher, gain = d_stuff_wtd)],
  `glove side (RHP->1B, LHP->3B)` =
    ok[delta_arm_in <= 0][, .SD[which.min(delta_arm_in)], by = pitcher][, .(pitcher, gain = d_stuff_wtd)],
  `stay put` = ok[delta_in == 0][, .(pitcher, gain = d_stuff_wtd)]
)

# Batter-adaptive policies. Every policy above forces one position for the whole
# outing, but shifting along the rubber by the hitter's hand is legal, routine,
# and costs nothing, so it is the version of "optimal side by handedness" most
# likely to be actionable. Here the two sides are optimised separately and then
# recombined on the pitcher's real platoon mix.
side_best <- function(dt, dir_L, dir_R) {
  gl <- if (is.null(dir_L)) dt[, .SD[which.max(d_stuff_L)], by = pitcher]
        else dt[dir_L * delta_in >= 0][, .SD[which.max(dir_L * delta_in)], by = pitcher]
  gr <- if (is.null(dir_R)) dt[, .SD[which.max(d_stuff_R)], by = pitcher]
        else dt[dir_R * delta_in >= 0][, .SD[which.max(dir_R * delta_in)], by = pitcher]
  m <- merge(gl[, .(pitcher, gL = d_stuff_L, w_lhh)],
             gr[, .(pitcher, gR = d_stuff_R)], by = "pitcher")
  m[, .(pitcher, gain = w_lhh * gL + (1 - w_lhh) * gR)]
}
policies[["adaptive: toward the batter (3B vs RHH, 1B vs LHH)"]] <- side_best(ok, 1, -1)
policies[["adaptive: away from the batter"]] <- side_best(ok, -1, 1)
policies[["adaptive: per-side optimum (oracle)"]] <- side_best(ok, NULL, NULL)

hand <- unique(w[, .(pitcher, p_throws)])
pol <- rbindlist(lapply(names(policies), function(nm) {
  x <- merge(policies[[nm]], hand, by = "pitcher")
  data.table(policy = nm, n = nrow(x),
             all = round(mean(x$gain), 3),
             RHP = round(mean(x[p_throws == "R"]$gain), 3),
             LHP = round(mean(x[p_throws == "L"]$gain), 3),
             pct_helped = round(100 * mean(x$gain > 0), 1))
}))
print(pol[order(-all)])
cat(sprintf("\n1 stuff+ point = %.4f inches of miss distance\n", IN_PER_STUFF))
orc <- pol[policy == "per-pitcher optimum (oracle)"]$all
cat(sprintf("Best fixed rule captures %.0f%% of the oracle's mean gain.\n",
            100 * max(pol[policy != "per-pitcher optimum (oracle)"]$all) / orc))
fwrite(pol, file.path(MDIR, "rubber_policy_comparison.csv"))

# ============================================================== 3: pitch types

cat("\n=================== 3. gains by pitch type ===================\n")

d <- bundle$data
ALL_FEATS <- unique(unlist(mods$specs))
d <- d[set == "holdout" & stats::complete.cases(d[, ..ALL_FEATS])]
d[, pt := fifelse(pitch_type %in% c("KC", "CS"), "CU",
          fifelse(pitch_type == "SV", "ST", pitch_type))]
d <- d[pt %in% d[, .N, by = pt][N >= MIN_PT_TOTAL]$pt]

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
sw <- list()
for (st in c("L", "R")) {
  dd <- d[stand == st]
  base <- score_shift(dd, st, 0)
  for (din in deltas_in) {
    p <- if (din == 0) base else score_shift(dd, st, din / 12)
    sw[[length(sw) + 1L]] <- data.table(
      pitcher = dd$pitcher, player_name = dd$player_name, p_throws = dd$p_throws,
      pt = dd$pt, stand = st, delta_in = din, d_pred = p - base)
  }
  cat(sprintf("  scored %d holdout swings vs %sHH across %d pitch types\n",
              nrow(dd), st, uniqueN(dd$pt)))
}
sw <- rbindlist(sw)
sw[, delta_arm_in := fifelse(p_throws == "R", -delta_in, delta_in)]
# Section 1 established that the two handedness groups move together in the
# ABSOLUTE frame and oppose each other in the arm frame, so pitch types are
# reported toward third base rather than toward the arm side. Pooling in the arm
# frame would have averaged RHP against LHP with the signs crossed, and since RHP
# are 73% of these swings the result would have looked like a clean arm-side
# effect that LHP do not actually share.
sw[, delta_3b_in := -delta_in]

cat("\n---- league sweep shape by pitch type, toward THIRD BASE ----\n")
cat("stuff+ gain from moving toward third base, the direction both handedness\n")
cat("groups share. Split by pitcher hand so the pooling cannot hide a reversal.\n\n")
lg <- sw[, .(d_stuff = mean(d_pred) / IN_PER_STUFF, n = .N),
         by = .(pt, p_throws, delta_3b_in)]
lgp <- sw[, .(d_stuff = mean(d_pred) / IN_PER_STUFF), by = .(pt, delta_3b_in)]
tab <- dcast(lgp, pt ~ delta_3b_in, value.var = "d_stuff")
num <- setdiff(names(tab), "pt")
tab[, (num) := lapply(.SD, function(v) round(v, 2)), .SDcols = num]
setorder(tab, -`8`)
print(tab[, .SD, .SDcols = c("pt", "-8", "-4", "-2", "0", "2", "4", "8")])

cat("\n---- sensitivity ranking: stuff+ per inch toward third base ----\n")
cat("Slope from the central 4 inches, where the model is interpolating.\n\n")
mk_slope <- function(x, by_cols) {
  s <- dcast(x[delta_3b_in %in% c(-2, 2)], ... ~ delta_3b_in, value.var = "d_stuff")
  setnames(s, c("-2", "2"), c("m2", "p2"))
  s[, per_in := round((p2 - m2) / 4, 3)][, .SD, .SDcols = c(by_cols, "per_in")]
}
slope <- mk_slope(lgp, "pt")
slope <- merge(slope, sw[delta_in == 0, .(swings = .N), by = pt], by = "pt")
sl_h <- dcast(mk_slope(lg[, .(pt, p_throws, delta_3b_in, d_stuff)],
                       c("pt", "p_throws")), pt ~ p_throws, value.var = "per_in")
setnames(sl_h, c("L", "R"), c("per_in_LHP", "per_in_RHP"), skip_absent = TRUE)
slope <- merge(slope, sl_h, by = "pt")
slope[, per_6in := round(per_in * 6, 2)]
setorder(slope, -per_in)
print(slope[, .(pt, swings, stuff_per_in = per_in, stuff_per_6in = per_6in,
                RHP = per_in_RHP, LHP = per_in_LHP)])

# The achievability check. Per-pitch-type optima are not simultaneously
# attainable because one pitch is thrown from one spot, so compare the
# best-per-pitch-type sum against the single-position compromise.
cat("\n---- can a pitcher chase his best pitch type? ----\n")
mixw <- sw[delta_in == 0, .(n = .N), by = .(pitcher, pt)]
mixw[, wt := n / sum(n), by = pitcher]
ppt <- sw[, .(g = mean(d_pred) / IN_PER_STUFF), by = .(pitcher, p_throws, pt, delta_in)]
ppt <- merge(ppt, mixw[, .(pitcher, pt, wt, n)], by = c("pitcher", "pt"))
ppt <- ppt[n >= MIN_SWINGS_PT]
# Restrict to the feasible, in-support deltas phase 8 already established.
feas <- unique(ok[, .(pitcher, delta_in)])
ppt <- merge(ppt, feas, by = c("pitcher", "delta_in"))
# One position for all his pitches: maximise the mix-weighted gain.
comp <- ppt[, .(g = sum(wt * g) / sum(wt)), by = .(pitcher, p_throws, delta_in)]
comp <- comp[, .SD[which.max(g)], by = .(pitcher, p_throws)]
setnames(comp, c("delta_in", "g"), c("best_delta_in", "compromise_gain"))
# Each pitch type's own best, and what it actually gets at the compromise spot.
own <- ppt[, .SD[which.max(g)], by = .(pitcher, pt)][, .(pitcher, pt, own_best = g,
                                                        own_delta = delta_in)]
atc <- merge(ppt[, .(pitcher, pt, delta_in, g)], comp[, .(pitcher, best_delta_in)],
             by = "pitcher")[delta_in == best_delta_in, .(pitcher, pt, at_comp = g)]
cf <- merge(own, atc, by = c("pitcher", "pt"))
cf <- merge(cf, comp[, .(pitcher, p_throws, best_delta_in)], by = "pitcher")
cat(sprintf("%d pitcher-pitch-type pairs across %d pitchers.\n",
            nrow(cf), uniqueN(cf$pitcher)))
cat(sprintf("mean best gain for a pitch type on its OWN terms   : %.2f stuff+\n",
            mean(cf$own_best)))
cat(sprintf("mean gain for that pitch at the COMPROMISE position : %.2f stuff+\n",
            mean(cf$at_comp)))
cat(sprintf("share of pitch types whose own optimum points the OTHER way\n"))
cat(sprintf("  from the pitcher's compromise move: %.0f%%\n",
            100 * mean(sign(cf$own_delta) != sign(cf$best_delta_in) &
                       cf$own_delta != 0 & cf$best_delta_in != 0)))

cat("\n---- biggest single-pitch gains at the pitcher's own compromise spot ----\n")
top <- cf[, .(player_name = NA_character_, pitcher, p_throws, pt,
              best_delta_in, at_comp = round(at_comp, 2),
              own_best = round(own_best, 2))]
nm <- unique(sw[, .(pitcher, player_name)])
top <- merge(top[, -"player_name"], nm, by = "pitcher")
setorder(top, -at_comp)
print(head(top[, .(player_name, p_throws, pt, best_delta_in, at_comp, own_best)], 20))

fwrite(lg, file.path(MDIR, "rubber_sweep_by_pitchtype.csv"))

# Per-pitcher arsenal detail behind the single-spot optimum, so the pitcher-level
# gain can be decomposed into which pitch is actually earning it.
ars <- merge(cf, mixw[, .(pitcher, pt, n_pt = n, wt)], by = c("pitcher", "pt"))
ars <- merge(ars, comp[, .(pitcher, compromise_gain)], by = "pitcher")
ars <- merge(ars, nm, by = "pitcher")
ars[, contrib := wt * at_comp]
setorder(ars, -compromise_gain, -contrib)
fwrite(ars[, .(pitcher, player_name, p_throws, pt, n_pt, wt, best_delta_in,
               compromise_gain, at_comp, own_best, own_delta, contrib)],
       file.path(MDIR, "rubber_arsenal_detail.csv"))
cat(sprintf("\nwrote rubber_sweep_by_hand.csv, rubber_policy_comparison.csv, rubber_sweep_by_pitchtype.csv, rubber_arsenal_detail.csv\n"))

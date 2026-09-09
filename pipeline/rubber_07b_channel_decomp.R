# ---------------------------------------------------------------------------
# Phase 7b: which feature channel carries the rubber-shift effect?
#
# Sliding a pitcher along the rubber touches three model inputs at once:
#
#   haa                  the horizontal approach angle the ball actually arrives
#                        at, recomputed exactly from the shifted vx0
#   release_pos_x        where the hand is in absolute space
#   rubber_x_in_rel_own  how far he is standing from his own usual spot
#
# The tempting move is to let only the first one respond, on the theory that HAA
# is the real geometric channel and release_pos_x mostly encodes pitcher identity
# (its training median is -1.52 ft across a -4.2..+4.3 ft span, so RHP occupy the
# negative mode and LHP the positive one, and a large shift walks a righty into
# the lefty mode).
#
# RESULT: that theory is wrong, and this script is what disproves it. Isolating
# HAA makes the pathology far WORSE, not better -- +22.98 stuff+ at -36 inches
# against +4.83 for the joint shift. The reason is that HAA is not an independent
# input. It is determined by release position and the velocity vector, so moving
# it alone manufactures a pitch that has a sweeper's approach angle, a fastball's
# movement and an unchanged release point: a combination that appears nowhere in
# training. The model then applies the full cross-sectional HAA-to-whiff slope,
# which is largely a proxy for pitch type rather than a causal gradient.
#
# Only the joint shift keeps the feature combination physically consistent, and
# it is the only variant that survives contact with the natural experiment. Over
# the in-support +/-6 inch region the implied slopes are:
#
#   all three (phase 7)   +0.00323 in/in   z = +0.7 vs measured   CONSISTENT
#   haa + release_pos_x   +0.00052 in/in   z = -1.3               consistent
#   haa only              +0.01538 in/in   z = +9.5               RULED OUT
#   haa + stance          +0.01808 in/in   z = +11.4              RULED OUT
#   release_pos_x only    -0.01372 in/in   z = -11.6              RULED OUT
#
# against a measured +0.00226 +/- 0.00138. So phase 7's existing joint shift is
# correct and should not be replaced with a physics-only channel. The U-shape at
# the edges of the full window is ordinary tree extrapolation, and the remedy for
# it is the in-support cap, not a change of channel.
#
# Inputs : data/statcast_model/stuff_platoon_data.rds
#          data/statcast_model/stuff_platoon_models.rds
# Output : data/statcast_model/rubber_channel_decomp.csv
#
# Usage: Rscript pipeline/rubber_07b_channel_decomp.R
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({ library(data.table); library(lightgbm) })

MDIR <- file.path("data", "statcast_model")
STEP_IN <- 3
SWEEP_IN <- 36
SUBSAMPLE <- 80000L   # mean curves converge long before the full holdout

bundle <- readRDS(file.path(MDIR, "stuff_platoon_data.rds"))
mods   <- readRDS(file.path(MDIR, "stuff_platoon_models.rds"))
BEST  <- mods$scale$best
SC_SD <- mods$scale$sd

d <- bundle$data
ALL_FEATS <- unique(unlist(mods$specs))
d <- d[set == "holdout" & stats::complete.cases(d[, ..ALL_FEATS])]
set.seed(1L)
if (nrow(d) > SUBSAMPLE) d <- d[sample(.N, SUBSAMPLE)]
cat(sprintf("channel decomposition on %d holdout swings\n", nrow(d)))

# Each channel is a set of inputs allowed to respond to the shift. Everything
# not listed stays at its observed value.
CHANNELS <- list(
  `haa only`                = c("haa"),
  `haa + stance`            = c("haa", "rubber_x_in_rel_own"),
  `haa + release_pos_x`     = c("haa", "release_pos_x"),
  `all three (phase 7)`     = c("haa", "release_pos_x", "rubber_x_in_rel_own"),
  `release_pos_x only`      = c("release_pos_x")
)

score_shift <- function(dd, st, delta_ft, active) {
  mo  <- mods$models[[paste(st, BEST, sep = "_")]]
  fts <- mo$feats
  x <- copy(dd)
  if ("release_pos_x" %in% active) x[, release_pos_x := release_pos_x + delta_ft]
  if ("rubber_x_in_rel_own" %in% active)
    x[, rubber_x_in_rel_own := rubber_x_in_rel_own + delta_ft * 12]
  if ("haa" %in% active) {
    x[, vy_p := vy0 + ay * t_plate]
    x[, vx_p_new := (vx0 - delta_ft / t_plate) + ax * t_plate]
    x[, haa := atan2(vx_p_new, -vy_p) * 180 / pi]
  }
  predict(mo$model, as.matrix(x[, ..fts]))
}

deltas_in <- seq(-SWEEP_IN, SWEEP_IN, by = STEP_IN)
out <- list()
for (cn in names(CHANNELS)) {
  active <- CHANNELS[[cn]]
  for (st in c("L", "R")) {
    dd <- d[stand == st]
    base <- score_shift(dd, st, 0, active)
    for (din in deltas_in) {
      p <- if (din == 0) base else score_shift(dd, st, din / 12, active)
      out[[length(out) + 1L]] <- data.table(
        channel = cn, stand = st, delta_in = din,
        d_stuff = mean(p - base) / (SC_SD / 10))
    }
  }
  cat(sprintf("  scored channel: %s\n", cn))
}
res <- rbindlist(out)

# Platoon-weight the two sides by the league mix so the curves are comparable.
w_lhh <- d[, mean(stand == "L")]
wide <- dcast(res, channel + delta_in ~ stand, value.var = "d_stuff")
wide[, wtd := w_lhh * L + (1 - w_lhh) * R]

cat("\n---- mean stuff+ gain by channel (platoon-weighted) ----\n")
print(dcast(wide[delta_in %in% seq(-36, 36, 6)], delta_in ~ channel, value.var = "wtd"))

# The diagnostic that matters: a channel is usable for extrapolation if the
# curve does not turn back up at both ends. Positive gain at BOTH extremes means
# the model is rewarding extremeness rather than direction.
cat("\n---- is the curve coherent, or U-shaped? ----\n")
for (cn in names(CHANNELS)) {
  z <- wide[channel == cn]
  lo <- z[delta_in == -36]$wtd; hi <- z[delta_in == 36]$wtd
  mid <- min(z[abs(delta_in) <= 6]$wtd)
  verdict <- if (lo > mid + 0.05 && hi > mid + 0.05)
    "U-SHAPED: rewards extremeness in both directions" else
    if (sign(lo) != sign(hi) || abs(lo - hi) > max(abs(c(lo, hi))))
      "DIRECTIONAL: one way helps, the other hurts" else "flat / weak"
  cat(sprintf("%-22s  at -36: %+6.2f   at +36: %+6.2f   -> %s\n", cn, lo, hi, verdict))
}

fwrite(wide[order(channel, delta_in)], file.path(MDIR, "rubber_channel_decomp.csv"))
cat(sprintf("\nwrote %s\n", file.path(MDIR, "rubber_channel_decomp.csv")))

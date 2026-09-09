#!/usr/bin/env Rscript

# Build the modeling table for the platoon-split stuff+ models.
#
# Target : miss_distance on competitive swings, same construction as
#          miss_grade_features.R -- contact is a ~0 miss, whiffs with no tracked
#          miss are unknown and drop out. Higher = the batter missed by more =
#          better pitch.
# Train  : 2023 (H2, when bat tracking begins) + 2024 + 2025
# Holdout: 2026
#
# The one structural departure from the existing stuff models. whiff_tjstuff.R
# mirrors left-handed pitchers (flips the sign of x-axis features) so that one
# model serves both hands. That mirroring is exactly what makes it blind to
# rubber position: after the flip, "six inches toward first base" and "six inches
# toward third base" are the same number, and the platoon asymmetry that makes
# rubber position interesting is folded away. So here LHP are NOT mirrored, every
# x-axis feature keeps its sign, and the models are fit separately versus LHH and
# versus RHH. Fitting separately is a full interaction with `stand`, which
# subsumes the rubber_x_in x stand and HAA x stand interactions the plan calls
# for -- no explicit product terms are needed.
#
# Feature groups, kept as named blocks so stuff_platoon_train.R can add them one
# at a time and measure what each is worth:
#   SHAPE     velo, spin rate, pfx_x/pfx_z, spin axis (sin/cos), ax/az, and the
#             diffs versus that pitcher's primary fastball
#   RELEASE   signed release_pos_x, release_pos_z, release_extension, arm_angle
#   APPROACH  horizontal and vertical approach angle at the plate
#   RUBBER    rubber_x_in (absolute, from phase 5 when calibrated) and
#             rubber_x_in_rel_own (within-pitcher inches, always available)
#   PITCHER   leave-one-season-out shrunk pitcher quality, so that release_pos_x
#             cannot quietly stand in for pitcher identity
#
# Why PITCHER matters more here than in the existing models. Signed release_pos_x
# is close to a pitcher fingerprint: arm slot, height, and handedness all read
# off it. A tree given release_pos_x and no pitcher term will happily use it to
# recover "who is this" and post a good RMSE that says nothing about position.
# The encoding is built leave-one-season-out from training seasons only, so a
# pitcher's own season never informs his own feature.
#
# Inputs : data/statcast_<season>/statcast_<season>_all.csv
#          data/rubber/rubber_position_pitcher_game.csv   (phase 5)
# Output : data/statcast_model/stuff_platoon_data.rds
#
# Usage: Rscript pipeline/stuff_platoon_features.R

suppressPackageStartupMessages({ library(data.table) })

OUT_DIR <- file.path("data", "statcast_model")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

SEASONS_TRAIN <- c(2023L, 2024L, 2025L)
SEASON_TEST   <- 2026L
FASTBALLS <- c("FF", "SI", "FC")
BREAKING  <- c("SL", "ST", "CU", "KC", "SV", "CS")
OFFSPEED  <- c("CH", "FS", "FO")
MISS_START <- as.Date("2023-07-14")

WHIFF <- c("swinging_strike", "swinging_strike_blocked", "foul_tip")
SWING <- c(WHIFF, "foul", "hit_into_play")

NEED <- c("pitcher", "player_name", "pitch_type", "game_type", "description",
          "game_date", "game_pk", "at_bat_number", "pitch_number",
          "balls", "strikes", "stand", "p_throws", "home_team", "miss_distance",
          "release_speed", "release_spin_rate", "release_extension",
          "release_pos_x", "release_pos_y", "release_pos_z", "spin_axis",
          "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
          "plate_x", "plate_z", "arm_angle", "delta_run_exp")

yf <- 17 / 12
cmean <- function(a) {
  a <- a[!is.na(a)]
  if (!length(a)) return(NA_real_)
  r <- a * pi / 180
  ((atan2(mean(sin(r)), mean(cos(r))) * 180 / pi) + 360) %% 360
}
circd <- function(a, b) { d <- abs(a - b) %% 360; pmin(d, 360 - d) }

load_season <- function(yr) {
  f <- file.path("data", sprintf("statcast_%d", yr), sprintf("statcast_%d_all.csv", yr))
  if (!file.exists(f)) { message("MISSING season file: ", f); return(NULL) }
  d <- fread(f, select = NEED, showProgress = FALSE)
  d[, season := yr]
  d <- d[game_type == "R" & !is.na(vx0) & pitch_type != "" & !is.na(release_pos_y) &
         stand %in% c("L", "R")]
  d[, game_date := as.Date(game_date)]
  if (yr == 2023L) d <- d[game_date >= MISS_START]

  # ---- approach angles at the front of the plate ----
  # Both are signed in the same frame as release_pos_x (positive x = first-base
  # side), so HAA is positive when the ball is travelling toward first base as it
  # crosses. This is the channel a rubber move acts through: shifting the release
  # point laterally while holding the plate location fixed rotates HAA.
  d[, t_plate := (-vy0 - sqrt(vy0^2 - 2 * ay * (release_pos_y - yf))) / ay]
  d[, `:=`(vx_p = vx0 + ax * t_plate,
           vy_p = vy0 + ay * t_plate,
           vz_p = vz0 + az * t_plate)]
  d[, haa := atan2(vx_p, -vy_p) * 180 / pi]
  d[, vaa := atan2(vz_p, sqrt(vx_p^2 + vy_p^2)) * 180 / pi]

  # ---- primary fastball anchor per pitcher x season ----
  fb <- d[pitch_type %in% FASTBALLS,
          .(nfb = .N, fb_velo = mean(release_speed, na.rm = TRUE),
            fb_ax = mean(ax, na.rm = TRUE), fb_az = mean(az, na.rm = TRUE),
            fb_pfx_x = mean(pfx_x, na.rm = TRUE), fb_pfx_z = mean(pfx_z, na.rm = TRUE),
            fb_axis = cmean(spin_axis)),
          by = .(pitcher, pitch_type)]
  fb[, rank := fifelse(pitch_type == "FF", 1L, fifelse(pitch_type == "SI", 2L, 3L))]
  fb <- fb[nfb >= 50][order(pitcher, rank)][, .SD[1], by = pitcher][
    , .(pitcher, fb_velo, fb_ax, fb_az, fb_pfx_x, fb_pfx_z, fb_axis)]
  d <- merge(d, fb, by = "pitcher", all.x = TRUE)
  d[, `:=`(speed_diff = release_speed - fb_velo,
           ax_diff = ax - fb_ax, az_diff = az - fb_az,
           pfx_x_diff = pfx_x - fb_pfx_x, pfx_z_diff = pfx_z - fb_pfx_z,
           axis_diff = circd(spin_axis, fb_axis))]

  d[, grp := fifelse(pitch_type %in% FASTBALLS, "fastball",
             fifelse(pitch_type %in% BREAKING, "breaking",
             fifelse(pitch_type %in% OFFSPEED, "offspeed", "other")))]
  d[, is_swing := description %in% SWING]
  d[, is_whiff := description %in% WHIFF]
  d <- d[is_swing == TRUE & !(is_whiff & is.na(miss_distance))]
  d[is.na(miss_distance), miss_distance := 0]

  out <- d[, .(season, game_pk, at_bat_number, pitch_number, game_date,
               pitcher, player_name, p_throws, stand, park = home_team,
               pitch_type, grp, balls, strikes,
               miss_distance, is_whiff, delta_run_exp,
               release_speed, release_spin_rate, release_extension,
               release_pos_x, release_pos_z, arm_angle,
               spin_axis, pfx_x, pfx_z, ax, az,
               speed_diff, ax_diff, az_diff, pfx_x_diff, pfx_z_diff, axis_diff,
               haa, vaa, t_plate, vx0, vy0, ay, plate_x, plate_z)]
  message(sprintf("  %d: %d competitive swings (%.1f%% whiffs), mean miss=%.2f",
                  yr, nrow(out), 100 * mean(out$is_whiff), mean(out$miss_distance)))
  out
}

message("loading seasons")
d <- rbindlist(lapply(c(SEASONS_TRAIN, SEASON_TEST), load_season),
               use.names = TRUE, fill = TRUE)

d[, sax := sin(spin_axis * pi / 180)]
d[, cax := cos(spin_axis * pi / 180)]
d[, set := fifelse(season == SEASON_TEST, "holdout", "train")]

# ------------------------------------------------------------ rubber position

pos_f <- file.path("data", "rubber", "rubber_position_pitcher_game.csv")
if (file.exists(pos_f)) {
  pos <- fread(pos_f, showProgress = FALSE)
  pcols <- intersect(c("game_pk", "pitcher", "stand", "rubber_x_in",
                       "rubber_x_in_rel_own", "release_pos_x_med", "source"),
                     names(pos))
  d <- merge(d, pos[, ..pcols], by = c("game_pk", "pitcher", "stand"), all.x = TRUE)
  setnames(d, "source", "rubber_source", skip_absent = TRUE)
  message(sprintf("rubber position joined: %.1f%% of swings have rubber_x_in_rel_own, %.1f%% absolute",
                  100 * mean(!is.na(d$rubber_x_in_rel_own)),
                  100 * mean(!is.na(d$rubber_x_in))))
} else {
  message("no phase 5 output at ", pos_f, " -- RUBBER features will be all NA")
  d[, `:=`(rubber_x_in = NA_real_, rubber_x_in_rel_own = NA_real_,
           release_pos_x_med = NA_real_, rubber_source = NA_character_)]
}

# --------------------------------------------------- pitcher quality encoding

# Leave-one-season-out, shrunk toward the league mean. Built from training
# seasons only so the holdout is never used to construct its own feature, and
# excluding the pitcher's own season so a good year cannot predict itself.
K_SHRINK <- 300  # swings of prior weight; ~ half a season of a starter's swings
tr <- d[set == "train"]
league_mu <- mean(tr$miss_distance)
ps <- tr[, .(s = sum(miss_distance), n = .N), by = .(pitcher, season)]
tot <- ps[, .(S = sum(s), N = sum(n)), by = pitcher]
ps <- merge(ps, tot, by = "pitcher")
ps[, pitcher_q := (S - s + K_SHRINK * league_mu) / (N - n + K_SHRINK)]

# Holdout gets the pitcher's full training history, which is genuinely
# out-of-sample for 2026.
hold_q <- tot[, .(pitcher, season = SEASON_TEST,
                  pitcher_q = (S + K_SHRINK * league_mu) / (N + K_SHRINK))]
q <- rbind(ps[, .(pitcher, season, pitcher_q)], hold_q)
d <- merge(d, q, by = c("pitcher", "season"), all.x = TRUE)
d[is.na(pitcher_q), pitcher_q := league_mu]   # debutants get the league prior

message(sprintf("pitcher_q: league mean %.3f, SD across rows %.3f",
                league_mu, sd(d$pitcher_q)))

# ------------------------------------------------------------ feature blocks

BLOCKS <- list(
  SHAPE    = c("release_speed", "release_spin_rate", "pfx_x", "pfx_z",
               "sax", "cax", "ax", "az",
               "speed_diff", "ax_diff", "az_diff", "pfx_x_diff", "pfx_z_diff",
               "axis_diff"),
  RELEASE  = c("release_pos_x", "release_pos_z", "release_extension", "arm_angle"),
  APPROACH = c("haa", "vaa"),
  RUBBER   = c("rubber_x_in_rel_own"),
  PITCHER  = c("pitcher_q")
)

saveRDS(list(data = d, blocks = BLOCKS,
             seasons_train = SEASONS_TRAIN, season_test = SEASON_TEST,
             league_mu = league_mu),
        file.path(OUT_DIR, "stuff_platoon_data.rds"))

message(sprintf("\nwrote %s : %d rows (train=%d, holdout=%d)",
                file.path(OUT_DIR, "stuff_platoon_data.rds"), nrow(d),
                sum(d$set == "train"), sum(d$set == "holdout")))
message("by pitcher hand x batter side:")
print(d[, .(n = .N, mean_miss = round(mean(miss_distance), 3)), by = .(p_throws, stand)][
  order(p_throws, stand)])
message("signed release_pos_x by pitcher hand (mirroring deliberately NOT applied):")
print(d[, .(mean_rel_x = round(mean(release_pos_x, na.rm = TRUE), 3)), by = p_throws])

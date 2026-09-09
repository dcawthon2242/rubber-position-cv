#!/usr/bin/env Rscript

# Train the platoon-split stuff+ models, and measure what release position and
# rubber position are actually worth on top of pitch shape.
#
# One model versus LHH and one versus RHH, each trained on 2023H2-2025 and
# scored on the 2026 holdout, following the LightGBM setup in
# miss_grade_train.R. Fitting separately is a full interaction with `stand`, so
# the rubber_x_in x stand and HAA x stand interactions the plan asks for are
# built in rather than added as product terms.
#
# Feature blocks go in cumulatively so each addition has a price tag:
#
#   B0  SHAPE + PITCHER            the honest baseline
#   B1  + RELEASE                  signed release_pos_x/z, extension, arm angle
#   B2  + APPROACH                 horizontal and vertical approach angle
#   B3  + RUBBER                   within-pitcher lateral position in inches
#   NP  SHAPE + RELEASE + APPROACH, PITCHER REMOVED
#
# NP is the diagnostic that motivates the whole design. If dropping the pitcher
# term IMPROVES holdout RMSE, the release features were serving as a pitcher
# fingerprint rather than as physics, and any counterfactual that moves them is
# really just moving the model to a different pitcher's identity. Comparing NP
# to B2 puts a number on how much of the release features' apparent value is
# identity leakage.
#
# Outputs: data/statcast_model/stuff_platoon_models.rds   (phase 8 scores with this)
#          data/statcast_model/stuff_platoon_metrics.csv
#          data/statcast_model/stuff_platoon_importance.csv
#
# Usage: Rscript pipeline/stuff_platoon_train.R

suppressPackageStartupMessages({ library(data.table); library(lightgbm) })
set.seed(1)

MDIR <- file.path("data", "statcast_model")
bundle <- readRDS(file.path(MDIR, "stuff_platoon_data.rds"))
d <- bundle$data
BLOCKS <- bundle$blocks

SPECS <- list(
  B0  = c(BLOCKS$SHAPE, BLOCKS$PITCHER),
  B1  = c(BLOCKS$SHAPE, BLOCKS$PITCHER, BLOCKS$RELEASE),
  B2h = c(BLOCKS$SHAPE, BLOCKS$PITCHER, BLOCKS$RELEASE, "haa"),
  B2  = c(BLOCKS$SHAPE, BLOCKS$PITCHER, BLOCKS$RELEASE, BLOCKS$APPROACH),
  B3  = c(BLOCKS$SHAPE, BLOCKS$PITCHER, BLOCKS$RELEASE, BLOCKS$APPROACH, BLOCKS$RUBBER),
  NP  = c(BLOCKS$SHAPE, BLOCKS$RELEASE, BLOCKS$APPROACH)
)
SPEC_LABEL <- c(B0  = "shape + pitcher",
                B1  = "+ release",
                B2h = "+ HAA only (no VAA)",
                B2  = "+ approach angles (HAA+VAA)",
                B3  = "+ rubber position",
                NP  = "no pitcher term (leakage probe)")

# B2h exists because the two approach angles are not equivalent for this
# question. HAA is the channel a rubber move acts through. VAA is largely a
# restatement of where the pitch ended up vertically, so admitting it turns the
# model into stuff-plus-location -- which is precisely what TJStuff+ excludes on
# purpose. Keeping HAA-only alongside the full version shows how much of the
# approach-angle gain is the rubber-relevant part versus location leaking in.

# Every spec must be judged on the SAME rows. rubber_x_in_rel_own is missing for
# roughly a sixth of swings (short relief outings below the phase 5 fastball
# minimum), so scoring B3 on its own available rows and B2 on its larger set
# compares two different populations -- which shows up as RMSE falling while R2
# also falls, a tell that the denominator moved rather than the model improving.
ALL_FEATS <- unique(unlist(SPECS))

train_lgb <- function(dtr, feats, nrounds = 3000) {
  dtr <- dtr[stats::complete.cases(dtr[, ..feats]) & is.finite(miss_distance)]
  n <- nrow(dtr)
  vi <- sample(n, floor(0.15 * n))
  dtrain <- lgb.Dataset(as.matrix(dtr[-vi, ..feats]), label = dtr$miss_distance[-vi])
  dval <- lgb.Dataset.create.valid(dtrain, as.matrix(dtr[vi, ..feats]),
                                   label = dtr$miss_distance[vi])
  lgb.train(params = list(objective = "regression", metric = "rmse",
                          learning_rate = 0.05, num_leaves = 31,
                          min_data_in_leaf = 200, feature_fraction = 0.8,
                          bagging_fraction = 0.8, bagging_freq = 1),
            data = dtrain, nrounds = nrounds, valids = list(val = dval),
            early_stopping_rounds = 60, verbose = -1)
}

met <- function(pred, act) {
  ok <- is.finite(pred) & is.finite(act)
  pred <- pred[ok]; act <- act[ok]
  c(rmse = sqrt(mean((pred - act)^2)),
    r2 = 1 - sum((act - pred)^2) / sum((act - mean(act))^2),
    spearman = suppressWarnings(cor(pred, act, method = "spearman")),
    n = length(pred))
}

n_before <- nrow(d)
d <- d[stats::complete.cases(d[, ..ALL_FEATS])]
cat(sprintf("common complete-case restriction: %d -> %d swings (%.1f%% kept)\n",
            n_before, nrow(d), 100 * nrow(d) / n_before))

models <- list(); metrics <- list(); imps <- list()

for (st in c("L", "R")) {
  tr <- d[set == "train" & stand == st]
  ho <- d[set == "holdout" & stand == st]
  cat(sprintf("\n=== versus %sHH : train %d swings, holdout %d ===\n",
              st, nrow(tr), nrow(ho)))

  for (sp in names(SPECS)) {
    feats <- SPECS[[sp]]
    m <- train_lgb(tr, feats)
    pred <- predict(m, as.matrix(ho[, ..feats]))
    mm <- met(pred, ho$miss_distance)
    metrics[[length(metrics) + 1L]] <- data.table(
      stand = st, spec = sp, label = SPEC_LABEL[[sp]], n_feat = length(feats),
      best_iter = m$best_iter, t(mm))
    models[[paste(st, sp, sep = "_")]] <- list(model = m, feats = feats)

    ii <- as.data.table(lgb.importance(m, percentage = TRUE))
    ii[, `:=`(stand = st, spec = sp)]
    imps[[length(imps) + 1L]] <- ii
    cat(sprintf("  %-3s %-32s rmse %.5f  r2 %.4f  iters %d\n",
                sp, SPEC_LABEL[[sp]], mm[["rmse"]], mm[["r2"]], m$best_iter))
  }
}

metrics <- rbindlist(metrics)
imps <- rbindlist(imps)

# ------------------------------------------------------------------- report

cat("\n================ 2026 holdout, target = miss_distance | swing ================\n")
print(metrics[, .(stand, spec, label, rmse = round(rmse, 5), r2 = round(r2, 4),
                  spearman = round(spearman, 4), n)])

cat("\n---- what each block buys, as holdout RMSE reduction ----\n")
ladder <- c("B0", "B1", "B2h", "B2", "B3")
delta <- metrics[spec %in% ladder]
delta[, spec := factor(spec, levels = ladder)]
setorder(delta, stand, spec)
delta[, gain_vs_prev := c(NA_real_, -diff(rmse)), by = stand]
delta[, gain_vs_B0 := rmse[spec == "B0"] - rmse, by = stand]
print(delta[, .(stand, spec, label, rmse = round(rmse, 5),
                gain_vs_prev = signif(gain_vs_prev, 3),
                gain_vs_B0 = signif(gain_vs_B0, 3))])

cat("\n---- pitcher-identity leakage probe ----\n")
lk <- dcast(metrics[spec %in% c("B2", "NP")], stand ~ spec, value.var = "rmse")
lk[, np_minus_b2 := NP - B2]
print(lk[, .(stand, B2 = round(B2, 5), NP = round(NP, 5),
             np_minus_b2 = signif(np_minus_b2, 3))])
if (any(lk$np_minus_b2 < 0)) {
  cat("  Dropping the pitcher term IMPROVED holdout RMSE. The release features\n")
  cat("  are carrying pitcher identity, so a counterfactual that shifts them is\n")
  cat("  partly relabelling the pitcher. Phase 8 must treat its gradient as\n")
  cat("  descriptive.\n")
} else {
  cat("  The pitcher term helps, so release features are not simply standing in\n")
  cat("  for pitcher identity.\n")
}

cat("\n---- importance of the position features in the full model (B3) ----\n")
POS <- c(BLOCKS$RELEASE, BLOCKS$APPROACH, BLOCKS$RUBBER)
pos_imp <- imps[spec == "B3" & Feature %in% POS,
                .(stand, Feature, gain_pct = round(100 * Gain, 2))]
print(dcast(pos_imp, Feature ~ stand, value.var = "gain_pct"))
cat(sprintf("\nposition features hold %.1f%% (vs LHH) / %.1f%% of total gain (vs RHH)\n",
            100 * imps[spec == "B3" & stand == "L" & Feature %in% POS, sum(Gain)],
            100 * imps[spec == "B3" & stand == "R" & Feature %in% POS, sum(Gain)]))

# ------------------------------------------------- stuff+ scale and deliverable

# Scale the winning spec to the familiar 100-mean, 10-SD stuff+ convention.
# Both platoon models are centred on the SAME holdout population so that a
# pitcher's vs-LHH and vs-RHH numbers stay on one comparable scale -- centring
# each separately would erase the platoon advantage the models exist to measure.
BEST <- "B3"
sc <- list()
allpred <- rbindlist(lapply(c("L", "R"), function(st) {
  mo <- models[[paste(st, BEST, sep = "_")]]
  best_feats <- mo$feats
  ho <- d[set == "holdout" & stand == st]
  ho[, pred_miss := predict(mo$model, as.matrix(ho[, ..best_feats]))]
  ho[, .(season, game_pk, at_bat_number, pitch_number, pitcher, player_name,
         p_throws, stand, pitch_type, grp, miss_distance, pred_miss)]
}))
mu <- mean(allpred$pred_miss); sg <- sd(allpred$pred_miss)
allpred[, stuff_plus := 100 + 10 * (pred_miss - mu) / sg]
sc$mu <- mu; sc$sd <- sg; sc$best <- BEST

cat(sprintf("\nstuff+ scale from %s holdout preds: mean %.4f in, SD %.4f in\n",
            BEST, mu, sg))
cat(sprintf("1 stuff+ point = %.4f inches of miss distance\n", sg / 10))
cat("\nholdout stuff+ by matchup (same-hand advantage should be visible):\n")
print(allpred[, .(n = .N, stuff_plus = round(mean(stuff_plus), 1),
                  actual_miss = round(mean(miss_distance), 3)),
              by = .(p_throws, stand)][order(p_throws, stand)])

saveRDS(list(models = models, specs = SPECS, spec_label = SPEC_LABEL,
             blocks = BLOCKS, metrics = metrics, importance = imps,
             scale = sc, leakage = lk),
        file.path(MDIR, "stuff_platoon_models.rds"))
fwrite(metrics, file.path(MDIR, "stuff_platoon_metrics.csv"))
fwrite(imps, file.path(MDIR, "stuff_platoon_importance.csv"))
fwrite(allpred, file.path(MDIR, "stuff_platoon_holdout_preds.csv"))
cat(sprintf("\nwrote %s\nwrote %s\nwrote %s\n",
            file.path(MDIR, "stuff_platoon_models.rds"),
            file.path(MDIR, "stuff_platoon_metrics.csv"),
            file.path(MDIR, "stuff_platoon_holdout_preds.csv")))

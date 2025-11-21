#!/usr/bin/env Rscript
library(glmmTMB)
library(broom.mixed)

RESPONSE_FAMILIES <- list(
  fish_activity = "poisson",
  fish_richness = "poisson",
  fish_present = "binomial",
  dolphin_activity = "nbinom2",
  dolphin_whistles = "nbinom2",
  dolphin_echolocation = "nbinom2",
  dolphin_burst_pulses = "nbinom2",
  dolphin_present = "binomial",
  vessel_present = "binomial"
)

GROUP_FILES <- list(
  fish = "data/processed/model_data_fish.csv",
  dolphin = "data/processed/model_data_dolphin.csv",
  vessel = "data/processed/model_data_vessel.csv",
  universal = "data/processed/model_data_universal.csv"
)

OUTPUT_DIR <- "data/processed/glmm_results"
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

component_models_for_response <- function(df, response) {
  df$month <- as.factor(df$month)
  df$hour <- as.factor(df$hour)
  df$station <- as.factor(df$station)
  fam_name <- RESPONSE_FAMILIES[[response]]
  fam_fun <- switch(fam_name,
    "poisson" = poisson(),
    "nbinom2" = nbinom2(),
    "binomial" = binomial(),
    poisson()
  )
  meta <- c("Date", "month", "hour", "station", "temp", "depth", response)
  idx <- setdiff(colnames(df), meta)
  idx <- idx[sapply(df[idx], is.numeric)]

  models <- list()
  # env only + station
  f_env_sta <- as.formula(paste(response, "~ temp + depth + month + hour + (1|station)"))
  models$env_station <- tryCatch(glmmTMB(f_env_sta, data=df, family=fam_fun), error=function(e) NULL)
  # indices only + station
  if (length(idx) > 0) {
    f_idx_sta <- as.formula(paste(response, "~", paste(idx, collapse=" + "), "+ month + hour + (1|station)"))
    models$idx_station <- tryCatch(glmmTMB(f_idx_sta, data=df, family=fam_fun), error=function(e) NULL)
  } else {
    models$idx_station <- NULL
  }
  # indices only, no station
  if (length(idx) > 0) {
    f_idx_nosta <- as.formula(paste(response, "~", paste(idx, collapse=" + "), "+ month + hour"))
    models$idx_no_station <- tryCatch(glmmTMB(f_idx_nosta, data=df, family=fam_fun), error=function(e) NULL)
  } else {
    models$idx_no_station <- NULL
  }
  # env only, no station
  f_env_nosta <- as.formula(paste(response, "~ temp + depth + month + hour"))
  models$env_no_station <- tryCatch(glmmTMB(f_env_nosta, data=df, family=fam_fun), error=function(e) NULL)
  # full (indices + env + station)
  if (length(idx) > 0) {
    f_full <- as.formula(paste(response, "~", paste(c(idx, "temp", "depth", "month", "hour"), collapse=" + "), "+ (1|station)"))
    models$full <- tryCatch(glmmTMB(f_full, data=df, family=fam_fun), error=function(e) NULL)
  } else {
    models$full <- NULL
  }
  return(models)
}

score_model <- function(model, df, response, family_name) {
  if (is.null(model)) return(list(AIC=NA, logLik=NA))
  aic <- AIC(model)
  ll  <- as.numeric(logLik(model))
  return(list(AIC=aic, logLik=ll))
}

run_ablation <- function(group, path) {
  cat(sprintf("\n=== COMPONENT ABLATION: %s ===\n", group))
  df <- read.csv(path)
  responses <- intersect(names(RESPONSE_FAMILIES), colnames(df))
  out_rows <- data.frame()
  for (resp in responses) {
    fam <- RESPONSE_FAMILIES[[resp]]
    models <- component_models_for_response(df, resp)
    for (name in names(models)) {
      sc <- score_model(models[[name]], df, resp, fam)
      out_rows <- rbind(out_rows, data.frame(
        group=group, response=resp, model=name,
        family=fam, AIC=sc$AIC, logLik=sc$logLik
      ))
    }
  }
  out_path <- file.path(OUTPUT_DIR, sprintf("component_ablation_%s.csv", group))
  write.csv(out_rows, out_path, row.names=FALSE)
  cat(sprintf("Saved: %s\n", out_path))
}

for (g in names(GROUP_FILES)) {
  if (file.exists(GROUP_FILES[[g]])) {
    run_ablation(g, GROUP_FILES[[g]])
  }
}

cat("\n✅ Component ablation finished.\n")
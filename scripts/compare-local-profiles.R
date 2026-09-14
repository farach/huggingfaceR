args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: compare-local-profiles.R <profile-output-dir>")
}
output_dir <- args[1]
baseline <- readRDS(file.path(output_dir, "baseline-results.rds"))
refactored <- readRDS(file.path(output_dir, "refactored-results.rds"))
stopifnot(
  identical(baseline$public_api, refactored$public_api),
  identical(baseline$hot_path, refactored$hot_path),
  isTRUE(all.equal(baseline$samples, refactored$samples, tolerance = 1e-6))
)

baseline_metrics <- utils::read.csv(file.path(output_dir, "baseline-metrics.csv"))
refactored_metrics <- utils::read.csv(file.path(output_dir, "refactored-metrics.csv"))
keys <- c("task", "method", "n_texts")
comparison <- merge(
  baseline_metrics, refactored_metrics, by = keys,
  suffixes = c("_baseline", "_refactored"), all = TRUE
)
stopifnot(
  nrow(comparison) == nrow(baseline_metrics),
  nrow(comparison) == nrow(refactored_metrics),
  !anyNA(comparison$median_s_baseline),
  !anyNA(comparison$median_s_refactored)
)
comparison$time_ratio <- comparison$median_s_refactored / comparison$median_s_baseline
utils::write.csv(comparison, file.path(output_dir, "comparison.csv"), row.names = FALSE)
jsonlite::write_json(
  list(
    status = "PASS", public_api_unchanged = TRUE,
    prediction_code_unchanged = TRUE, predictions_equal = TRUE,
    tolerance = 1e-6,
    timing_note = "Ratios are descriptive, not CI thresholds or evidence of a statistical speedup.",
    allocation_note = "Allocated R bytes exclude Python/PyTorch allocations and are not peak memory."
  ),
  file.path(output_dir, "comparison.json"),
  pretty = TRUE, auto_unbox = TRUE
)
print(comparison[c(keys, "median_s_baseline", "median_s_refactored", "time_ratio")],
      row.names = FALSE)

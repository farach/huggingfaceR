# Usage: Rscript compare-api-profiles.R <profile-output-dir>
#
# Fails unless the baseline and current versions returned the same outputs for
# every profiled scenario, then reports the paired timings.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: compare-api-profiles.R <profile-output-dir>")
}
output_dir <- args[1]
baseline <- readRDS(file.path(output_dir, "outputs", "baseline-outputs.rds"))
current <- readRDS(file.path(output_dir, "outputs", "current-outputs.rds"))
stopifnot(setequal(names(baseline), names(current)))

# Types, classes, names, and dimensions must match exactly; numeric values may
# differ only by floating-point rounding from a different summation order.
structure_of <- function(x) {
  out <- list(
    type = typeof(x), class = class(x), names = names(x),
    dim = dim(x), length = length(x)
  )
  if (is.list(x)) {
    out$children <- lapply(x, structure_of)
  }
  out
}
tolerance <- 1e-10
differences <- list()
for (key in sort(names(baseline))) {
  same_structure <- identical(structure_of(baseline[[key]]), structure_of(current[[key]]))
  values <- all.equal(baseline[[key]], current[[key]], tolerance = tolerance)
  if (!same_structure || !isTRUE(values)) {
    differences[[key]] <- c(
      if (!same_structure) "structure differs (type, class, names, or dimensions)",
      if (!isTRUE(values)) utils::head(values, 5L)
    )
  }
}

status <- if (length(differences) == 0L) "PASS" else "FAIL"
for (key in names(differences)) {
  cat("Output mismatch in ", key, ":\n  ", paste(differences[[key]], collapse = "\n  "),
      "\n", sep = "")
}

baseline_metrics <- utils::read.csv(file.path(output_dir, "baseline-metrics.csv"))
current_metrics <- utils::read.csv(file.path(output_dir, "current-metrics.csv"))
keys <- c("scenario", "n")
comparison <- merge(
  baseline_metrics, current_metrics, by = keys,
  suffixes = c("_baseline", "_current"), all = TRUE
)
stopifnot(
  nrow(comparison) == nrow(baseline_metrics),
  nrow(comparison) == nrow(current_metrics),
  !anyNA(comparison$median_s_baseline),
  !anyNA(comparison$median_s_current)
)
comparison$speedup <- comparison$median_s_baseline / comparison$median_s_current
comparison$allocation_ratio <-
  comparison$r_allocated_mib_current / comparison$r_allocated_mib_baseline
comparison <- comparison[order(comparison$scenario, comparison$n), ]
utils::write.csv(comparison, file.path(output_dir, "comparison.csv"), row.names = FALSE)

table <- data.frame(
  scenario = comparison$scenario,
  n = comparison$n,
  baseline_ms = round(comparison$median_s_baseline * 1000, 2),
  current_ms = round(comparison$median_s_current * 1000, 2),
  speedup = round(comparison$speedup, 2),
  baseline_mib = round(comparison$r_allocated_mib_baseline, 1),
  current_mib = round(comparison$r_allocated_mib_current, 1)
)
print(table, row.names = FALSE)

summary_path <- Sys.getenv("GITHUB_STEP_SUMMARY")
if (nzchar(summary_path)) {
  lines <- c(
    paste0("## API overhead profile: ", status),
    "",
    "Median warm R-side time with HTTP mocked (network and inference excluded).",
    "`floor_*` rows time only the compiled routine on the same data.",
    "",
    "| scenario | n | baseline ms | current ms | speedup | baseline MiB | current MiB |",
    "|---|---:|---:|---:|---:|---:|---:|",
    sprintf(
      "| %s | %d | %.2f | %.2f | %.2f | %.1f | %.1f |",
      table$scenario, as.integer(table$n), table$baseline_ms, table$current_ms,
      table$speedup, table$baseline_mib, table$current_mib
    )
  )
  cat(lines, file = summary_path, sep = "\n", append = TRUE)
}

jsonlite::write_json(
  list(
    status = status,
    outputs_compared = length(baseline),
    output_mismatches = differences,
    tolerance = tolerance,
    timing_note = "Speedups are descriptive paired measurements, not CI thresholds.",
    allocation_note = "Allocated R bytes are not peak memory."
  ),
  file.path(output_dir, "comparison.json"),
  pretty = TRUE, auto_unbox = TRUE
)

if (status != "PASS") {
  stop("The current version changed the output of: ",
       paste(names(differences), collapse = ", "))
}

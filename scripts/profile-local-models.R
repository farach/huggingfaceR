# Usage: Rscript profile-local-models.R baseline|refactored <R-library> <cache-dir> <output-dir> <commit>
# Run each version in a fresh process on the same runner and Python environment.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L || !args[1] %in% c("baseline", "refactored")) {
  stop("Usage: profile-local-models.R baseline|refactored <R-library> <cache-dir> <output-dir> <commit>")
}
label <- args[1]
library_dir <- normalizePath(args[2], mustWork = TRUE)
cache_dir <- normalizePath(args[3], mustWork = TRUE)
output_dir <- args[4]
source_commit <- args[5]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(library_dir, .libPaths()))
Sys.setenv(HF_HUB_OFFLINE = "1", TRANSFORMERS_OFFLINE = "1")
Sys.unsetenv(c("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
library(huggingfaceR)
stopifnot(
  identical(normalizePath(find.package("huggingfaceR")),
            normalizePath(file.path(library_dir, "huggingfaceR")))
)
if (!requireNamespace("bench", quietly = TRUE) ||
    !capabilities("Rprof") || !capabilities("profmem")) {
  stop("Profiling requires the developer package 'bench' and R profiling/memory-profiling support.")
}

hf_local_setup()
torch <- reticulate::import("torch")
torch$set_num_threads(2L)
models <- c(
  embed = "sentence-transformers/all-MiniLM-L6-v2",
  classify = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
revisions <- c(
  embed = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
  classify = "714eb0fa89d2f80546fda750413ed43d93601a13"
)
load_model <- function(task) {
  hf_load_local_model(
    models[[task]], task = task, revision = revisions[[task]],
    cache_dir = cache_dir, local_files_only = TRUE, device = "cpu"
  )
}
handles <- lapply(names(models), load_model)
names(handles) <- names(models)

reticulate::py_run_string(paste(
  "from time import perf_counter",
  "def hfr_profile_backend(model, texts, task, batch_size, iterations):",
  "    def predict():",
  "        if task == 'embed':",
  "            return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,",
  "                                normalize_embeddings=True, show_progress_bar=False)",
  "        return model(texts, batch_size=batch_size, truncation=True, top_k=1)",
  "    predict()",
  "    elapsed = []",
  "    for _ in range(iterations):",
  "        start = perf_counter()",
  "        predict()",
  "        elapsed.append(perf_counter() - start)",
  "    return elapsed",
  sep = "\n"
))

iterations <- 7L
metrics <- list()
samples <- list()
measure <- function(run, task, method, n_texts, iterations = 7L, memory = TRUE) {
  invisible(gc())
  measured <- bench::mark(
    work = run(), iterations = iterations, check = FALSE,
    filter_gc = FALSE, memory = memory
  )
  times <- as.numeric(measured$time[[1]])
  data.frame(
    version = label, task = task, method = method, n_texts = n_texts,
    iterations = length(times),
    median_s = stats::median(times),
    q25_s = unname(stats::quantile(times, 0.25)),
    q75_s = unname(stats::quantile(times, 0.75)),
    r_allocated_mib = if (memory) as.numeric(measured$mem_alloc[1]) / 1024^2 else NA_real_,
    stringsAsFactors = FALSE
  )
}
add_metrics <- function(row) {
  metrics[[length(metrics) + 1L]] <<- row
}
predict_local <- function(task, texts) {
  if (task == "embed") {
    hf_embed_local(texts, handles[[task]], batch_size = 32L, normalize = TRUE)
  } else {
    hf_classify_local(texts, handles[[task]], batch_size = 32L, truncation = TRUE)
  }
}

text_pool <- c(
  "The cat sat on the mat.",
  "A feline is resting on a rug.",
  "The database server needs an update before the next release.",
  "I loved this movie. The acting was wonderful!",
  "I hated this movie. It was a complete waste of time.",
  "The research team compared methods for grouping similar documents.",
  "The delivery arrived late and the package was damaged.",
  "This tool makes it easier to work with text in R."
)
for (task in names(models)) {
  for (n in c(1L, 32L, 256L)) {
    texts <- rep(text_pool, length.out = n)
    samples[[paste0(task, "_", n)]] <- predict_local(task, texts)
    native <- unlist(reticulate::py$hfr_profile_backend(
      handles[[task]][["backend"]], as.list(texts), task, 32L, iterations
    ), use.names = FALSE)
    stopifnot(length(native) == iterations, all(is.finite(native)), all(native > 0))
    add_metrics(data.frame(
      version = label, task = task, method = "python_native", n_texts = n,
      iterations = length(native), median_s = stats::median(native),
      q25_s = unname(stats::quantile(native, 0.25)),
      q75_s = unname(stats::quantile(native, 0.75)),
      r_allocated_mib = NA_real_, stringsAsFactors = FALSE
    ))
    add_metrics(measure(function() predict_local(task, texts), task, "wrapper", n))
  }
  add_metrics(measure(function() load_model(task), task, "cached_hub_load", 0L,
                      iterations = 5L, memory = FALSE))
}

# Isolate R validation/result shaping from both model execution and matrix creation.
measure_shaping <- function(n) {
  set.seed(37)
  embeddings <- matrix(stats::runif(n * 384L), nrow = n)
  texts <- rep("Synthetic row for R allocation measurement.", n)
  handle <- structure(list(task = "embed", backend = TRUE), class = "hf_local_model")
  testthat::local_mocked_bindings(
    hf_local_encode = function(...) embeddings, .package = "huggingfaceR"
  )
  measure(function() hf_embed_local(texts, handle), "embed", "r_output_shaping", n,
          iterations = 5L)
}
for (n in c(1000L, 10000L)) {
  add_metrics(measure_shaping(n))
}

profile_task <- function(task) {
  path <- file.path(output_dir, paste0(label, "-", task, ".Rprof"))
  texts <- rep(text_pool, length.out = 256L)
  tryCatch(
    {
      utils::Rprof(path, interval = 0.005)
      for (i in seq_len(5L)) {
        invisible(predict_local(task, texts))
      }
    },
    finally = utils::Rprof(NULL)
  )
  profile <- utils::summaryRprof(path)
  utils::write.csv(profile$by.total,
                   file.path(output_dir, paste0(label, "-", task, "-profile.csv")))
}
invisible(lapply(names(models), profile_task))

public <- c(
  "hf_local_setup", "hf_download_model", "hf_load_local_model",
  "hf_embed_local", "hf_classify_local", "print.hf_local_model"
)
hot_path <- c(
  "hf_embed_local", "hf_classify_local", "hf_local_encode", "hf_local_classify",
  "hf_local_python_list", "hf_local_call", "hf_local_to_r"
)
code_text <- function(expression) {
  paste(deparse(expression, width.cutoff = 500L,
                control = c("keepInteger", "keepNA")), collapse = "\n")
}
results <- list(
  samples = samples,
  public_api = setNames(lapply(public, function(name) {
    code_text(formals(getFromNamespace(name, "huggingfaceR")))
  }), public),
  hot_path = setNames(lapply(hot_path, function(name) {
    code_text(body(getFromNamespace(name, "huggingfaceR")))
  }), hot_path)
)
saveRDS(results, file.path(output_dir, paste0(label, "-results.rds")))
metrics <- do.call(rbind, metrics)
stopifnot(all(is.finite(metrics$median_s)), all(metrics$median_s > 0))
utils::write.csv(metrics, file.path(output_dir, paste0(label, "-metrics.csv")),
                 row.names = FALSE)
metadata <- reticulate::import("importlib.metadata")
packages <- c("torch", "transformers", "sentence-transformers", "huggingface-hub", "numpy")
jsonlite::write_json(
  list(
    version = label, source_commit = source_commit,
    package_path = find.package("huggingfaceR"),
    R_version = as.character(getRversion()),
    Python_version = reticulate::import("platform")$python_version(),
    packages = as.list(setNames(vapply(packages, metadata$version, character(1)), packages)),
    system = as.list(Sys.info()[c("sysname", "release", "machine")]),
    models = as.list(models), revisions = as.list(revisions), torch_threads = 2L,
    timing_scope = "Warm CPU execution; downloads, first imports and initial model loading excluded.",
    native_scope = "Python perf_counter around the backend, excluding transfer to R and tidy output.",
    allocation_scope = "R allocations only, not peak RSS, Python, or PyTorch memory.",
    interpretation = "Descriptive paired-run measurements on a shared runner; no statistical speedup claim."
  ),
  file.path(output_dir, paste0(label, "-metadata.json")),
  pretty = TRUE, auto_unbox = TRUE, na = "null", null = "null"
)
print(metrics, row.names = FALSE)

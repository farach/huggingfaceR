# Run against an installed package in a fresh R process, not devtools::load_all().
# Usage: Rscript test-local-models.R online|offline <cache-dir> <output-dir>
# Optional HF_LOCAL_R_LIBRARY selects an isolated R installation library.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L || !args[1] %in% c("online", "offline")) {
  stop("Usage: Rscript test-local-models.R online|offline <cache-dir> <output-dir>")
}
mode <- args[1]
cache_dir <- normalizePath(args[2], winslash = "/", mustWork = FALSE)
output_dir <- normalizePath(args[3], winslash = "/", mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

lib <- Sys.getenv("HF_LOCAL_R_LIBRARY")
if (nzchar(lib)) {
  .libPaths(c(lib, .libPaths()))
}
if (mode == "offline") {
  Sys.setenv(HF_HUB_OFFLINE = "1", TRANSFORMERS_OFFLINE = "1")
}
Sys.unsetenv(c("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
library(huggingfaceR)

package_path <- normalizePath(find.package("huggingfaceR"), winslash = "/")
if (nzchar(lib)) {
  stopifnot(identical(
    package_path,
    normalizePath(file.path(lib, "huggingfaceR"), winslash = "/")
  ))
}
stopifnot(
  !reticulate::py_available(initialize = FALSE),
  all(c("hf_download_model", "hf_embed_local", "hf_classify_local") %in%
        getNamespaceExports("huggingfaceR"))
)
hf_local_setup()
stopifnot(!reticulate::py_available(initialize = FALSE))
if (mode == "offline") {
  reticulate::py_run_string(paste(
    "import requests",
    "_hfr_http_calls = 0",
    "def _hfr_block_request(self, *args, **kwargs):",
    "    global _hfr_http_calls",
    "    _hfr_http_calls += 1",
    "    raise RuntimeError('Unexpected HTTP request during offline validation')",
    "requests.sessions.Session.request = _hfr_block_request",
    sep = "\n"
  ))
}

models <- c(
  embed = "sentence-transformers/all-MiniLM-L6-v2",
  classify = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  embed_default = "BAAI/bge-small-en-v1.5"
)
revisions <- c(
  embed = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
  classify = "714eb0fa89d2f80546fda750413ed43d93601a13",
  embed_default = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
paths <- vapply(names(models), function(task) {
  hf_download_model(
    models[[task]],
    revision = revisions[[task]],
    cache_dir = cache_dir,
    local_files_only = mode == "offline"
  )
}, character(1))
stopifnot(
  identical(unname(basename(paths)), unname(revisions)),
  file.exists(file.path(paths[["embed"]], "1_Pooling", "config.json")),
  all(file.exists(file.path(paths, "model.safetensors")))
)

torch <- reticulate::import("torch")
torch$set_num_threads(2L)
embedding_model <- hf_load_local_model(
  paths[["embed"]], task = "embed", local_files_only = TRUE
)
classifier <- hf_load_local_model(
  task = "classify", revision = revisions[["classify"]],
  cache_dir = cache_dir, local_files_only = TRUE
)
default_embedding_model <- hf_load_local_model(
  task = "embed", revision = revisions[["embed_default"]],
  cache_dir = cache_dir, local_files_only = TRUE
)
stopifnot(
  identical(classifier$model, models[["classify"]]),
  identical(default_embedding_model$model, models[["embed_default"]])
)
print(embedding_model)
print(classifier)

texts <- c(
  "The cat sat on the mat.",
  "A feline is resting on a rug.",
  "The database server needs an update.",
  NA_character_,
  "The cat sat on the mat.",
  "Un chat se repose pr\u00e8s de la fen\u00eatre."
)
embeddings <- hf_embed_local(
  texts, embedding_model, batch_size = 2L, normalize = TRUE
)
valid <- which(!is.na(texts))
embedding_matrix <- do.call(rbind, embeddings$embedding[valid])
stopifnot(
  identical(names(embeddings), c("text", "embedding", "n_dims")),
  identical(embeddings$text, texts),
  all(embeddings$n_dims[valid] == 384L),
  is.na(embeddings$n_dims[4]),
  is.null(embeddings$embedding[[4]]),
  all(is.finite(embedding_matrix)),
  all(abs(rowSums(embedding_matrix^2) - 1) < 1e-5),
  isTRUE(all.equal(embeddings$embedding[[1]], embeddings$embedding[[5]],
                  tolerance = 1e-6)),
  nrow(hf_embed_local(character(), embedding_model)) == 0L,
  is.null(hf_embed_local(NA_character_, embedding_model)$embedding[[1]])
)
singleton <- hf_embed_local(texts[1], embedding_model, normalize = TRUE)
stopifnot(isTRUE(all.equal(
  singleton$embedding[[1]], embeddings$embedding[[1]], tolerance = 1e-5
)))
similarity <- hf_similarity(embeddings[1:3, ])
stopifnot(similarity$similarity[1] > similarity$similarity[2])
default_embedding <- hf_embed_local(
  texts[1], default_embedding_model, normalize = TRUE
)
stopifnot(
  default_embedding$n_dims == 384L,
  all(is.finite(default_embedding$embedding[[1]]))
)

reviews <- c(
  "I loved this movie. The acting was wonderful!",
  "I hated this movie. It was a complete waste of time.",
  NA_character_,
  "I loved this movie. The acting was wonderful!",
  ""
)
classification <- hf_classify_local(reviews, classifier, batch_size = 2L)
single_classification <- hf_classify_local(reviews[1], classifier)
long_text <- paste(rep("good", 600L), collapse = " ")
truncated <- hf_classify_local(long_text, classifier)
untruncated <- tryCatch(
  hf_classify_local(long_text, classifier, truncation = FALSE),
  error = identity
)
stopifnot(
  identical(names(classification), c("text", "label", "score")),
  identical(classification$text, reviews),
  identical(classification$label[c(1, 2)], c("POSITIVE", "NEGATIVE")),
  all(classification$score[c(1, 2)] > 0.9),
  is.na(classification$label[3]),
  is.na(classification$score[3]),
  is.na(classification$label[5]),
  is.na(classification$score[5]),
  identical(classification$label[1], classification$label[4]),
  identical(single_classification$label, classification$label[1]),
  isTRUE(all.equal(single_classification$score, classification$score[1],
                  tolerance = 1e-5)),
  nrow(hf_classify_local(character(), classifier)) == 0L,
  is.na(hf_classify_local(NA_character_, classifier)$score),
  nrow(truncated) == 1L,
  is.finite(truncated$score),
  inherits(untruncated, "error")
)

missing_cache <- tryCatch(
  hf_download_model(
    models[["embed"]], revision = revisions[["embed"]],
    cache_dir = file.path(cache_dir, "missing-cache"),
    local_files_only = TRUE
  ),
  error = identity
)
stopifnot(inherits(missing_cache, "error"))
offline_http_calls <- NULL
if (mode == "offline") {
  offline_http_calls <- reticulate::py_eval("_hfr_http_calls")
  stopifnot(offline_http_calls == 0)
}

results <- list(
  embeddings = embeddings, classification = classification,
  default_embedding = default_embedding
)
if (mode == "offline") {
  online <- readRDS(file.path(output_dir, "online-results.rds"))
  stopifnot(isTRUE(all.equal(results, online, tolerance = 1e-6)))
}
saveRDS(results, file.path(output_dir, paste0(mode, "-results.rds")))
utils::write.csv(classification,
                 file.path(output_dir, paste0(mode, "-classification.csv")),
                 row.names = FALSE)
utils::write.csv(similarity,
                 file.path(output_dir, paste0(mode, "-similarity.csv")),
                 row.names = FALSE)
utils::write.csv(
  data.frame(text = texts[valid], round(embedding_matrix[, 1:8], 6)),
  file.path(output_dir, paste0(mode, "-embedding-preview.csv")),
  row.names = FALSE
)

metadata <- reticulate::import("importlib.metadata")
dependencies <- c(
  "torch", "transformers", "sentence-transformers", "huggingface-hub", "numpy"
)
report <- list(
  status = "PASS",
  mode = mode,
  package_path = package_path,
  package_version = as.character(packageVersion("huggingfaceR")),
  source_commit = Sys.getenv("HF_VALIDATION_GIT_SHA", unset = NA_character_),
  R_version = as.character(getRversion()),
  python = reticulate::py_config()$python,
  dependencies = as.list(setNames(
    vapply(dependencies, metadata$version, character(1)), dependencies
  )),
  models = as.list(models),
  revisions = as.list(revisions),
  paths = as.list(paths),
  embedding_dimensions = ncol(embedding_matrix),
  default_embedding_dimensions = default_embedding$n_dims[[1]],
  offline_http_calls = offline_http_calls,
  related_similarity = similarity$similarity[1],
  unrelated_similarity = similarity$similarity[2]
)
jsonlite::write_json(
  report, file.path(output_dir, paste0(mode, "-report.json")),
  pretty = TRUE, auto_unbox = TRUE, na = "null", null = "null"
)
print(classification)
print(similarity)
cat("\nLocal model validation completed:", mode, "\n")

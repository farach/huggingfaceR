#' Declare Optional Local Python Requirements
#'
#' Declare a shared Python environment for local embeddings and classification
#' using `reticulate::py_require()`. This is opt-in: loading `huggingfaceR` and
#' using its hosted API functions do not require or initialize Python.
#'
#' Requires optional `reticulate` version 1.41.0 or later. Declared requirements
#' are Python `>=3.10,<3.13`, transformers `>=4.41,<5`,
#' sentence-transformers `>=3,<6`, huggingface_hub `>=0.34,<1`,
#' torch `>=2.6,<3`, and numpy `>=1.26,<3`.
#'
#' Call this before initializing Python. Requirements are resolved lazily by
#' reticulate when Python is first used. An explicitly selected Python
#' installation, including `RETICULATE_PYTHON`, is never overridden. In a
#' manually managed environment, install these requirements yourself and
#' restart R after changing the environment. Local download and load operations
#' check the selected environment's versions using Python package metadata.
#'
#' @returns Invisibly, a list with `packages` and `python_version` requirements.
#' @seealso [hf_download_model()], [hf_load_local_model()]
#' @export
#' @md
#'
#' @examples
#' \dontrun{
#' requirements <- hf_local_setup()
#' requirements$packages
#' }
hf_local_setup <- function() {
  hf_local_require_reticulate()
  requirements <- hf_local_requirements()
  hf_local_try(
    withCallingHandlers(
      hf_local_py_require(
        packages = requirements$packages,
        python_version = requirements$python_version
      ),
      warning = function(w) {
        hf_local_abort(w, "declare local Python requirements",
                       advice = hf_local_environment_advice())
      }
    ),
    "declare local Python requirements",
    advice = hf_local_environment_advice()
  )
  invisible(requirements)
}

#' Download a Model Snapshot for Local Inference
#'
#' Use the official `huggingface_hub.snapshot_download()` cache to download a
#' model's configuration, tokenizer, and safetensors weights. Repository
#' subdirectories and immutable snapshot paths are preserved. Online calls
#' can download files; `local_files_only = TRUE` never fetches missing Hub files.
#'
#' Only standard models with safetensors weights are supported, not pickle
#' weights, adapters, or custom Python code. The download patterns are
#' `*.json`, `*.safetensors`, `*.txt`, `*.model`, and `*.tiktoken`, including
#' matching files in subdirectories. Embedding module stacks, when present,
#' must use a root Transformer followed by built-in Pooling or Normalize
#' modules. Missing required files or referenced weight shards cause an error.
#'
#' Python dependencies must already be available for fully offline operation;
#' `local_files_only` controls model files, not reticulate's environment setup.
#'
#' @param model Character string. Model ID on the Hugging Face Hub.
#' @param revision Character string. Branch, tag, or commit to download.
#'   Use a commit hash for reproducibility.
#' @param cache_dir Character string or `NULL`. Official Hugging Face cache
#'   directory. `NULL` uses the Hub library's configured cache.
#' @param token Character string or `NULL`. Optional authentication token.
#'   `NULL` uses `HF_TOKEN`, then legacy `HUGGING_FACE_HUB_TOKEN`, via
#'   the package's token helper.
#' @param local_files_only Logical. Use only already cached model files.
#'   An incomplete offline cache fails rather than going online.
#'
#' @returns A normalized, existing snapshot-directory character path.
#' @seealso [hf_local_setup()], [hf_load_local_model()]
#' @export
#' @md
#'
#' @examples
#' \dontrun{
#' path <- hf_download_model(hf_default_model("embed"))
#' model <- hf_load_local_model(path, task = "embed")
#'
#' # Reuse the official cache without downloading model files
#' path <- hf_download_model(hf_default_model("embed"), local_files_only = TRUE)
#' }
hf_download_model <- function(model, revision = "main", cache_dir = NULL,
                              token = NULL, local_files_only = FALSE) {
  hf_local_check_string(model, "model")
  hf_local_check_hub_id(model)
  hf_local_check_options(revision, cache_dir, token, local_files_only)
  token <- hf_get_token(token = token, required = FALSE)
  hf_local_check_string(token, "token", allow_null = TRUE)

  path <- hf_local_try(
    {
      hf_local_initialize()
      hf_local_snapshot_download(
        model = model,
        revision = revision,
        cache_dir = cache_dir,
        token = token,
        local_files_only = local_files_only
      )
    },
    if (local_files_only) "locate a cached model snapshot" else "download a model snapshot",
    token = token,
    advice = paste(
      "Offline mode needs a complete cached snapshot.",
      "To fetch missing files, explicitly call hf_download_model() while online."
    )
  )
  path <- hf_local_normalize_path(path)
  hf_local_check_snapshot(path)
  path
}

#' Load a Reusable Local Model
#'
#' Load an opt-in local embedding or text-classification model. The default
#' device is CPU. A Hub ID is first resolved by [hf_download_model()]; an
#' existing local directory skips downloading. All backend components load
#' from that exact directory with `local_files_only = TRUE` and
#' `trust_remote_code = FALSE`. There is no hosted inference fallback.
#'
#' Standard safetensors models are required. Embedding models must include
#' `modules.json` describing a root Transformer followed by built-in Pooling or
#' Normalize modules. Missing module metadata is rejected rather than falling
#' back to a different pooling configuration. Arbitrary module loaders, adapters,
#' and custom code are unsupported.
#'
#' Python handles are session-specific. Do not use `saveRDS()` to transfer a
#' loaded handle between R sessions. Save the snapshot path instead and call
#' `hf_load_local_model()` again in the new session.
#'
#' @param model Character string or `NULL`. A Hub model ID or an existing
#'   local model directory. `NULL` resolves [hf_default_model()] for `task`.
#' @param task Character string. Either `"embed"` or `"classify"`.
#' @inheritParams hf_download_model
#' @param device Character string. `"cpu"` by default. Other devices, such as
#'   `"cuda:0"` or `"mps"`, are passed to the user's compatible Python stack.
#'   This function does not install CUDA or select a GPU automatically.
#'
#' @returns An `hf_local_model` handle containing `task`, `model`, `source`
#'   (`"hub"` or `"local"`), `path`, resolved `revision` when known
#'   (otherwise `NULL`), `requested_revision`, `device`, and `backend`.
#'   Authentication tokens are not stored in the handle.
#' @seealso [hf_local_setup()], [hf_embed_local()], [hf_classify_local()]
#' @export
#' @md
#'
#' @examples
#' \dontrun{
#' model <- hf_load_local_model(task = "embed")
#' hf_embed_local(c("Hello world", "Goodbye world"), model)
#'
#' classifier <- hf_load_local_model(task = "classify")
#' hf_classify_local("I enjoy programming in R.", classifier)
#'
#' # Reload an already downloaded snapshot in a new session
#' snapshot_path <- model$path
#' model <- hf_load_local_model(snapshot_path, task = "embed")
#' }
hf_load_local_model <- function(model = NULL, task = c("embed", "classify"),
                                revision = "main", cache_dir = NULL,
                                token = NULL, local_files_only = FALSE,
                                device = "cpu") {
  task <- match.arg(task)
  if (is.null(model)) {
    model <- hf_default_model(task)
  }
  hf_local_check_string(model, "model")
  hf_local_check_string(device, "device")
  hf_local_check_options(revision, cache_dir, token, local_files_only)

  is_local <- dir.exists(model)
  if (is_local) {
    path <- hf_local_normalize_path(model)
    hf_local_check_snapshot(path)
  } else {
    hf_local_check_hub_id(model)
    path <- hf_download_model(
      model = model, revision = revision, cache_dir = cache_dir,
      token = token, local_files_only = local_files_only
    )
  }
  if (task == "embed" && !hf_local_has_file(file.path(path, "modules.json"))) {
    hf_local_snapshot_error(
      "modules.json is required for embeddings; download the complete Sentence Transformers snapshot instead of using a transformer-only cache."
    )
  }

  backend <- hf_local_try(
    {
      # Hub downloads have already initialized Python and validated the snapshot.
      if (is_local) {
        hf_local_initialize()
      }
      hf_local_load_backend(path = path, task = task, device = device)
    },
    "load a local model",
    token = token,
    advice = paste(
      "Use a complete standard safetensors snapshot and a compatible device",
      "(device = \"cpu\" is the default). No hosted inference was attempted."
    )
  )
  if (is.null(backend)) {
    stop("The local backend did not return a model.", call. = FALSE)
  }

  resolved_revision <- hf_local_snapshot_revision(path)
  if (is.null(resolved_revision) && !is_local &&
      grepl("^[[:xdigit:]]{40}$", revision)) {
    resolved_revision <- tolower(revision)
  }
  structure(
    list(
      task = task,
      model = model,
      source = if (is_local) "local" else "hub",
      path = path,
      revision = resolved_revision,
      requested_revision = if (is_local) NULL else revision,
      device = device,
      backend = backend
    ),
    class = "hf_local_model"
  )
}

#' @rdname hf_load_local_model
#' @param x An `hf_local_model` handle.
#' @param ... Additional arguments (currently unused).
#' @export
#' @md
print.hf_local_model <- function(x, ...) {
  cat("<hf_local_model> ", x[["task"]], "\n", sep = "")
  cat("  Model: ", x[["model"]], " (", x[["source"]], ")\n", sep = "")
  cat("  Revision: ", x[["revision"]] %||% "unknown", "\n", sep = "")
  cat("  Path: ", x[["path"]], "\n", sep = "")
  cat("  Device: ", x[["device"]], "\n", sep = "")
  invisible(x)
}

#' Generate Embeddings with a Loaded Local Model
#'
#' Run SentenceTransformers on a reusable local handle. Prediction never
#' downloads a model or falls back to hosted inference. Input order and
#' duplicates are preserved. Missing texts retain `NULL` embeddings and
#' `NA_integer_` dimensions; zero-length and all-missing inputs do not call
#' Python. Empty strings are valid texts. The model's own maximum sequence
#' length controls truncation of long texts.
#'
#' @param text Character vector of texts to embed.
#' @param model An `hf_local_model` handle loaded with `task = "embed"`.
#' @param batch_size Positive scalar integer. Python inference batch size.
#' @param normalize Logical. Normalize embeddings to unit length.
#' @param progress Logical. Display the Python encoding progress bar.
#'
#' @returns A tibble with `text`, `embedding` (a list of numeric vectors),
#'   and integer `n_dims`, matching the schema of [hf_embed()].
#' @seealso [hf_load_local_model()], [hf_classify_local()]
#' @export
#' @md
#'
#' @examples
#' \dontrun{
#' model <- hf_load_local_model(task = "embed")
#' hf_embed_local(c("Hello", NA, "Hello"), model, normalize = TRUE)
#' }
hf_embed_local <- function(text, model, batch_size = 32L, normalize = FALSE,
                           progress = FALSE) {
  hf_local_check_text(text)
  hf_local_check_handle(model, "embed")
  batch_size <- hf_local_check_batch_size(batch_size)
  hf_local_check_flag(normalize, "normalize")
  hf_local_check_flag(progress, "progress")

  result <- tibble::tibble(
    text = text,
    embedding = rep(list(NULL), length(text)),
    n_dims = rep(NA_integer_, length(text))
  )
  valid <- which(!is.na(text))
  if (!length(valid)) {
    return(result)
  }

  embeddings <- hf_local_try(
    hf_local_encode(
      backend = model[["backend"]],
      text = unname(as.list(text[valid])),
      batch_size = batch_size, normalize = normalize, progress = progress
    ),
    "generate local embeddings",
    advice = hf_local_prediction_advice()
  )
  if (!is.matrix(embeddings) || !is.numeric(embeddings) || is.complex(embeddings) ||
      nrow(embeddings) != length(valid) || ncol(embeddings) < 1L ||
      any(!is.finite(embeddings))) {
    stop(
      "Malformed local embeddings: expected a finite numeric matrix with one row per text and positive dimensions.",
      call. = FALSE
    )
  }
  result$embedding[valid] <- lapply(seq_along(valid), function(i) {
    as.numeric(embeddings[i, , drop = TRUE])
  })
  result$n_dims[valid] <- as.integer(ncol(embeddings))
  result
}

#' Classify Text with a Loaded Local Model
#'
#' Run a local text-classification pipeline, requesting the top label for each
#' text. Prediction never downloads a model or falls back to hosted inference.
#' Input order and duplicates are preserved. Missing and empty-string texts
#' retain missing labels and scores without being sent to Python. Zero-length
#' and entirely missing or empty inputs do not call the backend.
#'
#' Scores are model outputs, not calibrated measures of certainty. By default,
#' long texts are truncated to the tokenizer's supported maximum length.
#'
#' @param text Character vector of texts to classify.
#' @param model An `hf_local_model` handle loaded with `task = "classify"`.
#' @param batch_size Positive scalar integer. Python inference batch size.
#' @param truncation Logical. Truncate long inputs to the tokenizer's maximum
#'   length. With `FALSE`, overlong inputs may produce a backend error.
#'
#' @returns A tibble with `text`, character `label`, and numeric `score`,
#'   matching the schema of [hf_classify()].
#' @seealso [hf_load_local_model()], [hf_embed_local()]
#' @export
#' @md
#'
#' @examples
#' \dontrun{
#' model <- hf_load_local_model(task = "classify")
#' hf_classify_local(c("I like this.", NA, "I dislike this."), model)
#' }
hf_classify_local <- function(text, model, batch_size = 32L, truncation = TRUE) {
  hf_local_check_text(text)
  hf_local_check_handle(model, "classify")
  batch_size <- hf_local_check_batch_size(batch_size)
  hf_local_check_flag(truncation, "truncation")

  result <- tibble::tibble(
    text = text,
    label = rep(NA_character_, length(text)),
    score = rep(NA_real_, length(text))
  )
  valid <- which(!is.na(text) & nzchar(text))
  if (!length(valid)) {
    return(result)
  }

  predictions <- hf_local_try(
    hf_local_classify(
      backend = model[["backend"]],
      text = unname(as.list(text[valid])),
      batch_size = batch_size, truncation = truncation
    ),
    "classify text locally",
    advice = hf_local_prediction_advice()
  )
  rows <- hf_local_classification_rows(predictions, length(valid))
  result$label[valid] <- vapply(rows, function(row) row[["label"]], character(1))
  result$score[valid] <- vapply(rows, function(row) as.numeric(row[["score"]]), numeric(1))
  result
}

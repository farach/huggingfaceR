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
    hf_local_snapshot_download(
      model = model,
      revision = revision,
      cache_dir = cache_dir,
      token = token,
      local_files_only = local_files_only
    ),
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
#' Standard safetensors models are required. SentenceTransformers models may
#' contain a root Transformer followed by built-in Pooling or Normalize
#' modules; arbitrary module loaders, adapters, and custom code are unsupported.
#' Models without `modules.json` use SentenceTransformers' standard mean-pooling
#' fallback over the local transformer.
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
  } else {
    hf_local_check_hub_id(model)
    path <- hf_download_model(
      model = model, revision = revision, cache_dir = cache_dir,
      token = token, local_files_only = local_files_only
    )
    path <- hf_local_normalize_path(path)
  }
  hf_local_check_snapshot(path)

  backend <- hf_local_try(
    hf_local_load_backend(path = path, task = task, device = device),
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

hf_local_requirements <- function() {
  list(
    packages = c(
      "transformers>=4.41,<5",
      "sentence-transformers>=3,<6",
      "huggingface_hub>=0.34,<1",
      "torch>=2.6,<3",
      "numpy>=1.26,<3"
    ),
    python_version = ">=3.10,<3.13"
  )
}

hf_local_require_reticulate <- function() {
  version <- hf_local_reticulate_version()
  if (is.null(version) || utils::compareVersion(version, "1.41.0") < 0L) {
    stop(
      "Local inference requires optional package 'reticulate' (>= 1.41.0). Install or update it with install.packages('reticulate').",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

hf_local_reticulate_version <- function() {
  if (requireNamespace("reticulate", quietly = TRUE)) {
    as.character(utils::packageVersion("reticulate"))
  } else {
    NULL
  }
}

hf_local_py_require <- function(packages, python_version) {
  reticulate::py_require(
    packages = packages, python_version = python_version, action = "add"
  )
}

hf_local_import <- function(module) {
  hf_local_try(
    reticulate::import(module, convert = TRUE),
    paste0("import Python module '", module, "'"),
    advice = hf_local_environment_advice()
  )
}

hf_local_initialize <- function() {
  hf_local_setup()
  hf_local_try(
    hf_local_check_environment(),
    "validate the selected Python environment",
    advice = hf_local_environment_advice()
  )
  invisible(TRUE)
}

hf_local_check_environment <- function() {
  requirements <- hf_local_requirements()
  metadata <- hf_local_import("importlib.metadata")
  packaging <- hf_local_import("packaging.requirements")
  specifiers <- hf_local_import("packaging.specifiers")
  platform <- hf_local_import("platform")

  python_version <- platform$python_version()
  python_spec <- specifiers$SpecifierSet(requirements$python_version)
  if (!isTRUE(python_spec$contains(python_version))) {
    stop("Selected Python version ", python_version, " does not satisfy ",
         requirements$python_version, ".", call. = FALSE)
  }
  for (spec in requirements$packages) {
    requirement <- packaging$Requirement(spec)
    version <- metadata$version(requirement$name)
    if (!isTRUE(requirement$specifier$contains(version))) {
      stop("Selected Python package ", requirement$name, "==", version,
           " does not satisfy ", spec, ".", call. = FALSE)
    }
  }
  invisible(TRUE)
}

hf_local_call <- function(object, ...) {
  reticulate::py_call(object, ...)
}

hf_local_python_list <- function(text) {
  reticulate::r_to_py(unname(as.list(text)), convert = FALSE)
}

hf_local_to_r <- function(value) {
  if (inherits(value, "python.builtin.object")) {
    reticulate::py_to_r(value)
  } else {
    value
  }
}

hf_local_snapshot_download <- function(model, revision, cache_dir, token,
                                       local_files_only) {
  hf_local_initialize()
  hub <- hf_local_import("huggingface_hub")
  hf_local_to_r(hf_local_call(
    hub$snapshot_download,
    repo_id = model,
    revision = revision,
    cache_dir = cache_dir,
    token = token,
    local_files_only = local_files_only,
    allow_patterns = as.list(c("*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"))
  ))
}

hf_local_load_backend <- function(path, task, device) {
  hf_local_initialize()
  if (identical(task, "embed")) {
    sentence_transformers <- hf_local_import("sentence_transformers")
    return(hf_local_call(
      sentence_transformers$SentenceTransformer,
      model_name_or_path = path,
      device = device,
      local_files_only = TRUE,
      trust_remote_code = FALSE,
      model_kwargs = list(
        use_safetensors = TRUE, local_files_only = TRUE, trust_remote_code = FALSE
      ),
      tokenizer_kwargs = list(local_files_only = TRUE, trust_remote_code = FALSE),
      config_kwargs = list(local_files_only = TRUE, trust_remote_code = FALSE)
    ))
  }

  transformers <- hf_local_import("transformers")
  tokenizer <- hf_local_call(
    transformers$AutoTokenizer$from_pretrained,
    pretrained_model_name_or_path = path,
    local_files_only = TRUE,
    trust_remote_code = FALSE
  )
  weights <- hf_local_call(
    transformers$AutoModelForSequenceClassification$from_pretrained,
    pretrained_model_name_or_path = path,
    local_files_only = TRUE,
    trust_remote_code = FALSE,
    use_safetensors = TRUE
  )
  hf_local_call(
    transformers$pipeline,
    task = "text-classification", model = weights, tokenizer = tokenizer,
    device = device
  )
}

hf_local_encode <- function(backend, text, batch_size, normalize, progress) {
  hf_local_to_r(hf_local_call(
    backend$encode,
    sentences = hf_local_python_list(text),
    batch_size = batch_size,
    convert_to_numpy = TRUE,
    normalize_embeddings = normalize,
    show_progress_bar = progress
  ))
}

hf_local_classify <- function(backend, text, batch_size, truncation) {
  hf_local_to_r(hf_local_call(
    backend,
    hf_local_python_list(text),
    batch_size = batch_size,
    truncation = truncation,
    top_k = 1L
  ))
}

hf_local_check_string <- function(value, name, allow_null = FALSE) {
  if (allow_null && is.null(value)) {
    return(invisible(NULL))
  }
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !is.null(dim(value)) || !nzchar(trimws(value))) {
    stop("`", name, "` must be a non-empty character string",
         if (allow_null) " or NULL" else "", ".", call. = FALSE)
  }
  invisible(NULL)
}

hf_local_check_flag <- function(value, name) {
  if (!is.logical(value) || length(value) != 1L || is.na(value) ||
      !is.null(dim(value))) {
    stop("`", name, "` must be TRUE or FALSE.", call. = FALSE)
  }
  invisible(NULL)
}

hf_local_check_options <- function(revision, cache_dir, token, local_files_only) {
  hf_local_check_string(revision, "revision")
  hf_local_check_string(cache_dir, "cache_dir", allow_null = TRUE)
  hf_local_check_string(token, "token", allow_null = TRUE)
  hf_local_check_flag(local_files_only, "local_files_only")
}

hf_local_check_hub_id <- function(model) {
  if (!grepl("^[^/\\\\:[:space:]]+(/[^/\\\\:[:space:]]+)?$", model) ||
      grepl("^[.~]", model) || file.exists(model)) {
    stop("`model` must be a Hub model ID or, when loading, an existing local directory.",
         call. = FALSE)
  }
}

hf_local_check_text <- function(text) {
  if (!is.character(text) || !is.null(dim(text))) {
    stop("`text` must be a character vector.", call. = FALSE)
  }
}

hf_local_check_batch_size <- function(batch_size) {
  if (!is.numeric(batch_size) || is.complex(batch_size) || length(batch_size) != 1L ||
      !is.null(dim(batch_size)) || !is.finite(batch_size) ||
      batch_size < 1 || batch_size > .Machine$integer.max ||
      batch_size != floor(batch_size)) {
    stop("`batch_size` must be a positive scalar integer.", call. = FALSE)
  }
  as.integer(batch_size)
}

hf_local_check_handle <- function(model, task) {
  if (!inherits(model, "hf_local_model") || !is.list(model) ||
      is.null(model[["backend"]])) {
    stop("`model` must be an hf_local_model handle from hf_load_local_model().",
         call. = FALSE)
  }
  if (!identical(model[["task"]], task)) {
    stop("This function requires an hf_local_model with task = \"", task, "\".",
         call. = FALSE)
  }
}

hf_local_normalize_path <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !dir.exists(path)) {
    stop("The local model snapshot directory does not exist.", call. = FALSE)
  }
  normalizePath(path, mustWork = TRUE)
}

hf_local_snapshot_revision <- function(path) {
  revision <- basename(path)
  if (identical(basename(dirname(path)), "snapshots") &&
      grepl("^[[:xdigit:]]{40}$", revision)) {
    tolower(revision)
  } else {
    NULL
  }
}

hf_local_has_file <- function(path) {
  file.exists(path) & !dir.exists(path) & !is.na(file.size(path)) & file.size(path) > 0
}

hf_local_snapshot_error <- function(detail) {
  stop(
    "Incomplete or unsupported local model snapshot: ", detail,
    " Use a complete standard safetensors snapshot; no missing model files will be downloaded during loading.",
    call. = FALSE
  )
}

hf_local_relative_path <- function(path, relative) {
  if (!is.character(relative) || length(relative) != 1L || is.na(relative) ||
      !nzchar(relative) || grepl("^[\\\\/]|[\\\\:]", relative) ||
      any(strsplit(relative, "/", fixed = TRUE)[[1]] %in% c(".", "..", ""))) {
    hf_local_snapshot_error("a referenced path is not a repository-relative path.")
  }
  file.path(path, relative)
}

hf_local_read_json <- function(path) {
  tryCatch(
    jsonlite::fromJSON(path, simplifyVector = FALSE),
    error = function(e) {
      hf_local_abort(e, "read local model metadata",
                     advice = "Use a complete snapshot with valid JSON configuration files.")
    }
  )
}

hf_local_check_snapshot <- function(path) {
  if (!hf_local_has_file(file.path(path, "config.json"))) {
    hf_local_snapshot_error("config.json is missing.")
  }
  if (file.exists(file.path(path, "adapter_config.json"))) {
    hf_local_snapshot_error("adapter models are not supported.")
  }

  index_path <- file.path(path, "model.safetensors.index.json")
  if (file.exists(index_path)) {
    index <- hf_local_read_json(index_path)
    weights <- if (is.list(index)) index[["weight_map"]] else NULL
    if (!is.list(weights) || !length(weights) ||
        !all(vapply(weights, function(x) is.character(x) && length(x) == 1L &&
                    !is.na(x) && grepl("\\.safetensors$", x), logical(1)))) {
      hf_local_snapshot_error("the safetensors shard index is invalid.")
    }
    shards <- vapply(unique(unlist(weights, use.names = FALSE)), function(shard) {
      hf_local_relative_path(path, shard)
    }, character(1))
    if (!all(hf_local_has_file(shards))) {
      hf_local_snapshot_error("one or more safetensors weight shards are missing.")
    }
  } else if (!hf_local_has_file(file.path(path, "model.safetensors"))) {
    hf_local_snapshot_error("model.safetensors or its complete shard index is missing.")
  }

  files <- list.files(path)
  tokenizer_files <- files[
    files %in% c("tokenizer.json", "vocab.txt") |
      grepl("\\.(model|tiktoken)$", files)
  ]
  has_tokenizer <- any(hf_local_has_file(file.path(path, tokenizer_files))) ||
    all(hf_local_has_file(file.path(path, c("vocab.json", "merges.txt"))))
  if (!has_tokenizer) {
    hf_local_snapshot_error("standard tokenizer files are missing.")
  }
  hf_local_check_modules(path)
  invisible(TRUE)
}

hf_local_check_modules <- function(path) {
  modules_path <- file.path(path, "modules.json")
  if (file.exists(modules_path)) {
    modules <- hf_local_read_json(modules_path)
    allowed <- paste0("sentence_transformers.models.", c("Transformer", "Pooling", "Normalize"))
    if (!is.list(modules) || !length(modules)) {
      hf_local_snapshot_error("modules.json must contain a standard module stack.")
    }
    types <- vapply(modules, function(module) {
      type <- if (is.list(module)) module[["type"]] else NULL
      if (!is.character(type) || length(type) != 1L || is.na(type) ||
          !type %in% allowed) {
        hf_local_snapshot_error("only built-in Transformer, Pooling, and Normalize modules are supported.")
      }
      type
    }, character(1))
    if (types[[1]] != allowed[[1]] || sum(types == allowed[[1]]) != 1L ||
        !identical(modules[[1]][["path"]], "")) {
      hf_local_snapshot_error("the Transformer module must load from the snapshot root.")
    }
    for (i in seq_along(modules)[-1L]) {
      module_path <- hf_local_relative_path(path, modules[[i]][["path"]])
      if (types[[i]] == allowed[[2]] &&
          !hf_local_has_file(file.path(module_path, "config.json"))) {
        hf_local_snapshot_error("a Pooling module configuration is missing.")
      }
    }
  }

  configs <- list.files(path, pattern = "^sentence_.*_config\\.json$", full.names = TRUE)
  for (file in configs) {
    config <- hf_local_read_json(file)
    if (is.list(config) && !is.null(config[["tokenizer_name_or_path"]])) {
      hf_local_snapshot_error("the tokenizer cannot redirect to a different model or directory.")
    }
  }
  invisible(TRUE)
}

hf_local_classification_rows <- function(predictions, n) {
  is_record <- function(x) {
    is.list(x) && !is.data.frame(x) && !anyDuplicated(names(x)) &&
      all(c("label", "score") %in% names(x))
  }
  if (n == 1L && is_record(predictions)) {
    predictions <- list(predictions)
  }
  if (!is.list(predictions) || !is.null(names(predictions)) ||
      length(predictions) != n) {
    stop("Malformed local classification: expected one result per text.", call. = FALSE)
  }
  lapply(predictions, function(row) {
    if (!is_record(row) && is.list(row) && is.null(names(row)) && length(row) == 1L) {
      row <- row[[1]]
    }
    if (!is_record(row) || !is.character(row[["label"]]) ||
        length(row[["label"]]) != 1L || is.na(row[["label"]]) ||
        !is.null(dim(row[["label"]])) || !nzchar(row[["label"]]) ||
        !is.numeric(row[["score"]]) || is.complex(row[["score"]]) ||
        length(row[["score"]]) != 1L || !is.null(dim(row[["score"]])) ||
        !is.finite(row[["score"]])) {
      stop("Malformed local classification: each result needs a non-empty label and a finite numeric score.",
           call. = FALSE)
    }
    row
  })
}

hf_local_environment_advice <- function() {
  paste(
    "Run hf_local_setup() before initializing Python.",
    "For RETICULATE_PYTHON or an environment selected with reticulate::use_python()",
    "or use_virtualenv(), install the declared requirements there, or select a",
    "compatible environment and restart R. No selected environment is overridden."
  )
}

hf_local_prediction_advice <- function() {
  paste(
    "Reload the model with hf_load_local_model() in this R session and check",
    "the device and batch size. No hosted inference was attempted."
  )
}

hf_local_try <- function(expr, action, token = NULL, advice = NULL) {
  tryCatch(
    force(expr),
    error = function(e) hf_local_abort(e, action, token = token, advice = advice)
  )
}

hf_local_abort <- function(error, action, token = NULL, advice = NULL) {
  tokens <- unique(c(token, Sys.getenv(hf_token_env_vars(), unset = NA_character_)))
  tokens <- tokens[!is.na(tokens) & nzchar(tokens)]
  original_message <- conditionMessage(error)
  message <- original_message
  for (secret in tokens) {
    message <- gsub(secret, "<redacted>", message, fixed = TRUE)
  }
  call <- paste(deparse(conditionCall(error)), collapse = " ")
  secret_in_call <- any(vapply(tokens, function(secret) {
    grepl(secret, call, fixed = TRUE)
  }, logical(1)))
  if (identical(message, original_message) && !secret_in_call) {
    if (inherits(error, "hf_local_error")) {
      stop(error)
    }
    cause <- error
  } else {
    cause <- simpleError(message, call = NULL)
    cause$original_class <- class(error)
  }
  # Automatically captured call traces can reveal literal token arguments.
  condition <- simpleError(
    paste(c(paste0("Unable to ", action, "."), advice, paste0("Caused by: ", message)),
          collapse = "\n"),
    call = NULL
  )
  condition$parent <- cause
  class(condition) <- c("hf_local_error", class(condition))
  stop(condition)
}

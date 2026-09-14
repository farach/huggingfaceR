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
  metadata <- file.info(path, extra_cols = FALSE)
  !is.na(metadata$size) & !is.na(metadata$isdir) &
    !metadata$isdir & metadata$size > 0
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

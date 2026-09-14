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
    packages = packages,
    python_version = python_version,
    action = "add"
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
    stop(
      "Selected Python version ",
      python_version,
      " does not satisfy ",
      requirements$python_version,
      ".",
      call. = FALSE
    )
  }
  for (spec in requirements$packages) {
    requirement <- packaging$Requirement(spec)
    version <- metadata$version(requirement$name)
    if (!isTRUE(requirement$specifier$contains(version))) {
      stop(
        "Selected Python package ",
        requirement$name,
        "==",
        version,
        " does not satisfy ",
        spec,
        ".",
        call. = FALSE
      )
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

hf_local_callable_parameters <- function(callable) {
  inspect <- hf_local_import("inspect")
  builtins <- hf_local_import("builtins")
  signature <- hf_local_call(inspect$signature, callable)
  parameters <- hf_local_to_r(hf_local_call(
    builtins$list,
    signature$parameters
  ))
  as.character(unlist(parameters, use.names = FALSE))
}

hf_local_snapshot_download <- function(
  model,
  revision,
  cache_dir,
  token,
  local_files_only
) {
  hub <- hf_local_import("huggingface_hub")
  hf_local_to_r(hf_local_call(
    hub$snapshot_download,
    repo_id = model,
    revision = revision,
    cache_dir = cache_dir,
    token = token,
    local_files_only = local_files_only,
    allow_patterns = as.list(c(
      "*.json",
      "*.safetensors",
      "*.txt",
      "*.model",
      "*.tiktoken"
    ))
  ))
}

hf_local_load_backend <- function(path, task, device) {
  if (identical(task, "embed")) {
    sentence_transformers <- hf_local_import("sentence_transformers")
    constructor <- sentence_transformers$SentenceTransformer
    processor_argument <- intersect(
      c("processor_kwargs", "tokenizer_kwargs"),
      hf_local_callable_parameters(constructor)
    )
    if (!length(processor_argument)) {
      stop(
        "The installed SentenceTransformer constructor exposes neither processor_kwargs nor tokenizer_kwargs.",
        call. = FALSE
      )
    }
    arguments <- list(
      object = constructor,
      model_name_or_path = path,
      device = device,
      local_files_only = TRUE,
      trust_remote_code = FALSE,
      model_kwargs = list(
        use_safetensors = TRUE,
        local_files_only = TRUE,
        trust_remote_code = FALSE
      ),
      config_kwargs = list(local_files_only = TRUE, trust_remote_code = FALSE)
    )
    arguments[[processor_argument[[1]]]] <- list(
      local_files_only = TRUE,
      trust_remote_code = FALSE
    )
    return(do.call(hf_local_call, arguments))
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
    task = "text-classification",
    model = weights,
    tokenizer = tokenizer,
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

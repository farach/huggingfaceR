test_that("the documented local API signatures remain stable", {
  expect_identical(as.list(formals(hf_local_setup)), alist())
  expect_identical(as.list(formals(hf_download_model)), alist(
    model = , revision = "main", cache_dir = NULL,
    token = NULL, local_files_only = FALSE
  ))
  expect_identical(as.list(formals(hf_load_local_model)), alist(
    model = NULL, task = c("embed", "classify"), revision = "main",
    cache_dir = NULL, token = NULL, local_files_only = FALSE, device = "cpu"
  ))
  expect_identical(as.list(formals(hf_embed_local)), alist(
    text = , model = , batch_size = 32L, normalize = FALSE, progress = FALSE
  ))
  expect_identical(as.list(formals(hf_classify_local)), alist(
    text = , model = , batch_size = 32L, truncation = TRUE
  ))
})

test_that("hosted and empty local operations do not touch optional Python", {
  was_loaded <- "reticulate" %in% loadedNamespaces()
  testthat::local_mocked_bindings(
    hf_local_setup = local_test_unexpected,
    hf_local_import = local_test_unexpected,
    hf_local_initialize = local_test_unexpected,
    hf_local_encode = local_test_unexpected,
    hf_local_classify = local_test_unexpected,
    hf_local_snapshot_download = local_test_unexpected,
    hf_api_request = local_test_unexpected
  )

  expect_equal(nrow(hf_embed(character())), 0L)
  expect_equal(nrow(hf_classify(character())), 0L)
  expect_equal(nrow(hf_embed_local(character(), local_test_handle())), 0L)
  expect_equal(nrow(hf_classify_local(character(), local_test_handle("classify"))), 0L)
  expect_identical("reticulate" %in% loadedNamespaces(), was_loaded)
})

test_that("setup declares one shared compatible environment lazily", {
  calls <- list()
  testthat::local_mocked_bindings(
    hf_local_require_reticulate = function() invisible(TRUE),
    hf_local_py_require = function(packages, python_version) {
      calls[[length(calls) + 1L]] <<- list(
        packages = packages, python_version = python_version
      )
      invisible(NULL)
    },
    hf_local_import = local_test_unexpected,
    hf_local_check_environment = local_test_unexpected
  )
  withr::local_envvar(c(
    RETICULATE_PYTHON = "C:\\selected python\\python.exe",
    RETICULATE_USE_MANAGED_VENV = "no"
  ))

  result <- withVisible(hf_local_setup())
  expect_false(result$visible)
  expect_identical(result$value$python_version, ">=3.10,<3.13")
  expect_identical(result$value$packages, c(
    "transformers>=4.41,<5", "sentence-transformers>=3,<6",
    "huggingface_hub>=0.34,<1", "torch>=2.6,<3", "numpy>=1.26,<3"
  ))
  expect_identical(calls[[1]], result$value)
  expect_invisible(hf_local_setup())
  expect_identical(calls[[1]], calls[[2]])
  expect_identical(Sys.getenv("RETICULATE_PYTHON"), "C:\\selected python\\python.exe")
  expect_identical(Sys.getenv("RETICULATE_USE_MANAGED_VENV"), "no")
})

test_that("the optional reticulate minimum is checked before declaration", {
  version <- NULL
  testthat::local_mocked_bindings(
    hf_local_reticulate_version = function() version,
    hf_local_py_require = local_test_unexpected
  )
  expect_error(hf_local_setup(), "reticulate.*1.41.0")
  version <- "1.40.0"
  expect_error(hf_local_setup(), "reticulate.*1.41.0")
  version <- "1.41.0"
  expect_invisible(hf_local_require_reticulate())
})

test_that("setup errors and conflicting requirement warnings cannot report success", {
  testthat::local_mocked_bindings(
    hf_local_require_reticulate = function() invisible(TRUE),
    hf_local_py_require = function(...) stop("resolver failed")
  )
  error <- tryCatch(hf_local_setup(), error = identity)
  expect_s3_class(error, "hf_local_error")
  expect_match(conditionMessage(error), "resolver failed")
  expect_match(conditionMessage(error), "RETICULATE_PYTHON")
  expect_match(conditionMessage(error), "restart R")
  expect_identical(conditionMessage(error$parent), "resolver failed")

  testthat::local_mocked_bindings(
    hf_local_py_require = function(...) warning("Python version cannot be changed")
  )
  expect_error(hf_local_setup(), "Python version cannot be changed", class = "hf_local_error")
})

test_that("selected environment versions are checked with Python packaging metadata", {
  modules <- local_test_imports()
  testthat::local_mocked_bindings(hf_local_import = modules$import)

  expect_invisible(hf_local_check_environment())
  expect_identical(modules$checked$specs, hf_local_requirements()$packages)
  expect_identical(modules$checked$versions, c("4.57.0", "5.1.0", "0.35.0", "2.6.0+cpu", "2.2.6"))
})

test_that("incompatible selected Python and package versions fail explicitly", {
  modules <- local_test_imports(python = "3.13.0", incompatible = "python")
  testthat::local_mocked_bindings(
    hf_local_import = modules$import,
    hf_local_setup = function() invisible(hf_local_requirements())
  )
  expect_error(hf_local_initialize(), "3.13.0.*does not satisfy", class = "hf_local_error")
  expect_length(modules$checked$versions, 0L)

  modules <- local_test_imports(incompatible = "torch")
  testthat::local_mocked_bindings(hf_local_import = modules$import)
  expect_error(hf_local_initialize(), "torch==2.6.0\\+cpu.*does not satisfy")
})

test_that("environment initialization retains import failures and actionable remedies", {
  original <- simpleError("No module named 'packaging'")
  testthat::local_mocked_bindings(
    hf_local_setup = function() invisible(hf_local_requirements()),
    hf_local_check_environment = function() stop(original)
  )
  error <- tryCatch(hf_local_initialize(), error = identity)
  expect_s3_class(error, "hf_local_error")
  expect_identical(error$parent, original)
  expect_match(conditionMessage(error), "No module named 'packaging'")
  expect_match(conditionMessage(error), "install the declared requirements")
  expect_match(conditionMessage(error), "No selected environment is overridden")
})

test_that("snapshot downloads propagate explicit options to the official Hub boundary", {
  path <- local_test_fixture()
  calls <- list()
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_call = local_test_call,
    hf_local_import = function(module) {
      expect_identical(module, "huggingface_hub")
      list(snapshot_download = function(...) {
        calls[[length(calls) + 1L]] <<- list(...)
        path
      })
    }
  )
  withr::local_envvar(c(HF_TOKEN = "hf_environment_secret"))

  result <- hf_download_model(
    "example/tiny", revision = "refs/pr/17",
    cache_dir = "C:\\cache with spaces\\models",
    token = "hf_explicit_secret", local_files_only = TRUE
  )
  expect_identical(result, path)
  expect_identical(calls[[1]]$repo_id, "example/tiny")
  expect_identical(calls[[1]]$revision, "refs/pr/17")
  expect_identical(calls[[1]]$cache_dir, "C:\\cache with spaces\\models")
  expect_identical(calls[[1]]$token, "hf_explicit_secret")
  expect_true(calls[[1]]$local_files_only)
  expect_identical(calls[[1]]$allow_patterns, as.list(c(
    "*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"
  )))
  expect_false(any(grepl("bin|pickle|\\.py", unlist(calls[[1]]$allow_patterns))))
  expect_length(calls, 1L)
})

test_that("downloads honor HF_TOKEN and the legacy fallback without requiring authentication", {
  path <- local_test_fixture()
  tokens <- list()
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_snapshot_download = function(model, revision, cache_dir, token, local_files_only) {
      tokens[length(tokens) + 1L] <<- list(token)
      expect_identical(revision, "main")
      expect_null(cache_dir)
      expect_false(local_files_only)
      path
    }
  )
  withr::local_envvar(c(
    HF_TOKEN = "hf_current_secret", HUGGING_FACE_HUB_TOKEN = "hf_legacy_secret"
  ))
  expect_identical(hf_download_model("example/tiny"), path)
  expect_identical(tokens[[1]], "hf_current_secret")
  withr::local_envvar(c(HF_TOKEN = NA_character_))
  expect_identical(hf_download_model("example/tiny"), path)
  expect_identical(tokens[[2]], "hf_legacy_secret")
  withr::local_envvar(c(HUGGING_FACE_HUB_TOKEN = NA_character_))
  expect_identical(hf_download_model("example/tiny"), path)
  expect_null(tokens[[3]])
})

test_that("offline failures never retry online or fall back to hosted inference", {
  calls <- 0L
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_snapshot_download = function(model, revision, cache_dir, token, local_files_only) {
      calls <<- calls + 1L
      expect_true(local_files_only)
      stop("LocalEntryNotFoundError: no cached snapshot")
    },
    hf_local_load_backend = local_test_unexpected,
    hf_api_request = local_test_unexpected
  )
  expect_error(
    hf_load_local_model("example/tiny", local_files_only = TRUE),
    "no cached snapshot", class = "hf_local_error"
  )
  expect_identical(calls, 1L)
})

test_that("download failures redact tokens while retaining the backend explanation", {
  withr::local_envvar(c(
    HF_TOKEN = "hf_environment_secret", HUGGING_FACE_HUB_TOKEN = "hf_legacy_secret"
  ))
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_snapshot_download = function(...) {
      stop("Hub refused hf_explicit_secret, hf_environment_secret, and hf_legacy_secret")
    }
  )
  error <- tryCatch(
    hf_download_model("example/tiny", token = "hf_explicit_secret"),
    error = identity
  )
  expect_s3_class(error, "hf_local_error")
  expect_match(conditionMessage(error), "Hub refused")
  expect_match(conditionMessage(error), "<redacted>")
  expect_false(grepl("hf_explicit_secret|hf_environment_secret|hf_legacy_secret", conditionMessage(error)))
  expect_false(grepl("hf_explicit_secret|hf_environment_secret|hf_legacy_secret", conditionMessage(error$parent)))
})

test_that("tokens in error calls are not exposed by chained error printing", {
  error <- simpleError(
    "Hub refused authentication",
    call = quote(snapshot_download(token = "hf_secret_in_call"))
  )
  condition <- tryCatch(
    hf_local_abort(error, "download", token = "hf_secret_in_call"),
    error = identity
  )
  expect_match(conditionMessage(condition), "Hub refused authentication")
  expect_false(any(grepl("hf_secret_in_call", capture.output(print(condition)))))
  expect_null(conditionCall(condition$parent))
})

test_that("missing snapshot directories and incomplete offline files fail clearly", {
  path <- local_test_fixture()
  response_path <- file.path(path, "missing")
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_snapshot_download = function(...) response_path,
    hf_local_load_backend = local_test_unexpected,
    hf_api_request = local_test_unexpected
  )
  expect_error(hf_download_model("example/tiny"), "snapshot directory does not exist")
  response_path <- path
  unlink(file.path(path, "model.safetensors"))
  expect_error(
    hf_download_model("example/tiny", local_files_only = TRUE),
    "Incomplete.*safetensors"
  )
  expect_error(hf_load_local_model(path), "Incomplete.*safetensors")
  local_test_write(path, "model.safetensors", "mock weights")
  unlink(file.path(path, "tokenizer.json"))
  expect_error(hf_download_model("example/tiny", local_files_only = TRUE), "tokenizer files are missing")
  local_test_write(path, "tokenizer.json", "{}")
  unlink(file.path(path, "config.json"))
  expect_error(hf_download_model("example/tiny", local_files_only = TRUE), "config.json is missing")
})

test_that("sharded safetensors and repository subdirectories are preserved and checked", {
  path <- local_test_fixture()
  unlink(file.path(path, "model.safetensors"))
  local_test_write(path, "model.safetensors.index.json", paste0(
    '{"weight_map":{"a":"weights/model-00001.safetensors",',
    '"b":"weights/model-00002.safetensors"}}'
  ))
  local_test_write(path, file.path("weights", "model-00001.safetensors"), "first shard")
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_snapshot_download = function(...) path
  )
  expect_error(hf_download_model("example/tiny", local_files_only = TRUE), "shards are missing")
  local_test_write(path, file.path("weights", "model-00002.safetensors"), "second shard")
  expect_identical(hf_download_model("example/tiny", local_files_only = TRUE), path)
  expect_true(file.exists(file.path(path, "weights", "model-00001.safetensors")))
  expect_identical(basename(path), strrep("a", 40))

  local_test_write(path, "model.safetensors.index.json", '{"weight_map":{"a":"../outside.safetensors"}}')
  expect_error(hf_download_model("example/tiny", local_files_only = TRUE), "repository-relative")
  local_test_write(path, "model.safetensors.index.json", '{"weight_map":{"a":"pytorch_model.bin"}}')
  expect_error(hf_download_model("example/tiny", local_files_only = TRUE), "shard index is invalid")
})

test_that("standard sentence-transformer module files are checked before any backend call", {
  path <- local_test_fixture()
  unlink(file.path(path, "1_Pooling", "config.json"))
  local_test_write(path, "modules.json", paste0(
    '[{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},',
    '{"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"},',
    '{"idx":2,"name":"2","path":"2_Normalize","type":"sentence_transformers.models.Normalize"}]'
  ))
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_snapshot_download = function(...) path
  )
  expect_error(hf_download_model("example/tiny", local_files_only = TRUE), "Pooling.*missing")
  local_test_write(path, file.path("1_Pooling", "config.json"), '{"word_embedding_dimension":2}')
  expect_identical(hf_download_model("example/tiny", local_files_only = TRUE), path)

  local_test_write(path, "modules.json", '[{"type":"custom_model.Custom","path":""}]')
  expect_error(hf_download_model("example/tiny"), "only built-in")
  local_test_write(path, "modules.json", paste0(
    '[{"type":"sentence_transformers.models.Transformer","path":"0_Transformer"}]'
  ))
  expect_error(hf_download_model("example/tiny"), "snapshot root")
  unlink(file.path(path, "modules.json"))
  local_test_write(path, "sentence_bert_config.json", '{"tokenizer_name_or_path":"different/model"}')
  expect_error(hf_download_model("example/tiny"), "tokenizer cannot redirect")
})

test_that("adapters and invalid JSON do not trigger alternate model loading", {
  path <- local_test_fixture()
  testthat::local_mocked_bindings(
    hf_local_load_backend = local_test_unexpected,
    hf_local_snapshot_download = local_test_unexpected,
    hf_api_request = local_test_unexpected
  )
  local_test_write(path, "adapter_config.json", '{"base_model_name_or_path":"different/model"}')
  expect_error(hf_load_local_model(path), "adapter models are not supported")
  unlink(file.path(path, "adapter_config.json"))
  local_test_write(path, "modules.json", "not JSON")
  expect_error(hf_load_local_model(path), "read local model metadata", class = "hf_local_error")
})

test_that("Hub model handles record resolved rather than mutable requested revisions", {
  path <- local_test_fixture()
  backend <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    hf_download_model = function(model, revision, cache_dir, token, local_files_only) {
      expect_identical(model, "example/tiny")
      expect_identical(revision, "release")
      expect_identical(cache_dir, "selected-cache")
      expect_identical(token, "hf_not_stored")
      expect_true(local_files_only)
      path
    },
    hf_local_load_backend = function(path, task, device) {
      expect_identical(task, "embed")
      expect_identical(device, "cpu")
      backend
    }
  )
  result <- hf_load_local_model(
    "example/tiny", revision = "release", cache_dir = "selected-cache",
    token = "hf_not_stored", local_files_only = TRUE
  )
  expect_s3_class(result, "hf_local_model")
  expect_identical(result$task, "embed")
  expect_identical(result$model, "example/tiny")
  expect_identical(result$source, "hub")
  expect_identical(result$path, path)
  expect_identical(result$revision, strrep("a", 40))
  expect_identical(result$requested_revision, "release")
  expect_identical(result$device, "cpu")
  expect_identical(result$backend, backend)
  expect_false("token" %in% names(result))
  expect_false(any(grepl("hf_not_stored", capture.output(str(result)))))
})

test_that("NULL model resolution uses task defaults and local directories never download", {
  path <- local_test_fixture(snapshot = FALSE)
  expected_task <- "classify"
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_download_model = function(model, ...) {
      expect_identical(model, hf_default_model(expected_task))
      path
    },
    hf_local_load_backend = function(path, task, device) {
      expect_identical(task, expected_task)
      list(mock = TRUE)
    }
  )
  expect_identical(hf_load_local_model(task = "classify")$model, hf_default_model("classify"))
  expected_task <- "embed"
  expect_identical(hf_load_local_model()$model, hf_default_model("embed"))

  testthat::local_mocked_bindings(hf_download_model = local_test_unexpected)
  result <- hf_load_local_model(path, device = "cuda:0", token = "hf_not_used")
  expect_identical(result$source, "local")
  expect_identical(result$path, path)
  expect_null(result$revision)
  expect_null(result$requested_revision)
  expect_identical(result$device, "cuda:0")
})

test_that("the print method returns the handle invisibly without dumping Python objects", {
  path <- local_test_fixture()
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_load_backend = function(...) list(secret_backend_payload = "do not print")
  )
  model <- hf_load_local_model(path)
  output <- capture.output(visible <- withVisible(print(model)))
  expect_match(paste(output, collapse = "\n"), "<hf_local_model> embed")
  expect_match(paste(output, collapse = "\n"), "Device: cpu")
  expect_match(paste(output, collapse = "\n"), strrep("a", 40))
  expect_false(any(grepl("secret_backend_payload|do not print", output)))
  expect_false(visible$visible)
  expect_identical(visible$value, model)
})

test_that("mocked pinned embeddings preserve snapshot contents and offline results", {
  model_id <- "sentence-transformers/all-MiniLM-L6-v2"
  sha <- "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
  path <- local_test_fixture(revision = sha)
  cache <- dirname(dirname(dirname(path)))
  local_test_write(path, "modules.json", paste0(
    '[{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},',
    '{"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"},',
    '{"idx":2,"name":"2","path":"2_Normalize","type":"sentence_transformers.models.Normalize"}]'
  ))
  pooling <- '{"word_embedding_dimension":384,"pooling_mode_mean_tokens":true}'
  local_test_write(path, file.path("1_Pooling", "config.json"), pooling)
  withr::local_envvar(c(HF_TOKEN = "hf_unit_test", HUGGING_FACE_HUB_TOKEN = NA_character_))
  initialized <- FALSE
  declarations <- list()
  downloads <- logical()
  loads <- character()
  encodings <- 0L
  testthat::local_mocked_bindings(
    hf_local_callable_parameters = \(callable) "processor_kwargs",
    hf_local_reticulate_version = function() "1.46.0",
    hf_local_py_require = function(packages, python_version) {
      declarations[[length(declarations) + 1L]] <<- list(
        packages = packages, python_version = python_version
      )
    },
    hf_local_check_environment = function() {
      initialized <<- TRUE
      invisible(TRUE)
    },
    hf_local_call = local_test_call,
    hf_local_python_list = function(text) {
      expect_type(text, "list")
      expect_null(names(text))
      text
    },
    hf_local_import = function(module) {
      expect_true(initialized)
      switch(
        module,
        huggingface_hub = list(snapshot_download = function(...) {
          args <- list(...)
          expect_identical(args$repo_id, model_id)
          expect_identical(args$revision, sha)
          expect_identical(args$cache_dir, cache)
          expect_identical(args$token, "hf_unit_test")
          downloads <<- c(downloads, args$local_files_only)
          path
        }),
        sentence_transformers = list(SentenceTransformer = function(...) {
          args <- list(...)
          expect_identical(args$model_name_or_path, path)
          expect_identical(args$device, "cpu")
          expect_true(args$local_files_only)
          expect_false(args$trust_remote_code)
          expect_true(args$model_kwargs$use_safetensors)
          loads <<- c(loads, args$model_name_or_path)
          list(encode = function(sentences, batch_size, convert_to_numpy,
                                 normalize_embeddings, show_progress_bar) {
            expect_identical(batch_size, 2L)
            expect_true(convert_to_numpy)
            expect_true(normalize_embeddings)
            expect_false(show_progress_bar)
            encodings <<- encodings + 1L
            rows <- do.call(rbind, lapply(sentences, function(text) {
              seq_len(384L) * if (identical(text, "first")) 1 else -1
            }))
            rows / sqrt(rowSums(rows^2))
          })
        }),
        local_test_unexpected()
      )
    },
    hf_api_request = local_test_unexpected
  )

  expect_invisible(hf_local_setup())
  expect_false(initialized)
  expect_length(declarations, 1L)
  snapshot <- hf_download_model(model_id, revision = sha, cache_dir = cache)
  expect_identical(basename(snapshot), sha)
  expect_identical(readLines(file.path(snapshot, "1_Pooling", "config.json")), pooling)
  model <- hf_load_local_model(snapshot, task = "embed")
  expect_identical(model$revision, sha)
  expect_identical(model$source, "local")

  texts <- c("first", NA, "\u03b4", "first")
  online <- hf_embed_local(texts, model, batch_size = 2L, normalize = TRUE)
  expect_identical(online$text, texts)
  expect_identical(online$n_dims, c(384L, NA_integer_, 384L, 384L))
  expect_null(online$embedding[[2]])
  expected <- seq_len(384L) / sqrt(sum(seq_len(384L)^2))
  expect_equal(online$embedding, list(expected, NULL, -expected, expected))
  expect_identical(hf_embed_local(texts, model, batch_size = 2L, normalize = TRUE), online)
  expect_identical(downloads, FALSE)
  expect_identical(loads, path)

  cached <- hf_download_model(
    model_id, revision = sha, cache_dir = cache, local_files_only = TRUE
  )
  expect_identical(cached, snapshot)
  offline_model <- hf_load_local_model(
    model_id, task = "embed", revision = sha,
    cache_dir = cache, local_files_only = TRUE
  )
  offline <- hf_embed_local(texts, offline_model, batch_size = 2L, normalize = TRUE)
  expect_identical(offline, online)
  expect_identical(offline_model$revision, sha)
  expect_identical(offline_model$path, snapshot)
  expect_identical(downloads, c(FALSE, TRUE, TRUE))
  expect_identical(loads, rep(path, 2L))
  expect_identical(encodings, 3L)
  expect_true(all(vapply(declarations, identical, logical(1), declarations[[1]])))
})

test_that("mocked pinned classification reloads offline with matching scores and row order", {
  model_id <- "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
  sha <- "714eb0fa89d2f80546fda750413ed43d93601a13"
  path <- local_test_fixture(revision = sha)
  cache <- dirname(dirname(dirname(path)))
  withr::local_envvar(c(HF_TOKEN = "hf_unit_test", HUGGING_FACE_HUB_TOKEN = NA_character_))
  downloads <- logical()
  tokenizer_paths <- character()
  model_paths <- character()
  pipelines <- 0L
  declarations <- list()
  testthat::local_mocked_bindings(
    hf_local_reticulate_version = function() "1.46.0",
    hf_local_py_require = function(packages, python_version) {
      declarations[[length(declarations) + 1L]] <<- list(
        packages = packages, python_version = python_version
      )
    },
    hf_local_check_environment = function() invisible(TRUE),
    hf_local_call = local_test_call,
    hf_local_python_list = function(text) {
      expect_type(text, "list")
      expect_null(names(text))
      text
    },
    hf_local_import = function(module) {
      switch(
        module,
        huggingface_hub = list(snapshot_download = function(...) {
          args <- list(...)
          expect_identical(args$repo_id, model_id)
          expect_identical(args$revision, sha)
          expect_identical(args$cache_dir, cache)
          downloads <<- c(downloads, args$local_files_only)
          path
        }),
        transformers = list(
          AutoTokenizer = list(from_pretrained = function(...) {
            args <- list(...)
            expect_identical(args$pretrained_model_name_or_path, path)
            expect_true(args$local_files_only)
            expect_false(args$trust_remote_code)
            tokenizer_paths <<- c(tokenizer_paths, args$pretrained_model_name_or_path)
            list(kind = "mock_tokenizer", path = args$pretrained_model_name_or_path)
          }),
          AutoModelForSequenceClassification = list(from_pretrained = function(...) {
            args <- list(...)
            expect_identical(args$pretrained_model_name_or_path, path)
            expect_true(args$local_files_only)
            expect_false(args$trust_remote_code)
            expect_true(args$use_safetensors)
            model_paths <<- c(model_paths, args$pretrained_model_name_or_path)
            list(kind = "mock_model", path = args$pretrained_model_name_or_path)
          }),
          pipeline = function(task, model, tokenizer, device) {
            expect_identical(task, "text-classification")
            expect_identical(model, list(kind = "mock_model", path = path))
            expect_identical(tokenizer, list(kind = "mock_tokenizer", path = path))
            expect_identical(device, "cpu")
            pipelines <<- pipelines + 1L
            function(inputs, batch_size, truncation, top_k) {
              expect_identical(batch_size, 32L)
              expect_true(truncation)
              expect_identical(top_k, 1L)
              lapply(inputs, function(text) {
                record <- switch(
                  text,
                  "I love this." = list(label = "POSITIVE", score = 0.98),
                  "I hate this." = list(label = "NEGATIVE", score = 0.97),
                  local_test_unexpected()
                )
                list(record)
              })
            }
          }
        ),
        local_test_unexpected()
      )
    },
    hf_api_request = local_test_unexpected
  )

  model <- hf_load_local_model(model_id, task = "classify", revision = sha, cache_dir = cache)
  expect_identical(basename(model$path), sha)
  expect_identical(model$revision, sha)
  texts <- c("I love this.", NA, "I hate this.", "I love this.", "")
  online <- hf_classify_local(texts, model)
  expect_identical(online$text, texts)
  expect_identical(online$label, c("POSITIVE", NA, "NEGATIVE", "POSITIVE", NA))
  expect_identical(online$score, c(0.98, NA, 0.97, 0.98, NA))
  singleton <- hf_classify_local("I hate this.", model)
  expect_identical(singleton$label, "NEGATIVE")
  expect_identical(singleton$score, 0.97)
  expect_identical(downloads, FALSE)
  expect_identical(pipelines, 1L)

  offline_model <- hf_load_local_model(
    model_id, task = "classify", revision = sha,
    cache_dir = cache, local_files_only = TRUE
  )
  offline <- hf_classify_local(texts, offline_model)
  expect_identical(offline, online)
  expect_identical(offline_model$path, model$path)
  expect_identical(offline_model$revision, sha)
  expect_identical(downloads, c(FALSE, TRUE))
  expect_identical(tokenizer_paths, rep(path, 2L))
  expect_identical(model_paths, tokenizer_paths)
  expect_identical(pipelines, 2L)
  expect_true(all(vapply(declarations, identical, logical(1), hf_local_requirements())))
})

test_that("all classification components receive the exact snapshot path and safe flags", {
  path <- "C:\\Users\\O'Connor\\model cache\\snapshots\\abc\\\u6a21\u578b"
  tokenizer <- list(kind = "tokenizer")
  weights <- list(kind = "model")
  pipeline <- list(kind = "pipeline")
  calls <- character()
  testthat::local_mocked_bindings(
    hf_local_initialize = local_test_unexpected,
    hf_local_call = local_test_call,
    hf_local_import = function(module) {
      expect_identical(module, "transformers")
      list(
        AutoTokenizer = list(from_pretrained = function(...) {
          args <- list(...)
          calls <<- c(calls, "tokenizer")
          expect_identical(args$pretrained_model_name_or_path, path)
          expect_true(args$local_files_only)
          expect_false(args$trust_remote_code)
          expect_false("token" %in% names(args))
          tokenizer
        }),
        AutoModelForSequenceClassification = list(from_pretrained = function(...) {
          args <- list(...)
          calls <<- c(calls, "model")
          expect_identical(args$pretrained_model_name_or_path, path)
          expect_true(args$local_files_only)
          expect_false(args$trust_remote_code)
          expect_true(args$use_safetensors)
          expect_false("token" %in% names(args))
          weights
        }),
        pipeline = function(...) {
          args <- list(...)
          calls <<- c(calls, "pipeline")
          expect_identical(args$task, "text-classification")
          expect_identical(args$model, weights)
          expect_identical(args$tokenizer, tokenizer)
          expect_identical(args$device, "cpu")
          expect_false("token" %in% names(args))
          pipeline
        }
      )
    }
  )
  expect_identical(hf_local_load_backend(path, "classify", "cpu"), pipeline)
  expect_identical(calls, c("tokenizer", "model", "pipeline"))
})

test_that("embedding construction keeps all transformer components local and safetensors-only", {
  path <- "C:\\model cache\\snapshot 'quoted'\\\u6a21\u578b"
  backend <- list(kind = "embedding-model")
  testthat::local_mocked_bindings(
    hf_local_callable_parameters = \(callable) "processor_kwargs",
    hf_local_initialize = local_test_unexpected,
    hf_local_call = local_test_call,
    hf_local_import = function(module) {
      expect_identical(module, "sentence_transformers")
      list(SentenceTransformer = function(...) {
        args <- list(...)
        expect_identical(args$model_name_or_path, path)
        expect_identical(args$device, "mps")
        expect_true(args$local_files_only)
        expect_false(args$trust_remote_code)
        expect_true(args$model_kwargs$use_safetensors)
        for (name in c("model_kwargs", "processor_kwargs", "config_kwargs")) {
          expect_true(args[[name]]$local_files_only)
          expect_false(args[[name]]$trust_remote_code)
        }
        expect_false("token" %in% names(args))
        backend
      })
    }
  )
  expect_identical(hf_local_load_backend(path, "embed", "mps"), backend)
})

test_that("local loading reports backend and device errors without a fallback", {
  path <- local_test_fixture()
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_load_backend = function(...) stop("CUDA is unavailable"),
    hf_api_request = local_test_unexpected
  )
  expect_error(hf_load_local_model(path, device = "cuda:0"), "CUDA is unavailable")
  expect_error(hf_load_local_model(path, device = "cuda:0"), "No hosted inference")
  testthat::local_mocked_bindings(hf_local_load_backend = function(...) NULL)
  expect_error(hf_load_local_model(path), "did not return a model")
})

test_that("download and load arguments are validated before Python is needed", {
  testthat::local_mocked_bindings(
    hf_local_initialize = local_test_unexpected,
    hf_local_snapshot_download = local_test_unexpected,
    hf_local_load_backend = local_test_unexpected
  )
  for (model in list(NA_character_, "", character(), c("a", "b"), 1, "C:\\missing\\model")) {
    expect_error(hf_download_model(model), "`model`")
    expect_error(hf_load_local_model(model), "`model`")
  }
  expect_error(hf_download_model("example/tiny", revision = NA_character_), "`revision`")
  expect_error(hf_download_model("example/tiny", cache_dir = FALSE), "`cache_dir`")
  expect_error(hf_download_model("example/tiny", token = c("a", "b")), "`token`")
  expect_error(hf_download_model("example/tiny", local_files_only = NA), "`local_files_only`")
  expect_error(hf_load_local_model("example/tiny", device = 0), "`device`")
  expect_error(hf_load_local_model("example/tiny", task = "chat"), "arg")
})

test_that("nonexistent local-looking paths fail before the Hub download helper", {
  directory <- local_test_fixture(snapshot = FALSE)
  withr::local_dir(directory)
  paths <- c(
    file.path(directory, "missing"),
    "C:\\hf-local-missing\\model",
    "C:/hf-local-missing/model",
    "C:hf-local-missing",
    "relative\\missing",
    "\\hf-local-missing\\model",
    "/hf-local-missing/model",
    "./missing",
    "../missing",
    ".\\missing",
    "..\\missing",
    "~/hf-local-missing"
  )
  testthat::local_mocked_bindings(
    hf_download_model = local_test_unexpected,
    hf_local_snapshot_download = local_test_unexpected,
    hf_local_load_backend = local_test_unexpected,
    hf_local_initialize = local_test_unexpected,
    hf_api_request = local_test_unexpected
  )

  expect_silent({
    errors <- lapply(paths, function(path) {
      tryCatch(hf_load_local_model(path, token = "hf_path_secret"), error = identity)
    })
  })
  for (error in errors) {
    expect_s3_class(error, "error")
    expect_match(conditionMessage(error), "existing local directory")
    expect_false(grepl("hf_path_secret", conditionMessage(error), fixed = TRUE))
    expect_null(conditionCall(error))
  }
})

test_that("explicit downloads reject local path syntax before reaching Python", {
  paths <- c(
    "C:\\hf-local-missing\\model",
    "C:/hf-local-missing/model",
    "C:hf-local-missing",
    "relative\\missing",
    "\\\\server\\share\\missing",
    "/hf-local-missing/model",
    "./missing",
    "../missing",
    ".\\missing",
    "..\\missing",
    "~/hf-local-missing",
    "file:///hf-local-missing/model"
  )
  testthat::local_mocked_bindings(
    hf_local_snapshot_download = local_test_unexpected,
    hf_local_initialize = local_test_unexpected,
    hf_api_request = local_test_unexpected
  )
  for (path in paths) {
    expect_error(hf_download_model(path), "Hub model ID")
  }
})

test_that("existing quoted local paths are passed directly without token retention or logging", {
  directory <- local_test_fixture(snapshot = FALSE)
  backend <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_download_model = local_test_unexpected,
    hf_local_snapshot_download = local_test_unexpected,
    hf_local_load_backend = function(path, task, device) {
      expect_identical(path, directory)
      expect_identical(task, "classify")
      expect_identical(device, "cpu")
      backend
    }
  )
  expect_silent({
    model <- hf_load_local_model(directory, task = "classify", token = "hf_local_secret")
  })
  expect_identical(model$path, directory)
  expect_identical(model$backend, backend)
  expect_false("token" %in% names(model))
  expect_false(any(grepl("hf_local_secret", capture.output(str(model)), fixed = TRUE)))
  expect_false(any(grepl("hf_local_secret", capture.output(print(model)), fixed = TRUE)))
})

test_that("prediction requires reusable handles with the matching task", {
  testthat::local_mocked_bindings(
    hf_local_encode = local_test_unexpected,
    hf_local_classify = local_test_unexpected
  )
  expect_error(hf_embed_local("text", "example/tiny"), "hf_local_model handle")
  expect_error(hf_classify_local("text", NULL), "hf_local_model handle")
  expect_error(hf_embed_local("text", local_test_handle("classify")), 'task = "embed"')
  expect_error(hf_classify_local("text", local_test_handle()), 'task = "classify"')
  broken <- structure(list(task = "embed", backend = NULL), class = "hf_local_model")
  expect_error(hf_embed_local(character(), broken), "hf_local_model handle")
})

test_that("empty and all-missing embeddings preserve the hosted schema without backend calls", {
  testthat::local_mocked_bindings(hf_local_encode = local_test_unexpected)
  model <- local_test_handle()
  result <- hf_embed_local(character(), model)
  expect_s3_class(result, "tbl_df")
  expect_identical(names(result), c("text", "embedding", "n_dims"))
  expect_identical(result$text, character())
  expect_identical(result$embedding, list())
  expect_identical(result$n_dims, integer())
  result <- hf_embed_local(c(NA_character_, NA_character_), model)
  expect_identical(result$text, c(NA_character_, NA_character_))
  expect_identical(result$embedding, list(NULL, NULL))
  expect_identical(result$n_dims, c(NA_integer_, NA_integer_))
})

test_that("embeddings preserve Unicode duplicates missing rows and input order", {
  calls <- 0L
  texts <- c("caf\u00e9", NA, "\u6a21\u578b", "caf\u00e9", "")
  testthat::local_mocked_bindings(
    hf_local_encode = function(backend, text, batch_size, normalize, progress) {
      calls <<- calls + 1L
      expect_identical(text, as.list(texts[c(1, 3, 4, 5)]))
      expect_identical(batch_size, 2L)
      expect_true(normalize)
      expect_true(progress)
      matrix(c(1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9), nrow = 4L, byrow = TRUE)
    }
  )
  result <- hf_embed_local(texts, local_test_handle(), batch_size = 2, normalize = TRUE, progress = TRUE)
  expect_identical(result$text, texts)
  expect_identical(result$embedding, list(c(1, 2, 3), NULL, c(4, 5, 6), c(1, 2, 3), c(7, 8, 9)))
  expect_identical(result$n_dims, c(3L, NA_integer_, 3L, 3L, 3L))
  expect_identical(calls, 1L)
})

test_that("singleton embeddings remain matrices and convert each row to a numeric vector", {
  testthat::local_mocked_bindings(
    hf_local_encode = function(backend, text, batch_size, normalize, progress) {
      expect_identical(text, list("one"))
      expect_identical(batch_size, 32L)
      expect_false(normalize)
      expect_false(progress)
      matrix(7L, nrow = 1L, ncol = 1L)
    }
  )
  result <- hf_embed_local("one", local_test_handle())
  expect_identical(result$embedding, list(7))
  expect_identical(result$n_dims, 1L)
})

test_that("the encoding boundary forwards a list and explicit numpy and normalization options", {
  python_list <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    hf_local_python_list = function(text) {
      expect_identical(text, list("one"))
      python_list
    },
    hf_local_call = local_test_call
  )
  backend <- list(encode = function(sentences, batch_size, convert_to_numpy,
                                   normalize_embeddings, show_progress_bar) {
    expect_identical(sentences, python_list)
    expect_identical(batch_size, 8L)
    expect_true(convert_to_numpy)
    expect_true(normalize_embeddings)
    expect_false(show_progress_bar)
    matrix(c(1, 2), nrow = 1L)
  })
  expect_equal(
    hf_local_encode(backend, list("one"), 8L, TRUE, FALSE),
    matrix(c(1, 2), nrow = 1L)
  )
})

test_that("malformed embedding responses are errors instead of missing-result fallbacks", {
  response <- NULL
  testthat::local_mocked_bindings(hf_local_encode = function(...) response)
  bad <- list(
    NULL, c(1, 2), list(c(1, 2)), matrix(1, 2, 2), matrix(numeric(), 1, 0),
    matrix("x", 1, 1), matrix(TRUE, 1, 1), matrix(NA_real_, 1, 2),
    matrix(Inf, 1, 1), matrix(NaN, 1, 1), matrix(1 + 1i, 1, 1),
    array(1, dim = c(1, 1, 1))
  )
  for (value in bad) {
    response <- value
    expect_error(hf_embed_local("one", local_test_handle()), "Malformed local embeddings")
  }
})

test_that("empty and missing classification rows retain their schema without Python calls", {
  testthat::local_mocked_bindings(hf_local_classify = local_test_unexpected)
  model <- local_test_handle("classify")
  result <- hf_classify_local(character(), model)
  expect_s3_class(result, "tbl_df")
  expect_identical(names(result), c("text", "label", "score"))
  expect_identical(result$text, character())
  expect_identical(result$label, character())
  expect_identical(result$score, numeric())
  result <- hf_classify_local(c(NA, "", NA), model)
  expect_identical(result$text, c(NA, "", NA))
  expect_identical(result$label, rep(NA_character_, 3))
  expect_identical(result$score, rep(NA_real_, 3))
})

test_that("classification preserves Unicode duplicates and missing-row alignment", {
  texts <- c("\u5f88\u597d", NA, "bad", "\u5f88\u597d", "")
  calls <- 0L
  testthat::local_mocked_bindings(
    hf_local_classify = function(backend, text, batch_size, truncation) {
      calls <<- calls + 1L
      expect_identical(text, as.list(texts[c(1, 3, 4)]))
      expect_identical(batch_size, 2L)
      expect_false(truncation)
      list(
        list(list(label = "POS", score = 0.9)),
        list(list(label = "NEG", score = 0.8)),
        list(list(label = "POS", score = 0.9))
      )
    }
  )
  result <- hf_classify_local(texts, local_test_handle("classify"), batch_size = 2, truncation = FALSE)
  expect_identical(result$text, texts)
  expect_identical(result$label, c("POS", NA, "NEG", "POS", NA))
  expect_identical(result$score, c(0.9, NA, 0.8, 0.9, NA))
  expect_identical(calls, 1L)
})

test_that("singleton classification accepts dictionary flat and nested list shapes", {
  response <- NULL
  testthat::local_mocked_bindings(
    hf_local_classify = function(backend, text, batch_size, truncation) {
      expect_identical(text, list("one"))
      expect_identical(batch_size, 32L)
      expect_true(truncation)
      response
    }
  )
  record <- list(label = "POS", score = 0.75)
  for (shape in list(record, list(record), list(list(record)))) {
    response <- shape
    result <- hf_classify_local("one", local_test_handle("classify"))
    expect_identical(result$label, "POS")
    expect_identical(result$score, 0.75)
  }
})

test_that("flat classification batches and finite non-probability scores are retained", {
  testthat::local_mocked_bindings(
    hf_local_classify = function(...) {
      list(list(label = "LABEL_0", score = -2L), list(label = "LABEL_1", score = 4))
    }
  )
  result <- hf_classify_local(c("a", "b"), local_test_handle("classify"))
  expect_identical(result$label, c("LABEL_0", "LABEL_1"))
  expect_identical(result$score, c(-2, 4))
})

test_that("the classification boundary sends an explicit list and top_k equals one", {
  python_list <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    hf_local_python_list = function(text) {
      expect_identical(text, list("one"))
      python_list
    },
    hf_local_call = local_test_call
  )
  backend <- function(inputs, batch_size, truncation, top_k) {
    expect_identical(inputs, python_list)
    expect_identical(batch_size, 4L)
    expect_false(truncation)
    expect_identical(top_k, 1L)
    list(list(list(label = "POS", score = 0.8)))
  }
  expect_identical(
    hf_local_classify(backend, list("one"), 4L, FALSE),
    list(list(list(label = "POS", score = 0.8)))
  )
})

test_that("malformed classification output never becomes a silent NA result", {
  response <- NULL
  record <- list(label = "POS", score = 0.8)
  testthat::local_mocked_bindings(hf_local_classify = function(...) response)
  bad <- list(
    NULL, list(), "POS", list(record, record), list(list(record, record)),
    list(list(list(record))), list(label = "POS"), list(score = 0.8),
    list(label = NA_character_, score = 0.8), list(label = "", score = 0.8),
    list(label = c("A", "B"), score = 0.8), list(label = 1, score = 0.8),
    list(label = "POS", score = NA_real_), list(label = "POS", score = Inf),
    list(label = "POS", score = NaN), list(label = "POS", score = "0.8"),
    list(label = "POS", score = TRUE), list(label = "POS", score = 1 + 1i),
    list(label = "POS", score = c(0.1, 0.9)),
    list(label = matrix("POS", 1), score = 0.8),
    list(label = "POS", score = matrix(0.8, 1)),
    list(labels = "POS", scores = 0.8),
    structure(list("POS", "NEG", 0.8), names = c("label", "label", "score")),
    data.frame(label = "POS", score = 0.8)
  )
  for (value in bad) {
    response <- value
    expect_error(
      hf_classify_local("one", local_test_handle("classify")),
      "Malformed local classification"
    )
  }
  response <- list(record)
  expect_error(
    hf_classify_local(c("one", "two"), local_test_handle("classify")),
    "one result per text"
  )
})

test_that("prediction input types batches and flags are validated consistently", {
  testthat::local_mocked_bindings(
    hf_local_encode = local_test_unexpected,
    hf_local_classify = local_test_unexpected
  )
  embedding_model <- local_test_handle()
  classification_model <- local_test_handle("classify")
  for (text in list(NULL, 1, list("x"), factor("x"), matrix("x", 1))) {
    expect_error(hf_embed_local(text, embedding_model), "`text`")
    expect_error(hf_classify_local(text, classification_model), "`text`")
  }
  for (batch in list(NULL, 0, -1, 1.5, NA_real_, Inf, TRUE, "2", c(1, 2), 2^31, 1 + 1i)) {
    expect_error(hf_embed_local("x", embedding_model, batch_size = batch), "positive scalar integer")
    expect_error(hf_classify_local("x", classification_model, batch_size = batch), "positive scalar integer")
  }
  for (flag in list(NULL, NA, 1, c(TRUE, FALSE), "TRUE")) {
    expect_error(hf_embed_local("x", embedding_model, normalize = flag), "`normalize`")
    expect_error(hf_embed_local("x", embedding_model, progress = flag), "`progress`")
    expect_error(hf_classify_local("x", classification_model, truncation = flag), "`truncation`")
  }
})

test_that("prediction errors retain the original cause and never use hosted inference", {
  testthat::local_mocked_bindings(
    hf_local_encode = function(...) stop("stale Python object"),
    hf_local_classify = function(...) stop("input exceeds maximum length"),
    hf_api_request = local_test_unexpected,
    hf_local_snapshot_download = local_test_unexpected,
    hf_local_initialize = local_test_unexpected
  )
  expect_error(hf_embed_local("x", local_test_handle()), "stale Python object", class = "hf_local_error")
  expect_error(
    hf_classify_local("x", local_test_handle("classify"), truncation = FALSE),
    "input exceeds maximum length", class = "hf_local_error"
  )
  expect_error(hf_embed_local("x", local_test_handle()), "Reload the model")
})

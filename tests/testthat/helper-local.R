local_test_fixture <- function(snapshot = TRUE, revision = strrep("a", 40),
                               .local_envir = parent.frame()) {
  root <- withr::local_tempdir(pattern = "hf-", .local_envir = .local_envir)
  path <- if (snapshot) {
    file.path(root, "models--test", "snapshots", revision)
  } else {
    file.path(root, "model with spaces and 'quotes'")
  }
  if (!dir.create(path, recursive = TRUE)) {
    stop("Could not create a local model test fixture.")
  }
  local_test_write(path, "config.json", '{"model_type":"bert"}')
  local_test_write(path, "tokenizer.json", "{}")
  local_test_write(path, "model.safetensors", "mock safetensors weights")
  local_test_write(path, "modules.json", paste0(
    '[{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},',
    '{"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"}]'
  ))
  local_test_write(path, file.path("1_Pooling", "config.json"),
                   '{"word_embedding_dimension":384,"pooling_mode_mean_tokens":true}')
  normalizePath(path, mustWork = TRUE)
}

local_test_write <- function(path, name, contents) {
  file <- file.path(path, name)
  if (!dir.exists(dirname(file))) {
    dir.create(dirname(file), recursive = TRUE)
  }
  writeLines(contents, file, useBytes = TRUE)
  invisible(file)
}

local_test_handle <- function(task = "embed") {
  structure(
    list(task = task, backend = new.env(parent = emptyenv())),
    class = "hf_local_model"
  )
}

local_test_unexpected <- function(...) {
  stop("Unexpected Python, download, or hosted inference call.")
}

local_test_call <- function(object, ...) {
  object(...)
}

local_test_imports <- function(python = "3.12.4", incompatible = NULL) {
  versions <- c(
    transformers = "4.57.0",
    "sentence-transformers" = "5.1.0",
    huggingface_hub = "0.35.0",
    torch = "2.6.0+cpu",
    numpy = "2.2.6"
  )
  checked <- new.env(parent = emptyenv())
  checked$specs <- character()
  checked$versions <- character()
  list(
    checked = checked,
    import = function(module) {
      switch(
        module,
        "importlib.metadata" = list(version = function(name) unname(versions[[name]])),
        "packaging.requirements" = list(Requirement = function(spec) {
          name <- sub("[<>=].*$", "", spec)
          list(name = name, specifier = list(contains = function(version) {
            checked$specs <- c(checked$specs, spec)
            checked$versions <- c(checked$versions, version)
            !identical(name, incompatible)
          }))
        }),
        "packaging.specifiers" = list(SpecifierSet = function(spec) {
          expect_identical(spec, ">=3.10,<3.13")
          list(contains = function(version) {
            expect_identical(version, python)
            !identical(incompatible, "python")
          })
        }),
        "platform" = list(python_version = function() python),
        stop("Unexpected module: ", module)
      )
    }
  )
}

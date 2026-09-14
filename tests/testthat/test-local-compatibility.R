local_test_constructor_import <- function(constructor) {
  function(module) {
    switch(
      module,
      sentence_transformers = list(SentenceTransformer = constructor),
      inspect = list(signature = \(object) {
        list(parameters = as.list(formals(object)))
      }),
      builtins = list(list = names),
      local_test_unexpected()
    )
  }
}

test_that("modern constructors never receive the deprecated tokenizer argument", {
  backend <- list(kind = "modern")
  arguments <- NULL
  constructor <- function(
    ...,
    processor_kwargs = NULL,
    tokenizer_kwargs = NULL
  ) {
    if (!is.null(tokenizer_kwargs)) {
      stop("Deprecated tokenizer_kwargs reached the modern constructor")
    }
    arguments <<- c(list(...), list(processor_kwargs = processor_kwargs))
    backend
  }
  testthat::local_mocked_bindings(
    hf_local_call = local_test_call,
    hf_local_import = local_test_constructor_import(constructor)
  )

  expect_identical(hf_local_load_backend("snapshot", "embed", "cpu"), backend)
  expect_identical(
    arguments$processor_kwargs,
    list(local_files_only = TRUE, trust_remote_code = FALSE)
  )
  expect_null(arguments$tokenizer_kwargs)
  expect_identical(arguments$model_kwargs$use_safetensors, TRUE)
  expect_identical(arguments$config_kwargs$local_files_only, TRUE)
})

test_that("legacy constructors retain tokenizer arguments and local-only flags", {
  backend <- list(kind = "legacy")
  arguments <- NULL
  constructor <- function(..., tokenizer_kwargs = NULL) {
    arguments <<- c(list(...), list(tokenizer_kwargs = tokenizer_kwargs))
    if ("processor_kwargs" %in% names(arguments)) {
      stop("Unsupported processor_kwargs reached the legacy constructor")
    }
    backend
  }
  testthat::local_mocked_bindings(
    hf_local_call = local_test_call,
    hf_local_import = local_test_constructor_import(constructor)
  )

  expect_identical(hf_local_load_backend("snapshot", "embed", "cpu"), backend)
  expect_identical(
    arguments$tokenizer_kwargs,
    list(local_files_only = TRUE, trust_remote_code = FALSE)
  )
  expect_null(arguments$processor_kwargs)
})

test_that("unknown constructor interfaces fail rather than guessing an argument", {
  constructor <- function(...) list(kind = "unknown")
  testthat::local_mocked_bindings(
    hf_local_call = local_test_call,
    hf_local_import = local_test_constructor_import(constructor)
  )
  expect_snapshot(
    error = TRUE,
    hf_local_load_backend("snapshot", "embed", "cpu")
  )
})

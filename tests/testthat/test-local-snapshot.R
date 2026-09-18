test_that("embedding loads reject partial snapshots instead of changing pooling", {
  path <- local_test_fixture()
  unlink(file.path(path, "modules.json"))
  backend_calls <- 0L
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_snapshot_download = function(...) path,
    hf_local_load_backend = function(...) {
      backend_calls <<- backend_calls + 1L
      list(mock = TRUE)
    }
  )

  expect_error(
    hf_load_local_model(path, task = "embed", local_files_only = TRUE),
    "modules.json"
  )
  expect_error(
    hf_load_local_model("example/embedding", task = "embed", local_files_only = TRUE),
    "modules.json"
  )
  expect_identical(backend_calls, 0L)
})

test_that("classification does not require sentence embedding module metadata", {
  path <- local_test_fixture()
  unlink(file.path(path, "modules.json"))
  testthat::local_mocked_bindings(
    hf_local_initialize = function() invisible(TRUE),
    hf_local_load_backend = function(...) list(mock = TRUE)
  )

  expect_s3_class(
    hf_load_local_model(path, task = "classify", local_files_only = TRUE),
    "hf_local_model"
  )
})

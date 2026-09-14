test_that("each Hub load initializes and validates exactly once without a session cache", {
  path <- local_test_fixture()
  events <- character()
  validate <- hf_local_check_snapshot
  testthat::local_mocked_bindings(
    hf_local_initialize = function() {
      events <<- c(events, "initialize")
    },
    hf_local_snapshot_download = function(...) {
      events <<- c(events, "download")
      path
    },
    hf_local_check_snapshot = function(path) {
      events <<- c(events, "validate")
      validate(path)
    },
    hf_local_load_backend = function(...) {
      events <<- c(events, "load")
      list(mock = TRUE)
    }
  )

  expect_s3_class(hf_load_local_model("example/tiny"), "hf_local_model")
  expect_identical(events, c("initialize", "download", "validate", "load"))
  expect_s3_class(hf_load_local_model("example/tiny"), "hf_local_model")
  expect_identical(events, rep(c("initialize", "download", "validate", "load"), 2L))
})

test_that("directory loads validate before initialization and never download", {
  path <- local_test_fixture()
  events <- character()
  validate <- hf_local_check_snapshot
  testthat::local_mocked_bindings(
    hf_local_initialize = function() {
      events <<- c(events, "initialize")
    },
    hf_local_check_snapshot = function(path) {
      events <<- c(events, "validate")
      validate(path)
    },
    hf_local_load_backend = function(...) {
      events <<- c(events, "load")
      list(mock = TRUE)
    },
    hf_local_snapshot_download = local_test_unexpected
  )

  expect_s3_class(hf_load_local_model(path), "hf_local_model")
  expect_identical(events, c("validate", "initialize", "load"))
  unlink(file.path(path, "model.safetensors"))
  expect_error(hf_load_local_model(path), "Incomplete.*safetensors")
  expect_identical(events, c("validate", "initialize", "load", "validate"))
})

test_that("standalone downloads initialize and validate without constructing a model", {
  path <- local_test_fixture()
  initializations <- validations <- 0L
  validate <- hf_local_check_snapshot
  testthat::local_mocked_bindings(
    hf_local_initialize = function() initializations <<- initializations + 1L,
    hf_local_snapshot_download = function(...) path,
    hf_local_check_snapshot = function(path) {
      validations <<- validations + 1L
      validate(path)
    },
    hf_local_load_backend = local_test_unexpected
  )
  expect_identical(hf_download_model("example/tiny"), path)
  expect_identical(initializations, 1L)
  expect_identical(validations, 1L)
})

test_that("later operations recheck environment compatibility and propagate failures", {
  path <- local_test_fixture()
  initializations <- 0L
  testthat::local_mocked_bindings(
    hf_local_initialize = function() {
      initializations <<- initializations + 1L
      if (initializations > 1L) {
        stop("Selected Python environment no longer satisfies requirements")
      }
    },
    hf_local_snapshot_download = function(...) path,
    hf_local_load_backend = function(...) list(mock = TRUE)
  )
  expect_s3_class(hf_load_local_model("example/tiny"), "hf_local_model")
  expect_error(hf_load_local_model("example/tiny"), "no longer satisfies",
               class = "hf_local_error")
  expect_identical(initializations, 2L)
})

test_that("initialization failures retain token redaction at the public boundary", {
  testthat::local_mocked_bindings(
    hf_local_initialize = function() stop("Invalid environment hf_init_secret"),
    hf_local_snapshot_download = local_test_unexpected
  )
  error <- tryCatch(
    hf_download_model("example/tiny", token = "hf_init_secret"),
    error = identity
  )
  expect_s3_class(error, "hf_local_error")
  expect_match(conditionMessage(error), "Invalid environment <redacted>")
  expect_false(grepl("hf_init_secret", conditionMessage(error$parent), fixed = TRUE))
})

test_that("file metadata validation preserves empty directory missing and vector cases", {
  path <- local_test_fixture()
  empty <- file.path(path, "empty")
  expect_true(file.create(empty))
  paths <- c(file.path(path, "config.json"), empty, path,
             file.path(path, "missing"), NA_character_)
  expect_identical(hf_local_has_file(paths), c(TRUE, FALSE, FALSE, FALSE, FALSE))
  expect_identical(hf_local_has_file(character()), logical())
  expect_identical(hf_local_has_file(paths[c(1, 1, 4)]), c(TRUE, TRUE, FALSE))
})

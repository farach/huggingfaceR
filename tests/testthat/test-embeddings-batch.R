test_that("hf_embed_batch returns correct structure for empty input", {
  skip_on_cran()

  result <- hf_embed_batch(character())

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
  expect_true(all(c("text", "embedding", "n_dims", ".input_idx", ".error", ".error_msg") %in% names(result)))
})

test_that("hf_embed_batch returns tibble with correct columns", {
  skip_on_cran()

  # Mock test - just verify structure without API call
  # In real tests, this would call the API
  result <- hf_embed_batch(character())

  expected_cols <- c("text", "embedding", "n_dims", ".input_idx", ".error", ".error_msg")
  expect_true(all(expected_cols %in% names(result)))
})

test_that("hf_embed_chunks requires arrow package", {
  skip_on_cran()
  skip_if(requireNamespace("arrow", quietly = TRUE))

  expect_error(
    hf_embed_chunks("test", output_dir = tempfile()),
    "arrow"
  )
})

test_that("hf_embed_chunks handles empty input", {
  skip_on_cran()
  skip_if_not_installed("arrow")

  temp_dir <- tempfile()

  expect_message(
    result <- hf_embed_chunks(character(), output_dir = temp_dir),
    "No texts to process"
  )

  expect_equal(result, temp_dir)
})

test_that("hf_embed_chunks creates output directory", {
  skip_on_cran()
  skip_if_not_installed("arrow")

  temp_dir <- file.path(tempfile(), "nested", "dir")
  on.exit(unlink(dirname(dirname(temp_dir)), recursive = TRUE))

  # Empty input but should create directory logic
  hf_embed_chunks(character(), output_dir = temp_dir)

  # Directory not created for empty input (short-circuits)
  expect_false(dir.exists(temp_dir))
})

test_that("hf_embed_batch batch_size parameter works",
{
  skip_on_cran()

  # Just verify the parameter is accepted
  result <- hf_embed_batch(character(), batch_size = 50L)
  expect_s3_class(result, "tbl_df")
})

test_that("hf_embed_batch max_active parameter works", {
  skip_on_cran()

  # Just verify the parameter is accepted
  result <- hf_embed_batch(character(), max_active = 5L)
  expect_s3_class(result, "tbl_df")
})

test_that("hf_embed_batch assembles batches in input order and isolates failures", {
  sent <- list()
  testthat::local_mocked_bindings(
    hf_perform_batch = mock_batch_responses(function(inputs, i) {
      sent[[i]] <<- inputs
      if (i == 2L) {
        return(NULL)
      }
      jsonlite::toJSON(lapply(inputs, function(x) c(nchar(x) / 2, 0.25, -0.5)))
    })
  )

  texts <- c("a", "bb", "ccc", "dddd", "eeeee")
  result <- hf_embed_batch(
    texts, model = "org/model:hf-inference", batch_size = 2L, progress = FALSE
  )

  expect_equal(sent, list(c("a", "bb"), c("ccc", "dddd"), "eeeee"))
  expect_equal(result$text, texts)
  expect_equal(result$.input_idx, 1:5)
  expect_equal(result$.error, c(FALSE, FALSE, TRUE, TRUE, FALSE))
  expect_equal(
    result$.error_msg,
    c(NA, NA, "HTTP 503 Service Unavailable.", "HTTP 503 Service Unavailable.", NA)
  )
  expect_equal(result$embedding[[1]], c(0.5, 0.25, -0.5))
  expect_equal(result$embedding[[5]], c(2.5, 0.25, -0.5))
  expect_null(result$embedding[[3]])
  expect_equal(result$n_dims, c(3L, 3L, NA, NA, 3L))
})

test_that("hf_embed_batch pads or truncates responses with the wrong count", {
  counts <- c(1L, 3L)
  testthat::local_mocked_bindings(
    hf_perform_batch = mock_batch_responses(function(inputs, i) {
      jsonlite::toJSON(rep(list(c(0.5, 0.25)), counts[[i]]))
    })
  )

  warnings <- character()
  result <- withCallingHandlers(
    hf_embed_batch(
      c("a", "b", "c", "d"), model = "org/model:hf-inference",
      batch_size = 2L, progress = FALSE
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  expect_length(warnings, 2)
  expect_match(warnings, "count mismatch")
  expect_equal(result$n_dims, c(2L, NA, 2L, 2L))
  expect_null(result$embedding[[2]])
})

# Runs hf_embed_batch() over mocked responses with the fast parser on and off.
embed_with_each_parser <- function(texts, respond) {
  testthat::local_mocked_bindings(hf_perform_batch = mock_batch_responses(respond))
  run <- function(simdjson) {
    withr::with_options(list(huggingfaceR.simdjson = simdjson), {
      hf_embed_batch(
        texts, model = "org/model:hf-inference", batch_size = 2L, progress = FALSE
      )
    })
  }
  list(fast = run(TRUE), general = run(FALSE))
}

test_that("both embedding parsers return identical results", {
  skip_if_not_installed("RcppSimdJson")
  skip_if_not_installed("withr")

  results <- embed_with_each_parser(
    c("a", "bb", "ccc", "dddd", "eeeee"),
    function(inputs, i) {
      jsonlite::toJSON(
        lapply(inputs, function(x) c(nchar(x) / 3, -0.25, 1e-3)),
        digits = NA
      )
    }
  )

  expect_identical(results$fast, results$general)
  expect_equal(results$fast$embedding[[3]], c(1, -0.25, 1e-3))
})

test_that("embedding payloads with nulls use the general parser", {
  skip_if_not_installed("RcppSimdJson")
  skip_if_not_installed("withr")

  results <- embed_with_each_parser(
    c("a", "b"),
    function(inputs, i) "[[0.5,null,0.25],[0.1,0.2,0.3]]"
  )

  expect_identical(results$fast, results$general)
  expect_equal(results$fast$embedding[[1]], c(0.5, 0.25))
  expect_equal(results$fast$n_dims, c(2L, 3L))
})

test_that("batch requests share one provider resolution", {
  lookups <- 0L
  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) {
      lookups <<- lookups + 1L
      list(list(
        provider = "hf-inference", status = "live",
        task = NA_character_, provider_id = NA_character_
      ))
    }
  )

  reqs <- hf_build_batch_requests(
    "org/model", list("a", c("b", "c"), "d"),
    parameters = list(top_k = 2), token = "hf_test"
  )

  expect_equal(lookups, 1L)
  expect_length(reqs, 3)
  reference <- hf_build_request(
    "org/model", inputs = c("b", "c"), parameters = list(top_k = 2), token = "hf_test"
  )
  expect_equal(reqs[[2]]$url, reference$url)
  expect_equal(reqs[[2]]$body, reference$body)
  expect_equal(names(reqs[[2]]$policies), names(reference$policies))
  expect_equal(reqs[[3]]$body$data$inputs, "d")
})

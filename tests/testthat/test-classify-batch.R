test_that("hf_classify_batch returns correct structure for empty input", {
  skip_on_cran()

  result <- hf_classify_batch(character())

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
  expect_true(all(c("text", "label", "score", ".input_idx", ".error", ".error_msg") %in% names(result)))
})

test_that("hf_classify_batch returns tibble with correct columns", {
  skip_on_cran()

  result <- hf_classify_batch(character())

  expected_cols <- c("text", "label", "score", ".input_idx", ".error", ".error_msg")
  expect_true(all(expected_cols %in% names(result)))
})

test_that("hf_classify_chunks requires arrow package", {
  skip_on_cran()
  skip_if(requireNamespace("arrow", quietly = TRUE))

  expect_error(
    hf_classify_chunks("test", output_dir = tempfile()),
    "arrow"
  )
})

test_that("hf_classify_chunks handles empty input", {
  skip_on_cran()
  skip_if_not_installed("arrow")

  temp_dir <- tempfile()

  expect_message(
    result <- hf_classify_chunks(character(), output_dir = temp_dir),
    "No texts to process"
  )

  expect_equal(result, temp_dir)
})

test_that("hf_classify_zero_shot_batch requires labels", {
  skip_on_cran()

  expect_error(
    hf_classify_zero_shot_batch("test text", labels = character()),
    "At least one label"
  )
})

test_that("hf_classify_zero_shot_batch returns correct structure for empty input", {
  skip_on_cran()

  result <- hf_classify_zero_shot_batch(
    character(),
    labels = c("positive", "negative")
  )

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
  expect_true(all(c("text", "label", "score", ".input_idx", ".error", ".error_msg") %in% names(result)))
})

test_that("hf_classify_batch batch_size parameter works", {
  skip_on_cran()

  result <- hf_classify_batch(character(), batch_size = 50L)
  expect_s3_class(result, "tbl_df")
})

test_that("hf_classify_batch max_active parameter works", {
  skip_on_cran()

  result <- hf_classify_batch(character(), max_active = 5L)
  expect_s3_class(result, "tbl_df")
})

test_that("hf_classify_zero_shot_batch batch_size parameter works", {
  skip_on_cran()

  result <- hf_classify_zero_shot_batch(
    character(),
    labels = c("a", "b"),
    batch_size = 25L
  )
  expect_s3_class(result, "tbl_df")
})

test_that("hf_classify_batch assembles labels in input order", {
  testthat::local_mocked_bindings(
    hf_perform_batch = mock_batch_responses(function(inputs, i) {
      items <- lapply(inputs, function(x) {
        list(label = toupper(x), score = nchar(x) / 10)
      })
      if (length(inputs) == 1L) {
        # A single text returns every label; the first is the top label.
        items <- c(items, list(list(label = "OTHER", score = 0.01)))
      }
      if (i == 3L) {
        return(NULL)
      }
      jsonlite::toJSON(list(items), auto_unbox = TRUE)
    })
  )

  result <- hf_classify_batch(
    c("a", "bb", "ccc", "dddd", "eeeee"),
    model = "org/model:hf-inference", batch_size = 2L, progress = FALSE
  )

  expect_equal(result$text, c("a", "bb", "ccc", "dddd", "eeeee"))
  expect_equal(result$label, c("A", "BB", "CCC", "DDDD", NA))
  expect_equal(result$score, c(0.1, 0.2, 0.3, 0.4, NA))
  expect_equal(result$.input_idx, 1:5)
  expect_equal(result$.error, c(FALSE, FALSE, FALSE, FALSE, TRUE))
  expect_equal(result$.error_msg[5], "HTTP 503 Service Unavailable.")
})

test_that("hf_classify_zero_shot_batch parses each response shape", {
  testthat::local_mocked_bindings(
    hf_perform_batch = mock_batch_responses(function(inputs, i) {
      switch(inputs,
        current = '[{"label":"tech","score":0.7},{"label":"food","score":0.3}]',
        legacy = '{"labels":["food","tech"],"scores":[0.6,0.4]}',
        integer = '[{"label":"food","score":1},{"label":"tech","score":0}]',
        empty = "[]",
        fails = NULL
      )
    })
  )

  texts <- c("current", "legacy", "integer", "empty", "fails")
  result <- hf_classify_zero_shot_batch(
    texts, labels = c("tech", "food"),
    model = "org/model:hf-inference", progress = FALSE
  )

  expect_equal(result$text, rep(texts, each = 2))
  expect_equal(
    result$label,
    c("tech", "food", "food", "tech", "food", "tech", NA, NA, NA, NA)
  )
  expect_type(result$score, "double")
  expect_equal(result$score, c(0.7, 0.3, 0.6, 0.4, 1, 0, NA, NA, NA, NA))
  expect_equal(result$.input_idx, rep(1:5, each = 2))
  expect_equal(result$.error, rep(c(FALSE, TRUE), c(8, 2)))
  expect_equal(
    result$.error_msg,
    rep(c(NA, "HTTP 503 Service Unavailable."), c(8, 2))
  )
})

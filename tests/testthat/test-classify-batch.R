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

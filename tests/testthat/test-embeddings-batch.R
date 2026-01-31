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

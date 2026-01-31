test_that("batch_vector splits correctly", {
  # Empty input
  result <- batch_vector(character(), 10)
  expect_equal(length(result), 0)

  # Single batch
  result <- batch_vector(letters[1:5], 10)
  expect_equal(length(result), 1)
  expect_equal(result[[1]]$value, letters[1:5])
  expect_equal(result[[1]]$indices, 1:5)

  # Multiple batches
  result <- batch_vector(letters[1:10], 3)
  expect_equal(length(result), 4)  # 3 + 3 + 3 + 1
  expect_equal(result[[1]]$value, letters[1:3])
  expect_equal(result[[1]]$indices, 1:3)
  expect_equal(result[[4]]$value, letters[10])
  expect_equal(result[[4]]$indices, 10)
})

test_that("batch_vector preserves order", {
  x <- c("z", "y", "x", "w", "v")
  result <- batch_vector(x, 2)

  # Reconstruct original
  reconstructed <- unlist(purrr::map(result, ~ .x$value))
  expect_equal(reconstructed, x)

  # Indices are sequential
  all_indices <- unlist(purrr::map(result, ~ .x$indices))
  expect_equal(all_indices, 1:5)
})

test_that("hf_build_request creates valid request", {
  skip_on_cran()

  req <- hf_build_request(
    model_id = "test/model",
    inputs = "test input",
    token = "test_token"
  )

  expect_s3_class(req, "httr2_request")
  expect_true(grepl("test/model", req$url))
})

test_that("hf_build_request handles parameters", {
  skip_on_cran()

  req <- hf_build_request(
    model_id = "test/model",
    inputs = "test",
    parameters = list(candidate_labels = c("a", "b"))
  )

  expect_s3_class(req, "httr2_request")
})

test_that("hf_perform_batch returns empty tibble for empty input", {
  skip_on_cran()

  result <- hf_perform_batch(list(), integer(), max_active = 5)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
  expect_true(all(c(".input_idx", "response", ".error", ".error_msg") %in% names(result)))
})

test_that("hf_read_chunks errors on missing directory", {
  skip_on_cran()
  skip_if_not_installed("arrow")

  expect_error(
    hf_read_chunks("/nonexistent/directory"),
    "Directory not found"
  )
})

test_that("hf_read_chunks handles empty directory", {
  skip_on_cran()
  skip_if_not_installed("arrow")

  temp_dir <- tempfile()
  dir.create(temp_dir)
  on.exit(unlink(temp_dir, recursive = TRUE))

  expect_warning(
    result <- hf_read_chunks(temp_dir),
    "No parquet files found"
  )
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
})

test_that("hf_write_chunk and hf_read_chunks roundtrip", {
  skip_on_cran()
  skip_if_not_installed("arrow")

  temp_dir <- tempfile()
  on.exit(unlink(temp_dir, recursive = TRUE))

  # Write a chunk
  test_data <- tibble::tibble(
    text = c("hello", "world"),
    .input_idx = 1:2,
    .error = c(FALSE, FALSE)
  )

  hf_write_chunk(test_data, temp_dir, chunk_id = 1, prefix = "test_chunk")

  # Read it back
  result <- hf_read_chunks(temp_dir)

  expect_equal(nrow(result), 2)
  expect_equal(result$text, test_data$text)
  expect_equal(result$.input_idx, test_data$.input_idx)
})

test_that("hf_get_existing_chunks finds chunk files", {
  skip_on_cran()
  skip_if_not_installed("arrow")

  temp_dir <- tempfile()
  dir.create(temp_dir)
  on.exit(unlink(temp_dir, recursive = TRUE))

  # Create some chunk files
  test_data <- tibble::tibble(x = 1)
  hf_write_chunk(test_data, temp_dir, 1, prefix = "embed_chunk")
  hf_write_chunk(test_data, temp_dir, 3, prefix = "embed_chunk")
  hf_write_chunk(test_data, temp_dir, 5, prefix = "embed_chunk")

  existing <- hf_get_existing_chunks(temp_dir, prefix = "embed_chunk")

  expect_equal(sort(existing), c(1, 3, 5))
})

test_that("hf_get_existing_chunks returns empty for missing dir", {
  result <- hf_get_existing_chunks("/nonexistent/dir", prefix = "chunk")
  expect_equal(length(result), 0)
})

test_that("hf_embed returns tibble with embeddings", {
  skip_on_cran()
  
  # Test with empty input
  result <- hf_embed(character())
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
  expect_true("embedding" %in% names(result))
})

test_that("hf_embed handles NA values", {
  skip_on_cran()
  
  result <- hf_embed(NA_character_)
  expect_s3_class(result, "tbl_df")
  expect_true(is.na(result$text[1]))
  expect_true(is.null(result$embedding[[1]]))
})

test_that("hf_embed batches non-missing inputs", {
  skip_on_cran()

  calls <- 0L
  testthat::local_mocked_bindings(
    hf_embed_batch = function(text, model, token, batch_size, max_active, progress, endpoint_url) {
      calls <<- calls + 1L
      expect_equal(text, c("a", "b", "c"))
      expect_equal(batch_size, 3L)
      expect_equal(max_active, 1L)
      expect_false(progress)

      tibble::tibble(
        text = text,
        embedding = list(c(1, 0), c(0, 1), c(1, 1)),
        n_dims = c(2L, 2L, 2L),
        .input_idx = seq_along(text),
        .error = FALSE,
        .error_msg = NA_character_
      )
    }
  )

  result <- hf_embed(c("a", NA, "b", "c"))

  expect_equal(calls, 1L)
  expect_equal(result$text, c("a", NA, "b", "c"))
  expect_equal(result$n_dims, c(2L, NA_integer_, 2L, 2L))
  expect_true(is.null(result$embedding[[2]]))
})

test_that("hf_similarity requires embedding column", {
  skip_on_cran()
  
  df <- tibble::tibble(text = c("a", "b"))
  
  expect_error(
    hf_similarity(df),
    "Column 'embedding' not found"
  )
})

test_that("hf_similarity returns correct structure", {
  skip_on_cran()
  
  df <- tibble::tibble(
    text = c("a", "b"),
    embedding = list(c(1, 2, 3), c(4, 5, 6))
  )
  
  result <- hf_similarity(df)
  
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 1)  # n*(n-1)/2 = 2*1/2 = 1
  expect_true(all(c("text_1", "text_2", "similarity") %in% names(result)))
})

test_that("hf_similarity preserves pairwise results with NULL embeddings", {
  skip_on_cran()

  df <- tibble::tibble(
    text = c("a", "b", "c", "d"),
    embedding = list(c(1, 0), c(0, 1), NULL, c("x", "y"))
  )

  result <- hf_similarity(df)

  expect_equal(result$text_1, c("a", "a", "a", "b", "b", "c"))
  expect_equal(result$text_2, c("b", "c", "d", "c", "d", "d"))
  expect_equal(result$similarity, c(0, NA_real_, NA_real_, NA_real_, NA_real_, NA_real_))
})

test_that("hf_embed_umap requires uwot package", {
  skip_on_cran()
  
  # This test assumes uwot is not installed
  # If it is installed, this will pass differently
  if (!requireNamespace("uwot", quietly = TRUE)) {
    expect_error(
      hf_embed_umap(c("a", "b")),
      "uwot"
    )
  }
})

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

# Helper function
skip_on_cran <- function() {
  if (identical(Sys.getenv("NOT_CRAN"), "true")) {
    return(invisible(TRUE))
  }
  testthat::skip("Skipping on CRAN")
}

# These tests require Python/reticulate to be configured
skip_if_no_python <- function() {
  python_available <- tryCatch({
    reticulate::py_available(initialize = FALSE) &&
      !is.null(tryCatch(reticulate::import("transformers", delay_load = TRUE), error = function(e) NULL))
  }, error = function(e) FALSE)

  if (!python_available) {
    testthat::skip("Python/transformers not available")
  }
}

test_that("hf_list_datasets is still returning a character vector", {
  skip_on_cran()
  skip_if_no_python()

  imdb_datasets <- hf_list_datasets("imdb")
  expect_equal("character", class(imdb_datasets))
})

test_that("hf_list_datasets returns a vector longer than 1 for emotions pattern", {
  skip_on_cran()
  skip_if_no_python()

  emo_datasets <- hf_list_datasets('emo')
  expect_gt(length(emo_datasets), 1)
})

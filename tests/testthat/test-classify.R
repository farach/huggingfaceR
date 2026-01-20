test_that("hf_classify returns tibble", {
  skip_on_cran()
  
  # Test with empty input
  result <- hf_classify(character())
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
})

test_that("hf_classify handles NA values", {
  skip_on_cran()
  
  result <- hf_classify(NA_character_)
  expect_s3_class(result, "tbl_df")
  expect_true(is.na(result$label[1]))
})

test_that("hf_classify_zero_shot requires labels", {
  skip_on_cran()
  
  expect_error(
    hf_classify_zero_shot("test text", labels = character()),
    "At least one label"
  )
})

test_that("hf_classify_zero_shot returns correct structure", {
  skip_on_cran()
  
  # Test with empty input
  result <- hf_classify_zero_shot(
    character(),
    labels = c("positive", "negative")
  )
  
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
})

# Helper function
skip_on_cran <- function() {
  if (identical(Sys.getenv("NOT_CRAN"), "true")) {
    return(invisible(TRUE))
  }
  testthat::skip("Skipping on CRAN")
}

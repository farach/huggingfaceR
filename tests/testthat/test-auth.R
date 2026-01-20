test_that("hf_set_token sets token in environment", {
  # Set token
  hf_set_token("hf_test_token_123456789012345678901234567890")
  
  # Check it's in environment
  expect_equal(Sys.getenv("HUGGING_FACE_HUB_TOKEN"), 
               "hf_test_token_123456789012345678901234567890")
  
  # Clean up
  Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
})

test_that("hf_set_token validates token format", {
  # Should warn for unusual format
  expect_warning(hf_set_token("invalid_token"))
  
  # Clean up
  Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
})

test_that("hf_get_token retrieves token", {
  # Set a token
  Sys.setenv(HUGGING_FACE_HUB_TOKEN = "hf_test123")
  
  # Get it
  token <- hf_get_token()
  expect_equal(token, "hf_test123")
  
  # Clean up
  Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
})

test_that("hf_get_token requires token when needed", {
  # Unset token
  Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
  
  # Should error when required
  expect_error(hf_get_token(required = TRUE), "API token required")
})

# Skip API tests on CRAN
skip_on_cran <- function() {
  if (identical(Sys.getenv("NOT_CRAN"), "true")) {
    return(invisible(TRUE))
  }
  testthat::skip("Skipping on CRAN")
}

test_that("hf_whoami returns user info with valid token", {
  skip_on_cran()
  skip_if(Sys.getenv("HUGGING_FACE_HUB_TOKEN") == "", 
          "No HF token available")
  
  # This will only pass with a valid token
  result <- hf_whoami()
  
  expect_s3_class(result, "tbl_df")
  expect_true("name" %in% names(result))
})

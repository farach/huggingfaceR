test_that("hf_set_token sets token in environment", {
  old_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN", unset = NA)
  on.exit({
    if (is.na(old_token)) Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
    else Sys.setenv(HUGGING_FACE_HUB_TOKEN = old_token)
  }, add = TRUE)

  # Set token
  hf_set_token("hf_test_token_123456789012345678901234567890")
  
  # Check it's in environment
  expect_equal(Sys.getenv("HUGGING_FACE_HUB_TOKEN"), 
               "hf_test_token_123456789012345678901234567890")
  
  # Clean up
  Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
})

test_that("hf_set_token validates token format", {
  old_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN", unset = NA)
  on.exit({
    if (is.na(old_token)) Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
    else Sys.setenv(HUGGING_FACE_HUB_TOKEN = old_token)
  }, add = TRUE)

  # Should warn for unusual format
  expect_warning(hf_set_token("invalid_token"))
  
  # Clean up
  Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
})

test_that("hf_get_token retrieves token", {
  old_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN", unset = NA)
  on.exit({
    if (is.na(old_token)) Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
    else Sys.setenv(HUGGING_FACE_HUB_TOKEN = old_token)
  }, add = TRUE)

  # Set a token
  Sys.setenv(HUGGING_FACE_HUB_TOKEN = "hf_test123")
  
  # Get it
  token <- hf_get_token()
  expect_equal(token, "hf_test123")
  
  # Clean up
  Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
})

test_that("hf_get_token requires token when needed", {
  old_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN", unset = NA)
  on.exit({
    if (is.na(old_token)) Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
    else Sys.setenv(HUGGING_FACE_HUB_TOKEN = old_token)
  }, add = TRUE)

  # Unset token
  Sys.unsetenv("HUGGING_FACE_HUB_TOKEN")
  
  # Should error when required
  expect_error(hf_get_token(required = TRUE), "API token required")
})

test_that("hf_whoami returns user info with valid token", {
  skip_on_cran()
  skip_if(Sys.getenv("HUGGING_FACE_HUB_TOKEN") == "", 
          "No HF token available")
  
  # This will only pass with a valid token
  result <- hf_whoami()
  
  expect_s3_class(result, "tbl_df")
  expect_true("name" %in% names(result))
  expect_true("email" %in% names(result))
  expect_true("token_role" %in% names(result))
})

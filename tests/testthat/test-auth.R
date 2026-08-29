# Tests for token handling (R/auth.R).
#
# huggingfaceR resolves tokens from `HF_TOKEN` first and the legacy
# `HUGGING_FACE_HUB_TOKEN` second. Every test clears *both* variables so that a
# token set by one test cannot leak into another.

# Clears both token variables for the duration of a test and restores them
# afterwards, whatever the outcome.
local_clean_tokens <- function(env = parent.frame()) {
  vars <- hf_token_env_vars()
  old <- vapply(vars, function(n) Sys.getenv(n, unset = NA_character_), character(1))

  withr::defer(
    {
      for (n in vars) {
        if (is.na(old[[n]])) {
          Sys.unsetenv(n)
        } else {
          args <- list(old[[n]])
          names(args) <- n
          do.call(Sys.setenv, args)
        }
      }
    },
    envir = env
  )

  for (n in vars) Sys.unsetenv(n)
  invisible(NULL)
}

test_that("hf_set_token sets token in environment", {
  local_clean_tokens()

  hf_set_token("hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345")

  expect_equal(
    Sys.getenv("HUGGING_FACE_HUB_TOKEN"),
    "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345"
  )
})

test_that("hf_set_token populates both token variables", {
  local_clean_tokens()

  hf_set_token("hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345")

  expect_equal(
    Sys.getenv("HF_TOKEN"),
    "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345"
  )
  expect_equal(
    Sys.getenv("HUGGING_FACE_HUB_TOKEN"),
    "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345"
  )
})

test_that("hf_set_token validates token format", {
  local_clean_tokens()

  expect_warning(hf_set_token("invalid_token"))
})

test_that("hf_get_token retrieves token", {
  local_clean_tokens()

  Sys.setenv(HUGGING_FACE_HUB_TOKEN = "hf_test123")

  expect_equal(hf_get_token(), "hf_test123")
})

test_that("hf_get_token requires token when needed", {
  local_clean_tokens()

  expect_error(hf_get_token(required = TRUE), "API token required")
})

test_that("hf_get_token reads HF_TOKEN", {
  # HF_TOKEN is the variable used by current Hugging Face tooling, so a user who
  # followed the official docs must not be told they have no token.
  local_clean_tokens()

  Sys.setenv(HF_TOKEN = "hf_from_new_var")

  expect_equal(hf_get_token(), "hf_from_new_var")
})

test_that("HF_TOKEN takes priority over the legacy variable", {
  local_clean_tokens()

  Sys.setenv(HF_TOKEN = "hf_new")
  Sys.setenv(HUGGING_FACE_HUB_TOKEN = "hf_legacy")

  expect_equal(hf_get_token(), "hf_new")
})

test_that("an explicit token argument beats the environment", {
  local_clean_tokens()

  Sys.setenv(HF_TOKEN = "hf_from_env")

  expect_equal(hf_get_token("hf_explicit"), "hf_explicit")
})

test_that("hf_token_from_env returns NULL when nothing is set", {
  local_clean_tokens()

  expect_null(hf_token_from_env())
})

test_that("hf_whoami returns user info with valid token", {
  skip_on_cran()
  skip_if_offline()
  skip_if(is.null(hf_token_from_env()), "No HF token available")

  result <- hf_whoami()

  expect_s3_class(result, "tbl_df")
  expect_true("name" %in% names(result))
  expect_true("email" %in% names(result))
  expect_true("token_role" %in% names(result))
})

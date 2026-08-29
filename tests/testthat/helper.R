# Helper function used across test files
skip_on_cran <- function() {
  if (identical(Sys.getenv("NOT_CRAN"), "true")) {
    return(invisible(TRUE))
  }
  testthat::skip("Skipping on CRAN")
}

# Skips a live test when the Hub cannot be reached. Without this, a transient
# connection reset presents as a test failure rather than an unavailable
# dependency, which makes local runs unreliable.
skip_if_hub_unreachable <- function() {
  reachable <- tryCatch(
    {
      httr2::request("https://huggingface.co/api/datasets") |>
        httr2::req_url_query(limit = 1) |>
        httr2::req_timeout(15) |>
        httr2::req_retry(max_tries = 2) |>
        httr2::req_perform()
      TRUE
    },
    error = function(e) FALSE
  )

  if (!isTRUE(reachable)) {
    testthat::skip("Hugging Face Hub is unreachable")
  }
  invisible(TRUE)
}

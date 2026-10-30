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

# Stands in for hf_perform_batch() in offline tests. `respond(inputs, i)` returns
# the JSON body for the i-th request, or NULL to make that request fail.
mock_batch_responses <- function(respond) {
  function(reqs, input_indices, max_active = 10L, progress = TRUE) {
    responses <- lapply(seq_along(reqs), function(i) {
      body <- respond(reqs[[i]]$body$data$inputs, i)
      if (is.null(body)) {
        return(errorCondition("HTTP 503 Service Unavailable.", class = "httr2_http_503"))
      }
      httr2::response(
        status_code = 200,
        headers = list("Content-Type" = "application/json"),
        body = charToRaw(as.character(body))
      )
    })
    failed <- vapply(responses, inherits, logical(1), "error")
    tibble::tibble(
      .input_idx = input_indices,
      response = responses,
      .error = failed,
      .error_msg = ifelse(failed, "HTTP 503 Service Unavailable.", NA_character_)
    )
  }
}

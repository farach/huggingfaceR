#' @importFrom rlang .data
NULL

#' Split Vector into Batches
#'
#' Internal helper to split a vector into batches with index tracking.
#'
#' @param x Vector to split into batches.
#' @param batch_size Integer. Maximum size of each batch.
#'
#' @returns A list of lists, each containing:
#'   - `value`: The batch values
#'   - `indices`: The original indices of the values
#' @keywords internal
batch_vector <- function(x, batch_size) {
  if (length(x) == 0) {
    return(list())
  }

  n <- length(x)
  n_batches <- ceiling(n / batch_size)

  lapply(seq_len(n_batches), function(i) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, n)
    list(
      value = x[start_idx:end_idx],
      indices = start_idx:end_idx
    )
  })
}


#' Build a Hugging Face API Request
#'
#' Internal factory function to construct httr2 request objects for batch processing.
#'
#' @param model_id Character string. The model ID on Hugging Face Hub.
#' @param inputs The input data (usually character vector or list).
#' @param parameters Optional list of parameters for the inference.
#' @param token Character string or NULL. API token for authentication.
#' @param wait_for_model Deprecated no-op retained for internal compatibility.
#' @param use_cache Deprecated no-op retained for internal compatibility.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#'
#' @returns An httr2 request object ready for execution.
#' @keywords internal
hf_build_request <- function(model_id,
                              inputs,
                              parameters = NULL,
                              token = NULL,
                              wait_for_model = TRUE,
                              use_cache = TRUE,
                              endpoint_url = NULL) {

  parsed <- hf_parse_model(model_id)
  token <- hf_get_token(token, required = FALSE)

  body <- hf_inference_body(inputs, parameters)

  # Build request — provider routing, dedicated endpoint, or resolved serverless
  provider <- if (is.null(endpoint_url)) {
    hf_task_provider(parsed, token = token)
  } else {
    NULL
  }
  base_url <- hf_inference_url(parsed$model, provider, endpoint_url)
  req <- httr2::request(base_url)

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  req |>
    hf_req_bill_to() |>
    httr2::req_body_json(body) |>
    httr2::req_retry(max_tries = 3, is_transient = hf_is_transient) |>
    httr2::req_throttle(rate = 10 / 1)
}


# Builds one request per element of `inputs`. The token, provider route, and
# request policies are resolved once into a shared template, so only the JSON
# body differs between requests and no per-request Hub lookups or request
# construction are repeated.
hf_build_batch_requests <- function(model_id, inputs, parameters = NULL,
                                    token = NULL, endpoint_url = NULL) {
  template <- hf_build_request(
    model_id = model_id,
    inputs = NULL,
    parameters = parameters,
    token = token,
    endpoint_url = endpoint_url
  )

  lapply(inputs, function(value) {
    httr2::req_body_json(template, hf_inference_body(value, parameters))
  })
}


#' Perform Parallel Batch Requests
#'
#' Internal function to execute multiple API requests in parallel with error handling.
#'
#' `httr2::req_perform_parallel()` retries transient failures without honouring
#' `req_retry(max_tries)`, so a model that keeps returning 429 or 5xx responses
#' would never return. Each request's own retry policy is therefore disabled
#' and transient failures are retried here in bounded rounds instead, waiting
#' for the longest `Retry-After` header (or an exponential backoff) in between.
#'
#' @param reqs List of httr2 request objects.
#' @param input_indices Integer vector. Original indices corresponding to each request.
#' @param max_active Integer. Maximum concurrent requests. Default: 10.
#' @param progress Logical. Show progress bar. Default: TRUE.
#' @param max_tries Integer. Maximum attempts per request, including the first.
#'   Default: 3.
#'
#' @returns A tibble with columns:
#'   - `.input_idx`: Original input index
#'   - `response`: The response object (or error)
#'   - `.error`: Logical, TRUE if request failed
#'   - `.error_msg`: Error message or NA
#' @keywords internal
hf_perform_batch <- function(reqs, input_indices, max_active = 10L, progress = TRUE,
                             max_tries = 3L) {

  if (length(reqs) == 0) {
    return(tibble::tibble(
      .input_idx = integer(),
      response = list(),
      .error = logical(),
      .error_msg = character()
    ))
  }

  reqs <- lapply(reqs, hf_req_single_attempt)
  results <- vector("list", length(reqs))
  pending <- seq_along(reqs)

  for (attempt in seq_len(max_tries)) {
    round <- httr2::req_perform_parallel(
      reqs[pending],
      on_error = "continue",
      progress = progress,
      max_active = max_active
    )
    results[pending] <- round

    transient <- vapply(round, hf_is_transient_error, logical(1))
    if (attempt == max_tries || !any(transient)) {
      break
    }
    hf_sleep(hf_retry_delay(round[transient], attempt))
    pending <- pending[transient]
  }

  failed <- vapply(results, inherits, logical(1), "error")
  error_msg <- rep(NA_character_, length(results))
  error_msg[failed] <- vapply(results[failed], conditionMessage, character(1))

  tibble::tibble(
    .input_idx = input_indices,
    response = results,
    .error = failed,
    .error_msg = error_msg
  )
}


# Disables httr2's own retries so that hf_perform_batch() controls them.
hf_req_single_attempt <- function(req) {
  httr2::req_retry(req, max_tries = 1L, is_transient = function(resp) FALSE)
}


# TRUE for an HTTP error whose status is worth retrying.
hf_is_transient_error <- function(x) {
  inherits(x, "httr2_http") &&
    inherits(x$resp, "httr2_response") &&
    hf_is_transient(x$resp)
}


# Seconds to wait before the next retry round. The longest server-supplied
# `Retry-After` wins; otherwise the wait doubles each round. Both are capped.
hf_retry_delay <- function(errors, attempt, cap = 60) {
  after <- vapply(errors, function(e) {
    value <- suppressWarnings(httr2::resp_retry_after(e$resp))
    if (length(value) == 1 && is.finite(value)) value else NA_real_
  }, numeric(1))

  delay <- if (all(is.na(after))) 2^attempt else max(after, na.rm = TRUE)
  min(max(delay, 0), cap)
}


# Indirection over Sys.sleep() so tests can skip real waits.
hf_sleep <- function(seconds) {
  Sys.sleep(seconds)
}


#' Write Chunk to Parquet
#'
#' Internal function to write a data chunk to a parquet file.
#'
#' @param data A tibble to write.
#' @param output_dir Character string. Directory to write to.
#' @param chunk_id Integer. Chunk identifier.
#' @param prefix Character string. File name prefix. Default: "chunk".
#'
#' @returns Invisibly returns the file path written.
#' @keywords internal
hf_write_chunk <- function(data, output_dir, chunk_id, prefix = "chunk") {

  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' is required for chunk operations. Install with: install.packages('arrow')",
         call. = FALSE)
  }

  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  file_path <- file.path(output_dir, sprintf("%s_%04d.parquet", prefix, chunk_id))
  arrow::write_parquet(data, file_path)

  invisible(file_path)
}


#' Read All Chunks from Directory
#'
#' Read and combine all parquet chunk files from a directory.
#'
#' @param output_dir Character string. Directory containing chunk files.
#' @param pattern Character string. Glob pattern to match files. Default: "*.parquet".
#'
#' @returns A tibble combining all chunks, sorted by `.input_idx` if present.
#' @export
#'
#' @examples
#' \dontrun{
#' # After running hf_embed_chunks()
#' results <- hf_read_chunks("my_output_dir")
#' }
hf_read_chunks <- function(output_dir, pattern = "*.parquet") {

  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' is required for chunk operations. Install with: install.packages('arrow')",
         call. = FALSE)
  }

  if (!dir.exists(output_dir)) {
    stop(paste0("Directory not found: ", output_dir), call. = FALSE)
  }

  files <- list.files(output_dir, pattern = utils::glob2rx(pattern), full.names = TRUE)

  if (length(files) == 0) {
    cli::cli_warn("No parquet files found in {.path {output_dir}}")
    return(tibble::tibble())
  }

  # Read all files
  chunks <- purrr::map(files, arrow::read_parquet)
  combined <- dplyr::bind_rows(chunks)

  # Sort by input index if present
 if (".input_idx" %in% names(combined)) {
    combined <- dplyr::arrange(combined, .data$.input_idx)
  }

  combined
}


#' Get Existing Chunk IDs
#'
#' Internal function to find which chunks have already been written.
#'
#' @param output_dir Character string. Directory to check.
#' @param prefix Character string. File name prefix to match.
#'
#' @returns Integer vector of chunk IDs that already exist.
#' @keywords internal
hf_get_existing_chunks <- function(output_dir, prefix = "chunk") {

  if (!dir.exists(output_dir)) {
    return(integer())
  }

  files <- list.files(output_dir, pattern = paste0("^", prefix, "_\\d{4}\\.parquet$"))

  if (length(files) == 0) {
    return(integer())
  }

  # Extract chunk IDs from filenames
  matches <- regmatches(files, regexpr("\\d{4}", files))
  as.integer(matches)
}

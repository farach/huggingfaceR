#' Make a Request to Hugging Face Inference API
#'
#' Internal helper function to construct and execute API requests.
#'
#' @param model_id Character string. The model ID on Hugging Face Hub.
#' @param inputs The input data (usually character vector or list).
#' @param parameters Optional list of parameters for the inference.
#' @param token Character string or NULL. API token for authentication.
#' @param wait_for_model Deprecated no-op retained for internal compatibility.
#' @param use_cache Deprecated no-op retained for internal compatibility.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#'   When provided, requests are sent to this URL instead of the public
#'   Inference API. Use for dedicated Inference Endpoints.
#'
#' @returns The raw response from the API.
#' @keywords internal
hf_api_request <- function(model_id,
                           inputs,
                           parameters = NULL,
                           token = NULL,
                           wait_for_model = TRUE,
                           use_cache = TRUE,
                           endpoint_url = NULL) {
  hf_task_build(
    model = model_id,
    inputs = inputs,
    parameters = parameters,
    token = token,
    endpoint_url = endpoint_url
  ) |>
    httr2::req_perform()
}


#' Parse API Response to Tibble
#'
#' Internal helper to convert API JSON responses to tibbles.
#'
#' @param resp The response object from httr2.
#' @param input_text Optional. The original input text(s) to include in output.
#'
#' @returns A tibble with the parsed results.
#' @keywords internal
hf_parse_response <- function(resp, input_text = NULL) {
  
  result <- httr2::resp_body_json(resp)
  
  # Handle different response formats
  if (is.list(result) && !is.null(names(result))) {
    # Named list (single result)
    df <- tibble::as_tibble(result)
  } else if (is.list(result) && length(result) > 0) {
    # List of results
    if (is.list(result[[1]]) && !is.null(names(result[[1]]))) {
      # List of named lists
      df <- purrr::map_dfr(result, tibble::as_tibble)
    } else {
      # Simple list or nested structure
      df <- tibble::tibble(result = result)
    }
  } else {
    df <- tibble::tibble(result = list(result))
  }
  
  # Add input text if provided
  if (!is.null(input_text)) {
    df <- tibble::tibble(text = input_text) |>
      dplyr::bind_cols(df)
  }
  
  df
}


#' Vectorize API Calls
#'
#' Internal helper to apply an API function to multiple inputs efficiently.
#'
#' @param inputs Character vector of inputs.
#' @param fn Function to apply to each input.
#' @param ... Additional arguments passed to fn.
#' @param .progress Logical. Show progress bar?
#'
#' @returns Combined results from all API calls.
#' @keywords internal
hf_vectorize <- function(inputs, fn, ..., .progress = FALSE) {
  
  if (length(inputs) == 1) {
    return(fn(inputs, ...))
  }
  
  if (.progress) {
    cli::cli_progress_bar("Processing", total = length(inputs))
  }
  
  results <- purrr::map(inputs, function(x) {
    if (.progress) {
      cli::cli_progress_update()
    }
    fn(x, ...)
  })
  
  if (.progress) {
    cli::cli_progress_done()
  }
  
  dplyr::bind_rows(results)
}


#' Null-coalescing operator
#'
#' @name null_coalesce
#' @aliases %||%
#' @param x First value
#' @param y Second value (default)
#' @keywords internal
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}


# Returns one typed scalar from an API field, or `na` when it is absent.
hf_scalar <- function(x, na) {
  if (length(x) == 0L) {
    return(na)
  }
  x <- x[[1L]]
  if (!is.atomic(x) || length(x) != 1L || is.na(x)) {
    return(na)
  }
  as.vector(x, typeof(na))
}

hf_chr <- function(x) hf_scalar(x, NA_character_)
hf_dbl <- function(x) hf_scalar(x, NA_real_)

# Coerce scalar offsets from API responses into integer-or-NA values.
as_int_or_na <- function(x) hf_scalar(x, NA_integer_)


# Places per-item results back at the positions marked by `keep`. Positions
# that were skipped (for example `NA` inputs) stay NULL.
hf_scatter <- function(keep, values) {
  out <- vector("list", length(keep))
  out[keep] <- values
  out
}


# Builds a tibble with one row per record from per-input record lists.
#
# `records` holds one list of records (named lists) per input. With
# `keep_empty = TRUE`, an input with no records keeps one row whose fields are
# NA, so every input stays represented. With `keep_empty = FALSE` it adds no
# rows. `inputs` is a named list of input columns that are repeated to match.
# Each function in `fields` reads one record, receives NULL for filler rows, and
# must return a typed scalar. Columns are filled with one vapply() each instead
# of binding per-input tibbles.
hf_record_rows <- function(inputs, records, fields, keep_empty = TRUE) {
  records <- lapply(records, function(r) if (is.list(r)) r else list())
  counts <- lengths(records)
  rows <- if (keep_empty) pmax(counts, 1L) else counts
  idx <- rep.int(seq_along(records), rows)

  flat <- vector("list", length(idx))
  has_record <- rep.int(counts > 0L, rows)
  if (any(has_record)) {
    flat[has_record] <- do.call(c, unname(records))
  }
  flat <- lapply(flat, function(r) if (is.list(r)) r else NULL)

  cols <- c(
    lapply(inputs, function(x) x[idx]),
    lapply(fields, function(fn) vapply(flat, fn, fn(NULL), USE.NAMES = FALSE))
  )
  tibble::new_tibble(cols, nrow = length(idx))
}

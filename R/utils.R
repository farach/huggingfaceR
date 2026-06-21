#' Make a Request to Hugging Face Inference API
#'
#' Internal helper function to construct and execute API requests.
#'
#' @param model_id Character string. The model ID on Hugging Face Hub.
#' @param inputs The input data (usually character vector or list).
#' @param parameters Optional list of parameters for the inference.
#' @param token Character string or NULL. API token for authentication.
#' @param wait_for_model Logical. Wait for model to load if not ready.
#' @param use_cache Logical. Use cached results for identical inputs.
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

  parsed <- hf_parse_model(model_id)
  token <- hf_get_token(token, required = FALSE)

  # Build request body
  body <- list(inputs = inputs)
  if (!is.null(parameters)) {
    body$parameters <- parameters
  }
  if (wait_for_model) {
    body$options <- list(wait_for_model = TRUE)
  }
  if (!is.null(use_cache)) {
    if (is.null(body$options)) body$options <- list()
    body$options$use_cache <- use_cache
  }

  # Build request — provider routing, dedicated endpoint, or default serverless
  base_url <- hf_inference_url(parsed$model, parsed$provider, endpoint_url)
  req <- httr2::request(base_url)

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  req <- req |>
    httr2::req_body_json(body) |>
    httr2::req_retry(max_tries = 3, is_transient = hf_is_transient) |>
    httr2::req_error(body = hf_error_body(parsed$model))

  httr2::req_perform(req)
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

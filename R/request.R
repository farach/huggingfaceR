#' Parse a Model Spec into Model ID and Provider
#'
#' Splits a `"model-id:provider"` spec (for example
#' `"meta-llama/Llama-3.1-8B-Instruct:together"`) into its components. Hugging
#' Face model IDs never contain a colon, so a trailing `":provider"` segment is
#' interpreted as an Inference Provider selector.
#'
#' @param model Character string. A model ID, optionally suffixed with
#'   `":provider"`.
#'
#' @returns A list with elements `model` and `provider`. `provider` is `NULL`
#'   when no suffix is present.
#' @keywords internal
hf_parse_model <- function(model) {
  if (length(model) != 1 || is.na(model) || !grepl(":", model, fixed = TRUE)) {
    return(list(model = model, provider = NULL))
  }

  parts <- strsplit(model, ":", fixed = TRUE)[[1]]
  provider <- parts[length(parts)]
  model_id <- paste(parts[-length(parts)], collapse = ":")
  list(model = model_id, provider = provider)
}


#' Build an Inference Base URL
#'
#' Resolves the base URL for a serverless Inference Providers request, honoring an
#' explicit dedicated endpoint, an Inference Provider, or the default
#' `hf-inference` provider.
#'
#' @param model Character string. The model ID (without a provider suffix).
#' @param provider Character string or NULL. Inference provider (e.g.
#'   `"together"`). When NULL, the default `"hf-inference"` provider is used.
#' @param endpoint_url Character string or NULL. A dedicated Inference Endpoint
#'   URL. When supplied it takes precedence over provider routing.
#'
#' @returns A character scalar URL.
#' @keywords internal
hf_inference_url <- function(model, provider = NULL, endpoint_url = NULL) {
  if (!is.null(endpoint_url)) {
    return(sub("/$", "", endpoint_url))
  }

  provider <- provider %||% "hf-inference"
  if (identical(provider, "hf-inference")) {
    paste0("https://router.huggingface.co/hf-inference/models/", model)
  } else {
    paste0("https://router.huggingface.co/", provider, "/models/", model)
  }
}


#' Detect Transient HTTP Errors Worth Retrying
#'
#' @param resp An httr2 response object.
#'
#' @returns Logical. TRUE for status codes that indicate a transient failure
#'   (rate limiting, gateway errors, model warm-up).
#' @keywords internal
hf_is_transient <- function(resp) {
  httr2::resp_status(resp) %in% c(429L, 500L, 502L, 503L, 504L)
}


#' Shared Error-Body Translator for Inference Responses
#'
#' Returns a function suitable for `httr2::req_error(body = )` that converts
#' Hugging Face API error payloads into actionable messages. Centralizing this
#' keeps error messages consistent across every inference function.
#'
#' @param model_id Character string or NULL. Used to tailor "model not found"
#'   guidance.
#'
#' @returns A function of one argument (an httr2 response) returning a character
#'   message.
#' @keywords internal
hf_error_body <- function(model_id = NULL) {
  function(resp) {
    body <- tryCatch(
      httr2::resp_body_json(resp),
      error = function(e) list(error = httr2::resp_body_string(resp))
    )

    # Task APIs return {error: "string"}; chat completions return
    # {error: {message: "string"}}. Handle both shapes.
    err <- body$error
    error_msg <- if (is.list(err)) {
      err$message %||% "Unknown error"
    } else {
      err %||% "Unknown error"
    }

    if (grepl("not found", error_msg, ignore.case = TRUE) && !is.null(model_id)) {
      paste0(
        "Model '", model_id, "' was not found on the Inference API. ",
        "This usually means the model exists on the Hub but is not available ",
        "for serverless inference. Run hf_check_inference('", model_id, "') to ",
        "verify, or see https://huggingface.co/docs/inference-providers."
      )
    } else if (grepl("token|authoriz|authenticat", error_msg, ignore.case = TRUE)) {
      "Invalid or missing API token. Set one with hf_set_token()."
    } else if (grepl("rate limit", error_msg, ignore.case = TRUE)) {
      "Rate limit exceeded. Please wait or use an API token for higher limits."
    } else {
      paste0("API error: ", error_msg)
    }
  }
}


#' Perform a Task-Style Inference Request
#'
#' Internal engine shared by the task wrappers (summarization, translation,
#' named-entity recognition, question answering, table question answering). It
#' builds the request with provider routing, shared retry and error handling,
#' performs it, and returns the parsed JSON body.
#'
#' @param model Character string. Model ID, optionally suffixed with
#'   `":provider"`.
#' @param inputs The request `inputs` payload (a string or a list).
#' @param parameters Optional named list of task parameters. NULL elements are
#'   dropped, and an empty list is omitted from the request body.
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A dedicated Inference Endpoint
#'   URL.
#'
#' @returns The parsed JSON response (typically a list).
#' @keywords internal
hf_task_request <- function(model, inputs, parameters = NULL, token = NULL,
                            endpoint_url = NULL) {
  parsed <- hf_parse_model(model)
  token <- hf_get_token(token, required = FALSE)

  body <- list(inputs = inputs)
  parameters <- purrr::compact(parameters %||% list())
  if (length(parameters) > 0) {
    body$parameters <- parameters
  }

  url <- hf_inference_url(parsed$model, parsed$provider, endpoint_url)
  req <- httr2::request(url)
  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  resp <- req |>
    httr2::req_body_json(body) |>
    httr2::req_retry(max_tries = 3, is_transient = hf_is_transient) |>
    httr2::req_error(body = hf_error_body(parsed$model)) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

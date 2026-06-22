# Splits a `"model-id:provider"` spec into a model ID and optional provider.
hf_parse_model <- function(model) {
  if (length(model) != 1 || is.na(model) || !grepl(":", model, fixed = TRUE)) {
    return(list(model = model, provider = NULL))
  }

  parts <- strsplit(model, ":", fixed = TRUE)[[1]]
  provider <- parts[length(parts)]
  model_id <- paste(parts[-length(parts)], collapse = ":")
  list(model = model_id, provider = provider)
}


# Resolves the task-style inference URL for serverless/provider/endpoint routes.
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


# Resolves the OpenAI-compatible chat-completions URL.
hf_chat_url <- function(endpoint_url = NULL) {
  if (!is.null(endpoint_url)) {
    return(paste0(sub("/$", "", endpoint_url), "/v1/chat/completions"))
  }

  "https://router.huggingface.co/v1/chat/completions"
}


# Status codes that indicate transient failures worth retrying.
hf_is_transient <- function(resp) {
  httr2::resp_status(resp) %in% c(429L, 500L, 502L, 503L, 504L)
}


# Builds the JSON body used by task-style inference requests.
hf_inference_body <- function(inputs, parameters = NULL) {
  body <- list(inputs = inputs)
  parameters <- purrr::compact(parameters %||% list())
  if (length(parameters) > 0) {
    body$parameters <- parameters
  }
  body
}


# Builds the JSON body used by OpenAI-compatible chat-completions requests.
hf_chat_body <- function(model, messages, max_tokens = NULL, temperature = NULL,
                         ...) {
  body <- purrr::compact(list(
    model = model,
    messages = messages,
    max_tokens = max_tokens,
    temperature = temperature
  ))

  dots <- purrr::compact(list(...))
  if (length(dots) > 0) {
    body <- c(body, dots)
  }

  body
}


# Builds an authenticated chat-completions request without performing it.
hf_build_chat_request <- function(body, token = NULL, endpoint_url = NULL) {
  token <- hf_get_token(token, required = TRUE)

  httr2::request(hf_chat_url(endpoint_url)) |>
    httr2::req_auth_bearer_token(token) |>
    httr2::req_body_json(body) |>
    httr2::req_retry(max_tries = 3, is_transient = hf_is_transient) |>
    httr2::req_error(body = hf_error_body(body$model %||% NULL))
}


# Performs an OpenAI-compatible chat-completions request and parses JSON.
hf_perform_chat_request <- function(body, token = NULL, endpoint_url = NULL) {
  hf_build_chat_request(body, token = token, endpoint_url = endpoint_url) |>
    httr2::req_perform() |>
    httr2::resp_body_json()
}


# Performs a streaming chat request and reassembles text deltas.
hf_perform_chat_stream <- function(body, callback = NULL, token = NULL,
                                   endpoint_url = NULL) {
  body$stream <- TRUE
  req <- hf_build_chat_request(body, token = token, endpoint_url = endpoint_url)
  resp <- httr2::req_perform_connection(req)
  on.exit(resp$body$close(), add = TRUE)

  deltas <- character()
  repeat {
    event <- httr2::resp_stream_sse(resp)
    if (is.null(event)) {
      break
    }

    data <- paste(event$data %||% character(), collapse = "\n")
    if (!nzchar(data) || identical(data, "[DONE]")) {
      if (identical(data, "[DONE]")) break
      next
    }

    parsed <- jsonlite::fromJSON(data, simplifyVector = FALSE)
    if (length(parsed$choices %||% list()) == 0) {
      next
    }
    delta <- parsed$choices[[1]]$delta$content %||% ""
    if (nzchar(delta)) {
      deltas <- c(deltas, delta)
      if (is.null(callback)) {
        cat(delta)
      } else {
        callback(delta)
      }
    }
  }

  content <- paste0(deltas, collapse = "")
  list(
    choices = list(list(message = list(role = "assistant", content = content))),
    usage = list(completion_tokens = ceiling(nchar(content) / 4))
  )
}


# Shared translator from Hugging Face error payloads to actionable messages.
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
      err %||% body$message %||% body$reason %||% "Unknown error"
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


# Shared request path for task-style inference wrappers.
hf_task_request <- function(model, inputs, parameters = NULL, token = NULL,
                            endpoint_url = NULL) {
  parsed <- hf_parse_model(model)
  token <- hf_get_token(token, required = FALSE)

  body <- hf_inference_body(inputs, parameters)

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

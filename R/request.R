# Splits a `"model-id:provider"` spec into a model ID and optional provider.
#
# The suffix may also be a router routing policy ("auto", "fastest", "cheapest",
# "preferred"). Policies are reported through `policy` and left out of `provider`
# because they are resolved by the router rather than naming a URL path segment.
hf_parse_model <- function(model) {
  if (length(model) != 1 || is.na(model) || !grepl(":", model, fixed = TRUE)) {
    return(list(model = model, provider = NULL, policy = NULL))
  }

  parts <- strsplit(model, ":", fixed = TRUE)[[1]]
  suffix <- parts[length(parts)]
  model_id <- paste(parts[-length(parts)], collapse = ":")

  if (suffix %in% hf_routing_policies()) {
    return(list(model = model_id, provider = NULL, policy = suffix))
  }

  list(model = model_id, provider = suffix, policy = NULL)
}


# Resolves the task-style inference URL for serverless/provider/endpoint routes.
hf_inference_url <- function(model, provider = NULL, endpoint_url = NULL) {
  if (!is.null(endpoint_url)) {
    return(sub("/$", "", endpoint_url))
  }

  provider <- provider %||% "hf-inference"
  paste0("https://router.huggingface.co/", provider, "/models/", model)
}


# Resolves the provider for a task request, honouring an explicit `:provider`
# suffix and otherwise confirming that hf-inference serves the model.
#
# Raises an actionable error when the Hub reports that hf-inference does not
# serve the model, rather than sending a request that cannot succeed.
hf_task_provider <- function(parsed, token = NULL, task = NULL) {
  provider <- hf_resolve_provider(
    model = parsed$model,
    provider = parsed$provider,
    token = token,
    task = task
  )

  if (is.null(provider)) {
    stop(
      hf_no_provider_message(
        parsed$model,
        task = task,
        providers = hf_live_providers(parsed$model, token = token, task = task)
      ),
      call. = FALSE
    )
  }

  provider
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

  # The chat body carries the raw model spec, which may include a `:provider`
  # or `:policy` suffix that the router resolves. Strip it before using the ID
  # for Hub lookups in error messages.
  model_id <- hf_parse_model(body$model %||% NULL)$model

  httr2::request(hf_chat_url(endpoint_url)) |>
    httr2::req_auth_bearer_token(token) |>
    hf_req_bill_to() |>
    httr2::req_body_json(body) |>
    httr2::req_retry(max_tries = 3, is_transient = hf_is_transient) |>
    httr2::req_error(body = hf_error_body(model_id, task = "conversational"))
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
hf_error_body <- function(model_id = NULL, task = NULL) {
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

    # Only a 404 means "no such route for this model". Other statuses (for
    # example a 400 reporting an unsupported parameter) must not be reported as
    # a provider-availability problem.
    if (identical(httr2::resp_status(resp), 404L) && !is.null(model_id)) {
      hf_unroutable_message(model_id, task = task)
    } else if (grepl("token|authoriz|authenticat", error_msg, ignore.case = TRUE)) {
      paste0(
        "Invalid or missing API token. Set one with hf_set_token(). Inference ",
        "requires a token with the 'Make calls to Inference Providers' ",
        "permission: ",
        "https://huggingface.co/settings/tokens/new?ownUserPermissions=",
        "inference.serverless.write&tokenType=fineGrained"
      )
    } else if (grepl("rate limit|quota|credits", error_msg, ignore.case = TRUE)) {
      paste0(
        "Rate limit or credit allowance exceeded. Inference Providers include a ",
        "small monthly credit allowance; see ",
        "https://huggingface.co/docs/inference-providers/pricing."
      )
    } else {
      paste0("API error: ", error_msg)
    }
  }
}


# Builds a message naming the providers that actually serve a model, so a 404
# from the task API points somewhere useful.
hf_unroutable_message <- function(model_id, task = NULL) {
  providers <- tryCatch(
    hf_live_providers(model_id, task = task),
    error = function(e) character()
  )
  hf_no_provider_message(model_id, task = task, providers = providers)
}


# Adds the organization billing header when HF_BILL_TO is configured.
#
# Team and Enterprise accounts route usage to an organization by sending
# `X-HF-Bill-To`; without it, usage bills to the individual user.
hf_req_bill_to <- function(req, bill_to = NULL) {
  bill_to <- bill_to %||% Sys.getenv("HF_BILL_TO", unset = "")
  if (!nzchar(bill_to)) {
    return(req)
  }
  httr2::req_headers(req, "X-HF-Bill-To" = bill_to)
}


# Shared request path for task-style inference wrappers.
hf_task_request <- function(model, inputs, parameters = NULL, token = NULL,
                            endpoint_url = NULL, task = NULL) {
  parsed <- hf_parse_model(model)
  token <- hf_get_token(token, required = FALSE)

  body <- hf_inference_body(inputs, parameters)

  provider <- if (is.null(endpoint_url)) {
    hf_task_provider(parsed, token = token, task = task)
  } else {
    NULL
  }

  url <- hf_inference_url(parsed$model, provider, endpoint_url)
  req <- httr2::request(url)
  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  resp <- req |>
    hf_req_bill_to() |>
    httr2::req_body_json(body) |>
    httr2::req_retry(max_tries = 3, is_transient = hf_is_transient) |>
    httr2::req_error(body = hf_error_body(parsed$model, task = task)) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}


# Performs a task request with raw media bytes and parses JSON.
hf_binary_task_request <- function(model, input, token = NULL,
                                   endpoint_url = NULL,
                                   content_type = NULL,
                                   query = NULL,
                                   task = NULL) {
  media <- hf_media_input(input, content_type = content_type)
  parsed <- hf_parse_model(model)
  token <- hf_get_token(token, required = FALSE)

  provider <- if (is.null(endpoint_url)) {
    hf_task_provider(parsed, token = token, task = task)
  } else {
    NULL
  }

  req <- httr2::request(hf_inference_url(parsed$model, provider, endpoint_url)) |>
    hf_req_bill_to() |>
    httr2::req_headers("Content-Type" = media$content_type) |>
    httr2::req_body_raw(media$raw) |>
    httr2::req_retry(max_tries = 3, is_transient = hf_is_transient) |>
    httr2::req_error(body = hf_error_body(parsed$model, task = task))

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }
  query <- purrr::compact(query %||% list())
  if (length(query) > 0) {
    req <- do.call(httr2::req_url_query, c(list(req), query))
  }

  httr2::req_perform(req) |>
    httr2::resp_body_json()
}


# Performs a text-input task that returns binary data.
hf_binary_generation_request <- function(model, inputs, parameters = NULL,
                                         token = NULL, endpoint_url = NULL,
                                         task = NULL) {
  parsed <- hf_parse_model(model)
  token <- hf_get_token(token, required = FALSE)
  body <- hf_inference_body(inputs, parameters)

  provider <- if (is.null(endpoint_url)) {
    hf_task_provider(parsed, token = token, task = task)
  } else {
    NULL
  }

  req <- httr2::request(hf_inference_url(parsed$model, provider, endpoint_url)) |>
    hf_req_bill_to() |>
    httr2::req_body_json(body) |>
    httr2::req_retry(max_tries = 3, is_transient = hf_is_transient) |>
    httr2::req_error(body = hf_error_body(parsed$model, task = task))

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  resp <- httr2::req_perform(req)
  list(
    raw = httr2::resp_body_raw(resp),
    content_type = hf_clean_content_type(
      httr2::resp_header(resp, "content-type") %||% "application/octet-stream"
    )
  )
}

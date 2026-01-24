#' LLM Chat Interface
#'
#' Have a conversation with an open-source language model via the Inference Providers API.
#'
#' @param message Character string. The user message to send to the model.
#' @param system Character string or NULL. Optional system prompt to set behavior.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "HuggingFaceTB/SmolLM3-3B". Use `:provider` suffix to select
#'   a specific provider (e.g., "meta-llama/Llama-3-8B-Instruct:together").
#' @param max_tokens Integer. Maximum tokens to generate. Default: 500.
#' @param temperature Numeric. Sampling temperature (0-2). Default: 0.7.
#' @param token Character string or NULL. API token for authentication.
#' @param ... Additional parameters passed to the model.
#'
#' @returns A tibble with columns: role, content, model, tokens_used
#' @export
#'
#' @examples
#' \dontrun{
#' # Simple question
#' hf_chat("What is the capital of France?")
#'
#' # With system prompt
#' hf_chat(
#'   "Explain gradient descent",
#'   system = "You are a statistics professor. Use simple analogies."
#' )
#'
#' # Use a specific provider
#' hf_chat("Hello!", model = "meta-llama/Llama-3-8B-Instruct:together")
#' }
hf_chat <- function(message,
                    system = NULL,
                    model = "HuggingFaceTB/SmolLM3-3B",
                    max_tokens = 500,
                    temperature = 0.7,
                    token = NULL,
                    ...) {

  if (is.null(message) || nchar(trimws(message)) == 0) {
    stop("Message cannot be empty", call. = FALSE)
  }

  token <- hf_get_token(token, required = TRUE)

  # Build messages array
  messages <- list()
  if (!is.null(system)) {
    messages <- c(messages, list(list(role = "system", content = system)))
  }
  messages <- c(messages, list(list(role = "user", content = message)))

  # Build request body
  body <- list(
    model = model,
    messages = messages,
    max_tokens = max_tokens,
    temperature = temperature
  )

  # Add any extra parameters

  dots <- list(...)
  if (length(dots) > 0) {
    body <- c(body, dots)
  }

  # Make request to chat completions endpoint
  resp <- httr2::request("https://router.huggingface.co/v1/chat/completions") |>
    httr2::req_auth_bearer_token(token) |>
    httr2::req_body_json(body) |>
    httr2::req_retry(max_tries = 3) |>
    httr2::req_error(body = function(resp) {
      body <- tryCatch(
        httr2::resp_body_json(resp),
        error = function(e) list(error = list(message = httr2::resp_body_string(resp)))
      )
      error_msg <- body$error$message %||% body$error %||% "Unknown error"
      paste0("API error: ", error_msg)
    }) |>
    httr2::req_perform()

  result <- httr2::resp_body_json(resp)

  # Extract response
  choice <- result$choices[[1]]
  generated_text <- choice$message$content %||% ""
  tokens_used <- result$usage$completion_tokens %||% ceiling(nchar(generated_text) / 4)

  tibble::tibble(
    role = "assistant",
    content = generated_text,
    model = model,
    tokens_used = tokens_used
  )
}


#' Multi-turn Conversation
#'
#' Create and manage a multi-turn conversation with an LLM.
#'
#' @param system Character string or NULL. System prompt for the conversation.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "HuggingFaceTB/SmolLM3-3B".
#'
#' @returns A conversation object (list) that can be extended with chat().
#' @export
#'
#' @examples
#' \dontrun{
#' # Create conversation
#' convo <- hf_conversation(system = "You are a helpful R tutor.")
#'
#' # Add messages (see chat() method)
#' convo <- chat(convo, "How do I read a CSV?")
#' convo <- chat(convo, "What about Excel files?")
#'
#' # View history
#' convo$history
#' }
hf_conversation <- function(system = NULL,
                            model = "HuggingFaceTB/SmolLM3-3B") {
  
  structure(
    list(
      system = system,
      model = model,
      history = list()
    ),
    class = "hf_conversation"
  )
}


#' Continue a Conversation
#'
#' Add a message to an existing conversation and get a response.
#'
#' @param conversation An hf_conversation object from hf_conversation().
#' @param message Character string. The user message.
#' @param ... Additional parameters passed to the model.
#'
#' @returns Updated conversation object with new messages in history.
#' @export
#'
#' @examples
#' \dontrun{
#' convo <- hf_conversation()
#' convo <- chat(convo, "Tell me a joke")
#' }
chat <- function(conversation, message, ...) {
  UseMethod("chat")
}


#' @export
chat.hf_conversation <- function(conversation, message, ...) {

  token <- hf_get_token(NULL, required = TRUE)

  # Add user message to history
  conversation$history <- c(
    conversation$history,
    list(list(role = "user", content = message))
  )

  # Build messages array from full history
  messages <- list()
  if (!is.null(conversation$system)) {
    messages <- c(messages, list(list(role = "system", content = conversation$system)))
  }
  messages <- c(messages, conversation$history)

  # Build request body
  body <- list(
    model = conversation$model,
    messages = messages,
    max_tokens = 500L,
    ...
  )

  # Make request to chat completions endpoint
  resp <- httr2::request("https://router.huggingface.co/v1/chat/completions") |>
    httr2::req_auth_bearer_token(token) |>
    httr2::req_body_json(body) |>
    httr2::req_retry(max_tries = 3) |>
    httr2::req_error(body = function(resp) {
      result <- tryCatch(
        httr2::resp_body_json(resp),
        error = function(e) list(error = list(message = httr2::resp_body_string(resp)))
      )
      paste0("API error: ", result$error$message %||% result$error %||% "Unknown error")
    }) |>
    httr2::req_perform()

  result <- httr2::resp_body_json(resp)
  generated_text <- result$choices[[1]]$message$content %||% ""

  # Add assistant response to history
  conversation$history <- c(
    conversation$history,
    list(list(role = "assistant", content = generated_text))
  )

  conversation
}


#' @export
print.hf_conversation <- function(x, ...) {
  cat("Hugging Face Conversation\n")
  cat("Model:", x$model, "\n")
  if (!is.null(x$system)) {
    cat("System:", x$system, "\n")
  }
  cat("\nHistory (", length(x$history), " messages):\n", sep = "")
  
  for (msg in x$history) {
    cat("\n", toupper(msg$role), ": ", msg$content, sep = "")
  }
  
  invisible(x)
}

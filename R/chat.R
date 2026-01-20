#' LLM Chat Interface
#'
#' Have a conversation with an open-source language model via the Inference API.
#'
#' @param message Character string. The user message to send to the model.
#' @param system Character string or NULL. Optional system prompt to set behavior.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "mistralai/Mistral-7B-Instruct-v0.2".
#' @param max_tokens Integer. Maximum tokens to generate. Default: 500.
#' @param temperature Numeric. Sampling temperature (0-2). Default: 0.7.
#' @param token Character string or NULL. API token for authentication.
#' @param ... Additional parameters passed to the model.
#'
#' @returns A tibble with columns: role, content, model, tokens_used (estimated)
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
#' }
hf_chat <- function(message,
                    system = NULL,
                    model = "mistralai/Mistral-7B-Instruct-v0.2",
                    max_tokens = 500,
                    temperature = 0.7,
                    token = NULL,
                    ...) {
  
  if (is.null(message) || nchar(trimws(message)) == 0) {
    stop("Message cannot be empty", call. = FALSE)
  }
  
  # Build prompt with system message if provided
  prompt <- if (!is.null(system)) {
    paste0("<s>[INST] <<SYS>>\n", system, "\n<</SYS>>\n\n", message, " [/INST]")
  } else {
    paste0("<s>[INST] ", message, " [/INST]")
  }
  
  # Make API request
  resp <- hf_api_request(
    model_id = model,
    inputs = prompt,
    parameters = list(
      max_new_tokens = max_tokens,
      temperature = temperature,
      return_full_text = FALSE,
      ...
    ),
    token = token
  )
  
  result <- httr2::resp_body_json(resp)
  
  # Extract generated text
  generated_text <- if (is.list(result) && length(result) > 0) {
    result[[1]]$generated_text %||% ""
  } else {
    ""
  }
  
  # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
  tokens_used <- ceiling(nchar(generated_text) / 4)
  
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
#'   Default: "mistralai/Mistral-7B-Instruct-v0.2".
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
                            model = "mistralai/Mistral-7B-Instruct-v0.2") {
  
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
  
  # Add user message to history
  conversation$history <- c(
    conversation$history,
    list(list(role = "user", content = message))
  )
  
  # Build context from history
  # This is a simplified version; production would need proper chat templating
  context <- ""
  if (!is.null(conversation$system)) {
    context <- paste0("System: ", conversation$system, "\n\n")
  }
  
  for (msg in conversation$history) {
    if (msg$role == "user") {
      context <- paste0(context, "User: ", msg$content, "\n")
    } else {
      context <- paste0(context, "Assistant: ", msg$content, "\n")
    }
  }
  
  # Get response
  response <- hf_chat(
    message = message,
    system = conversation$system,
    model = conversation$model,
    ...
  )
  
  # Add assistant response to history
  conversation$history <- c(
    conversation$history,
    list(list(role = "assistant", content = response$content[1]))
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

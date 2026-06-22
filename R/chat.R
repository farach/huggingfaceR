#' LLM Chat Interface
#'
#' Have a conversation with an open-source language model via the Inference Providers API.
#' Tool calling support depends on the model/provider. The default chat model is
#' optimized for a low-friction first call; for tool-calling examples, use a
#' tool-capable model such as \code{"Qwen/Qwen2.5-72B-Instruct"}.
#'
#' @param message Character string. The user message to send to the model.
#' @param system Character string or NULL. Optional system prompt to set behavior.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "meta-llama/Llama-3.1-8B-Instruct". Use `:provider` suffix to select
#'   a specific provider (e.g., "meta-llama/Llama-3.1-8B-Instruct:together").
#' @param max_tokens Integer. Maximum tokens to generate. Default: 500.
#' @param temperature Numeric. Sampling temperature (0-2). Default: 0.7.
#' @param token Character string or NULL. API token for authentication.
#' @param tools A list of tool definitions created by \code{hf_tool()}, or NULL.
#' @param tool_choice Character string or list controlling tool use. The public
#'   Hugging Face router currently supports "auto" and "none"; a tool name,
#'   "required", or full list can be used with compatible custom endpoints.
#'   Default: NULL.
#' @param stream Logical. If TRUE, stream response deltas and return the
#'   reassembled final response. Default: FALSE.
#' @param callback Function called with each streamed text delta. When NULL and
#'   \code{stream = TRUE}, deltas are printed to the console.
#' @param image Optional image input for vision-capable chat models. Can be a
#'   URL, local file path, raw vector, or list/vector of those.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#'   When provided, requests are sent to this URL instead of the public
#'   Inference API. The endpoint must support the chat completions format.
#' @param ... Additional parameters passed to the model.
#'
#' @returns A tibble with columns: role, content, model, tokens_used, and
#'   tool_calls.
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
#'
#' # Stream response deltas
#' hf_chat("Reply with exactly: OK", stream = TRUE)
#' }
hf_chat <- function(message,
                    system = NULL,
                    model = hf_default_model("chat"),
                    max_tokens = 500,
                    temperature = 0.7,
                    token = NULL,
                    tools = NULL,
                    tool_choice = NULL,
                    stream = FALSE,
                    callback = NULL,
                    image = NULL,
                    endpoint_url = NULL,
                    ...) {

  if (!is.character(message) || length(message) != 1 || is.na(message) ||
      !nzchar(trimws(message))) {
    stop("`message` must be a non-empty character scalar.", call. = FALSE)
  }

  messages <- hf_chat_messages(message, system = system, image = image)

  body <- hf_chat_body(
    model = model,
    messages = messages,
    max_tokens = max_tokens,
    temperature = temperature,
    tools = hf_normalize_tools(tools),
    tool_choice = hf_normalize_tool_choice(tool_choice),
    ...
  )

  result <- if (isTRUE(stream)) {
    hf_perform_chat_stream(
      body,
      callback = callback,
      token = token,
      endpoint_url = endpoint_url
    )
  } else {
    hf_perform_chat_request(body, token = token, endpoint_url = endpoint_url)
  }

  hf_chat_response_tibble(result, model)
}


#' Define a Chat Tool
#'
#' Build an OpenAI-compatible function-calling tool definition for
#' \code{hf_chat()}. The \code{parameters} argument can be a lightweight named
#' character vector, for example \code{c(city = "string")}, or a full JSON Schema
#' object.
#'
#' @param name Tool/function name. Must be a non-empty character scalar.
#' @param description Human-readable description of what the tool does.
#' @param parameters A named character vector or JSON Schema list describing
#'   function arguments.
#'
#' @returns A list suitable for the \code{tools} argument of \code{hf_chat()}.
#' @export
#'
#' @examples
#' weather_tool <- hf_tool(
#'   "get_weather",
#'   "Get current weather for a city.",
#'   c(city = "string")
#' )
hf_tool <- function(name, description, parameters = list(
                    type = "object",
                    properties = list(),
                    additionalProperties = FALSE
                  )) {
  if (length(name) != 1 || !is.character(name) || is.na(name) || !nzchar(name)) {
    stop("`name` must be a non-empty character scalar.", call. = FALSE)
  }
  if (length(description) != 1 || !is.character(description) ||
      is.na(description) || !nzchar(description)) {
    stop("`description` must be a non-empty character scalar.", call. = FALSE)
  }

  schema <- if (is.list(parameters) && identical(parameters$type, "object") &&
                length(parameters$properties %||% list()) == 0) {
    parameters
  } else {
    hf_schema_from_spec(parameters)$response_format$json_schema$schema
  }
  list(
    type = "function",
    `function` = list(
      name = name,
      description = description,
      parameters = schema
    )
  )
}


#' Multi-turn Conversation
#'
#' Create and manage a multi-turn conversation with an LLM.
#'
#' @param system Character string or NULL. System prompt for the conversation.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "meta-llama/Llama-3.1-8B-Instruct".
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
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
                            model = hf_default_model("chat"),
                            endpoint_url = NULL) {

  structure(
    list(
      system = system,
      model = model,
      endpoint_url = endpoint_url,
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
chat.hf_conversation <- function(conversation, message, token = NULL,
                                 tools = NULL, tool_choice = NULL,
                                 max_tokens = 500L, temperature = NULL,
                                 stream = FALSE, callback = NULL,
                                 image = NULL, ...) {

  conversation$history <- c(
    conversation$history,
    list(hf_user_message(message, image = image))
  )

  messages <- hf_conversation_messages(conversation)

  body <- hf_chat_body(
    model = conversation$model,
    messages = messages,
    max_tokens = max_tokens,
    temperature = temperature,
    tools = hf_normalize_tools(tools),
    tool_choice = hf_normalize_tool_choice(tool_choice),
    ...
  )

  result <- if (isTRUE(stream)) {
    hf_perform_chat_stream(
      body,
      callback = callback,
      token = token,
      endpoint_url = conversation$endpoint_url
    )
  } else {
    hf_perform_chat_request(
      body,
      token = token,
      endpoint_url = conversation$endpoint_url
    )
  }

  conversation$history <- c(
    conversation$history,
    list(hf_assistant_message(result))
  )

  conversation
}


#' Run Tool Calls in a Conversation
#'
#' Execute tool calls requested by the last assistant message in an
#' \code{hf_conversation}, append tool-result messages, and ask the model for the
#' next response. The loop repeats until the model returns a response without
#' tool calls.
#'
#' @param conversation An \code{hf_conversation} object.
#' @param tools Named list of R functions, keyed by tool name.
#' @param max_turns Integer. Maximum tool-execution/model-response iterations.
#' @param token Character string or NULL. API token for authentication.
#' @param ... Additional parameters passed to the chat-completions request.
#'
#' @returns Updated \code{hf_conversation} object.
#' @export
#'
#' @examples
#' \dontrun{
#' tool <- hf_tool("add", "Add two numbers.", c(x = "number", y = "number"))
#' convo <- hf_conversation(model = "Qwen/Qwen2.5-72B-Instruct")
#' convo <- chat(convo, "What is 2 + 3?", tools = list(tool))
#' convo <- hf_run_tools(convo, list(add = function(x, y) x + y))
#' }
hf_run_tools <- function(conversation, tools, max_turns = 5L, token = NULL,
                         ...) {
  if (!inherits(conversation, "hf_conversation")) {
    stop("`conversation` must be an hf_conversation object.", call. = FALSE)
  }
  if (length(conversation$history) == 0) {
    stop("`conversation` must contain an assistant message with tool calls.",
         call. = FALSE)
  }
  if (!is.list(tools) || is.null(names(tools)) || any(!nzchar(names(tools)))) {
    stop("`tools` must be a named list of R functions.", call. = FALSE)
  }
  if (!all(vapply(tools, is.function, logical(1)))) {
    stop("Every element of `tools` must be a function.", call. = FALSE)
  }

  for (turn in seq_len(max_turns)) {
    last <- conversation$history[[length(conversation$history)]]
    tool_calls <- last$tool_calls %||% list()
    if (length(tool_calls) == 0) {
      return(conversation)
    }

    for (tool_call in tool_calls) {
      name <- tool_call$`function`$name %||% ""
      if (!name %in% names(tools)) {
        stop("No R function supplied for tool `", name, "`.", call. = FALSE)
      }
      args <- jsonlite::fromJSON(
        tool_call$`function`$arguments %||% "{}",
        simplifyVector = FALSE
      )
      value <- do.call(tools[[name]], args)
      content <- if (length(value) == 1 && is.atomic(value)) {
        as.character(value)
      } else {
        as.character(jsonlite::toJSON(value, auto_unbox = TRUE))
      }
      conversation$history <- c(conversation$history, list(list(
        role = "tool",
        tool_call_id = tool_call$id,
        name = name,
        content = content
      )))
    }

    body <- hf_chat_body(
      model = conversation$model,
      messages = hf_conversation_messages(conversation),
      ...
    )
    result <- hf_perform_chat_request(
      body,
      token = token,
      endpoint_url = conversation$endpoint_url
    )
    conversation$history <- c(
      conversation$history,
      list(hf_assistant_message(result))
    )
  }

  stop("Tool loop reached `max_turns` before a final response.", call. = FALSE)
}


#' Describe an Image
#'
#' Ask a vision-capable chat model to describe an image.
#'
#' @param image Image URL, local file path, raw vector, or a character vector/list
#'   of image URLs or paths.
#' @param prompt Prompt to send with each image. Default: "Describe this image."
#' @param model Character string. Vision-capable chat model ID.
#' @param max_tokens Integer. Maximum tokens to generate. Default: 200.
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param ... Additional parameters passed to \code{hf_chat()}.
#'
#' @returns A tibble with columns: image, description.
#' @export
#'
#' @examples
#' \dontrun{
#' image <- paste0(
#'   "https://huggingface.co/datasets/huggingface/",
#'   "documentation-images/resolve/main/cat.png"
#' )
#' hf_describe_image(image)
#' }
hf_describe_image <- function(image,
                              prompt = "Describe this image.",
                              model = hf_default_model("vision_chat"),
                              max_tokens = 200,
                              token = NULL,
                              endpoint_url = NULL,
                              ...) {
  images <- if (is.raw(image)) list(image) else as.list(image)
  purrr::map_dfr(images, function(single_image) {
    res <- hf_chat(
      prompt,
      model = model,
      max_tokens = max_tokens,
      token = token,
      image = single_image,
      endpoint_url = endpoint_url,
      ...
    )
    tibble::tibble(
      image = hf_image_label(single_image),
      description = res$content[[1]]
    )
  })
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
    content <- msg$content
    if (is.list(content)) {
      content <- "<multimodal content>"
    }
    cat("\n", toupper(msg$role), ": ", content %||% "", sep = "")
  }
  
  invisible(x)
}


# Build chat messages, including optional system and image content.
hf_chat_messages <- function(message, system = NULL, image = NULL) {
  messages <- list()
  if (!is.null(system)) {
    messages <- c(messages, list(list(role = "system", content = system)))
  }
  c(messages, list(hf_user_message(message, image = image)))
}


# Build a user message with text-only or multimodal content.
hf_user_message <- function(message, image = NULL) {
  list(role = "user", content = hf_user_content(message, image = image))
}


# Build a user message content payload.
hf_user_content <- function(message, image = NULL) {
  if (is.null(image)) {
    return(message)
  }

  images <- if (is.raw(image)) list(image) else as.list(image)
  c(
    list(list(type = "text", text = message)),
    purrr::map(images, function(single_image) {
      list(
        type = "image_url",
        image_url = list(url = hf_image_url(single_image))
      )
    })
  )
}


# Normalize local/raw/remote image inputs for vision-chat messages.
hf_image_url <- function(image) {
  if (is.character(image) && length(image) == 1 &&
      grepl("^https?://", image, ignore.case = TRUE)) {
    return(image)
  }

  if (is.character(image) && length(image) == 1) {
    if (!file.exists(image)) {
      stop("Image file not found: ", image, call. = FALSE)
    }
    raw <- readBin(image, what = "raw", n = file.info(image)$size)
    type <- hf_image_content_type(image)
    return(paste0("data:", type, ";base64,", jsonlite::base64_enc(raw)))
  }

  if (is.raw(image)) {
    return(paste0("data:image/png;base64,", jsonlite::base64_enc(image)))
  }

  stop("`image` must be a URL, local file path, or raw vector.", call. = FALSE)
}


# Infer common image content types from file extensions.
hf_image_content_type <- function(path) {
  ext <- tolower(tools::file_ext(path))
  switch(ext,
    jpg = "image/jpeg",
    jpeg = "image/jpeg",
    png = "image/png",
    gif = "image/gif",
    webp = "image/webp",
    "image/png"
  )
}


# Human-readable image label for returned tibbles.
hf_image_label <- function(image) {
  if (is.raw(image)) "<raw>" else as.character(image)
}


# Normalize tool definitions into the chat-completions payload shape.
hf_normalize_tools <- function(tools) {
  if (is.null(tools)) {
    return(NULL)
  }
  if (is.list(tools) && identical(tools$type, "function")) {
    tools <- list(tools)
  }
  if (!is.list(tools) || length(tools) == 0) {
    stop("`tools` must be a tool definition or list of definitions.", call. = FALSE)
  }
  for (tool in tools) {
    if (!is.list(tool) || !identical(tool$type, "function") ||
        is.null(tool$`function`$name)) {
      stop("Each tool must be created by `hf_tool()`.", call. = FALSE)
    }
  }
  tools
}


# Normalize OpenAI-compatible tool_choice shorthand.
hf_normalize_tool_choice <- function(tool_choice) {
  if (is.null(tool_choice) || is.list(tool_choice)) {
    return(tool_choice)
  }
  if (!is.character(tool_choice) || length(tool_choice) != 1 ||
      !nzchar(tool_choice)) {
    stop("`tool_choice` must be NULL, a character scalar, or a list.", call. = FALSE)
  }
  if (tool_choice %in% c("auto", "none", "required")) {
    return(tool_choice)
  }
  list(type = "function", `function` = list(name = tool_choice))
}


# Convert a chat-completions result into the public tibble shape.
hf_chat_response_tibble <- function(result, model) {
  choice <- result$choices[[1]]
  message <- choice$message
  generated_text <- message$content %||% ""
  tokens_used <- result$usage$completion_tokens %||% ceiling(nchar(generated_text) / 4)
  tibble::tibble(
    role = message$role %||% "assistant",
    content = generated_text,
    model = model,
    tokens_used = tokens_used,
    tool_calls = list(message$tool_calls %||% list())
  )
}


# Convert a chat-completions result into a conversation history message.
hf_assistant_message <- function(result) {
  message <- result$choices[[1]]$message
  out <- list(
    role = "assistant",
    content = message$content %||% ""
  )
  tool_calls <- message$tool_calls %||% list()
  if (length(tool_calls) > 0) {
    out$tool_calls <- tool_calls
  }
  out
}


# Build the full message list for a conversation.
hf_conversation_messages <- function(conversation) {
  messages <- list()
  if (!is.null(conversation$system)) {
    messages <- c(messages, list(list(role = "system", content = conversation$system)))
  }
  c(messages, conversation$history)
}

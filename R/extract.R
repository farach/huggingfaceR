#' Extract Structured Data from Text
#'
#' Convert unstructured text into tidy columns using a chat model with structured
#' JSON output. The \code{schema} argument can be a lightweight named character
#' vector such as \code{c(name = "string", score = "number")} or a full JSON
#' Schema list. The function returns one row per input text and one column per
#' schema field.
#'
#' @param text Character vector of text(s) to extract from.
#' @param schema A named character vector of field names and JSON types, or a
#'   JSON Schema list with object \code{properties}.
#' @param model Character string. Model ID from Hugging Face Hub. Default:
#'   "meta-llama/Llama-3.1-8B-Instruct".
#' @param strict Logical. Whether to request strict JSON Schema adherence.
#'   Default: TRUE.
#' @param system Character string. System prompt sent with each extraction
#'   request. Default: a concise extraction instruction.
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#'   The endpoint must support the chat completions format.
#' @param ... Additional parameters passed to the chat-completions request.
#'
#' @returns A tibble with one row per input and one column per schema field.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_extract(
#'   "Amelie is a chef in Paris.",
#'   c(name = "string", occupation = "string", city = "string")
#' )
#'
#' hf_extract(
#'   c("Great service.", "The delivery was late."),
#'   c(sentiment = "string", is_complaint = "boolean")
#' )
#' }
hf_extract <- function(text,
                       schema,
                       model = hf_default_model("chat"),
                       strict = TRUE,
                       system = paste(
                         "Extract the requested fields from the user's text.",
                         "Return only JSON that matches the schema."
                       ),
                       token = NULL,
                       endpoint_url = NULL,
                       ...) {

  spec <- hf_schema_from_spec(schema, strict = strict)

  if (length(text) == 0) {
    return(hf_extract_empty_tibble(spec))
  }

  schema_json <- jsonlite::toJSON(
    spec$response_format$json_schema$schema,
    auto_unbox = TRUE
  )
  system <- paste(
    system,
    "Use this JSON Schema:",
    schema_json
  )

  purrr::map_dfr(text, function(single_text) {
    if (is.na(single_text)) {
      return(hf_extract_empty_row(spec))
    }

    messages <- list(
      list(role = "system", content = system),
      list(role = "user", content = single_text)
    )

    body <- hf_chat_body(
      model = model,
      messages = messages,
      temperature = 0,
      response_format = spec$response_format,
      ...
    )

    result <- tryCatch(
      hf_perform_chat_request(
        body,
        token = token,
        endpoint_url = endpoint_url
      ),
      error = function(e) {
        if (!hf_json_schema_unsupported(e)) {
          stop(e)
        }
        body$response_format <- list(type = "json_object")
        hf_perform_chat_request(
          body,
          token = token,
          endpoint_url = endpoint_url
        )
      }
    )
    content <- result$choices[[1]]$message$content %||% ""
    parsed <- tryCatch(
      jsonlite::fromJSON(content, simplifyVector = FALSE),
      error = function(e) {
        stop(
          "Failed to parse structured output as JSON: ",
          conditionMessage(e),
          call. = FALSE
        )
      }
    )

    hf_extract_row(parsed, spec)
  })
}


# Detect providers that only support json_object structured output.
hf_json_schema_unsupported <- function(error) {
  grepl(
    "json_schema|response format|structured output",
    conditionMessage(error),
    ignore.case = TRUE
  )
}


# Normalize a lightweight field spec or full JSON Schema into response_format.
hf_schema_from_spec <- function(schema, strict = TRUE) {
  if (is.character(schema) && !is.null(names(schema)) &&
      all(nzchar(names(schema)))) {
    fields <- names(schema)
    field_types <- stats::setNames(
      vapply(schema, hf_schema_type, character(1)),
      fields
    )
    properties <- stats::setNames(
      lapply(field_types, function(type) list(type = type)),
      fields
    )
    schema <- list(
      type = "object",
      properties = properties,
      required = fields,
      additionalProperties = FALSE
    )
  } else if (is.list(schema)) {
    if (is.null(schema$type) && !is.null(schema$properties)) {
      schema$type <- "object"
    }
    if (!identical(schema$type, "object") || is.null(schema$properties)) {
      stop(
        "`schema` must be an object JSON Schema with `properties`.",
        call. = FALSE
      )
    }
    fields <- names(schema$properties)
    field_types <- stats::setNames(
      vapply(schema$properties, function(prop) {
        type <- prop$type %||% "string"
        if (length(type) > 1) type <- type[[1]]
        hf_schema_type(type)
      }, character(1)),
      fields
    )
    if (isTRUE(strict) && is.null(schema$additionalProperties)) {
      schema$additionalProperties <- FALSE
    }
  } else {
    stop(
      "`schema` must be a named character vector or JSON Schema list.",
      call. = FALSE
    )
  }

  if (length(fields) == 0 || any(!nzchar(fields))) {
    stop("`schema` must define at least one named field.", call. = FALSE)
  }

  list(
    fields = fields,
    field_types = field_types,
    response_format = list(
      type = "json_schema",
      json_schema = list(
        name = "hf_extract_schema",
        strict = strict,
        schema = schema
      )
    )
  )
}


# Map common R-style aliases to JSON Schema scalar types.
hf_schema_type <- function(type) {
  if (length(type) != 1 || !is.character(type) || is.na(type)) {
    type <- NA_character_
  }
  aliases <- c(
    character = "string",
    numeric = "number",
    double = "number",
    logical = "boolean"
  )
  alias <- unname(aliases[type])
  if (!is.na(alias)) {
    type <- alias
  }
  valid <- c("string", "number", "integer", "boolean", "array", "object")
  if (length(type) != 1 || !type %in% valid) {
    stop(
      "`schema` field types must be one of: ",
      paste(valid, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  type
}


# Build a single empty extraction row for NA inputs.
hf_extract_empty_row <- function(spec) {
  values <- purrr::map(spec$field_types, hf_extract_na_value)
  tibble::as_tibble(values)
}


# Build a zero-row extraction tibble with the requested column types.
hf_extract_empty_tibble <- function(spec) {
  values <- purrr::map(spec$field_types, function(type) {
    switch(type,
      string = character(),
      number = numeric(),
      integer = integer(),
      boolean = logical(),
      array = list(),
      object = list()
    )
  })
  tibble::as_tibble(values)
}


# Use typed missing values so bind_rows() keeps predictable column types.
hf_extract_na_value <- function(type) {
  switch(type,
    string = NA_character_,
    number = NA_real_,
    integer = NA_integer_,
    boolean = NA,
    array = list(NULL),
    object = list(NULL)
  )
}


# Coerce parsed JSON into one tibble row following the requested field order.
hf_extract_row <- function(parsed, spec) {
  values <- purrr::map2(
    spec$fields,
    spec$field_types,
    function(field, type) {
      value <- parsed[[field]]
      if (is.null(value)) {
        return(hf_extract_na_value(type))
      }
      switch(type,
        string = as.character(value),
        number = as.numeric(value),
        integer = as.integer(value),
        boolean = as.logical(value),
        array = list(if (is.list(value)) {
          unlist(value, recursive = FALSE, use.names = FALSE)
        } else {
          value
        }),
        object = list(value)
      )
    }
  )
  names(values) <- spec$fields
  tibble::as_tibble(values)
}

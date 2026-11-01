# Extract Structured Data from Text

Convert unstructured text into tidy columns using a chat model with
structured JSON output. The `schema` argument can be a lightweight named
character vector such as `c(name = "string", score = "number")` or a
full JSON Schema list. The function returns one row per input text and
one column per schema field.

## Usage

``` r
hf_extract(
  text,
  schema,
  model = hf_default_model("chat"),
  strict = TRUE,
  system = paste("Extract the requested fields from the user's text.",
    "Return only JSON that matches the schema."),
  token = NULL,
  endpoint_url = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- text:

  Character vector of text(s) to extract from.

- schema:

  A named character vector of field names and JSON types, or a JSON
  Schema list with object `properties`.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "meta-llama/Llama-3.1-8B-Instruct".

- strict:

  Logical. Whether to request strict JSON Schema adherence. Default:
  TRUE.

- system:

  Character string. System prompt sent with each extraction request.
  Default: a concise extraction instruction.

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL. The
  endpoint must support the chat completions format.

- max_active:

  Integer. Maximum number of inputs to send at once. Default:
  \`getOption("huggingfaceR.max_active", 1)\`, which sends one input at
  a time. Larger values send inputs concurrently. This is faster for
  long vectors but spends provider credits and rate limits faster.
  Results keep input order, transient failures (HTTP 429 and 5xx) are
  retried a bounded number of times, and the first failing input stops
  the call with its error, as in sequential mode. Set
  \`options(huggingfaceR.max_active = 4)\` to change the default for
  every per-input function.

- ...:

  Additional parameters passed to the chat-completions request.

## Value

A tibble with one row per input and one column per schema field.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_extract(
  "Amelie is a chef in Paris.",
  c(name = "string", occupation = "string", city = "string")
)

hf_extract(
  c("Great service.", "The delivery was late."),
  c(sentiment = "string", is_complaint = "boolean")
)
} # }
```

# Text Generation

Generate text from a prompt using a language model via the Inference
Providers API.

## Usage

``` r
hf_generate(
  prompt,
  model = hf_default_model("generate"),
  max_new_tokens = 50,
  temperature = 1,
  top_p = NULL,
  token = NULL,
  endpoint_url = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- prompt:

  Character vector of text prompt(s) to generate from.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "meta-llama/Llama-3.1-8B-Instruct".

- max_new_tokens:

  Integer. Maximum number of tokens to generate. Default: 50.

- temperature:

  Numeric. Sampling temperature (0-2). Default: 1.0.

- top_p:

  Numeric. Nucleus sampling parameter. Default: NULL.

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

  Additional parameters passed to the model.

## Value

A tibble with columns: prompt, generated_text

## Examples

``` r
if (FALSE) { # \dontrun{
# Simple text generation
hf_generate("Once upon a time in a land far away,")

# With different model
hf_generate("The future of AI is", model = "meta-llama/Llama-3-8B-Instruct:together")
} # }
```

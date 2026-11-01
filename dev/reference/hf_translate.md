# Translate Text

Translate text from one language to another using a translation model
via the Hugging Face Inference Providers API.

## Usage

``` r
hf_translate(
  text,
  model = hf_default_model("translate"),
  source = NULL,
  target = NULL,
  token = NULL,
  endpoint_url = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- text:

  Character vector of text(s) to translate.

- model:

  Character string. Model ID from the Hugging Face Hub. Append
  \`":provider"\` to select an inference provider. Default:
  "Helsinki-NLP/opus-mt-en-fr" (English to French).

- source:

  Character string or NULL. Source language code (model-specific;
  ignored by \`opus-mt-\*\` language-pair models).

- target:

  Character string or NULL. Target language code (model-specific;
  ignored by \`opus-mt-\*\` language-pair models).

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

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

  Additional arguments (currently unused).

## Value

A tibble with columns: text, translation.

## Details

The default model, \`Helsinki-NLP/opus-mt-en-fr\`, translates English to
French and is chosen for easy onboarding: it is small, fast, broadly
known, and encodes the translation direction in the model ID, so
\`hf_translate("Hello")\` works with no extra arguments. To translate a
different language pair, swap in another Helsinki-NLP \`opus-mt-\*\`
model (for example \`"Helsinki-NLP/opus-mt-en-es"\` for English to
Spanish).

Translation models vary in how they expect languages to be specified.
Language-pair models such as the Helsinki-NLP \`opus-mt-\*\` family
encode the direction in the model ID and ignore \`source\`/\`target\`.
Multilingual models such as NLLB (\`facebook/nllb-200-distilled-600M\`)
instead require \`source\` and \`target\` to be set to FLORES-200 codes
(for example "eng_Latn", "fra_Latn").

## See also

<https://huggingface.co/docs/inference-providers/tasks/translation>

## Examples

``` r
if (FALSE) { # \dontrun{
# Simplest call: English to French with the default model
hf_translate("Hello, how are you?")

# A different language pair (English to Spanish)
hf_translate("Hello, how are you?", model = "Helsinki-NLP/opus-mt-en-es")

# Multilingual model (FLORES-200 codes)
hf_translate(
  "Hello, how are you?",
  model = "facebook/nllb-200-distilled-600M",
  source = "eng_Latn",
  target = "fra_Latn"
)
} # }
```

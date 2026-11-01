# Use a Sentence Transformers pipeline to extract document(s)/sentence(s) embedding(s)

Use a Sentence Transformers pipeline to extract document(s)/sentence(s)
embedding(s)

## Usage

``` r
hf_sentence_encode(
  model,
  text,
  batch_size = 64L,
  show_progress_bar = TRUE,
  tidy = TRUE,
  ...
)
```

## Arguments

- model:

  Model object you loaded with \`hf_load_sentence_model()\`

- text:

  The text, or texts, you wish to embed/encode.

- batch_size:

  How many texts to embed at once.

- show_progress_bar:

  Whether to print a progress bar in the console or not.

- tidy:

  Whether to tidy the output into a tibble or not.

- ...:

  other args sent to the model's encode method, e.g. device = device

## Value

n-dimensional embeddings for every input \`text\`

## Examples

``` r
if (FALSE) { # \dontrun{
text <- c("There are things we do know, things we don't know, and then there is quantum mechanics.")
sentence_mod <- hf_load_sentence_model("paraphrase-MiniLM-L6-v2")
embeddings <- hf_sentence_encode(model = sentence_mod, text, show_progress_bar = TRUE)
} # }
```

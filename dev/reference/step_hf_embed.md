# Embedding Recipe Step

Create text embeddings using a Hugging Face model as part of a
tidymodels recipe. This step converts text columns into embedding
features for downstream modeling.

## Usage

``` r
step_hf_embed(
  recipe,
  ...,
  role = "predictor",
  trained = FALSE,
  model = hf_default_model("embed"),
  token = NULL,
  embeddings = NULL,
  skip = FALSE,
  id = recipes::rand_id("hf_embed")
)

# S3 method for class 'step_hf_embed'
tidy(x, ...)

# S3 method for class 'step_hf_embed'
tunable(x, ...)
```

## Arguments

- recipe:

  A recipe object.

- ...:

  One or more text column selectors (see recipes::selections()).

- role:

  Character string. Role for the new embedding variables. Default:
  "predictor".

- trained:

  Logical. Internal use only.

- model:

  Character string. Hugging Face model ID for embeddings. Default:
  "BAAI/bge-small-en-v1.5".

- token:

  Character string or NULL. API token for authentication.

- embeddings:

  List. Internal use only (stores embeddings during training).

- skip:

  Logical. Should step be skipped when baking? Default: FALSE.

- id:

  Character string. Unique ID for this step.

- x:

  A step_hf_embed object

## Value

An updated recipe object.

## Examples

``` r
if (FALSE) { # \dontrun{
library(tidymodels)
library(dplyr)

# Create a recipe with embeddings
rec <- recipe(sentiment ~ text, data = train_data) |>
  step_hf_embed(text, model = "BAAI/bge-small-en-v1.5")

# Use in a workflow
wf <- workflow() |>
  add_recipe(rec) |>
  add_model(logistic_reg()) |>
  fit(data = train_data)
} # }
```

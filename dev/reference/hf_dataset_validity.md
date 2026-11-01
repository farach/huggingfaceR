# Check Which Dataset Viewer Features Work for a Dataset

Report whether the Hugging Face dataset viewer can preview, page
through, search, filter, and summarize each dataset. Use it before
calling \[hf_load_dataset()\], \[hf_dataset_search()\],
\[hf_dataset_filter()\], or \[hf_dataset_statistics()\].

## Usage

``` r
hf_dataset_validity(dataset, token = NULL)
```

## Arguments

- dataset:

  Character vector of dataset IDs. Short names are resolved as in
  \[hf_load_dataset()\].

- token:

  Character string or NULL. API token for private or gated datasets.

## Value

A tibble with one row per dataset and logical columns `viewer` (rows can
be loaded), `preview`, `search`, `filter`, and `statistics`.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_dataset_validity(c("stanfordnlp/imdb", "openai/gdpval"))
} # }
```

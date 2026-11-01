# Resolve a Short Dataset Name to Full ID

Uses the Hub API to find the full org/name ID for a dataset.

## Usage

``` r
hf_resolve_dataset(dataset, token = NULL)
```

## Arguments

- dataset:

  Character string. Short dataset name.

- token:

  Character string or NULL. API token.

## Value

Character string with full dataset ID, or NULL if not found.

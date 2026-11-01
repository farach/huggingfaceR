# Detect Default Config for a Dataset

Query the splits endpoint to find the default config for a given dataset
and split.

## Usage

``` r
hf_detect_config(dataset, split, token = NULL)
```

## Arguments

- dataset:

  Character string. Dataset name.

- split:

  Character string. The split to find a config for.

- token:

  Character string or NULL. API token.

## Value

Character string with the config name.

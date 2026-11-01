# Get Dataset Information

Retrieve metadata about a dataset from Hugging Face Hub.

## Usage

``` r
hf_dataset_info(dataset, token = NULL)
```

## Arguments

- dataset:

  Character string. Dataset name.

- token:

  Character string or NULL. API token for private datasets.

## Value

A list with dataset information.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_dataset_info("imdb")
} # }
```

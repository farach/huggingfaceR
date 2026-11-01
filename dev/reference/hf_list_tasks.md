# List Available Tasks

List all available task types on Hugging Face.

## Usage

``` r
hf_list_tasks(pattern = NULL)
```

## Arguments

- pattern:

  Character string or NULL. Optional regex pattern to filter tasks.

## Value

A character vector of task names.

## Examples

``` r
if (FALSE) { # \dontrun{
# List all tasks
hf_list_tasks()

# Filter tasks
hf_list_tasks(pattern = "classification")
} # }
```

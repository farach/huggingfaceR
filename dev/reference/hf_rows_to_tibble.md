# Convert Dataset Server Rows to a Tibble

Internal helper converting the \`rows\` payload from the Hugging Face
datasets server into a tibble.

## Usage

``` r
hf_rows_to_tibble(rows)
```

## Arguments

- rows:

  A list of row payloads, each a named list of field values.

## Value

A tibble with one row per element of \`rows\`.

## Details

Row fields are not always scalars: many datasets contain variable-length
fields (for example \`reference_files\` on \`openai/gdpval\`).
Converting such a row directly with \[tibble::as_tibble()\] recycles the
row to the length of its longest field, silently multiplying rows. This
helper detects fields that are ever non-scalar and represents them as
list-columns, so one source row always yields exactly one output row.

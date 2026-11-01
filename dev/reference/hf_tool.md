# Define a Chat Tool

Build an OpenAI-compatible function-calling tool definition for
[`hf_chat()`](https://farach.github.io/huggingfaceR/dev/reference/hf_chat.md).
The `parameters` argument can be a lightweight named character vector,
for example `c(city = "string")`, or a full JSON Schema object.

## Usage

``` r
hf_tool(
  name,
  description,
  parameters = list(type = "object", properties = list(), additionalProperties = FALSE)
)
```

## Arguments

- name:

  Tool/function name. Must be a non-empty character scalar.

- description:

  Human-readable description of what the tool does.

- parameters:

  A named character vector or JSON Schema list describing function
  arguments.

## Value

A list suitable for the `tools` argument of
[`hf_chat()`](https://farach.github.io/huggingfaceR/dev/reference/hf_chat.md).

## Examples

``` r
weather_tool <- hf_tool(
  "get_weather",
  "Get current weather for a city.",
  c(city = "string")
)
```

# Run Tool Calls in a Conversation

Execute tool calls requested by the last assistant message in an
`hf_conversation`, append tool-result messages, and ask the model for
the next response. The loop repeats until the model returns a response
without tool calls.

## Usage

``` r
hf_run_tools(conversation, tools, max_turns = 5L, token = NULL, ...)
```

## Arguments

- conversation:

  An `hf_conversation` object.

- tools:

  Named list of R functions, keyed by tool name.

- max_turns:

  Integer. Maximum tool-execution/model-response iterations.

- token:

  Character string or NULL. API token for authentication.

- ...:

  Additional parameters passed to the chat-completions request.

## Value

Updated `hf_conversation` object.

## Examples

``` r
if (FALSE) { # \dontrun{
tool <- hf_tool("add", "Add two numbers.", c(x = "number", y = "number"))
convo <- hf_conversation(model = "Qwen/Qwen2.5-72B-Instruct")
convo <- chat(convo, "What is 2 + 3?", tools = list(tool))
convo <- hf_run_tools(convo, list(add = function(x, y) x + y))
} # }
```

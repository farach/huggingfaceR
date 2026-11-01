# Multi-turn Conversation

Create and manage a multi-turn conversation with an LLM.

## Usage

``` r
hf_conversation(
  system = NULL,
  model = hf_default_model("chat"),
  endpoint_url = NULL
)
```

## Arguments

- system:

  Character string or NULL. System prompt for the conversation.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "meta-llama/Llama-3.1-8B-Instruct".

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

## Value

A conversation object (list) that can be extended with chat().

## Examples

``` r
if (FALSE) { # \dontrun{
# Create conversation
convo <- hf_conversation(system = "You are a helpful R tutor.")

# Add messages (see chat() method)
convo <- chat(convo, "How do I read a CSV?")
convo <- chat(convo, "What about Excel files?")

# View history
convo$history
} # }
```

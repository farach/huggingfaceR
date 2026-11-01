# Continue a Conversation

Add a message to an existing conversation and get a response.

## Usage

``` r
chat(conversation, message, ...)
```

## Arguments

- conversation:

  An hf_conversation object from hf_conversation().

- message:

  Character string. The user message.

- ...:

  Additional parameters passed to the model.

## Value

Updated conversation object with new messages in history.

## Examples

``` r
if (FALSE) { # \dontrun{
convo <- hf_conversation()
convo <- chat(convo, "Tell me a joke")
} # }
```

#' Text Generation
#'
#' Generate text from a prompt using a language model via the Inference Providers API.
#'
#' @param prompt Character vector of text prompt(s) to generate from.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "meta-llama/Llama-3.1-8B-Instruct".
#' @param max_new_tokens Integer. Maximum number of tokens to generate. Default: 50.
#' @param temperature Numeric. Sampling temperature (0-2). Default: 1.0.
#' @param top_p Numeric. Nucleus sampling parameter. Default: NULL.
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#'   The endpoint must support the chat completions format.
#' @inheritParams hf_summarize
#' @param ... Additional parameters passed to the model.
#'
#' @returns A tibble with columns: prompt, generated_text
#' @export
#'
#' @examples
#' \dontrun{
#' # Simple text generation
#' hf_generate("Once upon a time in a land far away,")
#'
#' # With different model
#' hf_generate("The future of AI is", model = "meta-llama/Llama-3-8B-Instruct:together")
#' }
hf_generate <- function(prompt,
                        model = hf_default_model("generate"),
                        max_new_tokens = 50,
                        temperature = 1.0,
                        top_p = NULL,
                        token = NULL,
                        endpoint_url = NULL,
                        max_active = getOption("huggingfaceR.max_active", 1L),
                        ...) {

  if (length(prompt) == 0 || all(is.na(prompt))) {
    return(tibble::tibble(
      prompt = character(),
      generated_text = character()
    ))
  }

  keep <- !is.na(prompt)
  bodies <- lapply(prompt[keep], function(single_prompt) {
    hf_chat_body(
      model = model,
      messages = list(
        list(role = "user", content = single_prompt)
      ),
      max_tokens = max_new_tokens,
      temperature = temperature,
      top_p = top_p,
      ...
    )
  })
  results <- hf_scatter(keep, hf_chat_requests(
    bodies,
    token = token,
    endpoint_url = endpoint_url,
    max_active = max_active
  ))

  tibble::tibble(
    prompt = prompt,
    generated_text = vapply(results, function(result) {
      if (is.null(result)) {
        return(NA_character_)
      }
      hf_chr(result$choices[[1]]$message$content %||% "")
    }, NA_character_)
  )
}


#' Fill Mask
#'
#' Fill in a [MASK] token in text with predicted words.
#' Commonly used with BERT-style models.
#'
#' @param text Character vector of text(s) containing [MASK] token.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "google-bert/bert-base-uncased".
#' @param mask_token Character string. The mask token to use. Default: "[MASK]".
#'   Some models use different tokens like "<mask>".
#' @param top_k Integer. Number of top predictions to return. Default: 5.
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @inheritParams hf_summarize
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: text, token, score, filled (the complete text)
#' @export
#'
#' @examples
#' \dontrun{
#' # Fill in the blank
#' hf_fill_mask("The capital of France is [MASK].")
#'
#' # Get top predictions
#' hf_fill_mask("Paris is the [MASK] of France.", top_k = 3)
#'
#' # Use with different mask token
#' hf_fill_mask("The capital of France is <mask>.", mask_token = "<mask>")
#' }
hf_fill_mask <- function(text,
                         model = hf_default_model("fill_mask"),
                         mask_token = "[MASK]",
                         top_k = 5,
                         token = NULL,
                         endpoint_url = NULL,
                         max_active = getOption("huggingfaceR.max_active", 1L),
                         ...) {

  if (length(text) == 0 || all(is.na(text))) {
    return(tibble::tibble(
      text = character(),
      token = character(),
      score = numeric(),
      filled = character()
    ))
  }

  keep <- !is.na(text)
  has_mask <- grepl(mask_token, text, fixed = TRUE)
  for (single_text in text[keep & !has_mask]) {
    cli::cli_warn("Text does not contain mask token '{mask_token}': {single_text}")
  }
  keep <- keep & has_mask

  results <- hf_scatter(keep, hf_task_requests(
    model = model,
    inputs = text[keep],
    parameters = list(top_k = top_k),
    token = token,
    endpoint_url = endpoint_url,
    task = "fill-mask",
    max_active = max_active
  ))

  out <- hf_record_rows(
    inputs = list(text = text),
    records = lapply(results, hf_as_records),
    fields = list(
      token = function(r) hf_chr(r$token_str),
      score = function(r) hf_dbl(r$score),
      .found = function(r) !is.null(r)
    )
  )

  # A prediction without `token_str` still fills the mask with "", as before.
  fill <- ifelse(is.na(out$token), "", out$token)
  out$filled <- vapply(seq_len(nrow(out)), function(i) {
    if (!out$.found[[i]]) {
      return(NA_character_)
    }
    gsub(mask_token, fill[[i]], out$text[[i]], fixed = TRUE)
  }, NA_character_)
  out$.found <- NULL
  out
}

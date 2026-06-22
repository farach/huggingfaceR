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
                        ...) {

  if (length(prompt) == 0 || all(is.na(prompt))) {
    return(tibble::tibble(
      prompt = character(),
      generated_text = character()
    ))
  }

  # Process each prompt
  results <- purrr::map_dfr(prompt, function(single_prompt) {
    if (is.na(single_prompt)) {
      return(tibble::tibble(
        prompt = single_prompt,
        generated_text = NA_character_
      ))
    }

    body <- hf_chat_body(
      model = model,
      messages = list(
        list(role = "user", content = single_prompt)
      ),
      max_tokens = max_new_tokens,
      temperature = temperature,
      top_p = top_p,
      ...
    )

    result <- hf_perform_chat_request(body, token = token, endpoint_url = endpoint_url)
    generated_text <- result$choices[[1]]$message$content %||% ""

    tibble::tibble(
      prompt = single_prompt,
      generated_text = generated_text
    )
  })

  results
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
                         ...) {
  
  if (length(text) == 0 || all(is.na(text))) {
    return(tibble::tibble(
      text = character(),
      token = character(),
      score = numeric(),
      filled = character()
    ))
  }
  
  # Process each text
  results <- purrr::map_dfr(text, function(single_text) {
    if (is.na(single_text)) {
      return(tibble::tibble(
        text = single_text,
        token = NA_character_,
        score = NA_real_,
        filled = NA_character_
      ))
    }
    
    if (!grepl(mask_token, single_text, fixed = TRUE)) {
      cli::cli_warn("Text does not contain mask token '{mask_token}': {single_text}")
      return(tibble::tibble(
        text = single_text,
        token = NA_character_,
        score = NA_real_,
        filled = NA_character_
      ))
    }
    
    resp <- hf_api_request(
      model_id = model,
      inputs = single_text,
      parameters = list(top_k = top_k),
      token = token,
      endpoint_url = endpoint_url
    )
    
    result <- httr2::resp_body_json(resp)
    
    # Result is a list of predictions
    if (is.list(result) && length(result) > 0) {
      predictions <- purrr::map_dfr(result, function(pred) {
        filled_text <- gsub(mask_token, pred$token_str %||% "", single_text, fixed = TRUE)
        
        tibble::tibble(
          text = single_text,
          token = pred$token_str %||% NA_character_,
          score = pred$score %||% NA_real_,
          filled = filled_text
        )
      })
      predictions
    } else {
      tibble::tibble(
        text = single_text,
        token = NA_character_,
        score = NA_real_,
        filled = NA_character_
      )
    }
  })
  
  results
}

#' Text Generation
#'
#' Generate text continuation from a prompt using a language model.
#'
#' @param prompt Character vector of text prompt(s) to continue.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "gpt2".
#' @param max_new_tokens Integer. Maximum number of tokens to generate. Default: 50.
#' @param temperature Numeric. Sampling temperature (0-2). Default: 1.0.
#' @param top_p Numeric. Nucleus sampling parameter. Default: NULL.
#' @param top_k Integer. Top-k sampling parameter. Default: NULL.
#' @param return_full_text Logical. Return prompt + generated text. Default: FALSE.
#' @param num_return_sequences Integer. Number of sequences to generate. Default: 1.
#' @param token Character string or NULL. API token for authentication.
#' @param ... Additional parameters passed to the model.
#'
#' @returns A tibble with columns: completion_id, prompt, generated_text
#' @export
#'
#' @examples
#' \dontrun{
#' # Simple text generation
#' hf_generate("Once upon a time in a land far away,")
#'
#' # Multiple completions
#' hf_generate("The future of AI is", num_return_sequences = 3)
#' }
hf_generate <- function(prompt,
                        model = "gpt2",
                        max_new_tokens = 50,
                        temperature = 1.0,
                        top_p = NULL,
                        top_k = NULL,
                        return_full_text = FALSE,
                        num_return_sequences = 1,
                        token = NULL,
                        ...) {
  
  if (length(prompt) == 0 || all(is.na(prompt))) {
    return(tibble::tibble(
      completion_id = integer(),
      prompt = character(),
      generated_text = character()
    ))
  }
  
  # Process each prompt
  results <- purrr::map_dfr(prompt, function(single_prompt) {
    if (is.na(single_prompt)) {
      return(tibble::tibble(
        completion_id = 1L,
        prompt = single_prompt,
        generated_text = NA_character_
      ))
    }
    
    # Build parameters
    params <- list(
      max_new_tokens = max_new_tokens,
      temperature = temperature,
      return_full_text = return_full_text,
      num_return_sequences = num_return_sequences,
      ...
    )
    if (!is.null(top_p)) params$top_p <- top_p
    if (!is.null(top_k)) params$top_k <- top_k
    
    resp <- hf_api_request(
      model_id = model,
      inputs = single_prompt,
      parameters = params,
      token = token
    )
    
    result <- httr2::resp_body_json(resp)
    
    # Handle response format
    if (is.list(result) && length(result) > 0) {
      completions <- purrr::map_dfr(seq_along(result), function(i) {
        tibble::tibble(
          completion_id = i,
          prompt = single_prompt,
          generated_text = result[[i]]$generated_text %||% ""
        )
      })
      completions
    } else {
      tibble::tibble(
        completion_id = 1L,
        prompt = single_prompt,
        generated_text = NA_character_
      )
    }
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
#'   Default: "bert-base-uncased".
#' @param mask_token Character string. The mask token to use. Default: "[MASK]".
#'   Some models use different tokens like "<mask>".
#' @param top_k Integer. Number of top predictions to return. Default: 5.
#' @param token Character string or NULL. API token for authentication.
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
                         model = "bert-base-uncased",
                         mask_token = "[MASK]",
                         top_k = 5,
                         token = NULL,
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
      token = token
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

#' Text Generation
#'
#' Generate text from a prompt using a language model via the Inference Providers API.
#'
#' @param prompt Character vector of text prompt(s) to generate from.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "HuggingFaceTB/SmolLM3-3B".
#' @param max_new_tokens Integer. Maximum number of tokens to generate. Default: 50.
#' @param temperature Numeric. Sampling temperature (0-2). Default: 1.0.
#' @param top_p Numeric. Nucleus sampling parameter. Default: NULL.
#' @param token Character string or NULL. API token for authentication.
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
                        model = "HuggingFaceTB/SmolLM3-3B",
                        max_new_tokens = 50,
                        temperature = 1.0,
                        top_p = NULL,
                        token = NULL,
                        ...) {

  if (length(prompt) == 0 || all(is.na(prompt))) {
    return(tibble::tibble(
      prompt = character(),
      generated_text = character()
    ))
  }

  token <- hf_get_token(token, required = TRUE)

  # Process each prompt
  results <- purrr::map_dfr(prompt, function(single_prompt) {
    if (is.na(single_prompt)) {
      return(tibble::tibble(
        prompt = single_prompt,
        generated_text = NA_character_
      ))
    }

    # Build request body for chat completions
    body <- list(
      model = model,
      messages = list(
        list(role = "user", content = single_prompt)
      ),
      max_tokens = max_new_tokens,
      temperature = temperature
    )
    if (!is.null(top_p)) body$top_p <- top_p

    dots <- list(...)
    if (length(dots) > 0) body <- c(body, dots)

    resp <- httr2::request("https://router.huggingface.co/v1/chat/completions") |>
      httr2::req_auth_bearer_token(token) |>
      httr2::req_body_json(body) |>
      httr2::req_retry(max_tries = 3) |>
      httr2::req_error(body = function(resp) {
        result <- tryCatch(
          httr2::resp_body_json(resp),
          error = function(e) list(error = list(message = httr2::resp_body_string(resp)))
        )
        paste0("API error: ", result$error$message %||% result$error %||% "Unknown error")
      }) |>
      httr2::req_perform()

    result <- httr2::resp_body_json(resp)
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
                         model = "google-bert/bert-base-uncased",
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

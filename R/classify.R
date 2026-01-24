#' Text Classification
#'
#' Classify text using a Hugging Face model. Commonly used for sentiment analysis,
#' topic classification, etc.
#'
#' @param text Character vector of text(s) to classify.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "distilbert/distilbert-base-uncased-finetuned-sst-2-english" (sentiment analysis).
#' @param token Character string or NULL. API token for authentication.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: text, label, score
#' @export
#'
#' @examples
#' \dontrun{
#' # Sentiment analysis
#' hf_classify("I love R programming!")
#'
#' # Multiple texts
#' hf_classify(c("This is great!", "This is terrible."))
#'
#' # Use in a pipeline
#' library(dplyr)
#' reviews |>
#'   mutate(sentiment = hf_classify(review_text)) |>
#'   unnest(sentiment)
#' }
hf_classify <- function(text, 
                        model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                        token = NULL,
                        ...) {
  
  if (length(text) == 0 || all(is.na(text))) {
    return(tibble::tibble(text = character(), label = character(), score = numeric()))
  }
  
  # Process each text individually to get proper results
  results <- purrr::map_dfr(text, function(single_text) {
    if (is.na(single_text)) {
      return(tibble::tibble(text = single_text, label = NA_character_, score = NA_real_))
    }
    
    resp <- hf_api_request(
      model_id = model,
      inputs = single_text,
      token = token
    )
    
    result <- httr2::resp_body_json(resp)
    
    # The API returns a list of lists for classification
    # [[1]][[1]]$label, [[1]][[1]]$score, etc.
    if (is.list(result) && length(result) > 0) {
      # Get first classification result (highest score)
      classification <- result[[1]][[1]]
      
      tibble::tibble(
        text = single_text,
        label = classification$label %||% NA_character_,
        score = classification$score %||% NA_real_
      )
    } else {
      tibble::tibble(
        text = single_text,
        label = NA_character_,
        score = NA_real_
      )
    }
  })
  
  results
}


#' Zero-Shot Classification
#'
#' Classify text into custom categories without training a model.
#' The model determines which labels best describe the input text.
#'
#' @param text Character vector of text(s) to classify.
#' @param labels Character vector of candidate labels/categories.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "facebook/bart-large-mnli"
#' @param multi_label Logical. If TRUE, allows multiple labels per text.
#'   Default: FALSE (single label per text).
#' @param token Character string or NULL. API token for authentication.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: text, label, score (sorted by score descending)
#' @export
#'
#' @examples
#' \dontrun{
#' # Classify into custom categories
#' hf_classify_zero_shot(
#'   "I just bought a new laptop",
#'   labels = c("technology", "sports", "politics", "food")
#' )
#'
#' # Multi-label classification
#' hf_classify_zero_shot(
#'   "This laptop is great for gaming",
#'   labels = c("technology", "gaming", "entertainment"),
#'   multi_label = TRUE
#' )
#' }
hf_classify_zero_shot <- function(text,
                                   labels,
                                   model = "facebook/bart-large-mnli",
                                   multi_label = FALSE,
                                   token = NULL,
                                   ...) {
  
  if (length(text) == 0 || all(is.na(text))) {
    return(tibble::tibble(text = character(), label = character(), score = numeric()))
  }
  
  if (length(labels) == 0) {
    stop("At least one label must be provided", call. = FALSE)
  }
  
  # Process each text
  results <- purrr::map_dfr(text, function(single_text) {
    if (is.na(single_text)) {
      return(tibble::tibble(
        text = single_text,
        label = rep(NA_character_, length(labels)),
        score = rep(NA_real_, length(labels))
      ))
    }
    
    resp <- hf_api_request(
      model_id = model,
      inputs = single_text,
      parameters = list(
        candidate_labels = labels,
        multi_label = multi_label
      ),
      token = token
    )
    
    result <- httr2::resp_body_json(resp)

    # Zero-shot returns: [{label: ..., score: ...}, ...]
    if (is.list(result) && length(result) > 0 &&
        !is.null(result[[1]]$label)) {
      purrr::map_dfr(result, function(item) {
        tibble::tibble(
          text = single_text,
          label = item$label %||% NA_character_,
          score = item$score %||% NA_real_
        )
      })
    } else if (!is.null(result$labels) && !is.null(result$scores)) {
      # Legacy format: {labels: [...], scores: [...]}
      tibble::tibble(
        text = single_text,
        label = unlist(result$labels),
        score = unlist(result$scores)
      )
    } else {
      tibble::tibble(
        text = single_text,
        label = rep(NA_character_, length(labels)),
        score = rep(NA_real_, length(labels))
      )
    }
  })
  
  results
}

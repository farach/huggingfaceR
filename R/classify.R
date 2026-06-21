#' Text Classification
#'
#' Classify text using a Hugging Face model. Commonly used for sentiment analysis,
#' topic classification, etc. Vector inputs are sent in a single batched
#' Inference API request when possible, which is substantially faster than one
#' API request per text.
#'
#' @param text Character vector of text(s) to classify.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "distilbert/distilbert-base-uncased-finetuned-sst-2-english" (sentiment analysis).
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#'   When provided, requests are sent to this URL instead of the public
#'   Inference API. Use for models deployed on dedicated Inference Endpoints.
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
                        model = hf_default_model("classify"),
                        token = NULL,
                        endpoint_url = NULL,
                        ...) {
  
  if (length(text) == 0) {
    return(tibble::tibble(text = character(), label = character(), score = numeric()))
  }
  
  result <- tibble::tibble(
    text = text,
    label = rep(NA_character_, length(text)),
    score = rep(NA_real_, length(text))
  )

  valid_idx <- which(!is.na(text))
  if (length(valid_idx) == 0) {
    return(result)
  }

  batch_result <- hf_classify_batch(
    text = text[valid_idx],
    model = model,
    token = token,
    batch_size = length(valid_idx),
    max_active = 1L,
    progress = FALSE,
    endpoint_url = endpoint_url
  )

  if (any(batch_result$.error)) {
    stop(batch_result$.error_msg[which(batch_result$.error)[1]], call. = FALSE)
  }

  result$label[valid_idx] <- batch_result$label
  result$score[valid_idx] <- batch_result$score
  result
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
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
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
                                   model = hf_default_model("zero_shot"),
                                   multi_label = FALSE,
                                   token = NULL,
                                   endpoint_url = NULL,
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
      token = token,
      endpoint_url = endpoint_url
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

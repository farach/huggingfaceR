#' Generate Text Embeddings
#'
#' Generate dense vector representations (embeddings) for text using transformer models.
#' Useful for semantic similarity, clustering, and as features for ML models.
#'
#' @param text Character vector of text(s) to embed.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "BAAI/bge-small-en-v1.5" (384-dim embeddings).
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#'   When provided, requests are sent to this URL instead of the public
#'   Inference API. Use for models deployed on dedicated Inference Endpoints.
#' @param ... Additional arguments (currently unused).
#'
#' @returns A tibble with columns: text, embedding (list-column of numeric vectors), n_dims
#' @export
#'
#' @examples
#' \dontrun{
#' # Generate embeddings
#' embeddings <- hf_embed(c("Hello world", "Goodbye world"))
#'
#' # Access embedding vectors
#' embeddings$embedding[[1]]  # First embedding vector
#' }
hf_embed <- function(text,
                     model = "BAAI/bge-small-en-v1.5",
                     token = NULL,
                     endpoint_url = NULL,
                     ...) {
  
  if (length(text) == 0) {
    return(tibble::tibble(text = character(), embedding = list(), n_dims = integer()))
  }
  
  result <- tibble::tibble(
    text = text,
    embedding = rep(list(NULL), length(text)),
    n_dims = rep(NA_integer_, length(text))
  )

  valid_idx <- which(!is.na(text))
  if (length(valid_idx) == 0) {
    return(result)
  }

  batch_result <- hf_embed_batch(
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

  result$embedding[valid_idx] <- batch_result$embedding
  result$n_dims[valid_idx] <- batch_result$n_dims
  result
}


#' Compute Pairwise Similarity
#'
#' Compute cosine similarity between all pairs of embeddings.
#'
#' @param embeddings A tibble with an 'embedding' column (from hf_embed).
#' @param text_col Character string. Name of the text column. Default: "text".
#'
#' @returns A tibble with columns: text_1, text_2, similarity
#' @export
#'
#' @examples
#' \dontrun{
#' sentences <- c("I love cats", "I adore felines", "Dogs are great")
#' embeddings <- hf_embed(sentences)
#' similarities <- hf_similarity(embeddings)
#' }
hf_similarity <- function(embeddings, text_col = "text") {
  
  if (!text_col %in% names(embeddings)) {
    stop(paste0("Column '", text_col, "' not found in embeddings"), call. = FALSE)
  }
  
  if (!"embedding" %in% names(embeddings)) {
    stop("Column 'embedding' not found. Use hf_embed() to generate embeddings first.",
         call. = FALSE)
  }
  
  n <- nrow(embeddings)
  if (n < 2) {
    return(tibble::tibble(text_1 = character(), text_2 = character(), similarity = numeric()))
  }
  
  pairs <- utils::combn(n, 2)
  similarity <- rep(NA_real_, ncol(pairs))
  emb_lengths <- lengths(embeddings$embedding)
  numeric_embedding <- vapply(embeddings$embedding, is.numeric, logical(1))
  valid <- numeric_embedding & emb_lengths > 0

  if (sum(valid) >= 2 && length(unique(emb_lengths[valid])) == 1) {
    valid_idx <- which(valid)
    emb_matrix <- do.call(rbind, embeddings$embedding[valid_idx])
    norms <- sqrt(rowSums(emb_matrix^2))
    non_zero <- norms > 0

    if (any(non_zero)) {
      sim_matrix <- tcrossprod(emb_matrix) / outer(norms, norms)
      sim_matrix[!non_zero, ] <- NA_real_
      sim_matrix[, !non_zero] <- NA_real_

      valid_lookup <- integer(n)
      valid_lookup[valid_idx] <- seq_along(valid_idx)
      pair_valid <- valid[pairs[1, ]] & valid[pairs[2, ]]
      similarity[pair_valid] <- sim_matrix[
        cbind(valid_lookup[pairs[1, pair_valid]], valid_lookup[pairs[2, pair_valid]])
      ]
    }
  } else if (any(valid)) {
    similarity <- purrr::map2_dbl(pairs[1, ], pairs[2, ], function(i, j) {
      emb1 <- embeddings$embedding[[i]]
      emb2 <- embeddings$embedding[[j]]

      if (!is.numeric(emb1) || !is.numeric(emb2) || length(emb1) != length(emb2)) {
        return(NA_real_)
      }

      denom <- sqrt(sum(emb1^2)) * sqrt(sum(emb2^2))
      if (denom == 0) NA_real_ else sum(emb1 * emb2) / denom
    })
  }

  tibble::tibble(
    text_1 = embeddings[[text_col]][pairs[1, ]],
    text_2 = embeddings[[text_col]][pairs[2, ]],
    similarity = similarity
  )
}


#' Dimensionality Reduction with UMAP
#'
#' Reduce embedding dimensions to 2D using UMAP for visualization.
#' Requires the 'uwot' package to be installed.
#'
#' @param text Character vector of text(s) to embed and reduce.
#' @param model Character string. Model ID for generating embeddings.
#'   Default: "BAAI/bge-small-en-v1.5".
#' @param token Character string or NULL. API token for authentication.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#' @param n_neighbors Integer. UMAP n_neighbors parameter. Default: 15.
#' @param min_dist Numeric. UMAP min_dist parameter. Default: 0.1.
#' @param ... Additional arguments passed to uwot::umap().
#'
#' @returns A tibble with columns: text, umap_1, umap_2
#' @export
#'
#' @examples
#' \dontrun{
#' # Reduce and visualize
#' library(ggplot2)
#' texts <- c("cat", "dog", "kitten", "puppy", "car", "truck")
#' coords <- hf_embed_umap(texts)
#'
#' ggplot(coords, aes(umap_1, umap_2, label = text)) +
#'   geom_text() +
#'   theme_minimal()
#' }
hf_embed_umap <- function(text,
                          model = "BAAI/bge-small-en-v1.5",
                          token = NULL,
                          endpoint_url = NULL,
                          n_neighbors = 15,
                          min_dist = 0.1,
                          ...) {
  
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("Package 'uwot' is required for UMAP. Install it with: install.packages('uwot')",
         call. = FALSE)
  }
  
  # Generate embeddings
  embeddings <- hf_embed(text, model = model, token = token, endpoint_url = endpoint_url)
  
  # Filter out NULL embeddings
  valid_embeddings <- embeddings$embedding[!sapply(embeddings$embedding, is.null)]
  
  if (length(valid_embeddings) == 0) {
    stop("No valid embeddings generated. All inputs may be NA or invalid.",
         call. = FALSE)
  }
  
  # Convert list-column to matrix
  emb_matrix <- do.call(rbind, valid_embeddings)
  
  # Run UMAP
  umap_coords <- uwot::umap(
    emb_matrix,
    n_neighbors = n_neighbors,
    min_dist = min_dist,
    ...
  )
  
  # Return as tibble
  tibble::tibble(
    text = text,
    umap_1 = umap_coords[, 1],
    umap_2 = umap_coords[, 2]
  )
}

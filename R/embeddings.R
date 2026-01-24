#' Generate Text Embeddings
#'
#' Generate dense vector representations (embeddings) for text using transformer models.
#' Useful for semantic similarity, clustering, and as features for ML models.
#'
#' @param text Character vector of text(s) to embed.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "BAAI/bge-small-en-v1.5" (384-dim embeddings).
#' @param token Character string or NULL. API token for authentication.
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
                     ...) {
  
  if (length(text) == 0) {
    return(tibble::tibble(text = character(), embedding = list(), n_dims = integer()))
  }
  
  # Process each text
  results <- purrr::map_dfr(text, function(single_text) {
    if (is.na(single_text)) {
      return(tibble::tibble(
        text = single_text,
        embedding = list(NULL),
        n_dims = NA_integer_
      ))
    }
    
    resp <- hf_api_request(
      model_id = model,
      inputs = single_text,
      token = token
    )
    
    result <- httr2::resp_body_json(resp)

    # Embeddings are returned as a JSON array of numbers, which resp_body_json
    # parses as a list of single numerics. Convert to a numeric vector.
    emb_vec <- if (is.numeric(result)) {
      result
    } else if (is.list(result) && length(result) > 0 && is.numeric(result[[1]])) {
      unlist(result)
    } else {
      NULL
    }

    n_dims <- if (is.null(emb_vec)) NA_integer_ else length(emb_vec)

    tibble::tibble(
      text = single_text,
      embedding = list(emb_vec),
      n_dims = n_dims
    )
  })
  
  results
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
  
  # Compute all pairwise similarities
  results <- list()
  idx <- 1
  
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      emb1 <- embeddings$embedding[[i]]
      emb2 <- embeddings$embedding[[j]]
      
      if (is.null(emb1) || is.null(emb2)) {
        sim <- NA_real_
      } else {
        # Cosine similarity
        sim <- sum(emb1 * emb2) / (sqrt(sum(emb1^2)) * sqrt(sum(emb2^2)))
      }
      
      results[[idx]] <- tibble::tibble(
        text_1 = embeddings[[text_col]][i],
        text_2 = embeddings[[text_col]][j],
        similarity = sim
      )
      idx <- idx + 1
    }
  }
  
  dplyr::bind_rows(results)
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
                          n_neighbors = 15,
                          min_dist = 0.1,
                          ...) {
  
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("Package 'uwot' is required for UMAP. Install it with: install.packages('uwot')",
         call. = FALSE)
  }
  
  # Generate embeddings
  embeddings <- hf_embed(text, model = model, token = token)
  
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

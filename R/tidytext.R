#' Embed Text with Tidytext Compatibility
#'
#' Generate embeddings for text in a tidy data frame.
#' Designed to work seamlessly with tidytext workflows.
#'
#' @param data A data frame or tibble.
#' @param text_col Unquoted column name containing text to embed.
#' @param model Character string. Hugging Face model ID for embeddings.
#'   Default: "BAAI/bge-small-en-v1.5".
#' @param token Character string or NULL. API token for authentication.
#' @param keep_text Logical. Keep original text column? Default: TRUE.
#'
#' @returns The input data frame with added embedding and n_dims columns.
#' @export
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(tidytext)
#'
#' # Embed documents
#' docs <- tibble(
#'   doc_id = 1:3,
#'   text = c("I love R", "Python is great", "Julia is fast")
#' )
#'
#' docs_embedded <- docs |>
#'   hf_embed_text(text)
#'
#' # Find similar documents
#' docs_embedded |>
#'   hf_nearest_neighbors("I love R", k = 2)
#' }
hf_embed_text <- function(data, text_col, 
                          model = "BAAI/bge-small-en-v1.5",
                          token = NULL,
                          keep_text = TRUE) {
  
  text_col_name <- rlang::as_name(rlang::enquo(text_col))
  
  if (!text_col_name %in% names(data)) {
    stop(paste0("Column '", text_col_name, "' not found in data"), call. = FALSE)
  }
  
  # Generate embeddings
  embeddings <- hf_embed(
    text = data[[text_col_name]],
    model = model,
    token = token
  )
  
  # Add to data
  result <- data
  result$embedding <- embeddings$embedding
  result$n_dims <- embeddings$n_dims
  
  if (!keep_text) {
    result[[text_col_name]] <- NULL
  }
  
  result
}


#' Find Nearest Neighbors by Semantic Similarity
#'
#' Find the k most similar texts to a query text based on embedding similarity.
#'
#' @param data A data frame with an 'embedding' column (from hf_embed_text).
#' @param query Character string. The query text to compare against.
#' @param k Integer. Number of nearest neighbors to return. Default: 5.
#' @param text_col Character string. Name of text column. Default: "text".
#' @param model Character string. Model to use for query embedding.
#'   Should match the model used for data embeddings.
#' @param token Character string or NULL. API token for authentication.
#'
#' @returns A tibble with the k nearest neighbors, sorted by similarity (descending).
#' @export
#'
#' @examples
#' \dontrun{
#' docs_embedded |>
#'   hf_nearest_neighbors("machine learning", k = 5)
#' }
hf_nearest_neighbors <- function(data,
                                 query,
                                 k = 5,
                                 text_col = "text",
                                 model = "BAAI/bge-small-en-v1.5",
                                 token = NULL) {
  
  if (!"embedding" %in% names(data)) {
    stop("Data must have an 'embedding' column. Use hf_embed_text() first.",
         call. = FALSE)
  }
  
  # Embed query
  query_emb <- hf_embed(query, model = model, token = token)$embedding[[1]]
  
  # Compute similarities
  similarities <- purrr::map_dbl(data$embedding, function(emb) {
    if (is.null(emb) || is.null(query_emb)) {
      return(NA_real_)
    }
    # Cosine similarity
    sum(emb * query_emb) / (sqrt(sum(emb^2)) * sqrt(sum(query_emb^2)))
  })
  
  # Add similarity column and sort
  result <- data
  result$similarity <- similarities
  
  result |>
    dplyr::arrange(dplyr::desc(similarity)) |>
    dplyr::slice_head(n = k)
}


#' Cluster Texts by Semantic Similarity
#'
#' Perform k-means clustering on text embeddings.
#'
#' @param data A data frame with an 'embedding' column (from hf_embed_text).
#' @param k Integer. Number of clusters. Default: 3.
#' @param ... Additional arguments passed to stats::kmeans().
#'
#' @returns The input data frame with an added 'cluster' column.
#' @export
#'
#' @examples
#' \dontrun{
#' library(ggplot2)
#'
#' # Cluster documents
#' docs_clustered <- docs_embedded |>
#'   hf_cluster_texts(k = 3)
#'
#' # Reduce dimensions and visualize
#' library(uwot)
#' emb_matrix <- do.call(rbind, docs_clustered$embedding)
#' coords <- umap(emb_matrix)
#'
#' docs_clustered |>
#'   mutate(umap_1 = coords[, 1], umap_2 = coords[, 2]) |>
#'   ggplot(aes(umap_1, umap_2, color = factor(cluster))) +
#'   geom_point(size = 3)
#' }
hf_cluster_texts <- function(data, k = 3, ...) {
  
  if (!"embedding" %in% names(data)) {
    stop("Data must have an 'embedding' column. Use hf_embed_text() first.",
         call. = FALSE)
  }
  
  # Filter out NULL embeddings
  valid_idx <- !sapply(data$embedding, is.null)
  
  if (sum(valid_idx) == 0) {
    stop("No valid embeddings found. All embeddings are NULL.",
         call. = FALSE)
  }
  
  # Convert embeddings to matrix (only valid ones)
  emb_matrix <- do.call(rbind, data$embedding[valid_idx])
  
  # Perform k-means clustering
  clusters <- stats::kmeans(emb_matrix, centers = k, ...)
  
  # Add cluster assignments (only to valid rows)
  data$cluster <- NA_integer_
  data$cluster[valid_idx] <- clusters$cluster
  
  data
}


#' Extract Semantic Topics from Text
#'
#' Identify semantic topics by clustering embeddings and extracting
#' representative keywords from each cluster.
#'
#' @param data A data frame with text and embeddings.
#' @param text_col Character string. Name of text column.
#' @param k Integer. Number of topics/clusters. Default: 5.
#' @param top_n Integer. Number of top words per topic. Default: 10.
#'
#' @returns A tibble with topics and their top terms.
#' @export
#'
#' @examples
#' \dontrun{
#' library(tidytext)
#'
#' # Extract topics
#' topics <- docs_embedded |>
#'   hf_extract_topics(text_col = "text", k = 3, top_n = 5)
#' }
hf_extract_topics <- function(data,
                              text_col = "text",
                              k = 5,
                              top_n = 10) {
  
  if (!requireNamespace("tidytext", quietly = TRUE)) {
    stop("Package 'tidytext' is required. Install it with: install.packages('tidytext')",
         call. = FALSE)
  }
  
  # Cluster texts
  clustered <- hf_cluster_texts(data, k = k)
  
  # For each cluster, find most common words
  topics <- clustered |>
    dplyr::select(dplyr::all_of(c(text_col, "cluster"))) |>
    tidytext::unnest_tokens(word, dplyr::all_of(text_col)) |>
    dplyr::count(cluster, word, sort = TRUE) |>
    dplyr::group_by(cluster) |>
    dplyr::slice_head(n = top_n) |>
    dplyr::summarise(
      topic_terms = paste(word, collapse = ", "),
      .groups = "drop"
    )
  
  topics
}

#' Generate Text Embeddings
#'
#' Generate dense vector representations (embeddings) for text using transformer models.
#' Useful for semantic similarity, clustering, and as features for ML models.
#' Vector inputs are sent in a single batched Inference API request when possible,
#' which is substantially faster than one API request per text.
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
                     model = hf_default_model("embed"),
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
#' Compute cosine similarity between all pairs of embeddings. Embeddings are
#' grouped by dimension and each group is scored with one matrix
#' cross-product, so the work is done by compiled linear algebra rather than
#' by R code for each pair. Invalid, zero-length, zero-norm, non-finite, or
#' dimension-mismatched pairs return `NA_real_`.
#'
#' @param embeddings A tibble with an 'embedding' column (from hf_embed).
#' @param text_col Character string. Name of the text column. Default: "text".
#'
#' @returns A tibble with columns: text_1, text_2, similarity.
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

  # Every i < j pair, in the order utils::combn(n, 2) produces, built with
  # vectorized primitives instead of an interpreted loop over the pairs.
  first <- rep.int(seq_len(n - 1L), (n - 1L):1L)
  second <- sequence((n - 1L):1L, from = 2L:n)

  tibble::tibble(
    text_1 = embeddings[[text_col]][first],
    text_2 = embeddings[[text_col]][second],
    similarity = hf_pairwise_cosine(embeddings$embedding)
  )
}


# Cosine similarity for every i < j pair of `embeddings`, in utils::combn()
# order.
#
# Embeddings are grouped by dimension. Each group is normalized once and scored
# with a single BLAS cross-product. Every column of that symmetric matrix then
# fills one contiguous block of pairs, so no R code runs per pair. Pairs that
# cannot be compared (missing, non-numeric, zero-length, zero-norm,
# non-finite, or of different dimensions) are NA.
hf_pairwise_cosine <- function(embeddings) {
  n <- length(embeddings)
  similarity <- rep(NA_real_, n * (n - 1) / 2)
  sizes <- lengths(embeddings)
  comparable <- sizes > 0L & vapply(embeddings, is.numeric, logical(1))

  for (size in unique(sizes[comparable])) {
    members <- which(comparable & sizes == size)
    unit <- hf_unit_columns(embeddings[members], size)
    members <- members[unit$usable]
    k <- length(members)
    if (k < 2L) {
      next
    }

    scores <- crossprod(unit$matrix)
    for (a in seq_len(k - 1L)) {
      i <- members[a]
      later <- (a + 1L):k
      # Pair (i, j) sits at position (i - 1) * n - (i - 1) * i / 2 + (j - i).
      offset <- (i - 1) * n - (i - 1) * i / 2 - i
      similarity[offset + members[later]] <- scores[later, a]
    }
  }

  similarity
}


# Cosine similarity between each embedding and one query vector.
#
# Embeddings with the query's dimension are scored in blocks (by default about
# 2^22 values each) with a BLAS matrix-vector product, which keeps temporary
# memory bounded for large corpora. Missing, non-numeric, zero-norm,
# non-finite, or dimension-mismatched embeddings score NA.
hf_cosine_to_query <- function(embeddings, query, block_size = NULL) {
  similarity <- rep(NA_real_, length(embeddings))
  size <- length(query)
  if (!is.numeric(query) || size == 0L) {
    return(similarity)
  }

  query <- as.double(query)
  query_norm <- sqrt(sum(query * query))
  candidates <- which(
    lengths(embeddings) == size & vapply(embeddings, is.numeric, logical(1))
  )
  if (!is.finite(query_norm) || query_norm == 0 || length(candidates) == 0L) {
    return(similarity)
  }

  block_size <- block_size %||% max(1L, as.integer(2^22 %/% size))
  for (start in seq.int(1L, length(candidates), by = block_size)) {
    rows <- candidates[start:min(start + block_size - 1L, length(candidates))]
    unit <- hf_unit_columns(embeddings[rows], size)
    similarity[rows[unit$usable]] <- drop(crossprod(unit$matrix, query)) / query_norm
  }

  similarity
}


# Stacks equal-length numeric vectors as the columns of a matrix scaled to unit
# length. Vectors with a zero or non-finite norm cannot be normalized, so they
# are dropped and flagged as FALSE in `usable`.
hf_unit_columns <- function(vectors, size) {
  mat <- matrix(as.double(unlist(vectors, use.names = FALSE)), nrow = size)
  norms <- sqrt(colSums(mat * mat))
  usable <- is.finite(norms) & norms > 0
  if (!all(usable)) {
    mat <- mat[, usable, drop = FALSE]
    norms <- norms[usable]
  }
  list(matrix = mat / rep(norms, each = size), usable = usable)
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
                          model = hf_default_model("embed"),
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

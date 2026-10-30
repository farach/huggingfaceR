test_that("hf_embed returns tibble with embeddings", {
  skip_on_cran()
  
  # Test with empty input
  result <- hf_embed(character())
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)
  expect_true("embedding" %in% names(result))
})

test_that("hf_embed handles NA values", {
  skip_on_cran()
  
  result <- hf_embed(NA_character_)
  expect_s3_class(result, "tbl_df")
  expect_true(is.na(result$text[1]))
  expect_true(is.null(result$embedding[[1]]))
})

test_that("hf_embed batches non-missing inputs", {
  skip_on_cran()

  calls <- 0L
  testthat::local_mocked_bindings(
    hf_embed_batch = function(text, model, token, batch_size, max_active, progress, endpoint_url) {
      calls <<- calls + 1L
      expect_equal(text, c("a", "b", "c"))
      expect_equal(batch_size, 3L)
      expect_equal(max_active, 1L)
      expect_false(progress)

      tibble::tibble(
        text = text,
        embedding = list(c(1, 0), c(0, 1), c(1, 1)),
        n_dims = c(2L, 2L, 2L),
        .input_idx = seq_along(text),
        .error = FALSE,
        .error_msg = NA_character_
      )
    }
  )

  result <- hf_embed(c("a", NA, "b", "c"))

  expect_equal(calls, 1L)
  expect_equal(result$text, c("a", NA, "b", "c"))
  expect_equal(result$n_dims, c(2L, NA_integer_, 2L, 2L))
  expect_true(is.null(result$embedding[[2]]))
})

test_that("hf_similarity requires embedding column", {
  skip_on_cran()
  
  df <- tibble::tibble(text = c("a", "b"))
  
  expect_error(
    hf_similarity(df),
    "Column 'embedding' not found"
  )
})

test_that("hf_similarity returns correct structure", {
  skip_on_cran()
  
  df <- tibble::tibble(
    text = c("a", "b"),
    embedding = list(c(1, 2, 3), c(4, 5, 6))
  )
  
  result <- hf_similarity(df)
  
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 1)  # n*(n-1)/2 = 2*1/2 = 1
  expect_true(all(c("text_1", "text_2", "similarity") %in% names(result)))
})

test_that("hf_similarity preserves pairwise results with NULL embeddings", {
  skip_on_cran()

  df <- tibble::tibble(
    text = c("a", "b", "c", "d"),
    embedding = list(c(1, 0), c(0, 1), NULL, c("x", "y"))
  )

  result <- hf_similarity(df)

  expect_equal(result$text_1, c("a", "a", "a", "b", "b", "c"))
  expect_equal(result$text_2, c("b", "c", "d", "c", "d", "d"))
  expect_equal(result$similarity, c(0, NA_real_, NA_real_, NA_real_, NA_real_, NA_real_))
})

# Reference cosine similarity for one pair, written directly from the
# definition so the vectorized implementation is checked against it.
naive_cosine <- function(a, b) {
  usable <- function(x) {
    is.numeric(x) && length(x) > 0 && all(is.finite(x)) && sum(x^2) > 0
  }
  if (!usable(a) || !usable(b) || length(a) != length(b)) {
    return(NA_real_)
  }
  sum(a * b) / sqrt(sum(a^2) * sum(b^2))
}

test_that("hf_similarity matches a direct pairwise computation in combn order", {
  set.seed(20)
  vectors <- lapply(seq_len(25), function(i) stats::rnorm(6))
  df <- tibble::tibble(text = sprintf("t%02d", seq_along(vectors)), embedding = vectors)

  result <- hf_similarity(df)

  pairs <- utils::combn(length(vectors), 2)
  expected <- apply(pairs, 2, function(p) naive_cosine(vectors[[p[1]]], vectors[[p[2]]]))
  expect_equal(result$text_1, df$text[pairs[1, ]])
  expect_equal(result$text_2, df$text[pairs[2, ]])
  expect_equal(result$similarity, expected, tolerance = 1e-12)
})

test_that("hf_similarity scores only comparable pairs across mixed dimensions", {
  vectors <- list(
    c(1, 0), c(0, 1, 0), c(3, 4), NULL, c(0, 0), c(2, 0), c(1, 1, 0),
    c(NA, 1), "x", integer(0), c(1L, 2L), c(Inf, 1)
  )
  df <- tibble::tibble(text = as.character(seq_along(vectors)), embedding = vectors)

  result <- hf_similarity(df)

  pairs <- utils::combn(length(vectors), 2)
  expected <- apply(pairs, 2, function(p) naive_cosine(vectors[[p[1]]], vectors[[p[2]]]))
  expect_equal(nrow(result), ncol(pairs))
  expect_equal(result$text_1, df$text[pairs[1, ]])
  expect_equal(result$similarity, expected, tolerance = 1e-12)
})

test_that("hf_nearest_neighbors ranks rows by cosine similarity to the query", {
  testthat::local_mocked_bindings(
    hf_embed = function(text, model, token, ...) {
      tibble::tibble(text = text, embedding = list(c(1, 0, 0)), n_dims = 3L)
    }
  )
  data <- tibble::tibble(
    text = c("same", "orthogonal", "missing", "close", "wrong-size", "zero", "opposite"),
    embedding = list(
      c(2, 0, 0), c(0, 1, 0), NULL, c(1, 1, 0), c(1, 0), c(0, 0, 0), c(-1, 0, 0)
    )
  )

  result <- hf_nearest_neighbors(data, "query", k = 7)

  expect_equal(
    result$text,
    c("same", "close", "orthogonal", "opposite", "missing", "wrong-size", "zero")
  )
  expect_equal(
    result$similarity,
    c(1, 1 / sqrt(2), 0, -1, NA, NA, NA),
    tolerance = 1e-12
  )
  expect_equal(nrow(hf_nearest_neighbors(data, "query", k = 2)), 2)
})

test_that("hf_cosine_to_query gives the same scores for any block size", {
  set.seed(21)
  vectors <- lapply(seq_len(40), function(i) stats::rnorm(3))
  vectors[c(5, 17)] <- list(NULL, c(0, 0, 0))
  query <- c(0.5, -1, 2)
  expected <- vapply(vectors, function(v) naive_cosine(v, query), numeric(1))

  expect_equal(hf_cosine_to_query(vectors, query), expected, tolerance = 1e-12)
  expect_equal(
    hf_cosine_to_query(vectors, query, block_size = 7L), expected, tolerance = 1e-12
  )
  expect_equal(
    hf_cosine_to_query(vectors, query, block_size = 1L), expected, tolerance = 1e-12
  )
})

test_that("hf_cosine_to_query handles unusable queries and embeddings", {
  embeddings <- list(c(1, 0), NULL, c(0, 2), c(1, 1, 1), "a")

  expect_equal(
    hf_cosine_to_query(embeddings, c(1, 0)),
    c(1, NA, 0, NA, NA),
    tolerance = 1e-12
  )
  expect_equal(hf_cosine_to_query(embeddings, NULL), rep(NA_real_, 5))
  expect_equal(hf_cosine_to_query(embeddings, c(0, 0)), rep(NA_real_, 5))
  expect_equal(hf_cosine_to_query(list(), c(1, 0)), numeric())
})

test_that("hf_embed_umap requires uwot package", {
  skip_on_cran()
  
  # This test assumes uwot is not installed
  # If it is installed, this will pass differently
  if (!requireNamespace("uwot", quietly = TRUE)) {
    expect_error(
      hf_embed_umap(c("a", "b")),
      "uwot"
    )
  }
})

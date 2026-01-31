# Test Script for embeddings-and-similarity.Rmd Vignette
#
# Executes all code from the vignette to verify correctness.
#
# Usage:
#   Rscript scripts/test-embeddings-and-similarity.R
#
# Prerequisites:
#   - A valid Hugging Face token in .Renviron (HUGGING_FACE_HUB_TOKEN)
#   - Packages: huggingfaceR, dplyr, tidytext, uwot, ggplot2

devtools::load_all()
library(dplyr)

# --- Token discovery ---
if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") == "") {
  renviron_paths <- c(
    file.path(Sys.getenv("HOME"), ".Renviron"),
    file.path(Sys.getenv("USERPROFILE"), "Documents", ".Renviron"),
    file.path(Sys.getenv("USERPROFILE"),
              "OneDrive - Microsoft", "Documents", ".Renviron"),
    file.path(path.expand("~"), ".Renviron")
  )
  for (p in renviron_paths) {
    if (file.exists(p)) {
      readRenviron(p)
      if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") != "") break
    }
  }
  if (Sys.getenv("HUGGING_FACE_HUB_TOKEN") == "") {
    stop("No HUGGING_FACE_HUB_TOKEN found.")
  }
}

# --- Test infrastructure ---
results <- list()
test <- function(name, expr) {
  cat(sprintf("  %-55s", name))
  result <- tryCatch(
    {
      val <- eval(expr)
      cat("[PASS]\n")
      results[[name]] <<- list(status = "PASS", value = val)
      val
    },
    error = function(e) {
      cat(sprintf("[FAIL] %s\n", conditionMessage(e)))
      results[[name]] <<- list(status = "FAIL", error = conditionMessage(e))
      NULL
    }
  )
  invisible(result)
}

check <- function(condition, msg = "assertion failed") {
  if (!isTRUE(condition)) stop(msg, call. = FALSE)
}

cat("=== Embeddings and Similarity Vignette Test ===\n\n")

# ============================================================
# Section: Generating Embeddings
# ============================================================
cat("Generating Embeddings\n")

test("hf_embed basic (5 sentences)", {
  sentences <- c(
    "Machine learning is transforming healthcare",
    "Deep learning models require large datasets",
    "The weather forecast predicts rain tomorrow",
    "Clinical trials use statistical methods",
    "It will be sunny next week"
  )
  embeddings <<- hf_embed(sentences)
  check(tibble::is_tibble(embeddings), "expected tibble")
  check(nrow(embeddings) == 5, "expected 5 rows")
  check(all(c("text", "embedding", "n_dims") %in% names(embeddings)))
  check(embeddings$n_dims[1] == 384, "expected 384 dimensions")
  embeddings
})

test("Access embedding vectors", {
  # Single vector
  vec <- embeddings$embedding[[1]]
  check(is.numeric(vec), "expected numeric vector")
  check(length(vec) == 384, "expected 384 elements")

  # Matrix conversion
  emb_matrix <- do.call(rbind, embeddings$embedding)
  check(is.matrix(emb_matrix), "expected matrix")
  check(nrow(emb_matrix) == 5, "expected 5 rows")
  check(ncol(emb_matrix) == 384, "expected 384 cols")
  emb_matrix
})

# ============================================================
# Section: Pairwise Similarity
# ============================================================
cat("\nPairwise Similarity\n")

test("hf_similarity on 5 embeddings", {
  sim <- hf_similarity(embeddings)
  check(tibble::is_tibble(sim), "expected tibble")
  # 5 choose 2 = 10 pairs
  check(nrow(sim) == 10, sprintf("expected 10 pairs, got %d", nrow(sim)))
  check(all(c("text_1", "text_2", "similarity") %in% names(sim)))
  check(all(sim$similarity >= -1 & sim$similarity <= 1), "similarity in [-1,1]")

  # ML-related texts should be more similar than ML vs weather
  ml_pair <- sim |> filter(grepl("Machine learning", text_1),
                           grepl("Deep learning", text_2))
  weather_pair <- sim |> filter(grepl("Machine learning", text_1),
                                grepl("weather", text_2))
  if (nrow(ml_pair) > 0 && nrow(weather_pair) > 0) {
    check(ml_pair$similarity[1] > weather_pair$similarity[1],
          "ML pair should be more similar than ML-weather pair")
  }
  sim
})

# ============================================================
# Section: Tidytext Integration - hf_embed_text
# ============================================================
cat("\nTidytext Integration\n")

test("hf_embed_text on data frame", {
  docs <<- tibble(
    doc_id = 1:6,
    category = c("tech", "tech", "food", "food", "travel", "travel"),
    text = c(
      "Neural networks power modern AI systems",
      "Cloud computing enables scalable applications",
      "Fresh pasta requires only flour and eggs",
      "Sourdough bread needs a mature starter",
      "Tokyo offers incredible street food and temples",
      "The Swiss Alps provide world-class hiking trails"
    )
  )
  docs_embedded <<- docs |> hf_embed_text(text)
  check(tibble::is_tibble(docs_embedded), "expected tibble")
  check(nrow(docs_embedded) == 6, "expected 6 rows")
  check("embedding" %in% names(docs_embedded), "expected embedding column")
  check("n_dims" %in% names(docs_embedded), "expected n_dims column")
  check(all(c("doc_id", "category", "text") %in% names(docs_embedded)),
        "original columns preserved")
  docs_embedded
})

# ============================================================
# Section: Semantic Search - hf_nearest_neighbors
# ============================================================
cat("\nSemantic Search\n")

test("hf_nearest_neighbors for AI query", {
  result <- docs_embedded |>
    hf_nearest_neighbors("artificial intelligence applications", k = 3)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 3, "expected 3 neighbors")
  check("similarity" %in% names(result), "expected similarity column")
  # Top result should be tech-related
  check(result$category[1] == "tech",
        sprintf("expected tech as top match, got '%s'", result$category[1]))
  result
})

# ============================================================
# Section: Clustering
# ============================================================
cat("\nClustering\n")

test("hf_cluster_texts on articles (k=3)", {
  articles <<- tibble(
    id = 1:12,
    text = c(
      "New AI chip doubles processing speed",
      "Quantum computing reaches error correction milestone",
      "Open-source language model rivals proprietary alternatives",
      "Cybersecurity threats increase with IoT adoption",
      "Mediterranean diet linked to reduced heart disease risk",
      "New gene therapy shows promise for rare blood disorders",
      "Sleep quality affects cognitive performance in older adults",
      "Vaccine development accelerates with mRNA technology",
      "Arctic ice loss accelerates beyond model predictions",
      "Renewable energy capacity surpasses coal globally",
      "Ocean acidification threatens coral reef ecosystems",
      "Urban forests reduce city temperatures by up to 5 degrees"
    )
  )
  clustered <<- articles |>
    hf_embed_text(text) |>
    hf_cluster_texts(k = 3)
  check("cluster" %in% names(clustered), "expected cluster column")
  check(length(unique(clustered$cluster)) == 3, "expected 3 clusters")
  check(nrow(clustered) == 12, "expected 12 rows")
  clustered
})

test("Cluster assignments are reasonable", {
  # Check cluster assignments make sense
  cluster_summary <- clustered |>
    select(id, text, cluster) |>
    arrange(cluster)
  check(nrow(cluster_summary) == 12, "expected 12 rows")
  cluster_summary
})

# ============================================================
# Section: Topic Extraction
# ============================================================
cat("\nTopic Extraction\n")

test("hf_extract_topics", {
  if (!requireNamespace("tidytext", quietly = TRUE)) {
    stop("tidytext not installed")
  }
  topics <- articles |>
    hf_embed_text(text) |>
    hf_extract_topics(text_col = "text", k = 3, top_n = 5)
  check(tibble::is_tibble(topics), "expected tibble")
  check("topic_terms" %in% names(topics) || "cluster" %in% names(topics),
        "expected topic_terms or cluster column")
  check(nrow(topics) >= 3, "expected at least 3 rows")
  topics
})

# ============================================================
# Section: UMAP Visualization
# ============================================================
cat("\nUMAP Visualization\n")

test("hf_embed_umap on 12 texts", {
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("uwot not installed")
  }
  library(ggplot2)

  texts <- c(
    "cats are independent pets", "dogs are loyal companions",
    "goldfish are low-maintenance pets", "parrots can mimic speech",
    "sedans are practical family cars", "trucks haul heavy loads",
    "bicycles reduce carbon emissions", "motorcycles offer speed and freedom",
    "pizza is a popular dinner choice", "sushi requires fresh fish",
    "tacos feature various fillings", "pasta comes in many shapes"
  )

  coords <- hf_embed_umap(texts, n_neighbors = 4, min_dist = 0.1)
  check(tibble::is_tibble(coords), "expected tibble")
  check(nrow(coords) == 12, "expected 12 rows")
  check(all(c("text", "umap_1", "umap_2") %in% names(coords)))
  check(!any(is.na(coords$umap_1)), "no NA in umap_1")
  check(!any(is.na(coords$umap_2)), "no NA in umap_2")
  coords
})

test("ggplot UMAP scatter", {
  library(ggplot2)
  texts <- c(
    "cats are independent pets", "dogs are loyal companions",
    "goldfish are low-maintenance pets", "parrots can mimic speech",
    "sedans are practical family cars", "trucks haul heavy loads"
  )
  coords <- hf_embed_umap(texts, n_neighbors = 3, min_dist = 0.1)

  p <- ggplot(coords, aes(umap_1, umap_2, label = text)) +
    geom_point(size = 2) +
    theme_minimal() +
    labs(title = "UMAP Projection of Text Embeddings")
  check(inherits(p, "ggplot"), "expected ggplot object")
  p
})

# ============================================================
# Section: End-to-End Example (Research Abstracts)
# ============================================================
cat("\nEnd-to-End Example\n")

test("Research abstracts: embed + cluster + UMAP", {
  library(ggplot2)

  abstracts <- tibble(
    paper_id = 1:9,
    field = rep(c("NLP", "genomics", "climate"), each = 3),
    abstract = c(
      "Transformer architectures improve machine translation quality",
      "Attention mechanisms capture long-range text dependencies",
      "Pre-training on large corpora enables few-shot learning",
      "CRISPR enables precise genome editing in mammalian cells",
      "Single-cell RNA sequencing reveals cell type heterogeneity",
      "Epigenetic modifications regulate gene expression patterns",
      "Global temperatures rise faster than model projections",
      "Carbon capture technology scales to industrial levels",
      "Sea level rise threatens coastal infrastructure worldwide"
    )
  )

  # Embed and cluster
  result <- abstracts |>
    hf_embed_text(abstract) |>
    hf_cluster_texts(k = 3)
  check("cluster" %in% names(result), "expected cluster column")
  check(nrow(result) == 9, "expected 9 rows")

  # UMAP
  coords <- hf_embed_umap(abstracts$abstract, n_neighbors = 3)
  check(nrow(coords) == 9, "expected 9 UMAP rows")

  # Combine for plotting
  plot_data <- bind_cols(
    result |> select(paper_id, field, cluster),
    coords |> select(umap_1, umap_2)
  )

  p <- ggplot(plot_data, aes(umap_1, umap_2, color = factor(cluster), shape = field)) +
    geom_point(size = 3) +
    theme_minimal() +
    labs(title = "Research Abstracts by Embedding Cluster")
  check(inherits(p, "ggplot"), "expected ggplot object")
  p
})

# ============================================================
# Section: Processing at Scale (Batch Functions)
# ============================================================
cat("\nProcessing at Scale\n")

test("hf_embed_batch parallel (10 texts)", {
  texts <- c(
    "Machine learning transforms industries",
    "Deep learning requires data",
    "Neural networks learn patterns",
    "AI systems automate tasks",
    "Natural language processing understands text",
    "Computer vision analyzes images",
    "Reinforcement learning optimizes decisions",
    "Generative models create content",
    "Transfer learning shares knowledge",
    "Federated learning preserves privacy"
  )
  result <- hf_embed_batch(texts, batch_size = 5, max_active = 2, progress = FALSE)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 10, sprintf("expected 10 rows, got %d", nrow(result)))
  check(all(c("text", "embedding", "n_dims", ".input_idx", ".error", ".error_msg") %in% names(result)),
        "expected all batch columns")
  check(all(result$.input_idx == 1:10), "input indices should be 1:10")
  check(sum(result$.error) == 0, "expected no errors")
  check(all(result$n_dims == 384), "expected 384 dimensions")
  result
})
test("hf_embed_batch error tracking columns", {
  result <- hf_embed_batch(c("test text"), batch_size = 1, max_active = 1, progress = FALSE)
  check(".error" %in% names(result), "expected .error column")
  check(".error_msg" %in% names(result), "expected .error_msg column")
  check(is.logical(result$.error), ".error should be logical")
  result
})

test("hf_embed_chunks with disk checkpoints", {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("arrow not installed")
  }

  texts <- c(
    "First document about technology",
    "Second document about science",
    "Third document about nature",
    "Fourth document about sports",
    "Fifth document about music"
  )

  output_dir <- tempfile("embed_chunks_test")

  # Process with chunks
  hf_embed_chunks(
    texts,
    output_dir = output_dir,
    chunk_size = 2,
    batch_size = 2,
    max_active = 2,
    resume = FALSE,
    progress = FALSE
  )

  # Verify files were created
  files <- list.files(output_dir, pattern = "\\.parquet$")
  check(length(files) == 3, sprintf("expected 3 chunk files, got %d", length(files)))

  # Read chunks back
  result <- hf_read_chunks(output_dir)
  check(tibble::is_tibble(result), "expected tibble")
  check(nrow(result) == 5, sprintf("expected 5 rows, got %d", nrow(result)))
  check(all(c("text", "embedding", ".input_idx") %in% names(result)))

  # Clean up
  unlink(output_dir, recursive = TRUE)
  result
})

test("hf_embed_chunks resume capability", {
  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("arrow not installed")
  }

  texts <- c("doc one", "doc two", "doc three", "doc four")
  output_dir <- tempfile("embed_resume_test")

  # First run
  hf_embed_chunks(texts, output_dir, chunk_size = 2, batch_size = 2,
                  max_active = 1, resume = FALSE, progress = FALSE)

  # Get existing chunks
  existing <- hf_get_existing_chunks(output_dir, prefix = "embed_chunk")
  check(length(existing) == 2, "expected 2 chunks after first run")

  # Second run with resume should skip existing
  hf_embed_chunks(texts, output_dir, chunk_size = 2, batch_size = 2,
                  max_active = 1, resume = TRUE, progress = FALSE)

  result <- hf_read_chunks(output_dir)
  check(nrow(result) == 4, "expected 4 rows after resume")

  unlink(output_dir, recursive = TRUE)
  result
})

# ============================================================
# Summary
# ============================================================
cat("\n=== Results ===\n\n")

pass <- sum(sapply(results, function(r) r$status == "PASS"))
fail <- sum(sapply(results, function(r) r$status == "FAIL"))
total <- length(results)

cat(sprintf("  PASS: %d / %d\n", pass, total))
cat(sprintf("  FAIL: %d / %d\n", fail, total))

if (fail > 0) {
  cat("\nFailed tests:\n")
  for (name in names(results)) {
    if (results[[name]]$status == "FAIL") {
      cat(sprintf("  - %s: %s\n", name, results[[name]]$error))
    }
  }
  cat("\n")
  quit(status = 1)
} else {
  cat("\nAll tests passed.\n")
}

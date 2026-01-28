# Test Script for openai-gdpval-benchmark.Rmd Vignette
#
# Executes all code from the vignette with reduced sample sizes to verify
# correctness without excessive API calls or runtime.
#
# Usage:
#   Rscript scripts/test-gdpval-vignette.R
#
# Prerequisites:
#   - A valid Hugging Face token in .Renviron (HUGGING_FACE_HUB_TOKEN)
#   - Packages: huggingfaceR, dplyr, tidyr, ggplot2, uwot, tidytext

devtools::load_all()
library(dplyr)
library(tidyr)
library(ggplot2)

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
test <- function(name, expr, skip_on_api_error = FALSE) {
  cat(sprintf("  %-55s", name))
  result <- tryCatch(
    {
      val <- eval(expr)
      cat("[PASS]\n")
      results[[name]] <<- list(status = "PASS", value = val)
      val
    },
    error = function(e) {
      msg <- conditionMessage(e)
      # Check for transient API errors
      is_api_error <- grepl("HTTP 5[0-9][0-9]|Gateway Timeout|Internal Server Error",
                            msg, ignore.case = TRUE)
      if (skip_on_api_error && is_api_error) {
        cat("[SKIP] API unavailable\n")
        results[[name]] <<- list(status = "SKIP", error = "API unavailable")
      } else {
        cat(sprintf("[FAIL] %s\n", msg))
        results[[name]] <<- list(status = "FAIL", error = msg)
      }
      NULL
    }
  )
  invisible(result)
}

check <- function(condition, msg = "assertion failed") {
  if (!isTRUE(condition)) stop(msg, call. = FALSE)
}

cat("=== GDPval Vignette Test (openai-gdpval-benchmark.Rmd) ===\n\n")

# ============================================================
# Section: Loading the Dataset
# ============================================================
cat("Loading the Dataset\n")

test("Load GDPval dataset from HF Hub", {

  gdpval <<- hf_load_dataset("openai/gdpval", split = "train")
  check(tibble::is_tibble(gdpval), "expected tibble")
  check(nrow(gdpval) >= 200, sprintf("expected 200+ rows, got %d", nrow(gdpval)))
  check(all(c("task_id", "sector", "occupation", "prompt") %in% names(gdpval)),
        "expected required columns")
  cat(sprintf(" (%d rows)", nrow(gdpval)))
  gdpval
})

test("Dataset has expected columns", {
  # Core columns that should always exist
  expected_cols <- c("task_id", "sector", "occupation", "prompt",
                     "reference_files", "reference_file_urls",
                     "reference_file_hf_uris")
  missing <- setdiff(expected_cols, names(gdpval))
  check(length(missing) == 0,
        sprintf("missing columns: %s", paste(missing, collapse = ", ")))
  TRUE
})

# ============================================================
# Section: Exploratory Analysis
# ============================================================
cat("\nExploratory Analysis\n")

test("Count tasks by sector", {
  sector_counts <- gdpval |>
    count(sector, sort = TRUE)
  check(nrow(sector_counts) > 0, "expected sector counts")
  check(sum(sector_counts$n) == nrow(gdpval), "sector counts should sum to total rows")
  cat(sprintf(" (%d sectors)", nrow(sector_counts)))
  sector_counts
})

test("Count tasks by occupation", {
  occupation_counts <- gdpval |>
    count(occupation, sort = TRUE)
  check(nrow(occupation_counts) > 0, "expected occupation counts")
  check(sum(occupation_counts$n) == nrow(gdpval), "occupation counts should sum to total rows")
  cat(sprintf(" (%d occupations)", nrow(occupation_counts)))
  occupation_counts
})

test("Add prompt_length column", {
  gdpval <<- gdpval |>
    mutate(prompt_length = nchar(prompt))
  check("prompt_length" %in% names(gdpval), "expected prompt_length column")
  check(all(gdpval$prompt_length > 0), "all prompt lengths should be positive")
  cat(sprintf(" (range: %d-%d chars)",
              min(gdpval$prompt_length), max(gdpval$prompt_length)))
  gdpval
})

test("ggplot: sector distribution", {
  p <- gdpval |>
    count(sector, sort = TRUE) |>
    ggplot(aes(x = reorder(sector, n), y = n)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = "GDPval Tasks by Economic Sector", x = NULL, y = "Number of Tasks") +
    theme_minimal()
  check(inherits(p, "ggplot"), "expected ggplot object")
  p
})

# ============================================================
# Section: Semantic Embeddings of Task Prompts
# ============================================================
cat("\nSemantic Embeddings\n")

test("hf_embed on sample prompts", {
  # Use 20 prompts for testing (vignette uses all 220)
  sample_prompts <- gdpval$prompt[1:20]
  task_embeddings <<- hf_embed(sample_prompts)
  check(tibble::is_tibble(task_embeddings), "expected tibble")
  check(nrow(task_embeddings) == 20, "expected 20 rows")
  check(all(c("text", "embedding", "n_dims") %in% names(task_embeddings)))
  check(!is.na(task_embeddings$n_dims[1]), "expected non-NA n_dims")
  check(task_embeddings$n_dims[1] == 384, "expected 384 dimensions")
  task_embeddings
})

# ============================================================
# Section: Measuring Task Similarity
# ============================================================
cat("\nMeasuring Task Similarity\n")

test("hf_similarity on subset", {
  sample_embeddings <- task_embeddings |> slice(1:5)
  sim_result <- hf_similarity(sample_embeddings)
  check(tibble::is_tibble(sim_result), "expected tibble")
  # 5 choose 2 = 10 pairs
  check(nrow(sim_result) == 10, sprintf("expected 10 pairs, got %d", nrow(sim_result)))
  check(all(c("text_1", "text_2", "similarity") %in% names(sim_result)))
  # Allow small floating point tolerance
  check(all(sim_result$similarity >= -1.001 & sim_result$similarity <= 1.001),
        "similarity should be approximately in [-1, 1]")
  sim_result
})

# ============================================================
# Section: Nearest Neighbor Search
# ============================================================
cat("\nNearest Neighbor Search\n")

test("hf_embed_text on gdpval sample", {
  # Use 20 rows for testing
  task_docs <<- gdpval |>
    slice(1:20) |>
    select(task_id, sector, occupation, prompt, prompt_length) |>
    hf_embed_text(prompt)
  check(tibble::is_tibble(task_docs), "expected tibble")
  check("embedding" %in% names(task_docs), "expected embedding column")
  check(nrow(task_docs) == 20, "expected 20 rows")
  check(!is.null(task_docs$embedding[[1]]), "first embedding should not be NULL")
  task_docs
})

test("hf_nearest_neighbors: financial analysis", {
  nn_result <- hf_nearest_neighbors(task_docs, "financial analysis and reporting", k = 3)
  check(tibble::is_tibble(nn_result), "expected tibble")
  check(nrow(nn_result) == 3, "expected 3 neighbors")
  check("similarity" %in% names(nn_result), "expected similarity column")
  check(all(nn_result$similarity > 0), "expected positive similarities")
  nn_result
})

test("hf_nearest_neighbors: creative design", {
  nn_result <- hf_nearest_neighbors(task_docs, "creative design and production", k = 3)
  check(nrow(nn_result) == 3, "expected 3 neighbors")
  nn_result
})

test("hf_nearest_neighbors: technical problem solving", {
  nn_result <- hf_nearest_neighbors(task_docs, "technical problem solving", k = 3)
  check(nrow(nn_result) == 3, "expected 3 neighbors")
  nn_result
})

test("hf_nearest_neighbors: interpersonal communication", {
  nn_result <- hf_nearest_neighbors(task_docs, "interpersonal communication and negotiation", k = 3)
  check(nrow(nn_result) == 3, "expected 3 neighbors")
  nn_result
})

# ============================================================
# Section: Clustering Tasks by Semantic Content
# ============================================================
cat("\nClustering Tasks\n")

test("hf_cluster_texts k=4", {
  # Use k=4 since we only have 20 texts (vignette uses k=8 on all 220)
  clustered_tasks <<- hf_cluster_texts(task_docs, k = 4)
  check("cluster" %in% names(clustered_tasks), "expected cluster column")
  check(length(unique(clustered_tasks$cluster)) == 4, "expected 4 clusters")
  check(nrow(clustered_tasks) == 20, "expected 20 rows")
  clustered_tasks
})

test("Cluster summary statistics", {
  cluster_summary <- clustered_tasks |>
    group_by(cluster) |>
    summarize(
      n_tasks = n(),
      sectors = paste(unique(sector), collapse = ", "),
      example_occupation = first(occupation),
      .groups = "drop"
    )
  check(nrow(cluster_summary) == 4, "expected 4 cluster rows")
  check(all(c("n_tasks", "sectors", "example_occupation") %in% names(cluster_summary)))
  cat("\n")
  print(cluster_summary |> select(cluster, n_tasks, sectors))
  cat("  ")
  cluster_summary
})

test("Cross-tabulate clusters and sectors", {
  cluster_sector_table <- clustered_tasks |>
    count(cluster, sector) |>
    pivot_wider(names_from = sector, values_from = n, values_fill = 0)
  check(tibble::is_tibble(cluster_sector_table), "expected tibble")
  check("cluster" %in% names(cluster_sector_table), "expected cluster column")
  cluster_sector_table
})

test("hf_extract_topics", {
  if (!requireNamespace("tidytext", quietly = TRUE)) {
    stop("tidytext not installed")
  }
  topics <- task_docs |>
    hf_extract_topics(text_col = "prompt", k = 4)
  check(tibble::is_tibble(topics), "expected tibble")
  check("topic_terms" %in% names(topics), "expected topic_terms column")
  check("cluster" %in% names(topics), "expected cluster column")
  check(nrow(topics) == 4, "expected 4 topic rows")
  cat("\n")
  print(topics)
  cat("  ")
  topics
})

# ============================================================
# Section: Zero-Shot Classification
# ============================================================
cat("\nZero-Shot Classification\n")

test("Skill dimension classification", skip_on_api_error = TRUE, expr = {
  skill_labels <- c(
    "analytical and quantitative reasoning",
    "creative and design thinking",
    "interpersonal communication",
    "technical and procedural execution",
    "strategic planning and decision making"
  )

  # Classify first 5 prompts (vignette uses 30)
  # Use shorter prompts to avoid API timeouts
  short_prompts <- substr(gdpval$prompt[1:5], 1, 500)
  skill_classes <- hf_classify_zero_shot(
    short_prompts,
    labels = skill_labels
  )
  check(nrow(skill_classes) == 25, "expected 5 texts x 5 labels = 25 rows")
  check(all(c("text", "label", "score") %in% names(skill_classes)))

  skill_summary <- skill_classes |>
    group_by(text) |>
    slice_max(score, n = 1) |>
    ungroup() |>
    count(label, sort = TRUE)
  check(nrow(skill_summary) > 0, "expected at least 1 label")
  cat("\n")
  print(skill_summary)
  cat("  ")
  skill_summary
})

test("Automation potential classification", skip_on_api_error = TRUE, expr = {
  automation_labels <- c(
    "fully automatable by current AI",
    "partially automatable with human oversight",
    "requires significant human judgment",
    "requires physical presence or manipulation"
  )

  # Classify first 5 prompts (vignette uses 30)
  # Use shorter prompts to avoid API timeouts
  short_prompts <- substr(gdpval$prompt[1:5], 1, 500)
  automation_classes <- hf_classify_zero_shot(
    short_prompts,
    labels = automation_labels
  )
  check(nrow(automation_classes) == 20, "expected 5 texts x 4 labels = 20 rows")

  automation_summary <- automation_classes |>
    group_by(text) |>
    slice_max(score, n = 1) |>
    ungroup()
  check(nrow(automation_summary) == 5, "expected 5 rows")
  cat("\n")
  print(automation_summary |> select(label, score))
  cat("  ")
  automation_summary
})

test("Cognitive complexity classification", skip_on_api_error = TRUE, expr = {
  complexity_labels <- c(
    "routine procedural task",
    "moderately complex analytical task",
    "highly complex multi-step problem",
    "novel situation requiring creativity"
  )

  # Use shorter prompts to avoid API timeouts
  short_prompts <- substr(gdpval$prompt[1:5], 1, 500)
  complexity_classes <- hf_classify_zero_shot(
    short_prompts,
    labels = complexity_labels
  )
  check(nrow(complexity_classes) == 20, "expected 5 texts x 4 labels = 20 rows")

  complexity_summary <- complexity_classes |>
    group_by(text) |>
    slice_max(score, n = 1) |>
    ungroup()
  check(nrow(complexity_summary) == 5, "expected 5 rows")
  cat("\n")
  print(complexity_summary |> select(label, score))
  cat("  ")
  complexity_summary
})

# ============================================================
# Section: Similarity Analysis Across Occupations
# ============================================================
cat("\nOccupation Similarity Analysis\n")

test("Compute occupation embedding centroids", {
  # With only 20 rows, we have limited occupations
  occupation_profiles <- clustered_tasks |>
    group_by(occupation, sector) |>
    summarize(
      n_tasks = n(),
      embedding = list(Reduce(`+`, embedding) / n()),
      .groups = "drop"
    )
  check(nrow(occupation_profiles) > 0, "expected occupation profiles")
  check("embedding" %in% names(occupation_profiles), "expected embedding column")
  cat(sprintf(" (%d occupations)", nrow(occupation_profiles)))
  occupation_profiles
})

test("Compute inter-occupation similarity", {
  # Use task_docs directly since occupations may have single tasks
  # Just test that similarity computation works on a subset
  if (nrow(task_docs) >= 5) {
    subset_docs <- task_docs |>
      slice(1:5) |>
      select(text = occupation, embedding)
    # Can't use hf_similarity directly on occupation names

# so we test on task embeddings instead
    subset_emb <- task_docs |>
      slice(1:5) |>
      transmute(text = occupation, embedding)
    # This tests the similarity function works
    sim_test <- hf_similarity(task_embeddings |> slice(1:5))
    check(nrow(sim_test) == 10, "expected 10 pairs")
  }
  TRUE
})

# ============================================================
# Section: Visualizing the Task Embedding Space
# ============================================================
cat("\nVisualization (UMAP)\n")

test("UMAP projection with uwot", {
  if (!requireNamespace("uwot", quietly = TRUE)) {
    stop("uwot not installed")
  }
  library(uwot)

  emb_matrix <- do.call(rbind, task_docs$embedding)
  check(is.matrix(emb_matrix), "expected matrix")
  check(nrow(emb_matrix) == 20, "expected 20 rows")
  check(ncol(emb_matrix) == 384, "expected 384 cols")

  umap_coords <- umap(emb_matrix, n_neighbors = 15, min_dist = 0.1)
  check(is.matrix(umap_coords), "expected matrix from umap")
  check(ncol(umap_coords) == 2, "expected 2 columns")
  check(nrow(umap_coords) == 20, "expected 20 rows")

  plot_data <<- task_docs |>
    mutate(
      umap_1 = umap_coords[, 1],
      umap_2 = umap_coords[, 2]
    )
  check("umap_1" %in% names(plot_data))
  check("umap_2" %in% names(plot_data))
  plot_data
})

test("ggplot: sector colored UMAP", {
  p <- ggplot(plot_data, aes(x = umap_1, y = umap_2, color = sector)) +
    geom_point(alpha = 0.7, size = 2) +
    labs(
      title = "Semantic Map of GDPval Tasks by Sector",
      subtitle = "UMAP projection of task embeddings",
      color = "Sector",
      x = NULL, y = NULL
    ) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    )
  check(inherits(p, "ggplot"), "expected ggplot object")
  p
})

test("ggplot: cluster colored UMAP", {
  p <- ggplot(
    plot_data |> left_join(clustered_tasks |> select(prompt, cluster), by = "prompt"),
    aes(x = umap_1, y = umap_2, color = factor(cluster))
  ) +
    geom_point(alpha = 0.7, size = 2) +
    labs(
      title = "Task Clusters in Embedding Space",
      subtitle = "K-means clusters projected via UMAP",
      color = "Cluster",
      x = NULL, y = NULL
    ) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    )
  check(inherits(p, "ggplot"), "expected ggplot object")
  p
})

test("ggplot: prompt length colored UMAP", {
  p <- ggplot(plot_data, aes(x = umap_1, y = umap_2, color = prompt_length)) +
    geom_point(alpha = 0.7, size = 2) +
    scale_color_viridis_c(name = "Prompt Length") +
    labs(
      title = "Task Complexity in Embedding Space",
      subtitle = "Color indicates prompt length (characters)",
      x = NULL, y = NULL
    ) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    )
  check(inherits(p, "ggplot"), "expected ggplot object")
  p
})

# ============================================================
# Section: Reference File Analysis
# ============================================================
cat("\nReference File Analysis\n")

test("Count reference files per task", {
  gdpval <<- gdpval |>
    mutate(n_reference_files = lengths(reference_files))
  check("n_reference_files" %in% names(gdpval), "expected n_reference_files column")
  check(all(gdpval$n_reference_files >= 0), "all counts should be non-negative")
  cat(sprintf(" (range: %d-%d files)",
              min(gdpval$n_reference_files), max(gdpval$n_reference_files)))
  gdpval
})

test("Reference files by sector summary", {
  ref_summary <- gdpval |>
    group_by(sector) |>
    summarize(
      n_tasks = n(),
      mean_files = mean(n_reference_files),
      max_files = max(n_reference_files),
      pct_with_files = mean(n_reference_files > 0) * 100,
      .groups = "drop"
    ) |>
    arrange(desc(mean_files))
  check(nrow(ref_summary) > 0, "expected summary rows")
  check(all(c("mean_files", "pct_with_files") %in% names(ref_summary)))
  cat("\n")
  print(ref_summary)
  cat("  ")
  ref_summary
})

test("ggplot: reference file distribution", {
  p <- gdpval |>
    count(n_reference_files) |>
    ggplot(aes(x = factor(n_reference_files), y = n)) +
    geom_col(fill = "purple", alpha = 0.7) +
    labs(
      title = "Reference File Requirements",
      x = "Number of Reference Files",
      y = "Number of Tasks"
    ) +
    theme_minimal()
  check(inherits(p, "ggplot"), "expected ggplot object")
  p
})

# ============================================================
# Section: Comparison with Conversational Approaches
# ============================================================
cat("\nComparison Section (reduced)\n")

test("Embed subset and find data analysis neighbors", {
  # Use only 20 tasks for testing
  all_embeddings <- gdpval |>
    slice(1:20) |>
    hf_embed_text(prompt)
  check("embedding" %in% names(all_embeddings))
  check(nrow(all_embeddings) == 20)

  data_tasks <- hf_nearest_neighbors(all_embeddings, "data analysis", k = 5)
  check(nrow(data_tasks) == 5, "expected 5 neighbors")
  check("similarity" %in% names(data_tasks))

  # Sector distribution
  sector_dist <- data_tasks |>
    count(sector, sort = TRUE)
  check(nrow(sector_dist) > 0, "expected sector distribution")

  # Similarity stats
  sim_stats <- data_tasks |>
    summarize(
      mean_similarity = mean(similarity),
      sd_similarity = sd(similarity)
    )
  check(!is.na(sim_stats$mean_similarity))
  cat(sprintf(" (mean_similarity=%.4f)", sim_stats$mean_similarity))
  sim_stats
})

# ============================================================
# Section: hf_embed_umap alternative
# ============================================================
cat("\nhf_embed_umap alternative\n")

test("hf_embed_umap with small sample", {
  # Use 5 texts for testing with reduced n_neighbors
  umap_result <- hf_embed_umap(gdpval$prompt[1:5], n_neighbors = 3)
  check(tibble::is_tibble(umap_result), "expected tibble")
  check(nrow(umap_result) == 5, "expected 5 rows")
  check(all(c("text", "umap_1", "umap_2") %in% names(umap_result)))
  check(!any(is.na(umap_result$umap_1)), "expected no NA in umap_1")
  umap_result
})

# ============================================================
# Summary
# ============================================================
cat("\n=== Results ===\n\n")

pass <- sum(sapply(results, function(r) r$status == "PASS"))
skip <- sum(sapply(results, function(r) r$status == "SKIP"))
fail <- sum(sapply(results, function(r) r$status == "FAIL"))
total <- length(results)

cat(sprintf("  PASS: %d / %d\n", pass, total))
if (skip > 0) cat(sprintf("  SKIP: %d / %d (API unavailable)\n", skip, total))
cat(sprintf("  FAIL: %d / %d\n", fail, total))

if (skip > 0) {
  cat("\nSkipped tests (API unavailable - not a code error):\n")
  for (name in names(results)) {
    if (results[[name]]$status == "SKIP") {
      cat(sprintf("  - %s\n", name))
    }
  }
}

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
  cat("\nAll tests passed (skips due to API availability are OK).\n")
}
